import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment24_classification_aware"
DEFAULT_INITIAL_CHECKPOINT = (
    PROJECT_ROOT / "results" / "sae" / "experiment23_noise_residual_predictor"
    / "full_noise_predictor" / "hidden_linear.pt"
)
BLOCK_INDEX = 10
TOKENS = 197
PATCHES = 196
WIDTH = 768


class HiddenLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)
        self.position = nn.Embedding(PATCHES, WIDTH)

    def forward(self, patches):
        positions = torch.arange(PATCHES, device=patches.device)
        return self.linear(patches) + self.position(positions)[None]


class HiddenCache(Dataset):
    def __init__(self, directory):
        metadata = json.loads((directory / "metadata.json").read_text())
        self.samples = metadata["samples"]
        shape = (self.samples, TOKENS, WIDTH)
        self.clean = np.memmap(directory / "clean.f16", dtype=np.float16, mode="r", shape=shape)
        self.noise = np.memmap(directory / "noise.f16", dtype=np.float16, mode="r", shape=shape)
        self.labels = np.load(directory / "labels.npy")

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        return (
            np.asarray(self.clean[index], dtype=np.float32),
            np.asarray(self.noise[index], dtype=np.float32),
            int(self.labels[index]),
        )


def make_image_loader(samples, start, args):
    return DataLoader(
        PairedCorruptionDataset("noise", samples, start, args.seed),
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def build_cache(model, loader, device, directory):
    directory.mkdir(parents=True, exist_ok=True)
    samples = len(loader.dataset)
    shape = (samples, TOKENS, WIDTH)
    clean_cache = np.memmap(directory / "clean.f16", dtype=np.float16, mode="w+", shape=shape)
    noise_cache = np.memmap(directory / "noise.f16", dtype=np.float16, mode="w+", shape=shape)
    labels_cache = np.empty(samples, dtype=np.int16)
    offset = 0
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc=f"Caching {directory.name}"):
            batch = clean.shape[0]
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            end = offset + batch
            clean_cache[offset:end] = clean_hidden.cpu().numpy().astype(np.float16)
            noise_cache[offset:end] = noise_hidden.cpu().numpy().astype(np.float16)
            labels_cache[offset:end] = labels.numpy().astype(np.int16)
            offset = end
    clean_cache.flush()
    noise_cache.flush()
    np.save(directory / "labels.npy", labels_cache)
    (directory / "metadata.json").write_text(json.dumps({
        "samples": samples,
        "shape": list(shape),
        "dtype": "float16",
        "layer": "ViT Block 11 output",
    }, indent=2))


def objective(model, predictor, clean_hidden, noise_hidden, labels, config):
    noise_patches = noise_hidden[:, 1:]
    target = clean_hidden[:, 1:] - noise_patches
    prediction = predictor(noise_patches)
    candidate = torch.cat(
        [noise_hidden[:, :1], noise_patches + config["train_alpha"] * prediction], dim=1
    )
    corrected_logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, noise_hidden, BLOCK_INDEX)
        baseline_correct = baseline_logits.argmax(1) == labels
        baseline_margin = classification_margin(baseline_logits, labels)[1]
    corrected_margin = classification_margin(corrected_logits, labels)[1]
    residual_loss = F.smooth_l1_loss(prediction, target, beta=config["smooth_l1_beta"])
    classification_loss = F.cross_entropy(corrected_logits, labels)
    if baseline_correct.any():
        preservation_loss = F.relu(
            baseline_margin[baseline_correct] - corrected_margin[baseline_correct]
        ).mean()
    else:
        preservation_loss = corrected_margin.new_zeros(())
    total = (
        residual_loss
        + config["classification_weight"] * classification_loss
        + config["preservation_weight"] * preservation_loss
    )
    return total, residual_loss, classification_loss, preservation_loss


def run_epoch(model, predictor, loader, device, config, optimizer=None):
    training = optimizer is not None
    predictor.train(training)
    totals = np.zeros(4, dtype=np.float64)
    samples = 0
    for clean_hidden, noise_hidden, labels in loader:
        clean_hidden = clean_hidden.to(device)
        noise_hidden = noise_hidden.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        losses = objective(model, predictor, clean_hidden, noise_hidden, labels, config)
        if training:
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)
            optimizer.step()
        batch = labels.shape[0]
        totals += np.asarray([float(loss.detach()) for loss in losses]) * batch
        samples += batch
    return {
        key: float(value / samples)
        for key, value in zip(
            ["total", "residual", "classification", "preservation"], totals
        )
    }


def train_variant(model, train_data, validation_data, args, device, output_dir, weight):
    predictor = HiddenLinear().to(device)
    initialization = getattr(args, "initialization", "checkpoint")
    if initialization == "checkpoint":
        if args.initial_checkpoint is None:
            raise ValueError("checkpoint initialization requires initial_checkpoint")
        predictor.load_state_dict(
            torch.load(args.initial_checkpoint, map_location=device, weights_only=True)
        )
    elif initialization == "zero":
        nn.init.zeros_(predictor.linear.weight)
        nn.init.zeros_(predictor.linear.bias)
        nn.init.zeros_(predictor.position.weight)
    elif initialization != "random":
        raise ValueError(f"Unknown initialization: {initialization}")
    config = {
        "initialization": initialization,
        "classification_weight": weight,
        "preservation_weight": weight * args.preservation_ratio,
        "train_alpha": args.train_alpha,
        "smooth_l1_beta": args.smooth_l1_beta,
    }
    optimizer = AdamW(predictor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(train_data, batch_size=args.predictor_batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False, num_workers=0)
    checkpoint = output_dir / f"classification_weight_{weight:g}.pt"
    best_loss = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, predictor, train_loader, device, config, optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(model, predictor, validation_loader, device, config)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"weight={weight:g} epoch={epoch} train={train_metrics['total']:.5f} "
            f"validation={validation_metrics['total']:.5f}"
        )
        if validation_metrics["total"] < best_loss:
            best_loss = validation_metrics["total"]
            stale = 0
            torch.save(predictor.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    predictor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    predictor.eval()
    return predictor, config | {"best_validation_total": best_loss, "history": history}


def summarize(original_correct, values):
    correct = np.asarray(values["correct"], dtype=bool)
    recovered = int((~original_correct & correct).sum())
    damaged = int((original_correct & ~correct).sum())
    return {
        "noise4_accuracy": float(correct.mean()),
        "noise4_accuracy_gain": float(correct.mean() - original_correct.mean()),
        "mean_margin_change": float(np.mean(values["margin_change"])),
        "mean_true_logit_change": float(np.mean(values["true_logit_change"])),
        "predictions_recovered": recovered,
        "originally_correct_damaged": damaged,
        "mcnemar_exact_pvalue": float(binomtest(recovered, recovered + damaged, 0.5).pvalue)
        if recovered + damaged else 1.0,
    }


def evaluate(model, predictors, loader, device, configurations, controls=False):
    stores = {
        name: {key: [] for key in ["correct", "margin_change", "true_logit_change"]}
        for name in configurations
    }
    if controls:
        for name in ["experiment23_residual_only", "shuffled_prediction", "reverse_prediction", "paired_clean_oracle", "residual_identity"]:
            stores[name] = {key: [] for key in ["correct", "margin_change", "true_logit_change"]}
    original_correct, clean_correct, selected_clean_correct, original_margin = [], [], [], []
    selected = next(iter(configurations.values())) if controls else None
    residual_only = None
    if controls:
        residual_only = HiddenLinear().to(device)
        residual_only.load_state_dict(
            torch.load(DEFAULT_INITIAL_CHECKPOINT, map_location=device, weights_only=True)
        )
        residual_only.eval()
    with torch.no_grad():
        for clean_hidden, noise_hidden, labels in tqdm(loader, desc="Classification-aware evaluation"):
            clean_hidden = clean_hidden.to(device)
            noise_hidden = noise_hidden.to(device)
            labels = labels.to(device)
            clean_logits = downstream_from_layer(model, clean_hidden, BLOCK_INDEX)
            noise_logits = downstream_from_layer(model, noise_hidden, BLOCK_INDEX)
            noise_true, noise_margin = classification_margin(noise_logits, labels)
            original_correct.extend((noise_logits.argmax(1) == labels).cpu().tolist())
            clean_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            original_margin.extend(noise_margin.cpu().tolist())
            noise_patches = noise_hidden[:, 1:]
            clean_patches = clean_hidden[:, 1:]
            predictions = {
                name: predictor(noise_patches) for name, predictor in predictors.items()
            }
            for name, config in configurations.items():
                residual = config["alpha"] * predictions[config["predictor"]]
                logits = downstream_from_layer(
                    model, torch.cat([noise_hidden[:, :1], noise_patches + residual], dim=1), BLOCK_INDEX
                )
                true_logit, margin = classification_margin(logits, labels)
                stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                stores[name]["margin_change"].extend((margin - noise_margin).cpu().tolist())
                stores[name]["true_logit_change"].extend((true_logit - noise_true).cpu().tolist())
            if controls:
                prediction = predictions[selected["predictor"]]
                control_residuals = {
                    "experiment23_residual_only": 0.5 * residual_only(noise_patches),
                    "shuffled_prediction": selected["alpha"] * prediction.roll(1, dims=0),
                    "reverse_prediction": -selected["alpha"] * prediction,
                    "paired_clean_oracle": clean_patches - noise_patches,
                    "residual_identity": torch.zeros_like(prediction),
                }
                for name, residual in control_residuals.items():
                    logits = downstream_from_layer(
                        model, torch.cat([noise_hidden[:, :1], noise_patches + residual], dim=1), BLOCK_INDEX
                    )
                    true_logit, margin = classification_margin(logits, labels)
                    stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                    stores[name]["margin_change"].extend((margin - noise_margin).cpu().tolist())
                    stores[name]["true_logit_change"].extend((true_logit - noise_true).cpu().tolist())
                clean_prediction = predictors[selected["predictor"]](clean_hidden[:, 1:])
                clean_candidate = torch.cat([
                    clean_hidden[:, :1], clean_hidden[:, 1:] + selected["alpha"] * clean_prediction
                ], dim=1)
                selected_clean_correct.extend(
                    (downstream_from_layer(model, clean_candidate, BLOCK_INDEX).argmax(1) == labels).cpu().tolist()
                )
    original_correct = np.asarray(original_correct, dtype=bool)
    results = {
        "original_vit": {
            "noise4_accuracy": float(original_correct.mean()),
            "clean_accuracy": float(np.mean(clean_correct)),
            "mean_noise4_margin": float(np.mean(original_margin)),
        }
    }
    results.update({name: summarize(original_correct, values) for name, values in stores.items()})
    if controls:
        selected_name = next(iter(configurations))
        results[selected_name]["clean_accuracy_if_applied"] = float(np.mean(selected_clean_correct))
        results[selected_name]["clean_accuracy_gain_if_applied"] = float(
            np.mean(selected_clean_correct) - np.mean(clean_correct)
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 24: classification-aware Noise-4 residual predictor")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--train-start", type=int, default=25000)
    parser.add_argument("--validation-start", type=int, default=30000)
    parser.add_argument("--evaluation-start", type=int, default=35000)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--classification-weights", type=float, nargs="+", default=[0.01, 0.05, 0.1])
    parser.add_argument("--preservation-ratio", type=float, default=1.0)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--predictor-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    for split, samples, start in [
        ("train", args.train_samples, args.train_start),
        ("validation", args.validation_samples, args.validation_start),
        ("evaluation", args.evaluation_samples, args.evaluation_start),
    ]:
        cache_dir = output_dir / f"{split}_cache"
        if not (args.resume and (cache_dir / "metadata.json").exists()):
            build_cache(model, make_image_loader(samples, start, args), device, cache_dir)
    train_data = HiddenCache(output_dir / "train_cache")
    validation_data = HiddenCache(output_dir / "validation_cache")
    evaluation_data = HiddenCache(output_dir / "evaluation_cache")
    predictors, training = {}, {}
    for weight in args.classification_weights:
        name = f"weight_{weight:g}"
        predictors[name], training[name] = train_variant(
            model, train_data, validation_data, args, device, output_dir, weight
        )
    candidates = {
        f"{name}_alpha{alpha:g}": {"predictor": name, "alpha": alpha}
        for name in predictors for alpha in args.alphas
    }
    validation = evaluate(
        model, predictors,
        DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False),
        device, candidates
    )
    selected_name = max(
        candidates,
        key=lambda name: (validation[name]["noise4_accuracy"], validation[name]["mean_margin_change"]),
    )
    selected = candidates[selected_name]
    evaluation = evaluate(
        model, predictors,
        DataLoader(evaluation_data, batch_size=args.predictor_batch_size, shuffle=False),
        device, {"selected_classification_aware": selected}, controls=True
    )
    summary = {
        "configuration": vars(args) | {
            "initial_checkpoint": str(args.initial_checkpoint), "device": str(device),
            "vit_block": 11, "frozen_components": ["ViT", "SAE"],
        },
        "predictor_parameters": sum(parameter.numel() for parameter in next(iter(predictors.values())).parameters()),
        "training": training,
        "selected": {"validation_name": selected_name} | selected,
        "validation": validation,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "selected": summary["selected"], "evaluation": evaluation,
    }, indent=2))
    print(f"Saved Experiment 24 to {output_dir}")


if __name__ == "__main__":
    main()
