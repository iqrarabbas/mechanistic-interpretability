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

from scripts.compare_sae_level4 import load_sae
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment23_noise_residual_predictor"
DEFAULT_FEATURE_FILE = (
    PROJECT_ROOT / "results" / "sae" / "experiment22_noise_nearest_neighbor"
    / "full_noise_neighbors" / "key_features.npy"
)
BLOCK_INDEX = 10
PATCHES = 196
WIDTH = 768


class CachedPatches(Dataset):
    def __init__(self, directory):
        metadata = json.loads((directory / "metadata.json").read_text())
        shape = (metadata["samples"], metadata["patches_per_image"])
        self.sae = np.memmap(
            directory / "sae.f16", dtype=np.float16, mode="r",
            shape=shape + (metadata["sae_features"],),
        )
        self.hidden = np.memmap(
            directory / "hidden.f16", dtype=np.float16, mode="r", shape=shape + (WIDTH,)
        )
        self.cls = np.memmap(
            directory / "cls.f16", dtype=np.float16, mode="r", shape=(metadata["samples"], WIDTH)
        )
        self.target = np.memmap(
            directory / "target.f16", dtype=np.float16, mode="r", shape=shape + (WIDTH,)
        )
        self.positions = np.load(directory / "positions.npy")

    def __len__(self):
        return self.sae.shape[0]

    def __getitem__(self, index):
        return (
            np.asarray(self.sae[index], dtype=np.float32),
            np.asarray(self.hidden[index], dtype=np.float32),
            np.asarray(self.cls[index], dtype=np.float32),
            np.asarray(self.positions[index], dtype=np.int64),
            np.asarray(self.target[index], dtype=np.float32),
        )


class SAELinear(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.linear = nn.Linear(feature_count, WIDTH)
        self.position = nn.Embedding(PATCHES, WIDTH)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.position.weight)

    def forward(self, sae, hidden, cls, positions):
        return self.linear(sae) + self.position(positions)


class HiddenLinear(nn.Module):
    def __init__(self, feature_count):
        super().__init__()
        self.linear = nn.Linear(WIDTH, WIDTH)
        self.position = nn.Embedding(PATCHES, WIDTH)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.position.weight)

    def forward(self, sae, hidden, cls, positions):
        return self.linear(hidden) + self.position(positions)


class ConditionedMLP(nn.Module):
    def __init__(self, feature_count, bottleneck):
        super().__init__()
        self.position = nn.Embedding(PATCHES, 32)
        self.input = nn.Linear(feature_count + WIDTH + 32, bottleneck)
        self.output = nn.Linear(bottleneck, WIDTH)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, sae, hidden, cls, positions):
        repeated_cls = cls[:, None, :].expand(-1, sae.shape[1], -1)
        inputs = torch.cat([sae, repeated_cls, self.position(positions)], dim=-1)
        return self.output(F.gelu(self.input(inputs)))


def make_predictor(name, feature_count, bottleneck):
    if name == "sae_linear":
        return SAELinear(feature_count)
    if name == "hidden_linear":
        return HiddenLinear(feature_count)
    if name == "conditioned_mlp":
        return ConditionedMLP(feature_count, bottleneck)
    raise ValueError(name)


def make_image_loader(samples, start, args):
    return DataLoader(
        PairedCorruptionDataset("noise", samples, start, args.seed),
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def build_cache(model, sae, loader, device, directory, features, patches_per_image, seed):
    directory.mkdir(parents=True, exist_ok=True)
    samples = len(loader.dataset)
    shape = (samples, patches_per_image)
    arrays = {
        "sae": np.memmap(directory / "sae.f16", dtype=np.float16, mode="w+", shape=shape + (len(features),)),
        "hidden": np.memmap(directory / "hidden.f16", dtype=np.float16, mode="w+", shape=shape + (WIDTH,)),
        "cls": np.memmap(directory / "cls.f16", dtype=np.float16, mode="w+", shape=(samples, WIDTH)),
        "target": np.memmap(directory / "target.f16", dtype=np.float16, mode="w+", shape=shape + (WIDTH,)),
    }
    positions = np.empty(shape, dtype=np.int16)
    generator = torch.Generator().manual_seed(seed)
    feature_tensor = torch.as_tensor(features, dtype=torch.long, device=device)
    offset = 0
    with torch.no_grad():
        for clean, noise, _ in tqdm(loader, desc=f"Caching {directory.name}"):
            batch = clean.shape[0]
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            noise_patches = noise_hidden[:, 1:]
            latent = sae.encode(noise_patches.flatten(0, 1)).reshape(batch, PATCHES, -1)
            batch_positions = torch.stack([
                torch.randperm(PATCHES, generator=generator)[:patches_per_image]
                for _ in range(batch)
            ]).to(device)
            batch_index = torch.arange(batch, device=device)[:, None]
            end = offset + batch
            arrays["sae"][offset:end] = latent[batch_index, batch_positions][..., feature_tensor].cpu().numpy().astype(np.float16)
            arrays["hidden"][offset:end] = noise_patches[batch_index, batch_positions].cpu().numpy().astype(np.float16)
            arrays["cls"][offset:end] = noise_hidden[:, 0].cpu().numpy().astype(np.float16)
            arrays["target"][offset:end] = (
                clean_hidden[:, 1:][batch_index, batch_positions]
                - noise_patches[batch_index, batch_positions]
            ).cpu().numpy().astype(np.float16)
            positions[offset:end] = batch_positions.cpu().numpy().astype(np.int16)
            offset = end
    for array in arrays.values():
        array.flush()
    np.save(directory / "positions.npy", positions)
    metadata = {
        "samples": samples,
        "patches_per_image": patches_per_image,
        "sae_features": len(features),
        "target": "clean Block-11 patch minus Noise-4 Block-11 patch",
    }
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=2))


def predictor_loss(predictor, loader, device, loss_name, smooth_l1_beta, optimizer=None):
    training = optimizer is not None
    predictor.train(training)
    total_loss = 0.0
    observations = 0
    for sae, hidden, cls, positions, target in loader:
        sae, hidden, cls, positions, target = (
            sae.to(device), hidden.to(device), cls.to(device), positions.to(device), target.to(device)
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        prediction = predictor(sae, hidden, cls, positions)
        loss = (
            F.smooth_l1_loss(prediction, target, beta=smooth_l1_beta)
            if loss_name == "smooth_l1"
            else F.mse_loss(prediction, target)
        )
        if training:
            loss.backward()
            optimizer.step()
        total_loss += float(loss.detach()) * target.numel()
        observations += target.numel()
    return total_loss / observations


def train_predictor(name, train_data, validation_data, args, device, output_dir, feature_count):
    predictor = make_predictor(name, feature_count, args.bottleneck).to(device)
    optimizer = AdamW(predictor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(train_data, batch_size=args.predictor_batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False, num_workers=0)
    best_loss = float("inf")
    stale = 0
    history = []
    checkpoint = output_dir / f"{name}.pt"
    for epoch in range(1, args.epochs + 1):
        train_loss = predictor_loss(
            predictor, train_loader, device, args.loss, args.smooth_l1_beta, optimizer
        )
        with torch.no_grad():
            validation_loss = predictor_loss(
                predictor, validation_loader, device, args.loss, args.smooth_l1_beta
            )
        scheduler.step()
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": validation_loss})
        print(f"{name} epoch {epoch}: train={train_loss:.6f} validation={validation_loss:.6f}")
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale = 0
            torch.save(predictor.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    predictor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    predictor.eval()
    return predictor, {
        "objective": args.loss,
        "best_validation_loss": best_loss,
        "history": history,
    }


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


def evaluate(model, sae, predictors, loader, device, features, configurations, controls=False):
    stores = {
        name: {key: [] for key in ["correct", "margin_change", "true_logit_change"]}
        for name in configurations
    }
    if controls:
        for name in ["shuffled_prediction", "reverse_prediction", "paired_clean_oracle", "residual_identity"]:
            stores[name] = {key: [] for key in ["correct", "margin_change", "true_logit_change"]}
    original_correct, clean_correct = [], []
    original_margin = []
    selected_clean_correct = []
    feature_tensor = torch.as_tensor(features, dtype=torch.long, device=device)
    selected = next(iter(configurations.values())) if controls else None
    positions = torch.arange(PATCHES, device=device)
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Causal evaluation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            noise_true, noise_margin = classification_margin(noise_logits, labels)
            original_correct.extend((noise_logits.argmax(1) == labels).cpu().tolist())
            clean_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            original_margin.extend(noise_margin.cpu().tolist())
            noise_patches = noise_hidden[:, 1:]
            clean_patches = clean_hidden[:, 1:]
            noise_sae = sae.encode(noise_patches.flatten(0, 1)).reshape(batch, PATCHES, -1)[..., feature_tensor]
            batch_positions = positions[None].expand(batch, -1)
            predictions = {
                name: predictor(noise_sae, noise_patches, noise_hidden[:, 0], batch_positions)
                for name, predictor in predictors.items()
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
                residuals = {
                    "shuffled_prediction": selected["alpha"] * prediction.roll(1, dims=0),
                    "reverse_prediction": -selected["alpha"] * prediction,
                    "paired_clean_oracle": clean_patches - noise_patches,
                    "residual_identity": torch.zeros_like(prediction),
                }
                for name, residual in residuals.items():
                    logits = downstream_from_layer(
                        model, torch.cat([noise_hidden[:, :1], noise_patches + residual], dim=1), BLOCK_INDEX
                    )
                    true_logit, margin = classification_margin(logits, labels)
                    stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                    stores[name]["margin_change"].extend((margin - noise_margin).cpu().tolist())
                    stores[name]["true_logit_change"].extend((true_logit - noise_true).cpu().tolist())
                clean_sae = sae.encode(clean_patches.flatten(0, 1)).reshape(batch, PATCHES, -1)[..., feature_tensor]
                clean_prediction = predictors[selected["predictor"]](
                    clean_sae, clean_patches, clean_hidden[:, 0], batch_positions
                )
                clean_candidate = torch.cat(
                    [clean_hidden[:, :1], clean_patches + selected["alpha"] * clean_prediction], dim=1
                )
                clean_candidate_logits = downstream_from_layer(model, clean_candidate, BLOCK_INDEX)
                selected_clean_correct.extend((clean_candidate_logits.argmax(1) == labels).cpu().tolist())
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
    parser = argparse.ArgumentParser(description="Experiment 23: lightweight Noise-4 patch residual predictor")
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--train-start", type=int, default=25000)
    parser.add_argument("--validation-start", type=int, default=30000)
    parser.add_argument("--evaluation-start", type=int, default=35000)
    parser.add_argument("--patches-per-image", type=int, default=32)
    parser.add_argument("--feature-file", type=Path, default=DEFAULT_FEATURE_FILE)
    parser.add_argument("--architectures", nargs="+", default=["sae_linear", "hidden_linear", "conditioned_mlp"])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--bottleneck", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss", choices=["smooth_l1", "mse"], default="smooth_l1")
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--predictor-batch-size", type=int, default=16)
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
    features = np.load(args.feature_file)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_sae("noise", "base", "vanilla", device)
    for split, samples, start in [
        ("train", args.train_samples, args.train_start),
        ("validation", args.validation_samples, args.validation_start),
    ]:
        cache_dir = output_dir / f"{split}_cache"
        if not (args.resume and (cache_dir / "metadata.json").exists()):
            build_cache(
                model, sae, make_image_loader(samples, start, args), device,
                cache_dir, features, args.patches_per_image, args.seed + (split == "validation")
            )
    train_data = CachedPatches(output_dir / "train_cache")
    validation_data = CachedPatches(output_dir / "validation_cache")
    predictors, training = {}, {}
    for architecture in args.architectures:
        predictors[architecture], training[architecture] = train_predictor(
            architecture, train_data, validation_data, args, device, output_dir, len(features)
        )
    candidates = {
        f"{architecture}_alpha{alpha:g}": {"predictor": architecture, "alpha": alpha}
        for architecture in args.architectures for alpha in args.alphas
    }
    validation = evaluate(
        model, sae, predictors, make_image_loader(args.validation_samples, args.validation_start, args),
        device, features, candidates
    )
    selected_name = max(
        candidates,
        key=lambda name: (validation[name]["noise4_accuracy"], validation[name]["mean_margin_change"]),
    )
    selected = candidates[selected_name]
    evaluation = evaluate(
        model, sae, predictors, make_image_loader(args.evaluation_samples, args.evaluation_start, args),
        device, features, {"selected_predictor": selected}, controls=True
    )
    summary = {
        "configuration": vars(args) | {
            "feature_file": str(args.feature_file), "device": str(device), "vit_block": 11,
            "frozen_components": ["ViT", "SAE"],
        },
        "predictor_parameter_counts": {
            name: sum(parameter.numel() for parameter in predictor.parameters())
            for name, predictor in predictors.items()
        },
        "training": training,
        "selected": {"validation_name": selected_name} | selected,
        "validation": validation,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "parameter_counts": summary["predictor_parameter_counts"],
        "selected": summary["selected"], "evaluation": evaluation,
    }, indent=2))
    print(f"Saved Experiment 23 to {output_dir}")


if __name__ == "__main__":
    main()
