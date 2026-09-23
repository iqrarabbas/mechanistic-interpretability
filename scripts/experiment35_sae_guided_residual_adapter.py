import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenCache, HiddenLinear
from scripts.experiment25_multiseed_independent_confirmation import PairedExternalDataset, paired_summary
from scripts.experiment31_sae_adapter_causal_mediation import FEATURE_SOURCE


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment35_sae_guided_adapter"
DEFAULT_CACHE_ROOT = (
    PROJECT_ROOT / "results" / "sae" / "experiment25_multiseed_confirmation"
    / "imagenetv2_multiseed_confirmation" / "seed_2"
)
DEFAULT_ADAPTER = DEFAULT_CACHE_ROOT / "classification_weight_0.05.pt"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
BLOCK_INDEX = 10
PATCHES = 196
WIDTH = 768


class SelectedSAEFeatures(nn.Module):
    def __init__(self, sae, features, mean, scale):
        super().__init__()
        selected = torch.as_tensor(features, dtype=torch.long, device=sae.encoder.weight.device)
        self.register_buffer("weight", sae.encoder.weight[selected].detach().clone())
        self.register_buffer("bias", sae.encoder.bias[selected].detach().clone())
        self.register_buffer("mean", mean.detach().clone())
        self.register_buffer("scale", scale.detach().clone().clamp_min(1e-4))

    def activations(self, patches):
        return F.relu(F.linear(patches, self.weight, self.bias))

    def normalized(self, patches):
        return ((self.activations(patches) - self.mean) / self.scale).clamp(-10, 10)


class SAEGuidedAdapter(nn.Module):
    def __init__(self, base_adapter, feature_encoder):
        super().__init__()
        self.base_adapter = base_adapter
        self.feature_encoder = feature_encoder
        feature_count = feature_encoder.weight.shape[0]
        self.feature_projection = nn.Linear(feature_count, WIDTH)
        self.feature_gate = nn.Linear(feature_count, 1)
        nn.init.zeros_(self.feature_projection.weight)
        nn.init.zeros_(self.feature_projection.bias)
        nn.init.zeros_(self.feature_gate.weight)
        nn.init.zeros_(self.feature_gate.bias)
        self.base_adapter.requires_grad_(False)
        self.feature_encoder.requires_grad_(False)

    def forward(self, patches):
        abnormality = self.feature_encoder.normalized(patches)
        base = self.base_adapter(patches)
        gate = 2.0 * torch.sigmoid(self.feature_gate(abnormality))
        return gate * base + self.feature_projection(abnormality)


def feature_statistics(dataset, sae, features, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    selected = torch.as_tensor(features, dtype=torch.long, device=device)
    total = torch.zeros(len(features), dtype=torch.float64, device=device)
    squared = torch.zeros_like(total)
    count = 0
    with torch.no_grad():
        for clean_hidden, _, _ in loader:
            patches = clean_hidden[:, 1:].to(device)
            values = sae.encode(patches.flatten(0, 1))[:, selected].double()
            total += values.sum(0)
            squared += values.square().sum(0)
            count += values.shape[0]
    mean = total / count
    variance = (squared / count - mean.square()).clamp_min(1e-8)
    return mean.float(), variance.sqrt().float()


def objective(model, adapter, clean_hidden, noise_hidden, labels, args):
    clean_patches = clean_hidden[:, 1:]
    noise_patches = noise_hidden[:, 1:]
    target = clean_patches - noise_patches
    prediction = adapter(noise_patches)
    corrected_patches = noise_patches + args.train_alpha * prediction
    candidate = torch.cat([noise_hidden[:, :1], corrected_patches], dim=1)
    corrected_logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, noise_hidden, BLOCK_INDEX)
        baseline_correct = baseline_logits.argmax(1) == labels
        baseline_margin = classification_margin(baseline_logits, labels)[1]
        clean_features = adapter.feature_encoder.activations(clean_patches)
    corrected_margin = classification_margin(corrected_logits, labels)[1]
    residual_loss = F.smooth_l1_loss(prediction, target, beta=1.0)
    classification_loss = F.cross_entropy(corrected_logits, labels)
    corrected_features = adapter.feature_encoder.activations(corrected_patches)
    feature_loss = F.smooth_l1_loss(corrected_features, clean_features, beta=1.0)
    preservation = (
        F.relu(baseline_margin[baseline_correct] - corrected_margin[baseline_correct]).mean()
        if baseline_correct.any() else corrected_margin.new_zeros(())
    )
    total = (
        residual_loss
        + args.classification_weight * classification_loss
        + args.preservation_weight * preservation
        + args.feature_weight * feature_loss
    )
    return total, residual_loss, classification_loss, preservation, feature_loss


def run_epoch(model, adapter, loader, device, args, optimizer=None):
    adapter.train(optimizer is not None)
    adapter.base_adapter.eval()
    totals = np.zeros(5)
    samples = 0
    for clean_hidden, noise_hidden, labels in loader:
        clean_hidden = clean_hidden.to(device)
        noise_hidden = noise_hidden.to(device)
        labels = labels.to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        losses = objective(model, adapter, clean_hidden, noise_hidden, labels, args)
        if optimizer:
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
        batch = labels.shape[0]
        totals += np.asarray([float(loss.detach()) for loss in losses]) * batch
        samples += batch
    return dict(zip(
        ["total", "residual", "classification", "preservation", "feature"],
        (totals / samples).tolist(),
    ))


def train_variant(model, train_data, validation_data, sae, features, args, device, output_dir, name):
    mean, scale = feature_statistics(train_data, sae, features, device, args.predictor_batch_size)
    base = HiddenLinear().to(device)
    base.load_state_dict(torch.load(args.initial_adapter, map_location=device, weights_only=True))
    adapter = SAEGuidedAdapter(base, SelectedSAEFeatures(sae, features, mean, scale)).to(device)
    trainable = [parameter for parameter in adapter.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(train_data, batch_size=args.predictor_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False)
    checkpoint = output_dir / f"{name}.pt"
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, adapter, train_loader, device, args, optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(model, adapter, validation_loader, device, args)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(f"{name} epoch={epoch} validation={validation_metrics['total']:.5f}")
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            stale = 0
            torch.save(adapter.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    adapter.eval()
    return adapter, {"best_validation_total": best, "history": history}


def evaluate(model, adapters, loader, device, alpha, seed, repetitions):
    baseline_correct = []
    clean_correct = []
    stores = {name: {key: [] for key in ["correct", "clean_correct", "margin", "true"]} for name in adapters}
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Held-out SAE-guided evaluation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True)
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            noise_true, noise_margin = classification_margin(noise_logits, labels)
            baseline_correct.extend((noise_logits.argmax(1) == labels).cpu().tolist())
            clean_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            for name, adapter in adapters.items():
                noise_patches = noise_hidden[:, 1:]
                clean_patches = clean_hidden[:, 1:]
                noise_candidate = torch.cat([
                    noise_hidden[:, :1], noise_patches + alpha * adapter(noise_patches)
                ], dim=1)
                clean_candidate = torch.cat([
                    clean_hidden[:, :1], clean_patches + alpha * adapter(clean_patches)
                ], dim=1)
                logits = downstream_from_layer(model, noise_candidate, BLOCK_INDEX)
                clean_candidate_logits = downstream_from_layer(model, clean_candidate, BLOCK_INDEX)
                true_logit, margin = classification_margin(logits, labels)
                stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                stores[name]["clean_correct"].extend((clean_candidate_logits.argmax(1) == labels).cpu().tolist())
                stores[name]["margin"].extend((margin - noise_margin).cpu().tolist())
                stores[name]["true"].extend((true_logit - noise_true).cpu().tolist())
    baseline = np.asarray(baseline_correct, dtype=bool)
    clean = np.asarray(clean_correct, dtype=bool)
    results = {"baseline": {"noise4_accuracy": float(baseline.mean()), "clean_accuracy": float(clean.mean())}}
    arrays = {"baseline_correct": baseline, "clean_correct": clean}
    for index, (name, values) in enumerate(stores.items()):
        correct = np.asarray(values["correct"], dtype=bool)
        clean_variant = np.asarray(values["clean_correct"], dtype=bool)
        result = paired_summary(
            baseline, correct, np.asarray(values["margin"]), np.asarray(values["true"]),
            seed + index, repetitions,
        )
        result |= {
            "clean_accuracy": float(clean_variant.mean()),
            "clean_accuracy_gain": float(clean_variant.mean() - clean.mean()),
        }
        results[name] = result
        arrays[f"{name}_correct"] = correct
        arrays[f"{name}_clean_correct"] = clean_variant
    return results, arrays


def main():
    parser = argparse.ArgumentParser(description="Experiment 35: SAE-guided residual adapter")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--initial-adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--feature-source", type=Path, default=FEATURE_SOURCE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--feature-count", type=int, default=16)
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--feature-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--predictor-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--independent-samples", type=int, default=10000)
    parser.add_argument("--independent-start", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=35)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--max-cache-samples", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)
    sae.requires_grad_(False)
    source = json.loads(args.feature_source.read_text())
    harmful = source["selected"]["over_features"][:args.feature_count]
    generator = np.random.default_rng(args.seed)
    available = np.setdiff1d(np.arange(sae.latent_dim), harmful)
    random_features = generator.choice(available, args.feature_count, replace=False).tolist()
    train_data = HiddenCache(args.cache_root / "train_cache")
    validation_data = HiddenCache(args.cache_root / "validation_cache")
    if args.max_cache_samples:
        train_data = Subset(train_data, range(min(args.max_cache_samples, len(train_data))))
        validation_data = Subset(
            validation_data, range(min(args.max_cache_samples, len(validation_data)))
        )
    adapters = {}
    training = {}
    for name, features in {"harmful_sae_guided": harmful, "random_sae_guided": random_features}.items():
        adapters[name], training[name] = train_variant(
            model, train_data, validation_data, sae, features, args, device, output_dir, name
        )
    baseline_adapter = HiddenLinear().to(device)
    baseline_adapter.load_state_dict(torch.load(args.initial_adapter, map_location=device, weights_only=True))
    baseline_adapter.eval()
    dataset = PairedExternalDataset(
        args.image_root, args.independent_samples, args.independent_start, args.noise_seed
    )
    results, arrays = evaluate(
        model,
        {"existing_hidden_adapter": baseline_adapter} | adapters,
        DataLoader(dataset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.num_workers),
        device,
        args.alpha,
        args.noise_seed,
        args.bootstrap_repetitions,
    )
    np.savez_compressed(output_dir / "paired_outcomes.npz", **arrays)
    guided_gain = results["harmful_sae_guided"]["accuracy_gain"]
    existing_gain = results["existing_hidden_adapter"]["accuracy_gain"]
    random_gain = results["random_sae_guided"]["accuracy_gain"]
    summary = {
        "configuration": vars(args) | {
            "cache_root": str(args.cache_root.resolve()),
            "initial_adapter": str(args.initial_adapter.resolve()),
            "feature_source": str(args.feature_source.resolve()),
            "image_root": str(args.image_root.resolve()),
            "device": str(device),
            "harmful_features": harmful,
            "random_features": random_features,
            "frozen_components": ["ViT", "SAE", "existing residual adapter"],
            "trainable_added_parameters": sum(
                parameter.numel() for parameter in adapters["harmful_sae_guided"].parameters()
                if parameter.requires_grad
            ),
        },
        "training": training,
        "independent_evaluation": results,
        "comparison": {
            "harmful_guided_minus_existing_gain": float(guided_gain - existing_gain),
            "harmful_guided_minus_random_guided_gain": float(guided_gain - random_gain),
            "sae_guidance_supported": bool(guided_gain > existing_gain and guided_gain > random_gain),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"independent_evaluation": results, "comparison": summary["comparison"]}, indent=2))
    print(f"Saved Experiment 35 to {output_dir}")


if __name__ == "__main__":
    main()
