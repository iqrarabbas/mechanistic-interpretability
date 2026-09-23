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
from scripts.experiment25_multiseed_independent_confirmation import paired_summary
from scripts.experiment31_sae_adapter_causal_mediation import FEATURE_SOURCE
from scripts.experiment32_multirandom_sae_subspace_test import PairedExternalCorruptionDataset
from scripts.experiment35_sae_guided_residual_adapter import (
    DEFAULT_ADAPTER,
    DEFAULT_CACHE_ROOT,
    DEFAULT_IMAGE_ROOT,
    SelectedSAEFeatures,
    feature_statistics,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment36_sae_abnormality_gate"
BLOCK_INDEX = 10
WIDTH = 768


class FeatureGate(nn.Module):
    def __init__(self, base_adapter, encoder, max_scale=2.0):
        super().__init__()
        self.base_adapter = base_adapter
        self.encoder = encoder
        self.gate = nn.Linear(encoder.weight.shape[0], 1)
        self.max_scale = max_scale
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.base_adapter.requires_grad_(False)
        self.encoder.requires_grad_(False)

    def forward(self, patches):
        abnormality = self.encoder.normalized(patches)
        return self.base_adapter(patches) * (
            self.max_scale * torch.sigmoid(self.gate(abnormality))
        )


class HiddenProjectionEncoder(nn.Module):
    def __init__(self, projection, mean, scale):
        super().__init__()
        self.register_buffer("weight", projection)
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale.clamp_min(1e-4))

    def normalized(self, patches):
        values = F.linear(patches, self.weight)
        return ((values - self.mean) / self.scale).clamp(-10, 10)


class ConstantGate(nn.Module):
    def __init__(self, base_adapter):
        super().__init__()
        self.base_adapter = base_adapter
        self.logit = nn.Parameter(torch.zeros(()))
        self.base_adapter.requires_grad_(False)

    def forward(self, patches):
        return self.base_adapter(patches) * (2.0 * torch.sigmoid(self.logit))


def base_adapter(path, device):
    adapter = HiddenLinear().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    adapter.eval().requires_grad_(False)
    return adapter


def hidden_statistics(dataset, projection, device, batch_size):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = torch.zeros(projection.shape[0], dtype=torch.float64, device=device)
    squared = torch.zeros_like(total)
    count = 0
    with torch.no_grad():
        for clean, _, _ in loader:
            values = F.linear(clean[:, 1:].to(device), projection).flatten(0, 1).double()
            total += values.sum(0)
            squared += values.square().sum(0)
            count += values.shape[0]
    mean = total / count
    scale = (squared / count - mean.square()).clamp_min(1e-8).sqrt()
    return mean.float(), scale.float()


def loss(model, gate, clean_hidden, noise_hidden, labels, args):
    noise_patches = noise_hidden[:, 1:]
    target = clean_hidden[:, 1:] - noise_patches
    prediction = gate(noise_patches)
    candidate = torch.cat([
        noise_hidden[:, :1], noise_patches + args.train_alpha * prediction
    ], dim=1)
    logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, noise_hidden, BLOCK_INDEX)
        baseline_correct = baseline_logits.argmax(1) == labels
        baseline_margin = classification_margin(baseline_logits, labels)[1]
    margin = classification_margin(logits, labels)[1]
    residual = F.smooth_l1_loss(prediction, target, beta=1.0)
    classification = F.cross_entropy(logits, labels)
    preservation = (
        F.relu(baseline_margin[baseline_correct] - margin[baseline_correct]).mean()
        if baseline_correct.any() else margin.new_zeros(())
    )
    total = residual + args.classification_weight * classification + args.preservation_weight * preservation
    return total, residual, classification, preservation


def epoch(model, gate, loader, device, args, optimizer=None):
    gate.train(optimizer is not None)
    gate.base_adapter.eval()
    totals = np.zeros(4)
    samples = 0
    for clean, noise, labels in loader:
        clean = clean.to(device)
        noise = noise.to(device)
        labels = labels.to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        losses = loss(model, gate, clean, noise, labels, args)
        if optimizer:
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_([p for p in gate.parameters() if p.requires_grad], 1.0)
            optimizer.step()
        batch = labels.shape[0]
        totals += np.asarray([float(item.detach()) for item in losses]) * batch
        samples += batch
    return dict(zip(["total", "residual", "classification", "preservation"], (totals / samples).tolist()))


def train_gate(model, gate, train_data, validation_data, args, device, checkpoint):
    parameters = [parameter for parameter in gate.parameters() if parameter.requires_grad]
    optimizer = AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(train_data, batch_size=args.predictor_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False)
    best = float("inf")
    stale = 0
    history = []
    for number in range(1, args.epochs + 1):
        train_metrics = epoch(model, gate, train_loader, device, args, optimizer)
        with torch.no_grad():
            validation_metrics = epoch(model, gate, validation_loader, device, args)
        scheduler.step()
        history.append({"epoch": number, "train": train_metrics, "validation": validation_metrics})
        print(f"{checkpoint.stem} epoch={number} validation={validation_metrics['total']:.5f}")
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            stale = 0
            torch.save(gate.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    gate.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    gate.eval()
    return {"best_validation_total": best, "history": history}


def evaluate_condition(model, methods, loader, device, args, condition_index):
    original_correct = []
    clean_original_correct = []
    stores = {name: {key: [] for key in ["correct", "clean", "margin", "true"]} for name in methods}
    with torch.no_grad():
        for clean, corrupt, labels in tqdm(loader, desc="Gating evaluation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(pixel_values=torch.cat([clean, corrupt]).to(device), output_hidden_states=True)
            clean_logits, corrupt_logits = outputs.logits.split(batch)
            clean_hidden, corrupt_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            original_true, original_margin = classification_margin(corrupt_logits, labels)
            original_correct.extend((corrupt_logits.argmax(1) == labels).cpu().tolist())
            clean_original_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            for name, method in methods.items():
                corrupt_patches = corrupt_hidden[:, 1:]
                clean_patches = clean_hidden[:, 1:]
                logits = downstream_from_layer(model, torch.cat([
                    corrupt_hidden[:, :1], corrupt_patches + args.alpha * method(corrupt_patches)
                ], dim=1), BLOCK_INDEX)
                clean_candidate = downstream_from_layer(model, torch.cat([
                    clean_hidden[:, :1], clean_patches + args.alpha * method(clean_patches)
                ], dim=1), BLOCK_INDEX)
                true_logit, margin = classification_margin(logits, labels)
                stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                stores[name]["clean"].extend((clean_candidate.argmax(1) == labels).cpu().tolist())
                stores[name]["margin"].extend((margin - original_margin).cpu().tolist())
                stores[name]["true"].extend((true_logit - original_true).cpu().tolist())
    original = np.asarray(original_correct, dtype=bool)
    clean_original = np.asarray(clean_original_correct, dtype=bool)
    results = {"baseline_accuracy": float(original.mean()), "clean_baseline_accuracy": float(clean_original.mean()), "methods": {}}
    arrays = {"original_correct": original, "clean_original_correct": clean_original}
    for index, (name, values) in enumerate(stores.items()):
        correct = np.asarray(values["correct"], dtype=bool)
        clean = np.asarray(values["clean"], dtype=bool)
        record = paired_summary(
            original, correct, np.asarray(values["margin"]), np.asarray(values["true"]),
            args.seed + condition_index * 100 + index, args.bootstrap_repetitions,
        )
        record |= {"clean_accuracy": float(clean.mean()), "clean_accuracy_gain": float(clean.mean() - clean_original.mean())}
        results["methods"][name] = record
        arrays[f"{name}_correct"] = correct
    return results, arrays


def main():
    parser = argparse.ArgumentParser(description="Experiment 36: pure SAE abnormality gating")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--initial-adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--feature-source", type=Path, default=FEATURE_SOURCE)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--feature-count", type=int, default=16)
    parser.add_argument("--random-controls", type=int, default=10)
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--predictor-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=36)
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
    sae = load_fixed_sae("clean", device).eval()
    sae.requires_grad_(False)
    train_data = HiddenCache(args.cache_root / "train_cache")
    validation_data = HiddenCache(args.cache_root / "validation_cache")
    if args.max_cache_samples:
        train_data = Subset(train_data, range(min(args.max_cache_samples, len(train_data))))
        validation_data = Subset(validation_data, range(min(args.max_cache_samples, len(validation_data))))
    harmful = json.loads(args.feature_source.read_text())["selected"]["over_features"][:args.feature_count]
    generator = np.random.default_rng(args.seed)
    available = np.setdiff1d(np.arange(sae.latent_dim), harmful)
    feature_sets = {"harmful_sae_gate": harmful}
    for index in range(args.random_controls):
        feature_sets[f"random_sae_gate_{index}"] = generator.choice(
            available, args.feature_count, replace=False
        ).tolist()
    methods = {"existing_adapter": base_adapter(args.initial_adapter, device)}
    training = {}
    for name, features in feature_sets.items():
        mean, scale = feature_statistics(train_data, sae, features, device, args.predictor_batch_size)
        gate = FeatureGate(
            base_adapter(args.initial_adapter, device),
            SelectedSAEFeatures(sae, features, mean, scale),
        ).to(device)
        training[name] = train_gate(model, gate, train_data, validation_data, args, device, output_dir / f"{name}.pt")
        methods[name] = gate
    projection = F.normalize(torch.randn(args.feature_count, WIDTH, device=device), dim=1)
    hidden_mean, hidden_scale = hidden_statistics(train_data, projection, device, args.predictor_batch_size)
    hidden_gate = FeatureGate(
        base_adapter(args.initial_adapter, device),
        HiddenProjectionEncoder(projection, hidden_mean, hidden_scale),
    ).to(device)
    training["hidden_projection_gate"] = train_gate(
        model, hidden_gate, train_data, validation_data, args, device, output_dir / "hidden_projection_gate.pt"
    )
    methods["hidden_projection_gate"] = hidden_gate
    constant_gate = ConstantGate(base_adapter(args.initial_adapter, device)).to(device)
    training["constant_gate"] = train_gate(
        model, constant_gate, train_data, validation_data, args, device, output_dir / "constant_gate.pt"
    )
    methods["constant_gate"] = constant_gate
    conditions = [("noise4", "noise"), ("blur4", "blur")]
    evaluation = {}
    for condition_index, (condition, corruption) in enumerate(conditions):
        dataset = PairedExternalCorruptionDataset(
            args.image_root, corruption, 4, args.samples, 0, args.noise_seed
        )
        evaluation[condition], arrays = evaluate_condition(
            model, methods,
            DataLoader(dataset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.num_workers),
            device, args, condition_index,
        )
        np.savez_compressed(output_dir / f"{condition}_outcomes.npz", **arrays)
    comparison = {}
    for condition in evaluation:
        records = evaluation[condition]["methods"]
        harmful_gain = records["harmful_sae_gate"]["accuracy_gain"]
        random_gains = np.asarray([
            records[f"random_sae_gate_{index}"]["accuracy_gain"]
            for index in range(args.random_controls)
        ])
        comparison[condition] = {
            "existing_gain": records["existing_adapter"]["accuracy_gain"],
            "harmful_gate_gain": harmful_gain,
            "harmful_beats_existing": bool(harmful_gain > records["existing_adapter"]["accuracy_gain"]),
        }
        if random_gains.size:
            comparison[condition] |= {
                "random_gate_gain_mean": float(random_gains.mean()),
                "random_gate_gain_range": [float(random_gains.min()), float(random_gains.max())],
                "harmful_percentile_among_random": float(np.mean(random_gains < harmful_gain)),
                "harmful_beats_all_random": bool(harmful_gain > random_gains.max()),
            }
    summary = {
        "configuration": vars(args) | {
            "cache_root": str(args.cache_root.resolve()),
            "initial_adapter": str(args.initial_adapter.resolve()),
            "feature_source": str(args.feature_source.resolve()),
            "image_root": str(args.image_root.resolve()),
            "device": str(device),
            "harmful_features": harmful,
            "gate_trainable_parameters": 17,
            "frozen_components": ["ViT", "SAE", "existing residual adapter"],
            "no_added_residual_projection": True,
            "no_sae_alignment_loss": True,
        },
        "training": training,
        "evaluation": evaluation,
        "comparison": comparison,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"comparison": comparison}, indent=2))
    print(f"Saved Experiment 36 to {output_dir}")


if __name__ == "__main__":
    main()
