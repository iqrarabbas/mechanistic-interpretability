import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenCache
from scripts.experiment32_multirandom_sae_subspace_test import PairedExternalCorruptionDataset
from scripts.experiment35_sae_guided_residual_adapter import (
    DEFAULT_ADAPTER,
    DEFAULT_CACHE_ROOT,
    DEFAULT_IMAGE_ROOT,
    SelectedSAEFeatures,
    feature_statistics,
)
from scripts.experiment36_sae_abnormality_gate import (
    FeatureGate,
    base_adapter,
    evaluate_condition,
    train_gate,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment37_gate_grid"
DISCOVERY_ROOT = (
    PROJECT_ROOT / "results" / "sae" / "experiment17_noise_bidirectional_repair"
    / "noise_bidirectional_development_gpu"
)
BLOCK_INDEX = 10


def ranked_features():
    summary = json.loads((DISCOVERY_ROOT / "summary.json").read_text())
    discovery = json.loads((DISCOVERY_ROOT / "discovery_candidates.json").read_text())
    ordered = list(summary["selected"]["over_features"])
    ordered.extend(
        row["feature_index"] for row in discovery["over"]
        if row["feature_index"] not in ordered
    )
    ordered.extend(
        row["feature_index"] for row in discovery["under"]
        if row["feature_index"] not in ordered
    )
    return ordered


def validation_metrics(model, gate, loader, device, scale):
    gate.max_scale = scale
    original_correct = []
    corrected = []
    clean_original = []
    clean_corrected = []
    margins = []
    with torch.no_grad():
        for clean_hidden, noise_hidden, labels in loader:
            clean_hidden = clean_hidden.to(device)
            noise_hidden = noise_hidden.to(device)
            labels = labels.to(device)
            clean_logits = downstream_from_layer(model, clean_hidden, BLOCK_INDEX)
            noise_logits = downstream_from_layer(model, noise_hidden, BLOCK_INDEX)
            candidate_noise = torch.cat([
                noise_hidden[:, :1], noise_hidden[:, 1:] + gate(noise_hidden[:, 1:])
            ], dim=1)
            candidate_clean = torch.cat([
                clean_hidden[:, :1], clean_hidden[:, 1:] + gate(clean_hidden[:, 1:])
            ], dim=1)
            logits = downstream_from_layer(model, candidate_noise, BLOCK_INDEX)
            clean_candidate_logits = downstream_from_layer(model, candidate_clean, BLOCK_INDEX)
            original_correct.extend((noise_logits.argmax(1) == labels).cpu().tolist())
            corrected.extend((logits.argmax(1) == labels).cpu().tolist())
            clean_original.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            clean_corrected.extend((clean_candidate_logits.argmax(1) == labels).cpu().tolist())
            margins.extend(
                (classification_margin(logits, labels)[1] - classification_margin(noise_logits, labels)[1]).cpu().tolist()
            )
    original = np.mean(original_correct)
    accuracy = np.mean(corrected)
    clean_base = np.mean(clean_original)
    clean_accuracy = np.mean(clean_corrected)
    clean_drop = max(0.0, clean_base - clean_accuracy)
    return {
        "noise4_accuracy": float(accuracy),
        "noise4_gain": float(accuracy - original),
        "clean_accuracy": float(clean_accuracy),
        "clean_gain": float(clean_accuracy - clean_base),
        "mean_margin_change": float(np.mean(margins)),
        "selection_score": float(accuracy - 2.0 * clean_drop),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 37: validation-only SAE gate grid")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--initial-adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[8, 16, 32, 64])
    parser.add_argument("--max-scales", type=float, nargs="+", default=[1.5, 2.0, 2.5, 3.0])
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
    parser.add_argument("--seed", type=int, default=37)
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
    ordered = ranked_features()
    if max(args.feature_counts) > len(ordered):
        raise ValueError(f"Requested {max(args.feature_counts)} features, only {len(ordered)} ranked candidates")
    gates = {}
    training = {}
    grid = {}
    validation_loader = DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False)
    for count in args.feature_counts:
        features = ordered[:count]
        mean, scale = feature_statistics(train_data, sae, features, device, args.predictor_batch_size)
        gate = FeatureGate(
            base_adapter(args.initial_adapter, device),
            SelectedSAEFeatures(sae, features, mean, scale),
        ).to(device)
        name = f"features_{count}"
        training[name] = train_gate(
            model, gate, train_data, validation_data, args, device, output_dir / f"{name}.pt"
        )
        gates[count] = gate
        for scale_value in args.max_scales:
            key = f"features{count}_scale{scale_value:g}"
            grid[key] = validation_metrics(model, gate, validation_loader, device, scale_value) | {
                "feature_count": count, "max_scale": scale_value
            }
            print(key, grid[key])
    selected_name = max(grid, key=lambda name: (grid[name]["selection_score"], grid[name]["mean_margin_change"]))
    selected = grid[selected_name]
    selected_gate = gates[selected["feature_count"]]
    selected_gate.max_scale = selected["max_scale"]
    methods = {
        "existing_adapter": base_adapter(args.initial_adapter, device),
        "selected_sae_gate": selected_gate,
    }
    evaluation = {}
    for index, (condition, corruption) in enumerate([("noise4", "noise"), ("blur4", "blur")]):
        dataset = PairedExternalCorruptionDataset(
            args.image_root, corruption, 4, args.samples, 0, args.noise_seed
        )
        evaluation[condition], arrays = evaluate_condition(
            model, methods,
            DataLoader(dataset, batch_size=args.image_batch_size, shuffle=False, num_workers=args.num_workers),
            device, args, index,
        )
        np.savez_compressed(output_dir / f"{condition}_outcomes.npz", **arrays)
    summary = {
        "configuration": vars(args) | {
            "cache_root": str(args.cache_root.resolve()),
            "initial_adapter": str(args.initial_adapter.resolve()),
            "image_root": str(args.image_root.resolve()),
            "device": str(device),
            "selection_data": "seed-2 ImageNet validation cache only; held-out ImageNetV2 untouched until final evaluation",
            "ordered_feature_dictionary": ordered[:max(args.feature_counts)],
        },
        "training": training,
        "validation_grid": grid,
        "selected": {"name": selected_name} | selected,
        "held_out_evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected": summary["selected"], "held_out_evaluation": evaluation}, indent=2))
    print(f"Saved Experiment 37 to {output_dir}")


if __name__ == "__main__":
    main()
