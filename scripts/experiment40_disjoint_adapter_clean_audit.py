import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenCache, HiddenLinear
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment40_disjoint_adapter_clean_audit"
DEFAULT_ADAPTER_ROOT = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment39_disjoint_adapter_training"
    / "full_disjoint_zero_init_3seed_v1"
)
BLOCK_INDEX = 10


def bootstrap_interval(differences, seed, repetitions):
    generator = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = generator.integers(0, differences.size, differences.size)
        values[index] = differences[sample].mean()
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def summarize(original, corrected, seed, repetitions):
    recovered = int((~original & corrected).sum())
    damaged = int((original & ~corrected).sum())
    differences = corrected.astype(np.float64) - original.astype(np.float64)
    return {
        "samples": int(original.size),
        "original_clean_accuracy": float(original.mean()),
        "adapter_clean_accuracy": float(corrected.mean()),
        "clean_accuracy_gain": float(differences.mean()),
        "clean_accuracy_gain_95ci": bootstrap_interval(differences, seed, repetitions),
        "predictions_recovered": recovered,
        "originally_correct_damaged": damaged,
        "mcnemar_exact_pvalue": (
            float(binomtest(recovered, recovered + damaged, 0.5).pvalue)
            if recovered + damaged else 1.0
        ),
    }


def evaluate(model, adapter, dataset, batch_size, device, alpha):
    original, corrected = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for clean_hidden, _, labels in tqdm(loader, desc="Clean adapter audit"):
            clean_hidden = clean_hidden.to(device)
            labels = labels.to(device)
            original_logits = downstream_from_layer(model, clean_hidden, BLOCK_INDEX)
            patches = clean_hidden[:, 1:]
            candidate = torch.cat(
                [clean_hidden[:, :1], patches + alpha * adapter(patches)], dim=1
            )
            corrected_logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
            original.extend((original_logits.argmax(1) == labels).cpu().tolist())
            corrected.extend((corrected_logits.argmax(1) == labels).cpu().tolist())
    return np.asarray(original, dtype=bool), np.asarray(corrected, dtype=bool)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 40: paired clean-preservation audit for disjoint adapters"
    )
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)

    results = {}
    outcome_arrays = {}
    for seed in args.seeds:
        validation_record = splits[seed]["validation"]
        cache = HiddenCache(args.adapter_root / f"seed_{seed}" / "validation_cache")
        expected = validation_record["end"] - validation_record["start"]
        if len(cache) != expected:
            raise ValueError(f"Seed {seed} cache has {len(cache)} samples, expected {expected}")
        adapter = HiddenLinear().to(device)
        checkpoint = (
            args.adapter_root
            / f"seed_{seed}"
            / f"classification_weight_{args.classification_weight:g}.pt"
        )
        adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        adapter.eval().requires_grad_(False)
        original, corrected = evaluate(model, adapter, cache, args.batch_size, device, args.alpha)
        results[f"seed_{seed}"] = summarize(
            original, corrected, seed + 4000, args.bootstrap_repetitions
        )
        outcome_arrays[f"seed_{seed}_original_clean_correct"] = original
        outcome_arrays[f"seed_{seed}_adapter_clean_correct"] = corrected

    gains = np.asarray([results[f"seed_{seed}"]["clean_accuracy_gain"] for seed in args.seeds])
    summary = {
        "configuration": vars(args) | {
            "adapter_root": str(args.adapter_root.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "vit_block": 11,
            "imageNetV2_accessed": False,
            "status": "paired clean development audit only",
        },
        "validation_splits": {str(seed): splits[seed]["validation"] for seed in args.seeds},
        "results": results,
        "aggregate": {
            "mean_clean_accuracy_gain": float(gains.mean()),
            "all_clean_gains_nonnegative": bool((gains >= 0).all()),
        },
    }
    np.savez_compressed(output_dir / "paired_clean_outcomes.npz", **outcome_arrays)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"results": results, "aggregate": summary["aggregate"]}, indent=2))
    print(f"Saved Experiment 40 to {output_dir}")


if __name__ == "__main__":
    main()
