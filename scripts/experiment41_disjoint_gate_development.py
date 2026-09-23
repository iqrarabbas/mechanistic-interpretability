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
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenCache
from scripts.experiment35_sae_guided_residual_adapter import SelectedSAEFeatures, feature_statistics
from scripts.experiment36_sae_abnormality_gate import FeatureGate, base_adapter, train_gate
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment41_disjoint_gate_development"
DEFAULT_ADAPTER_ROOT = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment39_disjoint_adapter_training"
    / "full_disjoint_zero_init_3seed_v1"
)
DEFAULT_FEATURE_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment17_noise_bidirectional_repair"
    / "supervisor_disjoint_feature_dev_v1"
    / "summary.json"
)
BLOCK_INDEX = 10


def bootstrap_interval(differences, seed, repetitions):
    generator = np.random.default_rng(seed)
    values = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = generator.integers(0, differences.size, differences.size)
        values[index] = differences[sample].mean()
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def paired_comparison(reference, candidate, seed, repetitions):
    improved = int((~reference & candidate).sum())
    damaged = int((reference & ~candidate).sum())
    differences = candidate.astype(np.float64) - reference.astype(np.float64)
    return {
        "reference_accuracy": float(reference.mean()),
        "candidate_accuracy": float(candidate.mean()),
        "accuracy_difference": float(differences.mean()),
        "accuracy_difference_95ci": bootstrap_interval(differences, seed, repetitions),
        "reference_wrong_candidate_correct": improved,
        "reference_correct_candidate_wrong": damaged,
        "mcnemar_exact_pvalue": (
            float(binomtest(improved, improved + damaged, 0.5).pvalue)
            if improved + damaged else 1.0
        ),
    }


def evaluate(model, methods, dataset, batch_size, device, alpha, seed, repetitions):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    arrays = {"baseline_clean": [], "baseline_noise": []}
    for name in methods:
        arrays[f"{name}_clean"] = []
        arrays[f"{name}_noise"] = []
    with torch.no_grad():
        for clean_hidden, noise_hidden, labels in tqdm(loader, desc="Gate development evaluation"):
            clean_hidden = clean_hidden.to(device)
            noise_hidden = noise_hidden.to(device)
            labels = labels.to(device)
            arrays["baseline_clean"].extend(
                (downstream_from_layer(model, clean_hidden, BLOCK_INDEX).argmax(1) == labels).cpu().tolist()
            )
            arrays["baseline_noise"].extend(
                (downstream_from_layer(model, noise_hidden, BLOCK_INDEX).argmax(1) == labels).cpu().tolist()
            )
            for name, method in methods.items():
                for condition, hidden in (("clean", clean_hidden), ("noise", noise_hidden)):
                    patches = hidden[:, 1:]
                    candidate = torch.cat(
                        [hidden[:, :1], patches + alpha * method(patches)], dim=1
                    )
                    arrays[f"{name}_{condition}"].extend(
                        (downstream_from_layer(model, candidate, BLOCK_INDEX).argmax(1) == labels).cpu().tolist()
                    )
    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    results = {
        "baseline_clean_accuracy": float(arrays["baseline_clean"].mean()),
        "baseline_noise4_accuracy": float(arrays["baseline_noise"].mean()),
        "methods": {},
    }
    adapter_clean = arrays["adapter_clean"]
    adapter_noise = arrays["adapter_noise"]
    results["methods"]["adapter"] = {
        "clean_vs_baseline": paired_comparison(
            arrays["baseline_clean"], adapter_clean, seed, repetitions
        ),
        "noise_vs_baseline": paired_comparison(
            arrays["baseline_noise"], adapter_noise, seed + 1, repetitions
        ),
    }
    for offset, name in enumerate(methods):
        if name == "adapter":
            continue
        results["methods"][name] = {
            "clean_vs_baseline": paired_comparison(
                arrays["baseline_clean"], arrays[f"{name}_clean"], seed + 10 + offset, repetitions
            ),
            "noise_vs_baseline": paired_comparison(
                arrays["baseline_noise"], arrays[f"{name}_noise"], seed + 20 + offset, repetitions
            ),
            "clean_vs_adapter": paired_comparison(
                adapter_clean, arrays[f"{name}_clean"], seed + 30 + offset, repetitions
            ),
            "noise_vs_adapter": paired_comparison(
                adapter_noise, arrays[f"{name}_noise"], seed + 40 + offset, repetitions
            ),
        }
    return results, arrays


def ranked_over_features(path):
    record = json.loads(path.read_text())
    return [row["feature_index"] for row in record["top_over_features"]]


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 41: leakage-free top-8/top-16 SAE gate development"
    )
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--feature-source", type=Path, default=DEFAULT_FEATURE_SOURCE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-scale", type=float, default=2.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--predictor-batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--clean-penalty", type=float, default=2.0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cache-samples", type=int)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    ranking = ranked_over_features(args.feature_source)
    if max(args.feature_counts) > len(ranking):
        raise ValueError(f"Feature source has {len(ranking)} ranked over-features")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)
    sae.requires_grad_(False)
    training = {}
    validation = {}
    outcome_arrays = {}

    for seed in args.seeds:
        torch.manual_seed(seed + 4100)
        np.random.seed(seed + 4100)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_data = HiddenCache(args.adapter_root / f"seed_{seed}" / "train_cache")
        validation_data = HiddenCache(args.adapter_root / f"seed_{seed}" / "validation_cache")
        if args.max_cache_samples is not None:
            from torch.utils.data import Subset
            train_data = Subset(train_data, range(min(args.max_cache_samples, len(train_data))))
            validation_data = Subset(validation_data, range(min(args.max_cache_samples, len(validation_data))))
        adapter_path = (
            args.adapter_root
            / f"seed_{seed}"
            / f"classification_weight_{args.classification_weight:g}.pt"
        )
        frozen_adapter = base_adapter(adapter_path, device)
        methods = {"adapter": frozen_adapter}
        training[f"seed_{seed}"] = {}
        for count in args.feature_counts:
            name = f"gate_top{count}"
            features = ranking[:count]
            mean, scale = feature_statistics(
                train_data, sae, features, device, args.predictor_batch_size
            )
            encoder = SelectedSAEFeatures(sae, features, mean, scale)
            gate = FeatureGate(
                base_adapter(adapter_path, device), encoder, max_scale=args.max_scale
            ).to(device)
            checkpoint = seed_dir / f"{name}.pt"
            record_path = seed_dir / f"{name}_training.json"
            if args.resume and checkpoint.exists() and record_path.exists():
                gate.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                record = json.loads(record_path.read_text())
            else:
                record = train_gate(
                    model, gate, train_data, validation_data, args, device, checkpoint
                )
                record_path.write_text(json.dumps(record, indent=2))
            gate.eval()
            methods[name] = gate
            training[f"seed_{seed}"][name] = record | {"features": features}
        seed_results, arrays = evaluate(
            model,
            methods,
            validation_data,
            args.predictor_batch_size,
            device,
            args.alpha,
            seed + 4100,
            args.bootstrap_repetitions,
        )
        validation[f"seed_{seed}"] = seed_results
        for name, values in arrays.items():
            outcome_arrays[f"seed_{seed}_{name}"] = values

    selection = {}
    for count in args.feature_counts:
        name = f"gate_top{count}"
        noise = np.asarray([
            validation[f"seed_{seed}"]["methods"][name]["noise_vs_baseline"]["candidate_accuracy"]
            for seed in args.seeds
        ])
        clean = np.asarray([
            validation[f"seed_{seed}"]["methods"][name]["clean_vs_baseline"]["candidate_accuracy"]
            for seed in args.seeds
        ])
        clean_baseline = np.asarray([
            validation[f"seed_{seed}"]["baseline_clean_accuracy"] for seed in args.seeds
        ])
        score = float(noise.mean() - args.clean_penalty * max(0.0, clean_baseline.mean() - clean.mean()))
        selection[name] = {
            "mean_noise4_accuracy": float(noise.mean()),
            "mean_clean_accuracy": float(clean.mean()),
            "mean_clean_baseline_accuracy": float(clean_baseline.mean()),
            "selection_score": score,
        }
    selected_name = max(selection, key=lambda name: selection[name]["selection_score"])
    summary = {
        "configuration": vars(args) | {
            "adapter_root": str(args.adapter_root.resolve()),
            "feature_source": str(args.feature_source.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "vit_block": 11,
            "frozen_components": ["ViT", "clean SAE", "three residual adapters"],
            "imageNetV2_accessed": False,
            "status": "gate development only; final evaluation remains untouched",
        },
        "development_splits": {str(seed): splits[seed] for seed in args.seeds},
        "feature_ranking": ranking[: max(args.feature_counts)],
        "training": training,
        "validation": validation,
        "selection": selection,
        "selected_gate": selected_name,
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcome_arrays)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selection": selection, "selected_gate": selected_name}, indent=2))
    print(f"Saved Experiment 41 to {output_dir}")


if __name__ == "__main__":
    main()
