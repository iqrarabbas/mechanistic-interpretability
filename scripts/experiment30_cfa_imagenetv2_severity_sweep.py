import argparse
import json
from pathlib import Path

import numpy as np
import torch

from interpretability.cfa import load_source_statistics
from scripts.experiment28_cfa_same_setup import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_STATS,
    OFFICIAL_CFA_COMMIT,
    make_target_loader,
    paired_summary,
    run_cfa_loader,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment30_cfa_severity_sweep"
DEFAULT_LOCKED_ROOT = (
    PROJECT_ROOT / "results" / "sae" / "experiment27_corruption_transfer"
    / "fixed_noise_adapter_transfer" / "conditions"
)


def locked_arrays(root, condition, samples):
    path = root / f"{condition}.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing locked baseline/adapter outcomes: {path}")
    arrays = np.load(path)
    original = arrays["original_correct"][:samples].astype(bool)
    adapter = arrays["adapter_correct"][:samples].astype(bool)
    if original.size != samples or adapter.size != samples:
        raise ValueError(f"Requested {samples} locked outcomes, found {original.size} in {path}")
    return original, adapter


def aggregate(results, corruptions, severities, method):
    accuracies = []
    gains = []
    per_corruption = {}
    for corruption in corruptions:
        corruption_accuracies = [
            results[f"{corruption}{severity}"][method]["accuracy"]
            for severity in severities
        ]
        corruption_gains = [
            results[f"{corruption}{severity}"][method]["accuracy_gain"]
            for severity in severities
        ]
        accuracies.extend(corruption_accuracies)
        gains.extend(corruption_gains)
        per_corruption[corruption] = {
            "accuracy_by_severity": dict(zip(map(str, severities), corruption_accuracies)),
            "gain_by_severity": dict(zip(map(str, severities), corruption_gains)),
            "mean_accuracy": float(np.mean(corruption_accuracies)),
            "mean_gain": float(np.mean(corruption_gains)),
        }
    return {
        "mean_corruption_accuracy": float(np.mean(accuracies)),
        "mean_corruption_error": float(1.0 - np.mean(accuracies)),
        "mean_accuracy_gain": float(np.mean(gains)),
        "per_corruption": per_corruption,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 30: CFA official-batch severity sweep on shared ImageNetV2 corruptions"
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--locked-root", type=Path, default=DEFAULT_LOCKED_ROOT)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--corruptions", nargs="+", choices=["blur", "noise"], default=["blur", "noise"])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--full-max-moment", type=int, default=3)
    parser.add_argument("--class-max-moment", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if any(severity not in range(1, 6) for severity in args.severities):
        raise ValueError("Severities must be between 1 and 5")
    if not args.statistics.exists():
        raise FileNotFoundError(f"Missing CFA source statistics: {args.statistics}")

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    statistics = load_source_statistics(args.statistics, device)

    results = {}
    conditions = [
        (f"{corruption}{severity}", corruption, severity)
        for corruption in args.corruptions
        for severity in args.severities
    ]
    for index, (name, corruption, severity) in enumerate(conditions):
        result_path = condition_dir / f"{name}.json"
        outcomes_path = condition_dir / f"{name}.npz"
        if args.resume and result_path.exists() and outcomes_path.exists():
            results[name] = json.loads(result_path.read_text())
            print(f"Resumed {name}")
            continue

        original, adapter = locked_arrays(args.locked_root, name, args.samples)
        cfa_correct, diagnostics = run_cfa_loader(
            args,
            make_target_loader(args, corruption, severity),
            statistics,
            device,
            f"CFA {name} batch{args.batch_size}",
        )
        cfa = paired_summary(
            original,
            cfa_correct,
            args.noise_seed + index,
            args.bootstrap_repetitions,
        ) | diagnostics
        adapter_summary = paired_summary(
            original,
            adapter,
            args.noise_seed + 100 + index,
            args.bootstrap_repetitions,
        )
        record = {
            "corruption": corruption,
            "severity": severity,
            "baseline_accuracy": float(original.mean()),
            "fixed_adapter": adapter_summary,
            "cfa": cfa,
            "winner": "cfa" if cfa["accuracy"] > adapter_summary["accuracy"] else "fixed_adapter",
        }
        result_path.write_text(json.dumps(record, indent=2))
        np.savez_compressed(
            outcomes_path,
            original_correct=original,
            adapter_correct=adapter,
            cfa_correct=cfa_correct,
        )
        results[name] = record

    baseline_accuracies = [results[name]["baseline_accuracy"] for name, _, _ in conditions]
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "statistics": str(args.statistics.resolve()),
            "locked_root": str(args.locked_root.resolve()),
            "device": str(device),
            "official_cfa_repository_commit": OFFICIAL_CFA_COMMIT,
        },
        "protocol": {
            "same_backbone_images_order_and_corruptions": True,
            "cfa_reset_for_every_corruption_severity": True,
            "cfa_official_target_batch_size": args.batch_size == 64,
            "cfa_official_optimizer_defaults": (
                args.learning_rate == 0.001
                and args.momentum == 0.9
                and args.weight_decay == 0.0
            ),
            "resource_constraint": "ImageNet validation source statistics replace ImageNet train statistics",
            "claim": "controlled ImageNetV2 CFA comparison, not official ImageNet-C reproduction",
        },
        "conditions": results,
        "aggregate": {
            "baseline": {
                "mean_corruption_accuracy": float(np.mean(baseline_accuracies)),
                "mean_corruption_error": float(1.0 - np.mean(baseline_accuracies)),
            },
            "fixed_adapter": aggregate(results, args.corruptions, args.severities, "fixed_adapter"),
            "cfa": aggregate(results, args.corruptions, args.severities, "cfa"),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved Experiment 30 to {output_dir}")


if __name__ == "__main__":
    main()
