import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL, SAE_DIR, load_sae
from scripts.experiment4_non_oracle_sae_correction import make_loader, write_csv
from scripts.experiment5_sae_correction_strategies import evaluate


EXPERIMENT5_DIR = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment5_correction_strategies"
    / "full_strategies_5000"
)
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment6_fixed_alpha"


def load_frozen_correction(device):
    summary = json.loads((EXPERIMENT5_DIR / "summary.json").read_text())
    selected = summary["selected_configuration"]
    if selected["method"] != "affine" or len(selected["features"]) != 32:
        raise ValueError("Experiment 6 expects Experiment 5's frozen top-32 affine correction")
    with (EXPERIMENT5_DIR / "discovery_feature_statistics.csv").open() as source:
        rows = list(csv.DictReader(source))
    scale = torch.tensor([float(row["affine_scale"]) for row in rows], device=device)
    intercept = torch.tensor([float(row["affine_intercept"]) for row in rows], device=device)
    latent_dim = len(rows)
    statistics = {
        "scale": scale,
        "intercept": intercept,
        "mean_clean": torch.zeros(latent_dim, device=device),
        "clean_std": torch.ones(latent_dim, device=device),
        "change_sign": torch.ones(latent_dim, device=device),
    }
    return selected["features"], statistics


def main():
    parser = argparse.ArgumentParser(description="Experiment 6: fine fixed-alpha calibration")
    parser.add_argument("--validation-samples", type=int, default=5000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--validation-start", type=int, default=30000)
    parser.add_argument("--test-start", type=int, default=35000)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
    )
    parser.add_argument("--maximum-clean-loss", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    ranges = [
        (0, 30000, "all previous experiments"),
        (args.validation_start, args.validation_start + args.validation_samples, "alpha validation"),
        (args.test_start, args.test_start + args.test_samples, "locked test"),
    ]
    for left, first in enumerate(ranges):
        for second in ranges[left + 1 :]:
            if max(first[0], second[0]) < min(first[1], second[1]):
                raise ValueError(f"Data leakage: {first[2]} overlaps {second[2]}")
    if ranges[-1][1] > 50000:
        raise ValueError("Requested split exceeds ImageNet validation data")

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae, sae_metadata = load_sae(device)
    features, statistics = load_frozen_correction(device)

    configurations = {"baseline": {"method": "affine", "features": [], "alpha": 0.0}}
    for alpha in args.alphas:
        configurations[f"alpha_{alpha:g}"] = {
            "method": "affine",
            "features": features,
            "alpha": alpha,
        }
    validation_loader = make_loader(
        args.validation_samples,
        args.validation_start,
        args.seed,
        args.batch_size,
        args.num_workers,
    )
    validation = evaluate(model, sae, validation_loader, device, configurations, statistics)
    baseline = validation["baseline"]
    allowed = [
        name for name in configurations if name != "baseline"
        and baseline["clean_accuracy"] - validation[name]["clean_accuracy"]
        <= args.maximum_clean_loss + 1e-12
    ]
    if not allowed:
        raise RuntimeError("No alpha satisfies the clean-accuracy constraint")
    selected_name = max(
        allowed,
        key=lambda name: (validation[name]["blur_accuracy"], validation[name]["mean_blur_margin"]),
    )
    selected = configurations[selected_name]
    write_csv(
        output_dir / "validation_results.csv",
        [{"configuration": name} | values for name, values in validation.items()],
    )

    test_configurations = {
        "baseline": configurations["baseline"],
        "selected": selected,
        "reverse": selected | {"reverse": True},
    }
    test_loader = make_loader(
        args.test_samples,
        args.test_start,
        args.seed,
        args.batch_size,
        args.num_workers,
    )
    test = evaluate(model, sae, test_loader, device, test_configurations, statistics)
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "sae": str(SAE_DIR.relative_to(PROJECT_ROOT)),
            "sae_implementation": sae_metadata["implementation"],
            "frozen_source": str(EXPERIMENT5_DIR.relative_to(PROJECT_ROOT)),
            "features_and_mapping": "Frozen Experiment 5 top-32 affine correction",
        },
        "split_ranges": {name: [start, end] for start, end, name in ranges},
        "selected_alpha": selected["alpha"],
        "validation_baseline": baseline,
        "validation_selected": validation[selected_name],
        "locked_test": {
            "original_vit": test["original_vit"],
            "baseline": test["baseline"],
            "selected": test["selected"],
            "reverse": test["reverse"],
            "blur_accuracy_gain": test["selected"]["blur_accuracy"] - test["baseline"]["blur_accuracy"],
            "clean_accuracy_change": test["selected"]["clean_accuracy"] - test["baseline"]["clean_accuracy"],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Saved Experiment 6 to {output_dir}")


if __name__ == "__main__":
    main()
