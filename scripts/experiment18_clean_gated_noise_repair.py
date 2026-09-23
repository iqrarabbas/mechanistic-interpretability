import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae, make_dataset
from scripts.experiment11_quantile_sae_repair import calibrate_quantiles
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment17_noise_bidirectional_repair import evaluate


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment18_clean_gated_noise_repair"
EXPERIMENT17_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment17_noise_bidirectional_repair" / "noise_bidirectional_development_gpu"


def main():
    parser = argparse.ArgumentParser(description="Experiment 18: clean-gated Noise overactivation repair")
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--calibration-start", type=int, default=30000)
    parser.add_argument("--validation-start", type=int, default=30000)
    parser.add_argument("--evaluation-start", type=int, default=40000)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.95, 0.99, 0.995, 0.999])
    parser.add_argument("--patch-counts", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--clean-loss-penalty", type=float, default=4.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
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

    experiment17 = json.loads((EXPERIMENT17_ROOT / "summary.json").read_text())
    over_features = experiment17["selected"]["over_features"]
    under_features = experiment17["selected"]["under_features"]
    affine = np.load(EXPERIMENT17_ROOT / "affine_parameters.npz")
    scale, intercept = affine["scale"], affine["intercept"]
    calibration_loader = DataLoader(
        make_dataset(("clean", None, 0), args.calibration_samples, args.calibration_start, args.seed),
        batch_size=args.batch_size,
    )
    calibration = calibrate_quantiles(
        model, sae, calibration_loader, device, 4, args.quantiles, args.seed
    )
    thresholds = {key: value.to(device) for key, value in calibration["quantiles"].items()}

    candidates = {}
    for quantile in args.quantiles:
        for patch_count in args.patch_counts:
            for alpha in args.alphas:
                name = f"q{quantile:g}_patches{patch_count}_alpha{alpha:g}"
                candidates[name] = {
                    "over_features": over_features,
                    "under_features": under_features,
                    "mode": "over",
                    "patch_count": patch_count,
                    "alpha": alpha,
                    "upper_threshold": thresholds[str(quantile)],
                }
    validation_loader = DataLoader(
        PairedCorruptionDataset("noise", args.validation_samples, args.validation_start, args.seed),
        batch_size=args.batch_size,
    )
    validation = evaluate(model, sae, validation_loader, device, scale, intercept, candidates)
    clean_base = validation["original_vit"]["clean_correct"]
    selected_name = max(candidates, key=lambda name: (
        validation[name]["noise4_accuracy"]
        - args.clean_loss_penalty * max(0, clean_base - validation[name]["clean_accuracy"])
    ))
    selected = candidates[selected_name]
    selected_quantile = float(selected_name.split("_", 1)[0][1:])
    generator = np.random.default_rng(args.seed)
    random_features = generator.choice(sae.latent_dim, len(over_features), replace=False).tolist()
    configurations = {
        "gated_affine": selected,
        "ungated_affine": {key: value for key, value in selected.items() if key != "upper_threshold"},
        "quantile_clip_only": selected | {"correction": "clip"},
        "wrong_direction_control": selected | {"reverse": True},
        "random_feature_control": selected | {"over_features": random_features},
        "residual_identity": selected | {"alpha": 0.0},
    }
    evaluation_loader = DataLoader(
        PairedCorruptionDataset("noise", args.evaluation_samples, args.evaluation_start, args.seed),
        batch_size=args.batch_size,
    )
    evaluation = evaluate(model, sae, evaluation_loader, device, scale, intercept, configurations)
    summary = {
        "configuration": vars(args) | {
            "device": str(device),
            "status": "exploratory development experiment; ImageNetV2 is not used",
            "source_features": str(EXPERIMENT17_ROOT / "summary.json"),
        },
        "selected": {
            "name": selected_name,
            "quantile": selected_quantile,
            "patch_count": selected["patch_count"],
            "alpha": selected["alpha"],
            "over_features": over_features,
        },
        "validation": validation,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected": summary["selected"], "evaluation": evaluation}, indent=2))
    print(f"Saved Experiment 18 to {output_dir}")


if __name__ == "__main__":
    main()
