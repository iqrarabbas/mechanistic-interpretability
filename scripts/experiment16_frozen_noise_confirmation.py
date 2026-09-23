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
from scripts.experiment14_patch_budget_pareto import evaluate, pareto_front
from scripts.experiment15_independent_frozen_confirmation import (
    EXPERIMENT12_SUMMARY,
    EXPERIMENT13_SUMMARY,
    ExternalImageNetDataset,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment16_frozen_noise_confirmation"


def main():
    parser = argparse.ArgumentParser(description="Experiment 16: frozen ImageNetV2 Noise confirmation")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
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

    experiment12 = json.loads(EXPERIMENT12_SUMMARY.read_text())
    experiment13 = json.loads(EXPERIMENT13_SUMMARY.read_text())
    features = experiment13["selected"]["features"]
    combined_scores = (
        np.asarray(experiment12["discoveries"]["blur"]["score"])
        + np.asarray(experiment12["discoveries"]["noise"]["score"])
    )
    weights = torch.as_tensor(combined_scores[features], device=device).clamp_min(0)
    weights = weights / weights.mean().clamp_min(1e-8)
    calibration_loader = DataLoader(
        make_dataset(("clean", None, 0), 2000, 30000, 0),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    calibration = calibrate_quantiles(model, sae, calibration_loader, device, 4, [0.99], 0)
    threshold = calibration["quantiles"]["0.99"].to(device)
    generator = np.random.default_rng(args.seed)
    random_features = generator.choice(sae.latent_dim, 64, replace=False).tolist()
    frozen = {
        "features": features,
        "weights": weights,
        "alpha": 1.0,
        "patch_count": 32,
        "strategy": "top",
    }
    configurations = {
        "frozen_targeted": frozen,
        "residual_identity": frozen | {"alpha": 0.0},
        "random_feature_control": frozen | {
            "features": random_features, "weights": torch.ones(64, device=device)
        },
        "random_patch_control": frozen | {"strategy": "random"},
        "low_score_patch_control": frozen | {"strategy": "low"},
        "all_patch_control": frozen | {"strategy": "all", "patch_count": 196},
    }

    evaluation = {}
    for severity in range(6):
        condition = "clean" if severity == 0 else f"noise{severity}"
        dataset = ExternalImageNetDataset(
            args.image_root,
            severity,
            args.samples,
            args.start_index,
            corruption="noise",
            seed=args.seed,
        )
        if len(dataset) != args.samples:
            raise ValueError(f"Requested {args.samples} images, found {len(dataset)}")
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        evaluation[condition], arrays = evaluate(
            model, sae, loader, device, threshold, configurations
        )
        np.savez_compressed(output_dir / f"{condition}_paired_outcomes.npz", **arrays)

    noise4 = evaluation["noise4"]["frozen_targeted"]
    clean = evaluation["clean"]["frozen_targeted"]
    success = {
        "positive_noise4_gain": noise4["accuracy_gain_vs_original"] > 0,
        "noise4_ci_excludes_zero": noise4["accuracy_gain_95ci"][0] > 0,
        "noise4_mcnemar_below_0.05": noise4["mcnemar_exact_pvalue"] < 0.05,
        "clean_loss_within_0.1pp": clean["accuracy_gain_vs_original"] >= -0.001,
        "beats_random_features": noise4["accuracy"] > evaluation["noise4"]["random_feature_control"]["accuracy"],
        "beats_random_patches": noise4["accuracy"] > evaluation["noise4"]["random_patch_control"]["accuracy"],
    }
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "device": str(device),
            "frozen_without_retuning": True,
            "source_rule": "Experiment 15 Blur rule unchanged",
            "features": features,
            "patch_count": 32,
            "alpha": 1.0,
            "quantile": 0.99,
            "noise_seed": args.seed,
        },
        "evaluation": evaluation,
        "noise4_control_pareto": pareto_front(evaluation["noise4"]),
        "preregistered_success_criteria": success,
        "all_success_criteria_met": all(success.values()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"success": success, "clean": evaluation["clean"], "noise4": evaluation["noise4"]}, indent=2))
    print(f"Saved Experiment 16 to {output_dir}")


if __name__ == "__main__":
    main()
