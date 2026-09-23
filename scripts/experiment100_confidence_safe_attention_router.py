import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experiment99_attention_head_corruption_detector import (
    HeadRouter,
    LABELS,
    OUTPUT_ROOT as EXPERIMENT99_ROOT,
    SEEN,
    UNSEEN,
)
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment100_confidence_safe_attention_router"
DEFAULT_SOURCE = EXPERIMENT99_ROOT / "full_3seed_clean_noise_blur_v1"


def load_probabilities(seed_dir, prefix):
    calibration = torch.load(
        seed_dir / "clean_calibration_features.pt", map_location="cpu", weights_only=True
    )
    mean = calibration.mean(0)
    std = calibration.std(0).clamp_min(1e-6)
    payload = torch.load(
        seed_dir / "attention_head_router.pt", map_location="cpu", weights_only=True
    )
    router = HeadRouter(calibration.shape[1])
    router.load_state_dict(payload["router"])
    router.eval()
    output = {}
    with torch.no_grad():
        for family, conditions in SEEN.items():
            output[family] = {}
            for condition in conditions:
                key = "clean" if condition is None else condition
                features = torch.load(
                    seed_dir / f"{prefix}_{key}_features.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                output[family][key] = router((features - mean) / std).softmax(1)
    return output, router, mean, std


def predictions(probabilities, threshold=None):
    if threshold is None:
        return probabilities.argmax(1)
    nonclean = probabilities[:, 1:].argmax(1) + 1
    return torch.where(probabilities[:, 0] >= threshold, 0, nonclean)


def metrics(records, threshold=None, half="all"):
    condition_rows = {}
    family_correct = {family: [] for family in LABELS}
    all_correct = []
    for family, conditions in records.items():
        target = LABELS[family]
        for condition, values in conditions.items():
            midpoint = len(values) // 2
            selected = values[:midpoint] if half == "development" else values[midpoint:]
            if half == "all":
                selected = values
            predicted = predictions(selected, threshold)
            correct = predicted == target
            family_correct[family].append(correct)
            all_correct.append(correct)
            condition_rows[condition] = {
                "samples": len(selected),
                "accuracy": float(correct.float().mean()),
                "routing_fractions_clean_noise_blur": [
                    float((predicted == label).float().mean()) for label in range(3)
                ],
            }
    family_accuracy = {
        family: float(torch.cat(values).float().mean())
        for family, values in family_correct.items()
    }
    return {
        "overall_accuracy": float(torch.cat(all_correct).float().mean()),
        "macro_family_accuracy": float(np.mean(list(family_accuracy.values()))),
        "family_accuracy": family_accuracy,
        "conditions": condition_rows,
    }


def select_threshold(records, minimum_corruption_retention):
    original = metrics(records, None, "development")
    corruption_floor = minimum_corruption_retention * np.mean(
        [original["family_accuracy"]["noise"], original["family_accuracy"]["blur"]]
    )
    candidates = []
    for threshold in np.linspace(0.0, 1.0, 201):
        result = metrics(records, float(threshold), "development")
        corruption_accuracy = np.mean(
            [result["family_accuracy"]["noise"], result["family_accuracy"]["blur"]]
        )
        if corruption_accuracy >= corruption_floor:
            candidates.append((result["macro_family_accuracy"], result["family_accuracy"]["clean"], float(threshold), result))
    if not candidates:
        raise RuntimeError("No threshold satisfies corruption-retention constraint")
    candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
    _, _, threshold, result = candidates[0]
    return threshold, original, result, corruption_floor


def unseen_routing(seed_dir, router, mean, std, threshold):
    output = {}
    with torch.no_grad():
        for condition in UNSEEN:
            features = torch.load(
                seed_dir / f"unseen_{condition}_features.pt",
                map_location="cpu",
                weights_only=True,
            )
            probabilities = router((features - mean) / std).softmax(1)
            predicted = predictions(probabilities, threshold)
            output[condition] = {
                "samples": len(predicted),
                "routing_fractions_clean_noise_blur": [
                    float((predicted == label).float().mean()) for label in range(3)
                ],
                "mean_probabilities_clean_noise_blur": probabilities.mean(0).tolist(),
            }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--minimum-corruption-retention", type=float, default=0.95)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    results = {}
    for seed in args.seeds:
        seed_dir = args.source_run / f"seed_{seed}"
        records, router, mean, std = load_probabilities(seed_dir, "validation")
        threshold, original_dev, threshold_dev, corruption_floor = select_threshold(
            records, args.minimum_corruption_retention
        )
        results[str(seed)] = {
            "selected_clean_probability_threshold": threshold,
            "development": {
                "original_argmax": original_dev,
                "confidence_safe": threshold_dev,
                "minimum_mean_noise_blur_accuracy": corruption_floor,
            },
            "heldout_validation_half": {
                "original_argmax": metrics(records, None, "evaluation"),
                "confidence_safe": metrics(records, threshold, "evaluation"),
            },
            "unseen_routing": unseen_routing(seed_dir, router, mean, std, threshold),
        }
    summary = {
        "configuration": vars(args) | {
            "source_run": str(args.source_run.resolve()),
            "threshold_candidates": 201,
            "selection_data": "first half of each Experiment 99 validation condition",
            "evaluation_data": "second half of each Experiment 99 validation condition",
            "new_vit_inference": False,
            "imageNetV2_accessed": False,
            "router_uses_clean_counterpart": False,
            "status": "cached-feature development analysis; adapter not applied",
        },
        "results": results,
        "guardrails": [
            "Threshold selection and reported held-out evaluation use disjoint image halves.",
            "The threshold is selected independently for each seed.",
            "Unseen-corruption routing is descriptive because no correct expert label is defined.",
            "No final reserve, ImageNetV2, or ImageNet-Sketch data are accessed.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
