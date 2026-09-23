import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment3_sae_causal_intervention import downstream_logits
from scripts.experiment10_corruption_agnostic_sae_repair import encode, load_fixed_sae, make_dataset
from scripts.experiment11_quantile_sae_repair import calibrate_quantiles


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment13_patch_aware_repair"
EXPERIMENT12_SUMMARY = (
    PROJECT_ROOT / "results" / "sae" / "experiment12_failure_targeted_repair"
    / "full_failure_targeted_residual_gpu" / "summary.json"
)
CONDITIONS = [("clean", None, 0), ("blur4", "blur", 4)]


def make_patch_mask(scores, count, strategy):
    count = min(count, scores.shape[1])
    if strategy == "top":
        indices = torch.topk(scores, count, dim=1).indices
    elif strategy == "low":
        indices = torch.topk(scores, count, dim=1, largest=False).indices
    elif strategy == "random":
        indices = torch.topk(torch.rand_like(scores), count, dim=1).indices
    elif strategy == "all":
        return torch.ones_like(scores, dtype=torch.bool)
    else:
        raise ValueError(strategy)
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(1, indices, True)


def intervene(latent, threshold, features, weights, alpha, patch_count, strategy):
    selected = torch.as_tensor(features, device=latent.device, dtype=torch.long)
    excess = (latent[..., selected] - threshold[selected]).clamp_min(0)
    patch_scores = (excess * weights).sum(-1)
    patch_mask = make_patch_mask(patch_scores, patch_count, strategy)
    corrected = latent.clone()
    corrected[..., selected] -= alpha * excess * patch_mask[..., None]
    changed = (excess > 0) & patch_mask[..., None] & (alpha != 0)
    return corrected, patch_scores, changed


def evaluate(model, sae, loader, device, threshold, configurations):
    totals = {
        name: {"correct": 0, "margin": 0.0, "recovered": 0, "damaged": 0, "changed": 0, "patches": 0}
        for name in configurations
    }
    original = {"correct": 0, "margin": 0.0}
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, hidden, latent = encode(model, sae, images)
            decoded = sae.decode(latent)
            original_prediction = logits.argmax(1)
            original_confidence = logits.softmax(1).max(1).values
            original_margin = classification_margin(logits, labels)[1]
            original["correct"] += int((original_prediction == labels).sum())
            original["margin"] += float(original_margin.sum())
            for name, config in configurations.items():
                candidate, _, changed = intervene(latent, threshold, **{
                    key: config[key] for key in ["features", "weights", "alpha", "patch_count", "strategy"]
                })
                patches = hidden[:, 1:] + sae.decode(candidate) - decoded
                candidate_logits = downstream_logits(model, torch.cat([hidden[:, :1], patches], 1))
                if config.get("confidence_safe", False):
                    accepted = candidate_logits.softmax(1).max(1).values > original_confidence
                    final_logits = torch.where(accepted[:, None], candidate_logits, logits)
                    changed = changed & accepted[:, None, None]
                else:
                    final_logits = candidate_logits
                prediction = final_logits.argmax(1)
                margin = classification_margin(final_logits, labels)[1]
                totals[name]["correct"] += int((prediction == labels).sum())
                totals[name]["margin"] += float(margin.sum())
                totals[name]["recovered"] += int(((original_prediction != labels) & (prediction == labels)).sum())
                totals[name]["damaged"] += int(((original_prediction == labels) & (prediction != labels)).sum())
                totals[name]["changed"] += int(changed.sum())
                totals[name]["patches"] += int(changed.any(-1).sum())
            total += images.shape[0]
    results = {"original_vit": {"accuracy": original["correct"] / total, "mean_margin": original["margin"] / total}}
    for name, values in totals.items():
        accuracy = values["correct"] / total
        results[name] = {
            "accuracy": accuracy,
            "accuracy_gain_vs_original": accuracy - results["original_vit"]["accuracy"],
            "mean_margin": values["margin"] / total,
            "margin_change_vs_original": values["margin"] / total - results["original_vit"]["mean_margin"],
            "predictions_recovered": values["recovered"],
            "originally_correct_damaged": values["damaged"],
            "mean_changed_feature_patch_pairs": values["changed"] / total,
            "mean_changed_patches": values["patches"] / total,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 13: patch-aware residual SAE repair on Blur-4")
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--calibration-start", type=int, default=30000)
    parser.add_argument("--validation-start", type=int, default=40000)
    parser.add_argument("--evaluation-start", type=int, default=45000)
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--patch-counts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75, 1.0])
    parser.add_argument("--quantile", type=float, default=0.99)
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

    experiment12 = json.loads(EXPERIMENT12_SUMMARY.read_text())
    order = experiment12["combined_top_features"]
    blur_scores = np.asarray(experiment12["discoveries"]["blur"]["score"])
    noise_scores = np.asarray(experiment12["discoveries"]["noise"]["score"])
    combined_scores = blur_scores + noise_scores
    calibration_loader = DataLoader(
        make_dataset(("clean", None, 0), args.calibration_samples, args.calibration_start, args.seed),
        batch_size=args.batch_size,
    )
    calibration = calibrate_quantiles(model, sae, calibration_loader, device, 4, [args.quantile], args.seed)
    threshold = calibration["quantiles"][str(args.quantile)].to(device)

    candidates = {}
    for feature_count in args.feature_counts:
        features = order[:feature_count]
        weights = torch.as_tensor(combined_scores[features], device=device).clamp_min(0)
        weights = weights / weights.mean().clamp_min(1e-8)
        for patch_count in args.patch_counts:
            for alpha in args.alphas:
                name = f"features{feature_count}_patches{patch_count}_alpha{alpha:g}"
                candidates[name] = {
                    "features": features, "weights": weights, "alpha": alpha,
                    "patch_count": patch_count, "strategy": "top",
                }
    validation = {}
    for condition in CONDITIONS:
        loader = DataLoader(make_dataset(condition, args.validation_samples, args.validation_start, args.seed), batch_size=args.batch_size)
        validation[condition[0]] = evaluate(model, sae, loader, device, threshold, candidates)
    clean_baseline = validation["clean"]["original_vit"]["accuracy"]
    selected_name = max(candidates, key=lambda name: (
        validation["blur4"][name]["accuracy"]
        - 2 * max(0, clean_baseline - validation["clean"][name]["accuracy"])
    ))
    selected = candidates[selected_name]
    configurations = {
        "patch_targeted": selected,
        "patch_targeted_confidence_safe": selected | {"confidence_safe": True},
        "all_patch_baseline": selected | {"strategy": "all", "patch_count": 196},
        "random_patch_control": selected | {"strategy": "random"},
        "low_score_patch_control": selected | {"strategy": "low"},
    }
    evaluation = {}
    for condition in CONDITIONS:
        loader = DataLoader(make_dataset(condition, args.evaluation_samples, args.evaluation_start, args.seed), batch_size=args.batch_size)
        evaluation[condition[0]] = evaluate(model, sae, loader, device, threshold, configurations)
    serializable_selected = selected | {"weights": selected["weights"].tolist()}
    summary = {
        "configuration": vars(args) | {"device": str(device), "source_experiment": str(EXPERIMENT12_SUMMARY)},
        "selected": {"name": selected_name} | serializable_selected,
        "validation": validation,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected": summary["selected"], "evaluation": evaluation}, indent=2))
    print(f"Saved Experiment 13 to {output_dir}")


if __name__ == "__main__":
    main()
