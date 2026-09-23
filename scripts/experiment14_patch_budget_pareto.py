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
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment3_sae_causal_intervention import downstream_logits
from scripts.experiment10_corruption_agnostic_sae_repair import encode, load_fixed_sae, make_dataset
from scripts.experiment11_quantile_sae_repair import calibrate_quantiles
from scripts.experiment13_patch_aware_residual_repair import intervene


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment14_patch_budget_pareto"
EXPERIMENT12_SUMMARY = PROJECT_ROOT / "results" / "sae" / "experiment12_failure_targeted_repair" / "full_failure_targeted_residual_gpu" / "summary.json"
EXPERIMENT13_SUMMARY = PROJECT_ROOT / "results" / "sae" / "experiment13_patch_aware_repair" / "full_patch_aware_residual_gpu" / "summary.json"
CONDITIONS = [("clean", None, 0), ("blur4", "blur", 4)]


def bootstrap_gain_interval(differences, seed, repetitions=2000):
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for index in range(repetitions):
        means[index] = generator.choice(differences, len(differences), replace=True).mean()
    return np.quantile(means, [0.025, 0.975]).tolist()


def evaluate(model, sae, loader, device, threshold, configurations):
    original_correct, original_margin = [], []
    outcomes = {name: {"correct": [], "margin": [], "changed": [], "patches": []} for name in configurations}
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits, hidden, latent = encode(model, sae, images)
            decoded = sae.decode(latent)
            original_correct.append((logits.argmax(1) == labels).cpu())
            original_margin.append(classification_margin(logits, labels)[1].cpu())
            for name, config in configurations.items():
                candidate, _, changed = intervene(latent, threshold, **config)
                patches = hidden[:, 1:] + sae.decode(candidate) - decoded
                candidate_logits = downstream_logits(model, torch.cat([hidden[:, :1], patches], 1))
                outcomes[name]["correct"].append((candidate_logits.argmax(1) == labels).cpu())
                outcomes[name]["margin"].append(classification_margin(candidate_logits, labels)[1].cpu())
                outcomes[name]["changed"].append(changed.sum((1, 2)).cpu())
                outcomes[name]["patches"].append(changed.any(-1).sum(1).cpu())
    original_correct = torch.cat(original_correct).numpy().astype(bool)
    original_margin = torch.cat(original_margin).numpy()
    arrays = {"original_correct": original_correct, "original_margin": original_margin}
    results = {
        "original_vit": {"accuracy": float(original_correct.mean()), "mean_margin": float(original_margin.mean())}
    }
    for name, values in outcomes.items():
        correct = torch.cat(values["correct"]).numpy().astype(bool)
        margin = torch.cat(values["margin"]).numpy()
        changed = torch.cat(values["changed"]).numpy()
        patches = torch.cat(values["patches"]).numpy()
        recovered = int((~original_correct & correct).sum())
        damaged = int((original_correct & ~correct).sum())
        pvalue = float(binomtest(recovered, recovered + damaged, 0.5).pvalue) if recovered + damaged else 1.0
        differences = correct.astype(float) - original_correct.astype(float)
        results[name] = {
            "accuracy": float(correct.mean()),
            "accuracy_gain_vs_original": float(differences.mean()),
            "accuracy_gain_95ci": bootstrap_gain_interval(differences, 0),
            "mean_margin": float(margin.mean()),
            "margin_change_vs_original": float((margin - original_margin).mean()),
            "predictions_recovered": recovered,
            "originally_correct_damaged": damaged,
            "mcnemar_exact_pvalue": pvalue,
            "mean_changed_feature_patch_pairs": float(changed.mean()),
            "mean_changed_patches": float(patches.mean()),
        }
        arrays[f"{name}_correct"] = correct
        arrays[f"{name}_margin"] = margin
        arrays[f"{name}_changed_pairs"] = changed
        arrays[f"{name}_changed_patches"] = patches
    return results, arrays


def pareto_front(results):
    rows = []
    for name, values in results.items():
        if name == "original_vit":
            continue
        dominated = any(
            other != name
            and results[other]["accuracy_gain_vs_original"] >= values["accuracy_gain_vs_original"]
            and results[other]["mean_changed_patches"] <= values["mean_changed_patches"]
            and (
                results[other]["accuracy_gain_vs_original"] > values["accuracy_gain_vs_original"]
                or results[other]["mean_changed_patches"] < values["mean_changed_patches"]
            )
            for other in results if other != "original_vit"
        )
        if not dominated:
            rows.append({"name": name} | values)
    return sorted(rows, key=lambda row: row["mean_changed_patches"])


def main():
    parser = argparse.ArgumentParser(description="Experiment 14: Blur-4 patch-budget Pareto sweep")
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--calibration-start", type=int, default=30000)
    parser.add_argument("--validation-start", type=int, default=40000)
    parser.add_argument("--evaluation-start", type=int, default=45000)
    parser.add_argument("--patch-counts", type=int, nargs="+", default=[4, 8, 16, 32, 64, 196])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)

    experiment12 = json.loads(EXPERIMENT12_SUMMARY.read_text())
    experiment13 = json.loads(EXPERIMENT13_SUMMARY.read_text())
    features = experiment13["selected"]["features"]
    combined_scores = np.asarray(experiment12["discoveries"]["blur"]["score"]) + np.asarray(experiment12["discoveries"]["noise"]["score"])
    weights = torch.as_tensor(combined_scores[features], device=device).clamp_min(0)
    weights = weights / weights.mean().clamp_min(1e-8)
    loader = DataLoader(make_dataset(("clean", None, 0), args.calibration_samples, args.calibration_start, args.seed), batch_size=args.batch_size)
    calibration = calibrate_quantiles(model, sae, loader, device, 4, [args.quantile], args.seed)
    threshold = calibration["quantiles"][str(args.quantile)].to(device)
    configurations = {}
    for patch_count in args.patch_counts:
        strategy = "all" if patch_count == 196 else "top"
        for alpha in args.alphas:
            configurations[f"patches{patch_count}_alpha{alpha:g}"] = {
                "features": features, "weights": weights, "alpha": alpha,
                "patch_count": patch_count, "strategy": strategy,
            }

    validation, evaluation, arrays = {}, {}, {}
    for condition in CONDITIONS:
        loader = DataLoader(make_dataset(condition, args.validation_samples, args.validation_start, args.seed), batch_size=args.batch_size)
        validation[condition[0]], _ = evaluate(model, sae, loader, device, threshold, configurations)
        loader = DataLoader(make_dataset(condition, args.evaluation_samples, args.evaluation_start, args.seed), batch_size=args.batch_size)
        evaluation[condition[0]], arrays[condition[0]] = evaluate(model, sae, loader, device, threshold, configurations)
        np.savez_compressed(output_dir / f"{condition[0]}_paired_outcomes.npz", **arrays[condition[0]])

    clean_baseline = validation["clean"]["original_vit"]["accuracy"]
    selected_name = max(configurations, key=lambda name: (
        validation["blur4"][name]["accuracy"]
        - 2 * max(0, clean_baseline - validation["clean"][name]["accuracy"])
    ))
    summary = {
        "configuration": vars(args) | {
            "device": str(device),
            "status": "exploratory: validation/evaluation indices were examined in Experiment 13",
            "fixed_features": features,
        },
        "validation_selected": selected_name,
        "validation": validation,
        "evaluation": evaluation,
        "blur4_pareto_front": pareto_front(evaluation["blur4"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"validation_selected": selected_name, "blur4_pareto_front": summary["blur4_pareto_front"]}, indent=2))
    print(f"Saved Experiment 14 to {output_dir}")


if __name__ == "__main__":
    main()
