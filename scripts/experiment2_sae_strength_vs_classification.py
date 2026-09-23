import argparse
import csv
import json
import math
import platform
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification, __version__ as transformers_version

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment1_base_blur4_sae_identity_strength import (
    BASE_MODEL,
    EPSILON,
    OUTPUT_ROOT as EXPERIMENT1_ROOT,
    PairedDataset,
    SAE_DIR,
    load_sae,
)


OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment2_strength_vs_classification"
EXPERIMENT1_FULL = EXPERIMENT1_ROOT / "full_1000"


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def classification_margin(logits, labels):
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    alternatives = logits.clone()
    alternatives.scatter_(1, labels[:, None], float("-inf"))
    return true_logits, true_logits - alternatives.max(dim=1).values


def prediction_margin(logits):
    top_two = logits.topk(2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def correlation(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    pearson = pearsonr(x, y)
    spearman = spearmanr(x, y)
    try:
        interval = pearson.confidence_interval(confidence_level=0.95)
        low, high = float(interval.low), float(interval.high)
    except AttributeError:
        low, high = None, None
    return {
        "n": int(x.size),
        "pearson_r": float(pearson.statistic),
        "pearson_pvalue": float(pearson.pvalue),
        "pearson_ci95_low": low,
        "pearson_ci95_high": high,
        "spearman_rho": float(spearman.statistic),
        "spearman_pvalue": float(spearman.pvalue),
    }


def describe(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "standard_deviation": float(values.std(ddof=1)) if values.size > 1 else 0.0,
    }


def benjamini_hochberg(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(max=1)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def group_test(group_a, group_b):
    group_a = np.asarray(group_a, dtype=np.float64)
    group_b = np.asarray(group_b, dtype=np.float64)
    test = mannwhitneyu(group_b, group_a, alternative="two-sided")
    rank_biserial = 2 * float(test.statistic) / (len(group_a) * len(group_b)) - 1
    return {
        "mann_whitney_u": float(test.statistic),
        "pvalue": float(test.pvalue),
        "rank_biserial_effect_b_vs_a": rank_biserial,
    }


def scatter_plot(path, x, y, xlabel, ylabel, horizontal_zero=False):
    plt.figure(figsize=(7, 5))
    plt.scatter(x, y, s=10, alpha=0.35)
    if horizontal_zero:
        plt.axhline(0, color="black", linestyle="--")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Experiment 2: SAE strength vs classification")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--minimum-feature-support", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    minimum_support = args.minimum_feature_support or max(5, round(args.samples * 0.05))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedDataset(args.samples, args.start_index, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae, sae_metadata = load_sae(device)
    latent_dim = sae.latent_dim

    print(f"Device: {device}")
    print(f"ViT: {BASE_MODEL} (frozen)")
    print(f"SAE: {SAE_DIR.relative_to(PROJECT_ROOT)}")
    print("Layer/tokens: hidden_states[-2], 196 patches, CLS excluded")
    print("Active feature: z > 0")
    print(f"Latent dimension: {latent_dim}; k: not applicable")
    print("~990 active/shared means SAE dimensions per patch, averaged across 196 patches")
    print(f"Minimum per-feature image support: {minimum_support}")

    image_rows = []
    feature_delta = np.empty((args.samples, latent_dim), dtype=np.float32)
    feature_valid = np.empty((args.samples, latent_dim), dtype=np.bool_)
    clean_feature_sum = np.zeros(latent_dim, dtype=np.float64)
    blur_feature_sum = np.zeros(latent_dim, dtype=np.float64)
    clean_frequency = np.zeros(latent_dim, dtype=np.float64)
    blur_frequency = np.zeros(latent_dim, dtype=np.float64)
    row_offset = 0

    with torch.no_grad():
        for clean, blurred, labels, image_ids, paths in tqdm(loader):
            batch_size = clean.shape[0]
            images = torch.cat((clean, blurred), dim=0).to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            clean_logits, blur_logits = outputs.logits.split(batch_size)
            patches = outputs.hidden_states[-2][:, 1:, :]
            assert patches.shape[1:] == (196, 768)
            clean_patches, blur_patches = patches.split(batch_size)
            clean_z = sae.encode(clean_patches.flatten(0, 1)).reshape(batch_size, 196, latent_dim)
            blur_z = sae.encode(blur_patches.flatten(0, 1)).reshape(batch_size, 196, latent_dim)
            assert clean_z.shape == blur_z.shape == (batch_size, 196, latent_dim)
            clean_active = clean_z > 0
            blur_active = blur_z > 0
            shared = clean_active & blur_active
            union = clean_active | blur_active
            delta = blur_z - clean_z

            clean_mean = clean_z.mean(dim=1)
            blur_mean = blur_z.mean(dim=1)
            feature_delta[row_offset : row_offset + batch_size] = (blur_mean - clean_mean).cpu().numpy()
            feature_valid[row_offset : row_offset + batch_size] = shared.any(dim=1).cpu().numpy()
            clean_feature_sum += clean_mean.sum(dim=0).double().cpu().numpy()
            blur_feature_sum += blur_mean.sum(dim=0).double().cpu().numpy()
            clean_frequency += clean_active.sum(dim=(0, 1)).double().cpu().numpy()
            blur_frequency += blur_active.sum(dim=(0, 1)).double().cpu().numpy()

            clean_true, clean_margin = classification_margin(clean_logits, labels)
            blur_true, blur_margin = classification_margin(blur_logits, labels)
            clean_pred_margin = prediction_margin(clean_logits)
            blur_pred_margin = prediction_margin(blur_logits)
            clean_predictions = clean_logits.argmax(dim=1)
            blur_predictions = blur_logits.argmax(dim=1)

            for index in range(batch_size):
                clean_correct = bool(clean_predictions[index] == labels[index])
                blur_correct = bool(blur_predictions[index] == labels[index])
                if clean_correct and blur_correct:
                    group = "clean_correct_blur_correct"
                elif clean_correct and not blur_correct:
                    group = "clean_correct_blur_wrong"
                elif not clean_correct and blur_correct:
                    group = "clean_wrong_blur_correct"
                else:
                    group = "clean_wrong_blur_wrong"
                clean_vector = clean_z[index]
                blur_vector = blur_z[index]
                clean_norm = clean_vector.norm(dim=-1)
                blur_norm = blur_vector.norm(dim=-1)
                shared_changes = delta[index][shared[index]]
                shared_count = shared[index].sum(dim=-1)
                clean_count = clean_active[index].sum(dim=-1)
                image_rows.append(
                    {
                        "image_id": image_ids[index],
                        "image_path": paths[index],
                        "ground_truth": labels[index].item(),
                        "clean_prediction": clean_predictions[index].item(),
                        "blur_prediction": blur_predictions[index].item(),
                        "clean_correct": clean_correct,
                        "blur_correct": blur_correct,
                        "classification_group": group,
                        "clean_true_class_logit": clean_true[index].item(),
                        "blur_true_class_logit": blur_true[index].item(),
                        "delta_true_class_logit": (blur_true[index] - clean_true[index]).item(),
                        "clean_classification_margin": clean_margin[index].item(),
                        "blur_classification_margin": blur_margin[index].item(),
                        "delta_classification_margin": (blur_margin[index] - clean_margin[index]).item(),
                        "clean_prediction_margin": clean_pred_margin[index].item(),
                        "blur_prediction_margin": blur_pred_margin[index].item(),
                        "sae_cosine_similarity": F.cosine_similarity(clean_vector, blur_vector, dim=-1).mean().item(),
                        "feature_retention": (shared_count / clean_count.clamp_min(1)).float().mean().item(),
                        "jaccard_similarity": (shared_count / union[index].sum(dim=-1).clamp_min(1)).float().mean().item(),
                        "relative_sae_change": (delta[index].norm(dim=-1) / clean_norm.clamp_min(EPSILON)).mean().item(),
                        "clean_sae_norm": clean_norm.mean().item(),
                        "blur_sae_norm": blur_norm.mean().item(),
                        "norm_ratio": (blur_norm / clean_norm.clamp_min(EPSILON)).mean().item(),
                        "shared_feature_mean_signed_change": shared_changes.mean().item(),
                        "shared_feature_mean_absolute_change": shared_changes.abs().mean().item(),
                        "shared_features_percent_increased": (shared_changes > 0).float().mean().item() * 100,
                        "shared_features_percent_decreased": (shared_changes < 0).float().mean().item() * 100,
                    }
                )
            row_offset += batch_size

    assert row_offset == args.samples
    group_a_indices = np.array([r["classification_group"] == "clean_correct_blur_correct" for r in image_rows])
    group_b_indices = np.array([r["classification_group"] == "clean_correct_blur_wrong" for r in image_rows])
    if np.any([r["clean_classification_margin"] <= 0 for r in image_rows if r["clean_correct"]]):
        raise AssertionError("A clean-correct image has non-positive ground-truth margin.")
    if np.any([r["blur_classification_margin"] >= 0 for r in image_rows if r["classification_group"] == "clean_correct_blur_wrong"]):
        raise AssertionError("A clean-correct→blur-wrong image has non-negative blur margin.")

    metric_names = [
        "feature_retention",
        "jaccard_similarity",
        "sae_cosine_similarity",
        "relative_sae_change",
        "shared_feature_mean_absolute_change",
        "norm_ratio",
    ]
    delta_margin = np.array([r["delta_classification_margin"] for r in image_rows])
    delta_true = np.array([r["delta_true_class_logit"] for r in image_rows])
    correlations = {
        name: correlation([r[name] for r in image_rows], delta_margin)
        for name in metric_names
    }
    correlations["relative_sae_change_vs_delta_true_class_logit"] = correlation(
        [r["relative_sae_change"] for r in image_rows], delta_true
    )

    comparison_variables = [
        "clean_true_class_logit", "blur_true_class_logit", "delta_true_class_logit",
        "clean_classification_margin", "blur_classification_margin", "delta_classification_margin",
        "sae_cosine_similarity", "feature_retention", "relative_sae_change", "norm_ratio",
        "shared_feature_mean_absolute_change",
    ]
    group_a_rows = [r for r in image_rows if r["classification_group"] == "clean_correct_blur_correct"]
    group_b_rows = [r for r in image_rows if r["classification_group"] == "clean_correct_blur_wrong"]
    group_comparison = {}
    for name in comparison_variables:
        values_a = [r[name] for r in group_a_rows]
        values_b = [r[name] for r in group_b_rows]
        group_comparison[name] = {
            "group_a": describe(values_a),
            "group_b": describe(values_b),
            "test": group_test(values_a, values_b),
        }

    feature_rows = []
    eligible_indices = []
    raw_pearson_p = []
    raw_spearman_p = []
    total_patches = args.samples * 196
    for feature_index in range(latent_dim):
        valid = feature_valid[:, feature_index]
        valid_count = int(valid.sum())
        delta_values = feature_delta[:, feature_index]
        row = {
            "feature_index": feature_index,
            "valid_sample_count": valid_count,
            "mean_clean_activation": clean_feature_sum[feature_index] / args.samples,
            "mean_blur_activation": blur_feature_sum[feature_index] / args.samples,
            "mean_signed_change": float(delta_values.mean()),
            "mean_absolute_change": float(np.abs(delta_values).mean()),
            "mean_change_correct_to_correct": float(delta_values[group_a_indices].mean()),
            "mean_change_correct_to_wrong": float(delta_values[group_b_indices].mean()),
            "group_change_difference": float(delta_values[group_b_indices].mean() - delta_values[group_a_indices].mean()),
            "clean_activation_frequency": clean_frequency[feature_index] / total_patches,
            "blur_activation_frequency": blur_frequency[feature_index] / total_patches,
            "pearson_with_delta_margin": None,
            "pearson_pvalue": None,
            "pearson_qvalue": None,
            "spearman_with_delta_margin": None,
            "spearman_pvalue": None,
            "spearman_qvalue": None,
        }
        if valid_count >= minimum_support and np.std(delta_values[valid]) > 0:
            pearson = pearsonr(delta_values[valid], delta_margin[valid])
            spearman = spearmanr(delta_values[valid], delta_margin[valid])
            row.update(
                pearson_with_delta_margin=float(pearson.statistic),
                pearson_pvalue=float(pearson.pvalue),
                spearman_with_delta_margin=float(spearman.statistic),
                spearman_pvalue=float(spearman.pvalue),
            )
            eligible_indices.append(feature_index)
            raw_pearson_p.append(pearson.pvalue)
            raw_spearman_p.append(spearman.pvalue)
        feature_rows.append(row)

    pearson_q = benjamini_hochberg(raw_pearson_p)
    spearman_q = benjamini_hochberg(raw_spearman_p)
    for position, feature_index in enumerate(eligible_indices):
        feature_rows[feature_index]["pearson_qvalue"] = float(pearson_q[position])
        feature_rows[feature_index]["spearman_qvalue"] = float(spearman_q[position])

    eligible_rows = [row for row in feature_rows if row["spearman_with_delta_margin"] is not None]
    ranked_correlation = sorted(eligible_rows, key=lambda row: abs(row["spearman_with_delta_margin"]), reverse=True)
    ranked_group = sorted(feature_rows, key=lambda row: abs(row["group_change_difference"]), reverse=True)

    clean_correct_rows = [r for r in image_rows if r["clean_correct"]]
    x = np.array([[r["relative_sae_change"], r["sae_cosine_similarity"], r["norm_ratio"], r["shared_feature_mean_absolute_change"]] for r in clean_correct_rows])
    y = np.array([not r["blur_correct"] for r in clean_correct_rows], dtype=int)
    splits = min(5, int(np.bincount(y).min()))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=args.seed)
    estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=args.seed))
    probabilities = cross_val_predict(estimator, x, y, cv=cv, method="predict_proba")[:, 1]
    predictions = probabilities >= 0.5
    estimator.fit(x, y)
    logistic = estimator.named_steps["logisticregression"]
    prediction_analysis = {
        "samples": int(len(y)),
        "failures": int(y.sum()),
        "folds": splits,
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "accuracy": float(accuracy_score(y, predictions)),
        "standardized_coefficients": dict(zip(
            ["relative_sae_change", "sae_cosine_similarity", "norm_ratio", "shared_feature_mean_absolute_change"],
            logistic.coef_[0].tolist(),
        )),
    }

    write_csv(output_dir / "image_level_metrics.csv", image_rows, list(image_rows[0]))
    feature_fields = list(feature_rows[0])
    write_csv(output_dir / "per_feature_metrics.csv", feature_rows, feature_fields)
    write_csv(output_dir / "features_ranked_by_margin_correlation.csv", ranked_correlation, feature_fields)
    write_csv(output_dir / "features_ranked_by_group_difference.csv", ranked_group, feature_fields)

    relative = np.array([r["relative_sae_change"] for r in image_rows])
    shared_absolute = np.array([r["shared_feature_mean_absolute_change"] for r in image_rows])
    cosine = np.array([r["sae_cosine_similarity"] for r in image_rows])
    norm_ratio = np.array([r["norm_ratio"] for r in image_rows])
    scatter_plot(output_dir / "plot1_relative_change_vs_margin_change.png", relative, delta_margin, "Relative SAE change", "Δ classification margin", True)
    plt.figure(figsize=(7, 5)); plt.hist([r["relative_sae_change"] for r in group_a_rows], bins=40, alpha=.55, density=True, label="Correct→Correct"); plt.hist([r["relative_sae_change"] for r in group_b_rows], bins=40, alpha=.55, density=True, label="Correct→Wrong"); plt.legend(); plt.xlabel("Relative SAE change"); plt.ylabel("Density"); plt.tight_layout(); plt.savefig(output_dir / "plot2_group_relative_change.png", dpi=250); plt.close()
    plt.figure(figsize=(7, 5)); plt.hist([r["delta_classification_margin"] for r in group_a_rows], bins=40, alpha=.55, density=True, label="Correct→Correct"); plt.hist([r["delta_classification_margin"] for r in group_b_rows], bins=40, alpha=.55, density=True, label="Correct→Wrong"); plt.axvline(0, color="black", linestyle="--"); plt.legend(); plt.xlabel("Δ classification margin"); plt.ylabel("Density"); plt.tight_layout(); plt.savefig(output_dir / "plot3_group_margin_change.png", dpi=250); plt.close()
    scatter_plot(output_dir / "plot4_shared_abs_change_vs_margin_change.png", shared_absolute, delta_margin, "Mean absolute shared-feature change", "Δ classification margin", True)
    scatter_plot(output_dir / "plot5_cosine_vs_margin_change.png", cosine, delta_margin, "SAE cosine similarity", "Δ classification margin", True)
    scatter_plot(output_dir / "plot6_norm_ratio_vs_margin_change.png", norm_ratio, delta_margin, "Blur/Clean SAE norm ratio", "Δ classification margin", True)
    top_group = ranked_group[:20]; positions = np.arange(len(top_group)); width=.4; plt.figure(figsize=(10, 6)); plt.bar(positions-width/2, [r["mean_change_correct_to_correct"] for r in top_group], width, label="Correct→Correct"); plt.bar(positions+width/2, [r["mean_change_correct_to_wrong"] for r in top_group], width, label="Correct→Wrong"); plt.xticks(positions, [r["feature_index"] for r in top_group], rotation=75); plt.ylabel("Mean signed activation change"); plt.legend(); plt.tight_layout(); plt.savefig(output_dir / "plot7_top_group_difference_features.png", dpi=250); plt.close()
    top_corr = ranked_correlation[:20]; plt.figure(figsize=(9, 6)); plt.barh([str(r["feature_index"]) for r in reversed(top_corr)], [r["spearman_with_delta_margin"] for r in reversed(top_corr)]); plt.xlabel("Spearman correlation with Δ margin"); plt.ylabel("Feature index"); plt.tight_layout(); plt.savefig(output_dir / "plot8_top_margin_associated_features.png", dpi=250); plt.close()

    config = {
        **vars(args),
        "minimum_feature_support_effective": minimum_support,
        "model": BASE_MODEL,
        "sae_checkpoint": str(SAE_DIR.relative_to(PROJECT_ROOT)),
        "sae_implementation": sae_metadata["implementation"],
        "layer": "hidden_states[-2]",
        "tokens": "196 patches; CLS excluded",
        "active_definition": "z > 0",
        "feature_delta_definition": "mean Blur-4 minus clean activation across 196 patches per image",
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "scipy": scipy.__version__,
    }
    classification = {
        "clean_accuracy": float(np.mean([r["clean_correct"] for r in image_rows])),
        "blur4_accuracy": float(np.mean([r["blur_correct"] for r in image_rows])),
        "groups": {name: sum(r["classification_group"] == name for r in image_rows) for name in sorted(set(r["classification_group"] for r in image_rows))},
        "mean_delta_true_class_logit": float(delta_true.mean()),
        "mean_delta_classification_margin": float(delta_margin.mean()),
    }
    summary = {
        "configuration": config,
        "classification": classification,
        "correlations": correlations,
        "group_comparison": group_comparison,
        "logistic_regression": prediction_analysis,
        "eligible_features": len(eligible_rows),
        "fdr_significant_spearman_features_q05": sum(r["spearman_qvalue"] is not None and r["spearman_qvalue"] < .05 for r in eligible_rows),
        "top_features_by_margin_correlation": ranked_correlation[:20],
        "top_features_by_group_difference": ranked_group[:20],
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    if args.samples == 1000 and args.start_index == 11000:
        if abs(classification["clean_accuracy"] - .798) > .005 or abs(classification["blur4_accuracy"] - .633) > .005:
            raise AssertionError("Classification accuracy does not reproduce Experiment 1.")
        experiment1_ids = [r["image_id"] for r in csv.DictReader((EXPERIMENT1_FULL / "image_level_metrics.csv").open())]
        if experiment1_ids != [r["image_id"] for r in image_rows]:
            raise AssertionError("Experiment 2 image pairs differ from Experiment 1.")

    primary = correlations["relative_sae_change"]
    print(f"\nClean accuracy: {classification['clean_accuracy']:.2%}")
    print(f"Blur-4 accuracy: {classification['blur4_accuracy']:.2%}")
    print(f"Groups: {classification['groups']}")
    print(f"Mean Δ true-class logit: {classification['mean_delta_true_class_logit']:.6f}")
    print(f"Mean Δ margin: {classification['mean_delta_classification_margin']:.6f}")
    print(f"Relative SAE change vs Δ margin: Pearson={primary['pearson_r']:.4f}, Spearman={primary['spearman_rho']:.4f}")
    print(f"Eligible/FDR-significant features: {len(eligible_rows)}/{summary['fdr_significant_spearman_features_q05']}")
    print(f"Logistic ROC-AUC: {prediction_analysis['roc_auc']:.4f}")
    print(f"Saved Experiment 2 to {output_dir}")


if __name__ == "__main__":
    main()
