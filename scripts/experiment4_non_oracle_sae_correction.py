import argparse
import csv
import json
from pathlib import Path
import random
import sys

import numpy as np
from scipy.stats import pearsonr
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment1_base_blur4_sae_identity_strength import (
    BASE_MODEL,
    EPSILON,
    PairedDataset,
    SAE_DIR,
    load_sae,
)
from scripts.experiment2_sae_strength_vs_classification import (
    benjamini_hochberg,
    classification_margin,
)
from scripts.experiment3_sae_causal_intervention import downstream_logits


OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment4_non_oracle_correction"


def make_loader(samples, start_index, seed, batch_size, num_workers):
    dataset = PairedDataset(samples, start_index, seed)
    if len(dataset) != samples:
        raise ValueError(f"Requested {samples} images at {start_index}, found {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )


def encode_patches(sae, hidden):
    batch = hidden.shape[0]
    return sae.encode(hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)


def discovery_statistics(model, sae, loader, device):
    latent_dim = sae.latent_dim
    image_delta = np.empty((len(loader.dataset), latent_dim), dtype=np.float32)
    delta_margin = np.empty(len(loader.dataset), dtype=np.float32)
    clean_correct = np.empty(len(loader.dataset), dtype=np.bool_)
    blur_correct = np.empty(len(loader.dataset), dtype=np.bool_)
    sums = {
        name: torch.zeros(latent_dim, dtype=torch.float64)
        for name in ["x", "y", "xx", "xy", "yy"]
    }
    offset = 0
    with torch.no_grad():
        for clean, blur, labels, _, _ in tqdm(loader, desc="Discovery"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, blur]).to(device),
                output_hidden_states=True,
            )
            clean_logits, blur_logits = outputs.logits.split(batch)
            clean_hidden, blur_hidden = outputs.hidden_states[-2].split(batch)
            clean_z = encode_patches(sae, clean_hidden)
            blur_z = encode_patches(sae, blur_hidden)
            image_delta[offset : offset + batch] = (
                blur_z.mean(1) - clean_z.mean(1)
            ).cpu().numpy()
            clean_margin = classification_margin(clean_logits, labels)[1]
            blur_margin = classification_margin(blur_logits, labels)[1]
            delta_margin[offset : offset + batch] = (blur_margin - clean_margin).cpu().numpy()
            clean_correct[offset : offset + batch] = (clean_logits.argmax(1) == labels).cpu().numpy()
            blur_correct[offset : offset + batch] = (blur_logits.argmax(1) == labels).cpu().numpy()

            x = blur_z.double().sum((0, 1)).cpu()
            y = clean_z.double().sum((0, 1)).cpu()
            sums["x"] += x
            sums["y"] += y
            sums["xx"] += blur_z.double().square().sum((0, 1)).cpu()
            sums["xy"] += (blur_z.double() * clean_z.double()).sum((0, 1)).cpu()
            sums["yy"] += clean_z.double().square().sum((0, 1)).cpu()
            offset += batch
    return image_delta, delta_margin, clean_correct, blur_correct, sums


def fit_affine(sums, observations, ridge):
    mean_x = sums["x"].numpy() / observations
    mean_y = sums["y"].numpy() / observations
    var_x = sums["xx"].numpy() / observations - mean_x**2
    var_y = sums["yy"].numpy() / observations - mean_y**2
    covariance = sums["xy"].numpy() / observations - mean_x * mean_y
    scale = covariance / (var_x + ridge * np.maximum(var_x, EPSILON) + EPSILON)
    intercept = mean_y - scale * mean_x
    return scale.astype(np.float32), intercept.astype(np.float32), mean_x, mean_y, var_x, var_y


def feature_table(image_delta, delta_margin, clean_correct, blur_correct, moments, minimum_consistency):
    scale, intercept, mean_blur, mean_clean, var_blur, var_clean = moments
    eligible = np.std(image_delta, axis=0) > EPSILON
    indices = np.flatnonzero(eligible)
    correlations = np.zeros(image_delta.shape[1], dtype=np.float64)
    pvalues = np.ones(image_delta.shape[1], dtype=np.float64)
    for feature in tqdm(indices, desc="Feature associations"):
        result = pearsonr(image_delta[:, feature], delta_margin)
        correlations[feature] = result.statistic
        pvalues[feature] = result.pvalue
    qvalues = np.ones(image_delta.shape[1], dtype=np.float64)
    qvalues[indices] = benjamini_hochberg(pvalues[indices])
    signed_mean = image_delta.mean(0)
    sign = np.sign(signed_mean)
    consistency = np.maximum((image_delta > 0).mean(0), (image_delta < 0).mean(0))
    standardized = signed_mean / np.sqrt(np.maximum(var_clean, EPSILON))
    correct_to_correct = clean_correct & blur_correct
    correct_to_wrong = clean_correct & ~blur_correct
    mean_cc = image_delta[correct_to_correct].mean(0) if correct_to_correct.any() else np.zeros_like(signed_mean)
    mean_cw = image_delta[correct_to_wrong].mean(0) if correct_to_wrong.any() else np.zeros_like(signed_mean)
    failure_difference = mean_cw - mean_cc
    rows = []
    for feature in range(image_delta.shape[1]):
        rows.append({
            "feature_index": feature,
            "mean_clean_activation": float(mean_clean[feature]),
            "mean_blur_activation": float(mean_blur[feature]),
            "mean_blur_minus_clean": float(signed_mean[feature]),
            "standardized_change": float(standardized[feature]),
            "direction_consistency": float(consistency[feature]),
            "mean_change_correct_correct": float(mean_cc[feature]),
            "mean_change_correct_wrong": float(mean_cw[feature]),
            "failure_group_difference": float(failure_difference[feature]),
            "pearson_delta_margin": float(correlations[feature]),
            "pearson_pvalue": float(pvalues[feature]),
            "pearson_qvalue": float(qvalues[feature]),
            "affine_scale": float(scale[feature]),
            "affine_intercept": float(intercept[feature]),
        })
    candidates = [
        row for row in rows
        if row["pearson_qvalue"] < 0.05
        and row["direction_consistency"] >= minimum_consistency
        and row["pearson_delta_margin"] * row["mean_blur_minus_clean"] < 0
    ]
    candidates.sort(
        key=lambda row: abs(row["pearson_delta_margin"])
        * abs(row["standardized_change"])
        * row["direction_consistency"],
        reverse=True,
    )
    return rows, candidates


def corrected_latent(z, features, scale, intercept, alpha, reverse=False):
    corrected = z.clone()
    selected = torch.as_tensor(features, device=z.device, dtype=torch.long)
    predicted = z[..., selected] * scale[selected] + intercept[selected]
    direction = -1.0 if reverse else 1.0
    corrected[..., selected] += direction * alpha * (predicted - z[..., selected])
    return corrected


def evaluate(model, sae, loader, device, configurations, scale, intercept):
    scale_tensor = torch.as_tensor(scale, device=device)
    intercept_tensor = torch.as_tensor(intercept, device=device)
    totals = {
        name: {"clean_correct": 0, "blur_correct": 0, "clean_margin": 0.0, "blur_margin": 0.0}
        for name in configurations
    }
    totals["original_vit"] = {
        "clean_correct": 0,
        "blur_correct": 0,
        "clean_margin": 0.0,
        "blur_margin": 0.0,
    }
    total = 0
    with torch.no_grad():
        for clean, blur, labels, _, _ in tqdm(loader, desc="Evaluation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, blur]).to(device),
                output_hidden_states=True,
            )
            clean_original_logits, blur_original_logits = outputs.logits.split(batch)
            clean_hidden, blur_hidden = outputs.hidden_states[-2].split(batch)
            clean_z = encode_patches(sae, clean_hidden)
            blur_z = encode_patches(sae, blur_hidden)
            clean_original_margin = classification_margin(clean_original_logits, labels)[1]
            blur_original_margin = classification_margin(blur_original_logits, labels)[1]
            totals["original_vit"]["clean_correct"] += int(
                (clean_original_logits.argmax(1) == labels).sum()
            )
            totals["original_vit"]["blur_correct"] += int(
                (blur_original_logits.argmax(1) == labels).sum()
            )
            totals["original_vit"]["clean_margin"] += float(clean_original_margin.sum())
            totals["original_vit"]["blur_margin"] += float(blur_original_margin.sum())
            for name, config in configurations.items():
                features = config["features"]
                alpha = config["alpha"]
                reverse = config.get("reverse", False)
                clean_modified = corrected_latent(
                    clean_z, features, scale_tensor, intercept_tensor, alpha, reverse
                )
                blur_modified = corrected_latent(
                    blur_z, features, scale_tensor, intercept_tensor, alpha, reverse
                )
                clean_site = torch.cat([clean_hidden[:, :1], sae.decode(clean_modified)], 1)
                blur_site = torch.cat([blur_hidden[:, :1], sae.decode(blur_modified)], 1)
                clean_logits = downstream_logits(model, clean_site)
                blur_logits = downstream_logits(model, blur_site)
                clean_margin = classification_margin(clean_logits, labels)[1]
                blur_margin = classification_margin(blur_logits, labels)[1]
                totals[name]["clean_correct"] += int((clean_logits.argmax(1) == labels).sum())
                totals[name]["blur_correct"] += int((blur_logits.argmax(1) == labels).sum())
                totals[name]["clean_margin"] += float(clean_margin.sum())
                totals[name]["blur_margin"] += float(blur_margin.sum())
            total += batch
    return {
        name: {
            "clean_accuracy": values["clean_correct"] / total,
            "blur_accuracy": values["blur_correct"] / total,
            "mean_clean_margin": values["clean_margin"] / total,
            "mean_blur_margin": values["blur_margin"] / total,
            "samples": total,
        }
        for name, values in totals.items()
    }


def write_csv(path, rows):
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Experiment 4: non-oracle SAE Blur-4 correction")
    parser.add_argument("--discovery-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=1000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--discovery-start", type=int, default=12000)
    parser.add_argument("--validation-start", type=int, default=14000)
    parser.add_argument("--test-start", type=int, default=15000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--minimum-consistency", type=float, default=0.60)
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[1, 4, 16, 32])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--clean-loss-penalty", type=float, default=1.0)
    parser.add_argument("--random-controls", type=int, default=100)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ranges = [
        (0, 11000, "SAE train/validation"),
        (11000, 12000, "Experiments 1-3"),
        (args.discovery_start, args.discovery_start + args.discovery_samples, "discovery"),
        (args.validation_start, args.validation_start + args.validation_samples, "validation"),
        (args.test_start, args.test_start + args.test_samples, "test"),
    ]
    for left, first in enumerate(ranges):
        for second in ranges[left + 1 :]:
            if max(first[0], second[0]) < min(first[1], second[1]):
                raise ValueError(f"Data leakage: {first[2]} overlaps {second[2]}")

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae, sae_metadata = load_sae(device)

    discovery_loader = make_loader(
        args.discovery_samples, args.discovery_start, args.seed, args.batch_size, args.num_workers
    )
    image_delta, delta_margin, clean_correct, blur_correct, sums = discovery_statistics(
        model, sae, discovery_loader, device
    )
    observations = args.discovery_samples * 196
    moments = fit_affine(sums, observations, args.ridge)
    scale, intercept = moments[:2]
    feature_rows, candidates = feature_table(
        image_delta,
        delta_margin,
        clean_correct,
        blur_correct,
        moments,
        args.minimum_consistency,
    )
    if not candidates:
        raise RuntimeError("No features passed discovery criteria; do not tune on the test set")
    write_csv(output_dir / "discovery_feature_statistics.csv", feature_rows)
    write_csv(output_dir / "discovery_candidates.csv", candidates)

    configurations = {"baseline": {"features": [], "alpha": 0.0}}
    for count in args.feature_counts:
        selected = [row["feature_index"] for row in candidates[:count]]
        if not selected:
            continue
        for alpha in args.alphas:
            configurations[f"top{len(selected)}_alpha{alpha:g}"] = {
                "features": selected,
                "alpha": alpha,
            }
    validation_loader = make_loader(
        args.validation_samples, args.validation_start, args.seed, args.batch_size, args.num_workers
    )
    validation = evaluate(model, sae, validation_loader, device, configurations, scale, intercept)
    baseline_validation = validation["baseline"]
    eligible_names = [name for name in configurations if name != "baseline"]
    best_name = max(
        eligible_names,
        key=lambda name: validation[name]["blur_accuracy"]
        - args.clean_loss_penalty
        * max(0.0, baseline_validation["clean_accuracy"] - validation[name]["clean_accuracy"]),
    )
    best = configurations[best_name]

    rng = np.random.default_rng(args.seed)
    candidate_features = np.array([row["feature_index"] for row in candidates])
    all_features = np.arange(sae.latent_dim)
    test_configurations = {
        "baseline": configurations["baseline"],
        "selected": best,
        "reverse": best | {"reverse": True},
    }
    selected_set = set(best["features"])
    standardized_magnitudes = np.abs(
        np.array([row["standardized_change"] for row in feature_rows])
    )
    for draw in range(args.random_controls):
        random_features = rng.choice(all_features, size=len(best["features"]), replace=False).tolist()
        test_configurations[f"random_{draw:03d}"] = {
            "features": random_features,
            "alpha": best["alpha"],
        }
        matched_features = []
        unavailable = set(selected_set)
        for feature in best["features"]:
            distances = np.abs(standardized_magnitudes - standardized_magnitudes[feature])
            nearest = np.argsort(distances)
            pool = [index for index in nearest[:256] if index not in unavailable]
            chosen = int(rng.choice(pool))
            matched_features.append(chosen)
            unavailable.add(chosen)
        test_configurations[f"matched_{draw:03d}"] = {
            "features": matched_features,
            "alpha": best["alpha"],
        }
    permuted_scale = scale.copy()
    permuted_intercept = intercept.copy()
    permutation = rng.permutation(candidate_features)
    permuted_scale[candidate_features] = scale[permutation]
    permuted_intercept[candidate_features] = intercept[permutation]
    shuffled_pair_scale = np.zeros_like(scale)
    shuffled_pair_intercept = moments[3].astype(np.float32)

    test_loader = make_loader(
        args.test_samples, args.test_start, args.seed, args.batch_size, args.num_workers
    )
    test = evaluate(model, sae, test_loader, device, test_configurations, scale, intercept)
    feature_permuted = evaluate(
        model,
        sae,
        test_loader,
        device,
        {"feature_permuted_mapping": best},
        permuted_scale,
        permuted_intercept,
    )["feature_permuted_mapping"]
    shuffled_pair = evaluate(
        model,
        sae,
        test_loader,
        device,
        {"shuffled_pair_mapping": best},
        shuffled_pair_scale,
        shuffled_pair_intercept,
    )["shuffled_pair_mapping"]
    random_blur = np.array([test[name]["blur_accuracy"] for name in test if name.startswith("random_")])
    matched_blur = np.array([test[name]["blur_accuracy"] for name in test if name.startswith("matched_")])
    selected_gain = test["selected"]["blur_accuracy"] - test["baseline"]["blur_accuracy"]
    random_gains = random_blur - test["baseline"]["blur_accuracy"]
    matched_gains = matched_blur - test["baseline"]["blur_accuracy"]
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "sae": str(SAE_DIR.relative_to(PROJECT_ROOT)),
            "sae_implementation": sae_metadata["implementation"],
            "site": "hidden_states[-2], patch tokens; preserve original CLS",
        },
        "split_ranges": {name: [start, end] for start, end, name in ranges},
        "discovery": {
            "candidate_count": len(candidates),
            "clean_accuracy": float(clean_correct.mean()),
            "blur_accuracy": float(blur_correct.mean()),
            "top_candidates": candidates[:32],
        },
        "validation": validation,
        "selected_configuration": {"name": best_name} | best,
        "locked_test": {
            "original_vit": test["original_vit"],
            "baseline": test["baseline"],
            "selected": test["selected"],
            "reverse": test["reverse"],
            "feature_permuted_mapping": feature_permuted,
            "shuffled_pair_mapping": shuffled_pair,
            "blur_accuracy_gain": selected_gain,
            "clean_accuracy_change_if_applied_to_clean": test["selected"]["clean_accuracy"] - test["baseline"]["clean_accuracy"],
            "random_control_mean_gain": float(random_gains.mean()),
            "random_control_std_gain": float(random_gains.std(ddof=1)) if len(random_gains) > 1 else None,
            "random_empirical_pvalue": float((1 + np.sum(random_gains >= selected_gain)) / (1 + len(random_gains))),
            "magnitude_matched_mean_gain": float(matched_gains.mean()),
            "magnitude_matched_std_gain": float(matched_gains.std(ddof=1)) if len(matched_gains) > 1 else None,
            "magnitude_matched_empirical_pvalue": float((1 + np.sum(matched_gains >= selected_gain)) / (1 + len(matched_gains))),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    validation_rows = [{"configuration": name} | values for name, values in validation.items()]
    write_csv(output_dir / "validation_results.csv", validation_rows)
    print(json.dumps(summary["locked_test"], indent=2))
    print(f"Saved Experiment 4 to {output_dir}")


if __name__ == "__main__":
    main()
