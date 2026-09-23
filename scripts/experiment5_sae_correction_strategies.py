import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL, SAE_DIR, load_sae
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment3_sae_causal_intervention import downstream_logits
from scripts.experiment4_non_oracle_sae_correction import (
    discovery_statistics,
    encode_patches,
    feature_table,
    fit_affine,
    make_loader,
    write_csv,
)


OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment5_correction_strategies"


def correct_latent(z, config, statistics):
    if not config["features"] or config["alpha"] == 0:
        return z
    corrected = z.clone()
    selected = torch.as_tensor(config["features"], device=z.device, dtype=torch.long)
    values = z[..., selected]
    mean_clean = statistics["mean_clean"][selected]
    clean_std = statistics["clean_std"][selected]
    method = config["method"]
    if method == "affine":
        target = values * statistics["scale"][selected] + statistics["intercept"][selected]
    elif method == "clean_mean":
        target = mean_clean.expand_as(values)
    elif method == "conditional":
        threshold = config["threshold"]
        signs = statistics["change_sign"][selected]
        upper = mean_clean + threshold * clean_std
        lower = (mean_clean - threshold * clean_std).clamp_min(0)
        target = torch.where(signs > 0, torch.minimum(values, upper), torch.maximum(values, lower))
    else:
        raise ValueError(f"Unknown correction method: {method}")
    direction = -1.0 if config.get("reverse", False) else 1.0
    corrected[..., selected] += direction * config["alpha"] * (target - values)
    return corrected


def evaluate(model, sae, loader, device, configurations, statistics):
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
            clean_original, blur_original = outputs.logits.split(batch)
            clean_hidden, blur_hidden = outputs.hidden_states[-2].split(batch)
            clean_z = encode_patches(sae, clean_hidden)
            blur_z = encode_patches(sae, blur_hidden)
            for kind, logits in [("clean", clean_original), ("blur", blur_original)]:
                margin = classification_margin(logits, labels)[1]
                totals["original_vit"][f"{kind}_correct"] += int((logits.argmax(1) == labels).sum())
                totals["original_vit"][f"{kind}_margin"] += float(margin.sum())
            for name, config in configurations.items():
                clean_modified = correct_latent(clean_z, config, statistics)
                blur_modified = correct_latent(blur_z, config, statistics)
                clean_logits = downstream_logits(
                    model, torch.cat([clean_hidden[:, :1], sae.decode(clean_modified)], 1)
                )
                blur_logits = downstream_logits(
                    model, torch.cat([blur_hidden[:, :1], sae.decode(blur_modified)], 1)
                )
                for kind, logits in [("clean", clean_logits), ("blur", blur_logits)]:
                    margin = classification_margin(logits, labels)[1]
                    totals[name][f"{kind}_correct"] += int((logits.argmax(1) == labels).sum())
                    totals[name][f"{kind}_margin"] += float(margin.sum())
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


def configuration_name(method, count, alpha, threshold=None):
    suffix = f"_threshold{threshold:g}" if threshold is not None else ""
    return f"{method}_top{count}_alpha{alpha:g}{suffix}"


def main():
    parser = argparse.ArgumentParser(description="Experiment 5: stronger non-oracle SAE corrections")
    parser.add_argument("--discovery-samples", type=int, default=3000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--test-samples", type=int, default=5000)
    parser.add_argument("--discovery-start", type=int, default=20000)
    parser.add_argument("--validation-start", type=int, default=23000)
    parser.add_argument("--test-start", type=int, default=25000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--minimum-consistency", type=float, default=0.55)
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 0.75, 1.0, 1.25])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    parser.add_argument("--clean-loss-penalty", type=float, default=2.0)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ranges = [
        (0, 20000, "all previous work"),
        (args.discovery_start, args.discovery_start + args.discovery_samples, "discovery"),
        (args.validation_start, args.validation_start + args.validation_samples, "validation"),
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
    moments = fit_affine(sums, args.discovery_samples * 196, args.ridge)
    feature_rows, strict_candidates = feature_table(
        image_delta, delta_margin, clean_correct, blur_correct, moments, args.minimum_consistency
    )
    ranked = [
        row for row in feature_rows
        if row["direction_consistency"] >= args.minimum_consistency
        and row["pearson_delta_margin"] * row["mean_blur_minus_clean"] < 0
    ]
    ranked.sort(
        key=lambda row: abs(row["pearson_delta_margin"])
        * abs(row["standardized_change"])
        * row["direction_consistency"],
        reverse=True,
    )
    if len(ranked) < max(args.feature_counts):
        raise RuntimeError(f"Only {len(ranked)} ranked features; reduce --feature-counts")
    write_csv(output_dir / "discovery_feature_statistics.csv", feature_rows)
    write_csv(output_dir / "ranked_candidates.csv", ranked)

    scale, intercept, _, mean_clean, _, var_clean = moments
    statistics = {
        "scale": torch.as_tensor(scale, device=device),
        "intercept": torch.as_tensor(intercept, device=device),
        "mean_clean": torch.as_tensor(mean_clean.astype(np.float32), device=device),
        "clean_std": torch.as_tensor(np.sqrt(np.maximum(var_clean, 1e-8)).astype(np.float32), device=device),
        "change_sign": torch.as_tensor(np.sign(image_delta.mean(0)).astype(np.float32), device=device),
    }
    configurations = {"baseline": {"method": "affine", "features": [], "alpha": 0.0}}
    for count in args.feature_counts:
        features = [row["feature_index"] for row in ranked[:count]]
        for alpha in args.alphas:
            for method in ["affine", "clean_mean"]:
                name = configuration_name(method, count, alpha)
                configurations[name] = {"method": method, "features": features, "alpha": alpha}
            for threshold in args.thresholds:
                name = configuration_name("conditional", count, alpha, threshold)
                configurations[name] = {
                    "method": "conditional",
                    "features": features,
                    "alpha": alpha,
                    "threshold": threshold,
                }

    validation_loader = make_loader(
        args.validation_samples, args.validation_start, args.seed, args.batch_size, args.num_workers
    )
    validation = evaluate(model, sae, validation_loader, device, configurations, statistics)
    baseline = validation["baseline"]
    eligible = [name for name in configurations if name != "baseline"]
    best_name = max(
        eligible,
        key=lambda name: validation[name]["blur_accuracy"]
        - args.clean_loss_penalty * max(0, baseline["clean_accuracy"] - validation[name]["clean_accuracy"]),
    )
    best = configurations[best_name]
    write_csv(
        output_dir / "validation_results.csv",
        [{"configuration": name} | values for name, values in validation.items()],
    )

    rng = np.random.default_rng(args.seed)
    test_configurations = {
        "baseline": configurations["baseline"],
        "selected": best,
        "reverse": best | {"reverse": True},
    }
    all_features = np.arange(sae.latent_dim)
    magnitudes = np.abs(np.array([row["standardized_change"] for row in feature_rows]))
    selected_set = set(best["features"])
    for draw in range(args.random_controls):
        test_configurations[f"random_{draw:03d}"] = best | {
            "features": rng.choice(all_features, len(best["features"]), replace=False).tolist()
        }
        matched = []
        unavailable = set(selected_set)
        for feature in best["features"]:
            nearest = np.argsort(np.abs(magnitudes - magnitudes[feature]))
            pool = [index for index in nearest[:512] if index not in unavailable]
            chosen = int(rng.choice(pool))
            matched.append(chosen)
            unavailable.add(chosen)
        test_configurations[f"matched_{draw:03d}"] = best | {"features": matched}

    test_loader = make_loader(
        args.test_samples, args.test_start, args.seed, args.batch_size, args.num_workers
    )
    test = evaluate(model, sae, test_loader, device, test_configurations, statistics)
    selected_gain = test["selected"]["blur_accuracy"] - test["baseline"]["blur_accuracy"]
    random_gains = np.array([
        values["blur_accuracy"] - test["baseline"]["blur_accuracy"]
        for name, values in test.items() if name.startswith("random_")
    ])
    matched_gains = np.array([
        values["blur_accuracy"] - test["baseline"]["blur_accuracy"]
        for name, values in test.items() if name.startswith("matched_")
    ])
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "sae": str(SAE_DIR.relative_to(PROJECT_ROOT)),
            "sae_implementation": sae_metadata["implementation"],
            "site": "hidden_states[-2], patch tokens; preserve original CLS",
        },
        "split_ranges": {name: [start, end] for start, end, name in ranges},
        "discovery": {
            "strict_fdr_candidates": len(strict_candidates),
            "ranked_candidate_pool": len(ranked),
            "top_candidates": ranked[:32],
        },
        "selected_configuration": {"name": best_name} | best,
        "validation_selected": validation[best_name],
        "locked_test": {
            "original_vit": test["original_vit"],
            "baseline": test["baseline"],
            "selected": test["selected"],
            "reverse": test["reverse"],
            "blur_accuracy_gain": selected_gain,
            "clean_accuracy_change": test["selected"]["clean_accuracy"] - test["baseline"]["clean_accuracy"],
            "random_control_mean_gain": float(random_gains.mean()),
            "random_empirical_pvalue": float((1 + np.sum(random_gains >= selected_gain)) / (1 + len(random_gains))),
            "magnitude_matched_mean_gain": float(matched_gains.mean()),
            "magnitude_matched_empirical_pvalue": float((1 + np.sum(matched_gains >= selected_gain)) / (1 + len(matched_gains))),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    print(json.dumps(summary["locked_test"], indent=2))
    print(f"Saved Experiment 5 to {output_dir}")


if __name__ == "__main__":
    main()
