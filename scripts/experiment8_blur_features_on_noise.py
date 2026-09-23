import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.compare_sae_level4 import load_sae
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment3_sae_causal_intervention import downstream_logits
from scripts.experiment7_noise_finetuning_feature_mechanisms import NoisePairedDataset
from scripts.experiment4_non_oracle_sae_correction import write_csv


BASE_MODEL = "google/vit-base-patch16-224"
BLUR_RESULTS = PROJECT_ROOT / "results" / "sae" / "experiment5_correction_strategies" / "full_strategies_5000"
NOISE_RESULTS = PROJECT_ROOT / "results" / "sae" / "experiment7_noise_finetuning_features" / "full_noise_ft_5000"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment8_blur_features_on_noise"


def read_feature_rows(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    return [
        {
            key: int(value) if key == "feature_index" else float(value)
            for key, value in row.items()
        }
        for row in rows
    ]


def align_blur_to_noise(blur_sae, noise_sae, blur_features, noise_rows):
    blur_indices = torch.tensor(blur_features, dtype=torch.long)
    blur_directions = F.normalize(blur_sae.decoder.weight[:, blur_indices].T.float(), dim=1)
    noise_directions = F.normalize(noise_sae.decoder.weight.T.float(), dim=1)
    similarities = blur_directions @ noise_directions.T
    used = set()
    rows = []
    for position, blur_feature in enumerate(blur_features):
        order = torch.argsort(similarities[position], descending=True).tolist()
        noise_feature = next(index for index in order if index not in used)
        used.add(noise_feature)
        noise_row = noise_rows[noise_feature]
        rows.append({
            "blur_feature": blur_feature,
            "matched_noise_feature": noise_feature,
            "decoder_cosine": float(similarities[position, noise_feature]),
            "noise_mean_change": noise_row["mean_blur_minus_clean"],
            "noise_direction_consistency": noise_row["direction_consistency"],
            "noise_margin_correlation": noise_row["pearson_delta_margin"],
            "noise_margin_qvalue": noise_row["pearson_qvalue"],
            "noise_failure_associated": bool(
                noise_row["pearson_qvalue"] < 0.05
                and noise_row["direction_consistency"] >= 0.60
                and noise_row["pearson_delta_margin"] * noise_row["mean_blur_minus_clean"] < 0
            ),
        })
    return rows


def correct(z, features, scale, intercept, alpha, reverse=False):
    if not features or alpha == 0:
        return z
    output = z.clone()
    selected = torch.as_tensor(features, device=z.device, dtype=torch.long)
    values = z[..., selected]
    target = values * scale[selected] + intercept[selected]
    direction = -1 if reverse else 1
    output[..., selected] += direction * alpha * (target - values)
    return output


def evaluate(model, sae, loader, device, configurations, scale, intercept):
    totals = {
        name: {"clean_correct": 0, "noise_correct": 0, "clean_margin": 0.0, "noise_margin": 0.0}
        for name in configurations
    }
    totals["original_vit"] = {
        "clean_correct": 0,
        "noise_correct": 0,
        "clean_margin": 0.0,
        "noise_margin": 0.0,
    }
    total = 0
    with torch.no_grad():
        for clean, noisy, labels, _ in tqdm(loader, desc="Locked test"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noisy]).to(device),
                output_hidden_states=True,
            )
            clean_original, noise_original = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[-2].split(batch)
            clean_z = sae.encode(clean_hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)
            noise_z = sae.encode(noise_hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)
            for kind, logits in [("clean", clean_original), ("noise", noise_original)]:
                margin = classification_margin(logits, labels)[1]
                totals["original_vit"][f"{kind}_correct"] += int((logits.argmax(1) == labels).sum())
                totals["original_vit"][f"{kind}_margin"] += float(margin.sum())
            for name, config in configurations.items():
                clean_modified = correct(clean_z, scale=scale, intercept=intercept, **config)
                noise_modified = correct(noise_z, scale=scale, intercept=intercept, **config)
                clean_logits = downstream_logits(
                    model, torch.cat([clean_hidden[:, :1], sae.decode(clean_modified)], 1)
                )
                noise_logits = downstream_logits(
                    model, torch.cat([noise_hidden[:, :1], sae.decode(noise_modified)], 1)
                )
                for kind, logits in [("clean", clean_logits), ("noise", noise_logits)]:
                    margin = classification_margin(logits, labels)[1]
                    totals[name][f"{kind}_correct"] += int((logits.argmax(1) == labels).sum())
                    totals[name][f"{kind}_margin"] += float(margin.sum())
            total += batch
    return {
        name: {
            "clean_accuracy": values["clean_correct"] / total,
            "noise4_accuracy": values["noise_correct"] / total,
            "mean_clean_margin": values["clean_margin"] / total,
            "mean_noise_margin": values["noise_margin"] / total,
            "samples": total,
        }
        for name, values in totals.items()
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 8: transfer Blur-harmful SAE features to Noise-4")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--start-index", type=int, default=45000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.75)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start_index < 45000 or args.start_index + args.samples > 50000:
        raise ValueError("Use the fresh ImageNet range 45000–49999")

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blur_summary = json.loads((BLUR_RESULTS / "summary.json").read_text())
    blur_features = blur_summary["selected_configuration"]["features"]
    noise_rows = read_feature_rows(NOISE_RESULTS / "base_feature_statistics.csv")
    noise_candidates = read_feature_rows(NOISE_RESULTS / "base_failure_candidates.csv")
    blur_sae = load_sae("blur", "base", "vanilla", torch.device("cpu"))
    noise_sae = load_sae("noise", "base", "vanilla", device)
    alignments = align_blur_to_noise(blur_sae, noise_sae.to("cpu"), blur_features, noise_rows)
    noise_sae = noise_sae.to(device)
    write_csv(output_dir / "blur_to_noise_feature_alignment.csv", alignments)

    matched_all = [row["matched_noise_feature"] for row in alignments]
    matched_significant = [
        row["matched_noise_feature"] for row in alignments if row["noise_failure_associated"]
    ]
    noise_specific = [row["feature_index"] for row in noise_candidates[:32]]
    scale = torch.tensor([row["affine_scale"] for row in noise_rows], device=device)
    intercept = torch.tensor([row["affine_intercept"] for row in noise_rows], device=device)
    configurations = {
        "baseline": {"features": [], "alpha": 0.0},
        "blur_transferred_all32": {"features": matched_all, "alpha": args.alpha},
        "blur_transferred_noise_significant": {
            "features": matched_significant,
            "alpha": args.alpha,
        },
        "noise_specific_top32": {"features": noise_specific, "alpha": args.alpha},
        "reverse_blur_transferred_all32": {
            "features": matched_all,
            "alpha": args.alpha,
            "reverse": True,
        },
    }
    rng = np.random.default_rng(args.seed)
    all_features = np.arange(noise_sae.latent_dim)
    magnitudes = np.abs(np.array([row["standardized_change"] for row in noise_rows]))
    unavailable = set(matched_all)
    for draw in range(args.random_controls):
        configurations[f"random_{draw:03d}"] = {
            "features": rng.choice(all_features, len(matched_all), replace=False).tolist(),
            "alpha": args.alpha,
        }
        matched_control = []
        used = set(unavailable)
        for feature in matched_all:
            nearest = np.argsort(np.abs(magnitudes - magnitudes[feature]))
            pool = [index for index in nearest[:512] if index not in used]
            chosen = int(rng.choice(pool))
            matched_control.append(chosen)
            used.add(chosen)
        configurations[f"magnitude_matched_{draw:03d}"] = {
            "features": matched_control,
            "alpha": args.alpha,
        }

    dataset = NoisePairedDataset(args.samples, args.start_index, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    results = evaluate(model, noise_sae, loader, device, configurations, scale, intercept)
    baseline = results["baseline"]["noise4_accuracy"]
    random_gains = np.array([
        values["noise4_accuracy"] - baseline
        for name, values in results.items() if name.startswith("random_")
    ])
    matched_gains = np.array([
        values["noise4_accuracy"] - baseline
        for name, values in results.items() if name.startswith("magnitude_matched_")
    ])
    transferred_gain = results["blur_transferred_all32"]["noise4_accuracy"] - baseline
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "blur_feature_source": str(BLUR_RESULTS.relative_to(PROJECT_ROOT)),
            "noise_mapping_source": str(NOISE_RESULTS.relative_to(PROJECT_ROOT)),
            "test_data": "Fresh indices; no clean counterpart used to compute correction",
        },
        "mechanism_overlap": {
            "blur_features": len(blur_features),
            "matched_features": len(matched_all),
            "matched_noise_failure_associated": len(matched_significant),
            "mean_decoder_cosine": float(np.mean([row["decoder_cosine"] for row in alignments])),
            "alignments": alignments,
        },
        "locked_test": {
            key: results[key]
            for key in [
                "original_vit",
                "baseline",
                "blur_transferred_all32",
                "blur_transferred_noise_significant",
                "noise_specific_top32",
                "reverse_blur_transferred_all32",
            ]
        } | {
            "blur_transferred_gain": transferred_gain,
            "random_control_mean_gain": float(random_gains.mean()),
            "random_empirical_pvalue": float((1 + np.sum(random_gains >= transferred_gain)) / (1 + len(random_gains))),
            "magnitude_matched_mean_gain": float(matched_gains.mean()),
            "magnitude_matched_empirical_pvalue": float((1 + np.sum(matched_gains >= transferred_gain)) / (1 + len(matched_gains))),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    print(json.dumps(summary["locked_test"], indent=2))
    print(f"Saved Experiment 8 to {output_dir}")


if __name__ == "__main__":
    main()
