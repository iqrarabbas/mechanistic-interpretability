import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment3_sae_causal_intervention import downstream_logits
from scripts.experiment10_corruption_agnostic_sae_repair import (
    CONDITIONS,
    encode,
    load_fixed_sae,
    make_dataset,
)


OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment11_quantile_repair"
MAX_DIAGNOSTIC_SAMPLES = 200_000


def calibrate_quantiles(model, sae, loader, device, patches_per_image, quantiles, seed):
    generator = torch.Generator().manual_seed(seed)
    patch_indices = torch.randperm(196, generator=generator)[:patches_per_image]
    samples = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Clean quantile calibration"):
            _, _, latent = encode(model, sae, images.to(device))
            samples.append(latent[:, patch_indices].flatten(0, 1).cpu())
    activations = torch.cat(samples).float()
    split = max(1, activations.shape[0] // 2)
    fit_activations = activations[:split]
    reference_activations = activations[split:]
    if reference_activations.numel() == 0:
        reference_activations = fit_activations
    thresholds = {str(value): torch.empty(sae.latent_dim) for value in quantiles}
    for start in tqdm(range(0, sae.latent_dim, 512), desc="Empirical feature quantiles"):
        end = min(start + 512, sae.latent_dim)
        block = fit_activations[:, start:end]
        values = torch.quantile(block, torch.tensor(quantiles), dim=0)
        for position, quantile in enumerate(quantiles):
            thresholds[str(quantile)][start:end] = values[position]
    expected_rates = {
        str(quantile): float((reference_activations > thresholds[str(quantile)]).float().mean())
        for quantile in quantiles
    }
    return {
        "quantiles": thresholds,
        "expected_rates": expected_rates,
        "sampled_patch_activations": activations.shape[0],
        "quantile_fit_activations": fit_activations.shape[0],
        "rate_reference_activations": reference_activations.shape[0],
        "patches_per_image": patches_per_image,
    }


def quantile_repair(latent, threshold, config):
    exceedance = (latent - threshold).clamp_min(0)
    if "features" in config:
        feature_mask = torch.zeros(latent.shape[-1], dtype=torch.bool, device=latent.device)
        feature_mask[config["features"]] = True
        exceedance = exceedance * feature_mask
    mask = exceedance > 0
    observed_rate = mask.float().mean((1, 2))
    expected_rate = config["expected_rate"]
    if config["adaptive"]:
        fraction = ((observed_rate - expected_rate).clamp_min(0) / config["adaptive_width"]).clamp(0, 1)
        alpha = config["alpha_max"] * fraction
    else:
        alpha = latent.new_full((latent.shape[0],), config["alpha_max"])
    corrected = latent - alpha[:, None, None] * exceedance
    return corrected, {
        "mask": mask,
        "observed_rate": observed_rate,
        "alpha": alpha,
        "exceedance": exceedance,
    }


def evaluate_condition(model, sae, loader, device, thresholds, configurations, diagnostics=False):
    totals = {
        name: {
            "correct": 0,
            "margin": 0.0,
            "recovered": 0,
            "damaged": 0,
            "corrected": 0,
            "activations": 0,
            "accepted": 0,
            "images_corrected": 0,
            "alpha": 0.0,
            "observed_rate": 0.0,
        }
        for name in configurations
    }
    original = {"correct": 0, "margin": 0.0}
    reconstruction = {"correct": 0, "margin": 0.0}
    feature_counts = {name: torch.zeros(sae.latent_dim, dtype=torch.int64) for name in configurations}
    exceedance_samples = {name: [] for name in configurations}
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits, hidden, latent = encode(model, sae, images)
            original_prediction = logits.argmax(1)
            original_margin = classification_margin(logits, labels)[1]
            original_confidence = logits.softmax(1).max(1).values
            decoded_latent = sae.decode(latent)
            reconstruction_logits = downstream_logits(
                model, torch.cat([hidden[:, :1], decoded_latent], 1)
            )
            reconstruction_margin = classification_margin(reconstruction_logits, labels)[1]
            original["correct"] += int((original_prediction == labels).sum())
            original["margin"] += float(original_margin.sum())
            reconstruction["correct"] += int((reconstruction_logits.argmax(1) == labels).sum())
            reconstruction["margin"] += float(reconstruction_margin.sum())
            for name, config in configurations.items():
                threshold = thresholds[str(config["quantile"])]
                candidate, info = quantile_repair(latent, threshold, config)
                decoded_candidate = sae.decode(candidate)
                if config.get("residual_intervention", False):
                    candidate_patches = hidden[:, 1:] + decoded_candidate - decoded_latent
                else:
                    candidate_patches = decoded_candidate
                candidate_logits = downstream_logits(
                    model, torch.cat([hidden[:, :1], candidate_patches], 1)
                )
                if config.get("confidence_safe", False):
                    candidate_confidence = candidate_logits.softmax(1).max(1).values
                    accepted = candidate_confidence > original_confidence
                    final_logits = torch.where(accepted[:, None], candidate_logits, logits)
                else:
                    accepted = torch.ones(images.shape[0], dtype=torch.bool, device=device)
                    final_logits = candidate_logits
                final_prediction = final_logits.argmax(1)
                final_margin = classification_margin(final_logits, labels)[1]
                totals[name]["correct"] += int((final_prediction == labels).sum())
                totals[name]["margin"] += float(final_margin.sum())
                totals[name]["recovered"] += int(
                    ((original_prediction != labels) & (final_prediction == labels)).sum()
                )
                totals[name]["damaged"] += int(
                    ((original_prediction == labels) & (final_prediction != labels)).sum()
                )
                image_corrected = (info["alpha"] > 0) & accepted
                accepted_mask = info["mask"] & image_corrected[:, None, None]
                totals[name]["corrected"] += int(accepted_mask.sum())
                totals[name]["activations"] += accepted_mask.numel()
                totals[name]["accepted"] += int(accepted.sum())
                totals[name]["images_corrected"] += int(image_corrected.sum())
                totals[name]["alpha"] += float((info["alpha"] * accepted).sum())
                totals[name]["observed_rate"] += float(info["observed_rate"].sum())
                if diagnostics:
                    feature_counts[name] += accepted_mask.sum((0, 1)).cpu()
                    flat = info["exceedance"][accepted_mask]
                    current = sum(chunk.numel() for chunk in exceedance_samples[name])
                    remaining = MAX_DIAGNOSTIC_SAMPLES - current
                    if flat.numel() and remaining > 0:
                        take = min(10_000, remaining)
                        stride = max(1, flat.numel() // take)
                        exceedance_samples[name].append(flat[::stride][:take].cpu())
            total += images.shape[0]
    results = {
        "original_vit": {
            "accuracy": original["correct"] / total,
            "mean_true_class_margin": original["margin"] / total,
        },
        "sae_reconstruction": {
            "accuracy": reconstruction["correct"] / total,
            "mean_true_class_margin": reconstruction["margin"] / total,
        },
    }
    analyses = {}
    for name, values in totals.items():
        results[name] = {
            "accuracy": values["correct"] / total,
            "accuracy_gain_vs_original": values["correct"] / total - original["correct"] / total,
            "mean_true_class_margin": values["margin"] / total,
            "margin_change_vs_original": values["margin"] / total - original["margin"] / total,
            "predictions_recovered": values["recovered"],
            "originally_correct_damaged": values["damaged"],
            "percent_activations_corrected": 100 * values["corrected"] / values["activations"],
            "percent_images_accepted": 100 * values["accepted"] / total,
            "percent_images_corrected": 100 * values["images_corrected"] / total,
            "mean_image_alpha": values["alpha"] / total,
            "mean_quantile_exceedance_rate": values["observed_rate"] / total,
        }
        if diagnostics:
            sampled = torch.cat(exceedance_samples[name]) if exceedance_samples[name] else torch.zeros(1)
            analyses[name] = {
                "positive_exceedance_quantiles": dict(zip(
                    ["q50", "q90", "q99"],
                    torch.quantile(sampled, torch.tensor([0.5, 0.9, 0.99])).tolist(),
                )),
                "top_corrected_features": [
                    {"feature": int(index), "corrected_count": int(feature_counts[name][index])}
                    for index in torch.argsort(feature_counts[name], descending=True)[:32]
                ],
            }
    return results, analyses


def main():
    parser = argparse.ArgumentParser(description="Experiment 11: corruption-agnostic empirical-quantile SAE repair")
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--calibration-start", type=int, default=30000)
    parser.add_argument("--validation-start", type=int, default=32000)
    parser.add_argument("--evaluation-start", type=int, default=35000)
    parser.add_argument("--patches-per-calibration-image", type=int, default=4)
    parser.add_argument("--quantiles", type=float, nargs="+", default=[0.99, 0.995, 0.999])
    parser.add_argument("--alpha-maxes", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    parser.add_argument("--adaptive-widths", type=float, nargs="+", default=[0.0005, 0.001, 0.002])
    parser.add_argument("--clean-loss-penalty", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)

    calibration_path = output_dir / "clean_quantiles.pt"
    if args.resume and calibration_path.exists():
        calibration = torch.load(calibration_path, map_location="cpu", weights_only=True)
    else:
        data = make_dataset(("clean", None, 0), args.calibration_samples, args.calibration_start, args.seed)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        calibration = calibrate_quantiles(
            model, sae, loader, device, args.patches_per_calibration_image, args.quantiles, args.seed
        )
        torch.save(calibration, calibration_path)
    thresholds = {key: value.to(device) for key, value in calibration["quantiles"].items()}
    expected_rates = calibration["expected_rates"]

    candidates = {}
    for quantile in args.quantiles:
        for alpha_max in args.alpha_maxes:
            for width in args.adaptive_widths:
                name = f"q{quantile:g}_alpha{alpha_max:g}_width{width:g}"
                candidates[name] = {
                    "quantile": quantile,
                    "expected_rate": expected_rates[str(quantile)],
                    "alpha_max": alpha_max,
                    "adaptive_width": width,
                    "adaptive": True,
                }
    validation = {}
    for condition in CONDITIONS:
        data = make_dataset(condition, args.validation_samples, args.validation_start, args.seed)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        validation[condition[0]], _ = evaluate_condition(
            model, sae, loader, device, thresholds, candidates
        )
    corruptions = [name for name, _, _ in CONDITIONS if name != "clean"]
    clean_baseline = validation["clean"]["original_vit"]["accuracy"]
    def score(name):
        mean_corrupt = np.mean([validation[condition][name]["accuracy"] for condition in corruptions])
        clean_loss = max(0, clean_baseline - validation["clean"][name]["accuracy"])
        return mean_corrupt - args.clean_loss_penalty * clean_loss
    selected_name = max(candidates, key=score)
    selected = candidates[selected_name]
    (output_dir / "validation_checkpoint.json").write_text(json.dumps({
        "selected": {"name": selected_name} | selected,
        "validation": validation,
    }, indent=2))

    configurations = {
        "quantile_fixed": selected | {"adaptive": False},
        "quantile_adaptive": selected,
        "quantile_adaptive_confidence_safe": selected | {"confidence_safe": True},
    }
    evaluation = {}
    analyses = {}
    for condition in CONDITIONS:
        data = make_dataset(condition, args.evaluation_samples, args.evaluation_start, args.seed)
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        evaluation[condition[0]], analyses[condition[0]] = evaluate_condition(
            model, sae, loader, device, thresholds, configurations, diagnostics=True
        )
        (output_dir / "evaluation_checkpoint.json").write_text(json.dumps({
            "completed_conditions": list(evaluation), "evaluation": evaluation, "analysis": analyses
        }, indent=2))
    adaptive_accuracies = [evaluation[name]["quantile_adaptive"]["accuracy"] for name in corruptions]
    top_blur = {row["feature"] for row in analyses["blur4"]["quantile_adaptive"]["top_corrected_features"]}
    top_noise = {row["feature"] for row in analyses["noise4"]["quantile_adaptive"]["top_corrected_features"]}
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "sae": "checkpoints/sae/clean_base_vanilla_paper",
            "status": "exploratory; local ImageNet validation images were used previously",
        },
        "selected_validation_configuration": {"name": selected_name} | selected,
        "validation": validation,
        "evaluation": evaluation,
        "analysis": analyses,
        "automatic_feature_selection": {
            "blur4_noise4_top32_overlap": len(top_blur & top_noise),
            "blur4_noise4_top32_jaccard": len(top_blur & top_noise) / max(1, len(top_blur | top_noise)),
            "shared": sorted(top_blur & top_noise),
            "blur_only": sorted(top_blur - top_noise),
            "noise_only": sorted(top_noise - top_blur),
        },
        "aggregate": {
            "mean_corruption_accuracy": float(np.mean(adaptive_accuracies)),
            "worst_case_corruption_accuracy": float(np.min(adaptive_accuracies)),
            "clean_accuracy_change": evaluation["clean"]["quantile_adaptive"]["accuracy_gain_vs_original"],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    print(json.dumps({
        "selected": summary["selected_validation_configuration"],
        "aggregate": summary["aggregate"],
        "evaluation": {name: values["quantile_adaptive"] for name, values in evaluation.items()},
    }, indent=2))
    print(f"Saved Experiment 11 to {output_dir}")


if __name__ == "__main__":
    main()
