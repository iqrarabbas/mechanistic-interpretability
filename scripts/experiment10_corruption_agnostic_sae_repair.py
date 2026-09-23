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

from data.imagenet_dataset import ImageNetDataset
from scripts.compare_sae_level4 import load_sae
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment3_sae_causal_intervention import downstream_logits
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from interpretability.sae import VanillaReLUSAE


DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment10_agnostic_repair"
BLUR_REFERENCE = PROJECT_ROOT / "results" / "sae" / "experiment5_correction_strategies" / "full_strategies_5000"
MAX_ABNORMALITY_SAMPLES = 200_000
CONDITIONS = [
    ("clean", None, 0),
    ("blur1", "blur", 1),
    ("blur2", "blur", 2),
    ("blur4", "blur", 4),
    ("noise1", "noise", 1),
    ("noise2", "noise", 2),
    ("noise4", "noise", 4),
]


def load_fixed_sae(source, device, clean_tag=""):
    if source != "clean":
        return load_sae(source, "base", "vanilla", device)
    suffix = "clean_base_vanilla_paper"
    if clean_tag:
        suffix += f"_{clean_tag}"
    directory = PROJECT_ROOT / "checkpoints" / "sae" / suffix
    metadata_path = directory / "training.json"
    weights_path = directory / "model.pt"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing clean-only SAE at {directory}. Run scripts/train_clean_sae.py first."
        )
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("training_distribution") != "clean_only":
        raise ValueError(f"{directory} is not a clean-only SAE checkpoint")
    sae = VanillaReLUSAE(expansion_factor=metadata["expansion_factor"])
    sae.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return sae.to(device).eval()


def make_dataset(condition, samples, start_index, seed):
    name, corruption, severity = condition
    return ImageNetDataset(
        DATASET_DIR,
        max_samples=samples,
        start_index=start_index,
        corruption=corruption,
        blur_severity=severity if corruption == "blur" else 4,
        noise_severity=severity if corruption == "noise" else 4,
        corruption_seed=seed,
    )


def encode(model, sae, images):
    outputs = model(pixel_values=images, output_hidden_states=True)
    hidden = outputs.hidden_states[-2]
    batch = images.shape[0]
    latent = sae.encode(hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)
    return outputs.logits, hidden, latent


def calibrate_clean(model, sae, loader, device, patches_per_image, seed):
    generator = torch.Generator().manual_seed(seed)
    patch_indices = torch.randperm(196, generator=generator)[:patches_per_image]
    samples = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Clean calibration"):
            _, _, latent = encode(model, sae, images.to(device))
            samples.append(latent[:, patch_indices].flatten(0, 1).cpu())
    activations = torch.cat(samples).float()
    latent_dim = activations.shape[1]
    median = torch.empty(latent_dim)
    q25 = torch.empty(latent_dim)
    q75 = torch.empty(latent_dim)
    mad = torch.empty(latent_dim)
    standard_deviation = activations.std(dim=0)
    for start in tqdm(range(0, latent_dim, 512), desc="Robust feature statistics"):
        end = min(start + 512, latent_dim)
        block = activations[:, start:end]
        quantiles = torch.quantile(block, torch.tensor([0.25, 0.5, 0.75]), dim=0)
        q25[start:end], median[start:end], q75[start:end] = quantiles
        mad[start:end] = torch.quantile((block - quantiles[1]).abs(), 0.5, dim=0)
    scale = torch.maximum(1.4826 * mad, (q75 - q25) / 1.349)
    scale = torch.maximum(scale, 0.1 * standard_deviation).clamp_min(1e-4)
    active_frequency = (activations > 0).float().mean(0)
    return {
        "median": median,
        "mad": mad,
        "q25": q25,
        "q75": q75,
        "standard_deviation": standard_deviation,
        "scale": scale,
        "active_frequency": active_frequency,
        "sampled_patch_activations": int(activations.shape[0]),
    }


def repair(latent, stats, config):
    median = stats["median"]
    scale = stats["scale"]
    deviation = latent - median
    abnormality = deviation.abs() / scale
    active = latent > 0
    method = config["method"]
    if method == "reference_affine":
        selected = torch.as_tensor(config["features"], device=latent.device, dtype=torch.long)
        corrected = latent.clone()
        values = latent[..., selected]
        target = values * config["scale"][selected] + config["intercept"][selected]
        corrected[..., selected] += config["alpha_max"] * (target - values)
        mask = torch.zeros_like(latent, dtype=torch.bool)
        mask[..., selected] = True
        return corrected, {
            "abnormality": abnormality,
            "corrected_mask": mask,
            "alpha": latent.new_full((latent.shape[0],), config["alpha_max"]),
        }
    if method == "global":
        gate = active.float()
        alpha = latent.new_full((latent.shape[0], 1, 1), config["alpha_max"])
    else:
        gate = ((abnormality - config["tau"]) / config["kappa"]).clamp(0, 1)
        gate = gate * active
        if method == "gated_fixed":
            alpha = latent.new_full((latent.shape[0], 1, 1), config["alpha_max"])
        elif method == "adaptive":
            active_count = active.sum((1, 2)).clamp_min(1)
            image_abnormality = gate.sum((1, 2)) / active_count
            fraction = (image_abnormality / config["adaptive_scale"]).clamp(0, 1)
            alpha = (config["alpha_max"] * fraction)[:, None, None]
        else:
            raise ValueError(method)
    corrected = latent - alpha * gate * deviation
    diagnostics = {
        "abnormality": abnormality,
        "corrected_mask": gate > 0,
        "alpha": alpha.flatten(),
    }
    return corrected, diagnostics


def evaluate_condition(model, sae, loader, device, stats, configurations, collect_features=False):
    totals = {
        name: {
            "correct": 0,
            "margin": 0.0,
            "recovered": 0,
            "damaged": 0,
            "corrected": 0,
            "activations": 0,
            "alpha": 0.0,
        }
        for name in configurations
    }
    original = {"correct": 0, "margin": 0.0}
    reconstruction = {"correct": 0, "margin": 0.0}
    abnormality_samples = []
    feature_counts = torch.zeros(sae.latent_dim, dtype=torch.int64)
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False):
            images = images.to(device)
            labels = labels.to(device)
            logits, hidden, latent = encode(model, sae, images)
            original_margin = classification_margin(logits, labels)[1]
            original_prediction = logits.argmax(1)
            decoded = sae.decode(latent)
            reconstruction_logits = downstream_logits(
                model, torch.cat([hidden[:, :1], decoded], 1)
            )
            reconstruction_margin = classification_margin(reconstruction_logits, labels)[1]
            original["correct"] += int((original_prediction == labels).sum())
            original["margin"] += float(original_margin.sum())
            reconstruction["correct"] += int((reconstruction_logits.argmax(1) == labels).sum())
            reconstruction["margin"] += float(reconstruction_margin.sum())
            for name, config in configurations.items():
                corrected, diagnostics = repair(latent, stats, config)
                corrected_logits = downstream_logits(
                    model, torch.cat([hidden[:, :1], sae.decode(corrected)], 1)
                )
                corrected_margin = classification_margin(corrected_logits, labels)[1]
                corrected_prediction = corrected_logits.argmax(1)
                totals[name]["correct"] += int((corrected_prediction == labels).sum())
                totals[name]["margin"] += float(corrected_margin.sum())
                totals[name]["recovered"] += int(
                    ((original_prediction != labels) & (corrected_prediction == labels)).sum()
                )
                totals[name]["damaged"] += int(
                    ((original_prediction == labels) & (corrected_prediction != labels)).sum()
                )
                totals[name]["corrected"] += int(diagnostics["corrected_mask"].sum())
                totals[name]["activations"] += diagnostics["corrected_mask"].numel()
                totals[name]["alpha"] += float(diagnostics["alpha"].sum())
                if collect_features and config["method"] == "adaptive":
                    feature_counts += diagnostics["corrected_mask"].sum((0, 1)).cpu()
                    flat = diagnostics["abnormality"][diagnostics["corrected_mask"]]
                    sampled_so_far = sum(chunk.numel() for chunk in abnormality_samples)
                    remaining = MAX_ABNORMALITY_SAMPLES - sampled_so_far
                    if flat.numel() and remaining > 0:
                        take = min(10_000, remaining)
                        stride = max(1, flat.numel() // take)
                        abnormality_samples.append(flat[::stride][:take].cpu())
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
    for name, values in totals.items():
        results[name] = {
            "accuracy": values["correct"] / total,
            "accuracy_gain_vs_original": values["correct"] / total - original["correct"] / total,
            "mean_true_class_margin": values["margin"] / total,
            "margin_change_vs_original": values["margin"] / total - original["margin"] / total,
            "predictions_recovered": values["recovered"],
            "originally_correct_damaged": values["damaged"],
            "percent_activations_corrected": 100 * values["corrected"] / values["activations"],
            "mean_image_alpha": values["alpha"] / total,
        }
    analysis = {}
    if collect_features:
        values = torch.cat(abnormality_samples) if abnormality_samples else torch.zeros(1)
        analysis = {
            "abnormality_quantiles": dict(zip(
                ["q50", "q90", "q95", "q99"],
                torch.quantile(values, torch.tensor([0.5, 0.9, 0.95, 0.99])).tolist(),
            )),
            "top_corrected_features": [
                {"feature": int(index), "corrected_count": int(feature_counts[index])}
                for index in torch.argsort(feature_counts, descending=True)[:32]
            ],
        }
    return results, analysis


def config_name(config):
    return "_".join(f"{key}{value:g}" if isinstance(value, float) else f"{key}{value}" for key, value in config.items())


def main():
    parser = argparse.ArgumentParser(description="Experiment 10: corruption-agnostic SAE abnormality repair")
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--calibration-start", type=int, default=30000)
    parser.add_argument("--validation-start", type=int, default=32000)
    parser.add_argument("--evaluation-start", type=int, default=35000)
    parser.add_argument("--patches-per-calibration-image", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--taus", type=float, nargs="+", default=[3.0, 5.0])
    parser.add_argument("--kappas", type=float, nargs="+", default=[2.0, 5.0])
    parser.add_argument("--alpha-maxes", type=float, nargs="+", default=[0.25, 0.5])
    parser.add_argument("--adaptive-scales", type=float, nargs="+", default=[0.01, 0.025, 0.05, 0.10])
    parser.add_argument("--clean-loss-penalty", type=float, default=2.0)
    parser.add_argument("--sae-source", choices=["clean", "blur", "noise"], default="clean")
    parser.add_argument("--clean-sae-tag", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", required=True)
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
    sae = load_fixed_sae(args.sae_source, device, args.clean_sae_tag)

    statistics_path = output_dir / "clean_feature_statistics.pt"
    if args.resume and statistics_path.exists():
        print(f"Resuming with existing clean statistics from {statistics_path}")
        stats_cpu = torch.load(statistics_path, map_location="cpu", weights_only=True)
    else:
        calibration_data = make_dataset(("clean", None, 0), args.calibration_samples, args.calibration_start, args.seed)
        calibration_loader = DataLoader(
            calibration_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        stats_cpu = calibrate_clean(
            model,
            sae,
            calibration_loader,
            device,
            args.patches_per_calibration_image,
            args.seed,
        )
    stats = {key: value.to(device) if torch.is_tensor(value) else value for key, value in stats_cpu.items()}
    torch.save(stats_cpu, statistics_path)

    candidates = {}
    for tau in args.taus:
        for kappa in args.kappas:
            for alpha_max in args.alpha_maxes:
                for adaptive_scale in args.adaptive_scales:
                    config = {
                        "method": "adaptive",
                        "tau": tau,
                        "kappa": kappa,
                        "alpha_max": alpha_max,
                        "adaptive_scale": adaptive_scale,
                    }
                    candidates[config_name(config)] = config
    validation = {}
    for condition in CONDITIONS:
        dataset = make_dataset(condition, args.validation_samples, args.validation_start, args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        validation[condition[0]], _ = evaluate_condition(
            model, sae, loader, device, stats, candidates
        )
    clean_baseline = validation["clean"]["original_vit"]["accuracy"]
    corruptions = [condition[0] for condition in CONDITIONS if condition[0] != "clean"]
    def score(name):
        mean_corrupt = np.mean([validation[condition][name]["accuracy"] for condition in corruptions])
        clean_loss = max(0, clean_baseline - validation["clean"][name]["accuracy"])
        return mean_corrupt - args.clean_loss_penalty * clean_loss
    selected_name = max(candidates, key=score)
    selected = candidates[selected_name]
    (output_dir / "validation_checkpoint.json").write_text(json.dumps({
        "selected_validation_configuration": {"name": selected_name} | selected,
        "validation": validation,
    }, indent=2))

    final_configs = {
        "global": {"method": "global", "alpha_max": selected["alpha_max"]},
        "gated_fixed": selected | {"method": "gated_fixed"},
        "adaptive": selected,
    }
    if args.sae_source == "blur" and BLUR_REFERENCE.exists():
        reference_summary = json.loads((BLUR_REFERENCE / "summary.json").read_text())
        with (BLUR_REFERENCE / "discovery_feature_statistics.csv").open() as source:
            import csv

            reference_rows = list(csv.DictReader(source))
        final_configs["previous_blur_top32"] = {
            "method": "reference_affine",
            "features": reference_summary["selected_configuration"]["features"],
            "alpha_max": 0.75,
            "scale": torch.tensor(
                [float(row["affine_scale"]) for row in reference_rows], device=device
            ),
            "intercept": torch.tensor(
                [float(row["affine_intercept"]) for row in reference_rows], device=device
            ),
        }
    evaluation = {}
    analyses = {}
    for condition in CONDITIONS:
        dataset = make_dataset(condition, args.evaluation_samples, args.evaluation_start, args.seed)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        evaluation[condition[0]], analyses[condition[0]] = evaluate_condition(
            model, sae, loader, device, stats, final_configs, collect_features=True
        )
        (output_dir / "evaluation_checkpoint.json").write_text(json.dumps({
            "completed_conditions": list(evaluation),
            "evaluation": evaluation,
            "analysis": analyses,
        }, indent=2))
    adaptive_accuracies = [evaluation[name]["adaptive"]["accuracy"] for name in corruptions]
    top_sets = {
        name: {row["feature"] for row in analyses[name]["top_corrected_features"]}
        for name in analyses
    }
    blur_noise_intersection = top_sets["blur4"] & top_sets["noise4"]
    blur_noise_union = top_sets["blur4"] | top_sets["noise4"]
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "sae": (
                "checkpoints/sae/clean_base_vanilla_paper"
                + (f"_{args.clean_sae_tag}" if args.clean_sae_tag else "")
                if args.sae_source == "clean"
                else f"checkpoints/sae/{args.sae_source}4_base_vanilla_paper"
            ),
            "status": "exploratory; all local ImageNet validation images were used previously",
            "limitation": (
                "None from corruption exposure: the selected SAE was trained only on clean activations"
                if args.sae_source == "clean"
                else "The fixed SAE was originally trained with one corruption family; only the repair rule is corruption-agnostic"
            ),
        },
        "mathematical_rule": {
            "center": "per-feature clean median",
            "scale": "max(1.4826*MAD, IQR/1.349, 0.1*standard deviation, 1e-4)",
            "gate": "clip((abnormality-tau)/kappa, 0, 1), active ReLU features only",
            "image_alpha": "alpha_max * clip(mean gate intensity over active features/adaptive_scale, 0, 1)",
        },
        "selected_validation_configuration": {"name": selected_name} | selected,
        "validation": validation,
        "evaluation": evaluation,
        "analysis": analyses,
        "automatic_feature_selection": {
            "blur4_noise4_top32_overlap": len(blur_noise_intersection),
            "blur4_noise4_top32_jaccard": len(blur_noise_intersection) / max(1, len(blur_noise_union)),
            "shared_top_features": sorted(blur_noise_intersection),
            "blur4_only_top_features": sorted(top_sets["blur4"] - top_sets["noise4"]),
            "noise4_only_top_features": sorted(top_sets["noise4"] - top_sets["blur4"]),
        },
        "aggregate": {
            "mean_corruption_accuracy": float(np.mean(adaptive_accuracies)),
            "worst_case_corruption_accuracy": float(np.min(adaptive_accuracies)),
            "clean_accuracy_change": evaluation["clean"]["adaptive"]["accuracy_gain_vs_original"],
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    print(json.dumps({
        "selected": summary["selected_validation_configuration"],
        "aggregate": summary["aggregate"],
        "evaluation": {name: values["adaptive"] for name, values in evaluation.items()},
    }, indent=2))
    print(f"Saved Experiment 10 to {output_dir}")


if __name__ == "__main__":
    main()
