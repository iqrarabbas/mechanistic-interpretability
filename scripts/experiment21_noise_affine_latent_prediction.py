import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.compare_sae_level4 import load_sae
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL, EPSILON
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment21_noise_affine_prediction"
BLOCK_INDEX = 10


def encode(sae, patches):
    batch, patch_count, width = patches.shape
    return sae.encode(patches.reshape(-1, width)).reshape(batch, patch_count, -1)


def collect_affine_statistics(model, sae, loader, device):
    sums = {
        name: torch.zeros(sae.latent_dim, dtype=torch.float64)
        for name in ["x", "y", "xx", "xy", "yy"]
    }
    observations = 0
    with torch.no_grad():
        for clean, noise, _ in tqdm(loader, desc="Affine discovery"):
            batch = clean.shape[0]
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device),
                output_hidden_states=True,
            )
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            clean_latent = encode(sae, clean_hidden[:, 1:]).double()
            noise_latent = encode(sae, noise_hidden[:, 1:]).double()
            sums["x"] += noise_latent.sum((0, 1)).cpu()
            sums["y"] += clean_latent.sum((0, 1)).cpu()
            sums["xx"] += noise_latent.square().sum((0, 1)).cpu()
            sums["xy"] += (noise_latent * clean_latent).sum((0, 1)).cpu()
            sums["yy"] += clean_latent.square().sum((0, 1)).cpu()
            observations += noise_latent.shape[0] * noise_latent.shape[1]
    return sums, observations


def fit_affine(sums, observations, ridge):
    mean_noise = sums["x"].numpy() / observations
    mean_clean = sums["y"].numpy() / observations
    var_noise = sums["xx"].numpy() / observations - np.square(mean_noise)
    var_clean = sums["yy"].numpy() / observations - np.square(mean_clean)
    covariance = sums["xy"].numpy() / observations - mean_noise * mean_clean
    denominator = var_noise + ridge * np.maximum(var_noise, EPSILON) + EPSILON
    scale = covariance / denominator
    intercept = mean_clean - scale * mean_noise
    correlation = covariance / np.sqrt(
        np.maximum(var_noise, EPSILON) * np.maximum(var_clean, EPSILON)
    )
    expected_delta = scale * mean_noise + intercept - mean_noise
    score = np.abs(expected_delta) * np.sqrt(np.maximum(var_clean, EPSILON)) * np.abs(correlation)
    ranking = np.argsort(-score)
    return {
        "scale": scale.astype(np.float32),
        "intercept": intercept.astype(np.float32),
        "mean_clean": mean_clean.astype(np.float32),
        "mean_noise": mean_noise.astype(np.float32),
        "correlation": correlation.astype(np.float32),
        "score": score.astype(np.float32),
        "ranking": ranking.astype(np.int64),
    }


def corrected_patches(sae, patches, latent, parameters, config):
    selected = config["features"]
    candidate = latent.clone()
    values = latent[..., selected]
    if config["method"] == "affine":
        target = values * parameters["scale"][selected] + parameters["intercept"][selected]
    elif config["method"] == "mean":
        target = parameters["mean_clean"][selected].expand_as(values)
    elif config["method"] == "shuffled":
        shuffled = config["shuffled_features"]
        target = values * parameters["scale"][shuffled] + parameters["intercept"][shuffled]
    else:
        raise ValueError(config["method"])
    direction = -1.0 if config.get("reverse", False) else 1.0
    candidate[..., selected] += direction * config["alpha"] * (target - values)
    decoded_delta = sae.decode(candidate.flatten(0, 1)) - sae.decode(latent.flatten(0, 1))
    return patches + decoded_delta.reshape_as(patches)


def evaluate(model, sae, loader, device, parameters, configurations, include_oracle=False):
    tensor_parameters = {
        key: torch.as_tensor(value, device=device)
        for key, value in parameters.items()
        if key in {"scale", "intercept", "mean_clean"}
    }
    stores = {
        name: {
            key: [] for key in [
                "noise_correct", "clean_correct", "noise_margin_change",
                "clean_margin_change", "noise_true_logit_change",
            ]
        }
        for name in configurations
    }
    if include_oracle:
        stores["paired_clean_oracle"] = {
            key: [] for key in stores[next(iter(stores))]
        }
    original = {
        key: [] for key in ["noise_correct", "clean_correct", "noise_margin", "clean_margin"]
    }
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Evaluation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device),
                output_hidden_states=True,
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            clean_true, clean_margin = classification_margin(clean_logits, labels)
            noise_true, noise_margin = classification_margin(noise_logits, labels)
            original["clean_correct"].extend((clean_logits.argmax(1) == labels).cpu().tolist())
            original["noise_correct"].extend((noise_logits.argmax(1) == labels).cpu().tolist())
            original["clean_margin"].extend(clean_margin.cpu().tolist())
            original["noise_margin"].extend(noise_margin.cpu().tolist())
            clean_patches, noise_patches = clean_hidden[:, 1:], noise_hidden[:, 1:]
            clean_latent = encode(sae, clean_patches)
            noise_latent = encode(sae, noise_patches)
            for name, config in configurations.items():
                corrected_noise = corrected_patches(
                    sae, noise_patches, noise_latent, tensor_parameters, config
                )
                corrected_clean = corrected_patches(
                    sae, clean_patches, clean_latent, tensor_parameters, config
                )
                for kind, hidden, patches, baseline_margin, baseline_true in [
                    ("noise", noise_hidden, corrected_noise, noise_margin, noise_true),
                    ("clean", clean_hidden, corrected_clean, clean_margin, clean_true),
                ]:
                    logits = downstream_from_layer(
                        model, torch.cat([hidden[:, :1], patches], dim=1), BLOCK_INDEX
                    )
                    true_logit, margin = classification_margin(logits, labels)
                    stores[name][f"{kind}_correct"].extend(
                        (logits.argmax(1) == labels).cpu().tolist()
                    )
                    stores[name][f"{kind}_margin_change"].extend(
                        (margin - baseline_margin).cpu().tolist()
                    )
                    if kind == "noise":
                        stores[name]["noise_true_logit_change"].extend(
                            (true_logit - baseline_true).cpu().tolist()
                        )
            if include_oracle:
                oracle_patches = noise_patches + clean_patches - noise_patches
                logits = downstream_from_layer(
                    model, torch.cat([noise_hidden[:, :1], oracle_patches], dim=1), BLOCK_INDEX
                )
                true_logit, margin = classification_margin(logits, labels)
                stores["paired_clean_oracle"]["noise_correct"].extend(
                    (logits.argmax(1) == labels).cpu().tolist()
                )
                stores["paired_clean_oracle"]["noise_margin_change"].extend(
                    (margin - noise_margin).cpu().tolist()
                )
                stores["paired_clean_oracle"]["noise_true_logit_change"].extend(
                    (true_logit - noise_true).cpu().tolist()
                )
                stores["paired_clean_oracle"]["clean_correct"].extend(
                    (clean_logits.argmax(1) == labels).cpu().tolist()
                )
                stores["paired_clean_oracle"]["clean_margin_change"].extend(
                    torch.zeros_like(clean_margin).cpu().tolist()
                )
    original_noise = np.asarray(original["noise_correct"], dtype=bool)
    original_clean = np.asarray(original["clean_correct"], dtype=bool)
    results = {
        "original_vit": {
            "noise4_accuracy": float(original_noise.mean()),
            "clean_accuracy": float(original_clean.mean()),
            "mean_noise4_margin": float(np.mean(original["noise_margin"])),
            "mean_clean_margin": float(np.mean(original["clean_margin"])),
        }
    }
    for name, values in stores.items():
        noise_correct = np.asarray(values["noise_correct"], dtype=bool)
        clean_correct = np.asarray(values["clean_correct"], dtype=bool)
        recovered = int((~original_noise & noise_correct).sum())
        damaged = int((original_noise & ~noise_correct).sum())
        results[name] = {
            "noise4_accuracy": float(noise_correct.mean()),
            "noise4_accuracy_gain": float(noise_correct.mean() - original_noise.mean()),
            "clean_accuracy_if_applied": float(clean_correct.mean()),
            "clean_accuracy_gain_if_applied": float(clean_correct.mean() - original_clean.mean()),
            "operational_clean_accuracy": float(original_clean.mean()),
            "mean_noise4_margin_change": float(np.mean(values["noise_margin_change"])),
            "mean_clean_margin_change_if_applied": float(np.mean(values["clean_margin_change"])),
            "mean_noise4_true_logit_change": float(np.mean(values["noise_true_logit_change"])),
            "predictions_recovered": recovered,
            "originally_correct_damaged": damaged,
            "mcnemar_exact_pvalue": float(
                binomtest(recovered, recovered + damaged, 0.5).pvalue
            ) if recovered + damaged else 1.0,
        }
    return results


def make_loader(samples, start, batch_size, workers, seed):
    return DataLoader(
        PairedCorruptionDataset("noise", samples, start, seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )


def main():
    parser = argparse.ArgumentParser(description="Experiment 21: noisy-only affine SAE latent prediction")
    parser.add_argument("--discovery-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--discovery-start", type=int, default=25000)
    parser.add_argument("--validation-start", type=int, default=30000)
    parser.add_argument("--evaluation-start", type=int, default=35000)
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[32, 128, 512, 2048, 24576])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--clean-loss-penalty", type=float, default=2.0)
    parser.add_argument("--sae-source", choices=["clean", "noise"], default="noise")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
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
    sae = (
        load_fixed_sae("clean", device)
        if args.sae_source == "clean"
        else load_sae("noise", "base", "vanilla", device)
    )

    discovery_loader = make_loader(
        args.discovery_samples, args.discovery_start, args.batch_size, args.num_workers, args.seed
    )
    sums, observations = collect_affine_statistics(model, sae, discovery_loader, device)
    parameters = fit_affine(sums, observations, args.ridge)
    np.savez_compressed(
        output_dir / "affine_parameters.npz",
        **{key: value for key, value in parameters.items() if key != "ranking"},
        ranking=parameters["ranking"],
    )
    counts = sorted(set(min(count, sae.latent_dim) for count in args.feature_counts))
    candidates = {}
    for count in counts:
        features = parameters["ranking"][:count].tolist()
        for alpha in args.alphas:
            candidates[f"affine_features{count}_alpha{alpha:g}"] = {
                "method": "affine", "features": features, "alpha": alpha
            }
    validation = evaluate(
        model,
        sae,
        make_loader(args.validation_samples, args.validation_start, args.batch_size, args.num_workers, args.seed),
        device,
        parameters,
        candidates,
    )
    clean_baseline = validation["original_vit"]["clean_accuracy"]
    selected_name = max(
        candidates,
        key=lambda name: validation[name]["noise4_accuracy"]
        - args.clean_loss_penalty * max(
            0.0, clean_baseline - validation[name]["clean_accuracy_if_applied"]
        ),
    )
    selected = candidates[selected_name]
    generator = np.random.default_rng(args.seed)
    shuffled = np.asarray(selected["features"])[generator.permutation(len(selected["features"]))].tolist()
    controls = {
        "selected_affine": selected,
        "mean_clean_control": selected | {"method": "mean"},
        "reverse_direction_control": selected | {"reverse": True},
        "shuffled_parameter_control": selected | {
            "method": "shuffled", "shuffled_features": shuffled
        },
        "residual_identity": selected | {"alpha": 0.0},
    }
    evaluation = evaluate(
        model,
        sae,
        make_loader(args.evaluation_samples, args.evaluation_start, args.batch_size, args.num_workers, args.seed),
        device,
        parameters,
        controls,
        include_oracle=True,
    )
    summary = {
        "configuration": vars(args) | {
            "device": str(device),
            "vit_block": 11,
            "method": "ViT and SAE frozen; noisy image only at inference",
            "split_status": "non-overlapping development/validation/evaluation subsets",
        },
        "discovery": {
            "patch_observations": observations,
            "selected_feature_count_candidates": counts,
        },
        "selected": {"validation_name": selected_name} | selected,
        "validation": validation,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected": summary["selected"], "evaluation": evaluation}, indent=2))
    print(f"Saved Experiment 21 to {output_dir}")


if __name__ == "__main__":
    main()
