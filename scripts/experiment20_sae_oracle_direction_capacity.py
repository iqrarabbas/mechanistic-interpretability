import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.compare_sae_level4 import load_sae
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment20_oracle_direction_capacity"
BLOCK_INDEX = 10


def decode_delta(sae, clean_hidden, noise_hidden):
    batch, patches, width = clean_hidden.shape
    clean_latent = sae.encode(clean_hidden.reshape(-1, width))
    noise_latent = sae.encode(noise_hidden.reshape(-1, width))
    clean_decoded = sae.decode(clean_latent).reshape(batch, patches, width)
    noise_decoded = sae.decode(noise_latent).reshape(batch, patches, width)
    return clean_decoded - noise_decoded, clean_decoded, noise_decoded


def vector_metrics(estimate, target):
    estimate_flat = estimate.flatten(1)
    target_flat = target.flatten(1)
    estimate_norm = estimate_flat.norm(dim=1)
    target_norm = target_flat.norm(dim=1).clamp_min(1e-8)
    dot = (estimate_flat * target_flat).sum(1)
    cosine = dot / (estimate_norm.clamp_min(1e-8) * target_norm)
    beta = dot / estimate_norm.square().clamp_min(1e-8)
    return {
        "direction_cosine": cosine,
        "relative_error": (estimate_flat - target_flat).norm(dim=1) / target_norm,
        "norm_ratio": estimate_norm / target_norm,
        "target_projection_coefficient": dot / target_norm.square(),
        "aligned_energy_fraction": cosine.square(),
        "mean_patch_cosine": F.cosine_similarity(estimate, target, dim=-1).mean(1),
        "optimal_estimate_scale": beta,
    }


def summarize_values(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "standard_deviation": float(array.std()),
        "median": float(np.median(array)),
        "q05": float(np.quantile(array, 0.05)),
        "q95": float(np.quantile(array, 0.95)),
    }


def summarize_intervention(original_correct, values):
    correct = np.asarray(values["correct"], dtype=bool)
    recovered = int((~original_correct & correct).sum())
    damaged = int((original_correct & ~correct).sum())
    return {
        "accuracy": float(correct.mean()),
        "accuracy_gain": float(correct.mean() - original_correct.mean()),
        "mean_margin_change": float(np.mean(values["margin_change"])),
        "mean_true_logit_change": float(np.mean(values["true_logit_change"])),
        "predictions_recovered": recovered,
        "originally_correct_damaged": damaged,
        "mcnemar_exact_pvalue": float(
            binomtest(recovered, recovered + damaged, 0.5).pvalue
        ) if recovered + damaged else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 20: can clean or mixed-Noise SAE coordinates represent the Block-11 clean-restoration direction?"
    )
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--start-index", type=int, default=40000)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
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
    saes = {
        "clean_only_sae": load_fixed_sae("clean", device),
        "mixed_clean_noise4_sae": load_sae("noise", "base", "vanilla", device),
    }
    loader = DataLoader(
        PairedCorruptionDataset("noise", args.samples, args.start_index, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    metric_names = [
        "direction_cosine",
        "relative_error",
        "norm_ratio",
        "target_projection_coefficient",
        "aligned_energy_fraction",
        "mean_patch_cosine",
        "optimal_estimate_scale",
        "clean_reconstruction_relative_error",
        "noise_reconstruction_relative_error",
    ]
    representation = {
        name: {metric: [] for metric in metric_names} for name in saes
    }
    group_metrics = {
        name: {
            group: {metric: [] for metric in metric_names[:-2]}
            for group in ["both_correct", "clean_correct_noise_wrong", "both_wrong", "clean_wrong_noise_correct"]
        }
        for name in saes
    }
    definitions = {}
    for alpha in args.alphas:
        definitions[f"true_patch_oracle_alpha{alpha:g}"] = ("true", alpha, "raw")
        for name in saes:
            definitions[f"{name}_alpha{alpha:g}"] = (name, alpha, "raw")
    for name in saes:
        definitions[f"{name}_optimal_scalar"] = (name, 1.0, "scaled")
        definitions[f"{name}_shuffled"] = (name, 1.0, "shuffled")
    interventions = {
        name: {"correct": [], "margin_change": [], "true_logit_change": []}
        for name in definitions
    }
    original_correct = []
    original_margin = []
    original_true_logit = []
    clean_correct_store = []
    downstream_parity_error = []

    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="SAE direction capacity"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device),
                output_hidden_states=True,
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            clean_patches = clean_hidden[:, 1:]
            noise_patches = noise_hidden[:, 1:]
            true_delta = clean_patches - noise_patches
            noise_true, noise_margin = classification_margin(noise_logits, labels)
            clean_correct = clean_logits.argmax(1) == labels
            noise_correct = noise_logits.argmax(1) == labels
            original_correct.extend(noise_correct.cpu().tolist())
            clean_correct_store.extend(clean_correct.cpu().tolist())
            original_margin.extend(noise_margin.cpu().tolist())
            original_true_logit.extend(noise_true.cpu().tolist())
            parity_logits = downstream_from_layer(model, noise_hidden, BLOCK_INDEX)
            downstream_parity_error.extend(
                (parity_logits - noise_logits).abs().amax(1).cpu().tolist()
            )

            estimates = {"true": true_delta}
            scales = {}
            for name, sae in saes.items():
                estimate, clean_decoded, noise_decoded = decode_delta(
                    sae, clean_patches, noise_patches
                )
                estimates[name] = estimate
                metrics = vector_metrics(estimate, true_delta)
                scales[name] = metrics["optimal_estimate_scale"]
                metrics["clean_reconstruction_relative_error"] = (
                    (clean_decoded - clean_patches).flatten(1).norm(dim=1)
                    / clean_patches.flatten(1).norm(dim=1).clamp_min(1e-8)
                )
                metrics["noise_reconstruction_relative_error"] = (
                    (noise_decoded - noise_patches).flatten(1).norm(dim=1)
                    / noise_patches.flatten(1).norm(dim=1).clamp_min(1e-8)
                )
                clean_correct_list = clean_correct.cpu().tolist()
                noise_correct_list = noise_correct.cpu().tolist()
                groups = [
                    "both_correct" if clean_correct_list[index] and noise_correct_list[index]
                    else "clean_correct_noise_wrong" if clean_correct_list[index]
                    else "clean_wrong_noise_correct" if noise_correct_list[index]
                    else "both_wrong"
                    for index in range(batch)
                ]
                for metric, values in metrics.items():
                    cpu_values = values.cpu().tolist()
                    representation[name][metric].extend(cpu_values)
                    if metric in group_metrics[name]["both_correct"]:
                        for index, group in enumerate(groups):
                            group_metrics[name][group][metric].append(cpu_values[index])

            for definition, (source, alpha, mode) in definitions.items():
                applied = estimates[source]
                if mode == "scaled":
                    applied = applied * scales[source][:, None, None]
                elif mode == "shuffled":
                    applied = applied.roll(1, dims=0)
                candidate_hidden = torch.cat(
                    [noise_hidden[:, :1], noise_patches + alpha * applied], dim=1
                )
                logits = downstream_from_layer(model, candidate_hidden, BLOCK_INDEX)
                true_logit, margin = classification_margin(logits, labels)
                interventions[definition]["correct"].extend(
                    (logits.argmax(1) == labels).cpu().tolist()
                )
                interventions[definition]["margin_change"].extend(
                    (margin - noise_margin).cpu().tolist()
                )
                interventions[definition]["true_logit_change"].extend(
                    (true_logit - noise_true).cpu().tolist()
                )

    original_correct = np.asarray(original_correct, dtype=bool)
    clean_correct_array = np.asarray(clean_correct_store, dtype=bool)
    representation_summary = {
        name: {metric: summarize_values(values) for metric, values in metrics.items()}
        for name, metrics in representation.items()
    }
    group_summary = {
        name: {
            group: {
                metric: summarize_values(values) if values else None
                for metric, values in metrics.items()
            }
            for group, metrics in groups.items()
        }
        for name, groups in group_metrics.items()
    }
    intervention_summary = {
        name: summarize_intervention(original_correct, values)
        for name, values in interventions.items()
    }
    summary = {
        "configuration": vars(args) | {
            "device": str(device),
            "vit_block": 11,
            "intervention_scope": "patch tokens only; noisy CLS preserved",
            "status": "paired oracle diagnostic; not an inference-time method",
        },
        "sae_checkpoints": {
            "clean_only_sae": "checkpoints/sae/clean_base_vanilla_paper",
            "mixed_clean_noise4_sae": "checkpoints/sae/noise4_base_vanilla_paper",
        },
        "baseline": {
            "clean_accuracy": float(clean_correct_array.mean()),
            "noise4_accuracy": float(original_correct.mean()),
            "mean_noise4_margin": float(np.mean(original_margin)),
            "mean_noise4_true_logit": float(np.mean(original_true_logit)),
            "maximum_downstream_replay_logit_error": float(np.max(downstream_parity_error)),
        },
        "direction_representation": representation_summary,
        "direction_representation_by_prediction_group": group_summary,
        "causal_recovery": intervention_summary,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "baseline": summary["baseline"],
        "direction_means": {
            name: {metric: values["mean"] for metric, values in metrics.items()}
            for name, metrics in representation_summary.items()
        },
        "causal_recovery": intervention_summary,
    }, indent=2))
    print(f"Saved Experiment 20 to {output_dir}")


if __name__ == "__main__":
    main()
