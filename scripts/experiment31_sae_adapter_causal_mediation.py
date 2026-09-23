import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment25_multiseed_independent_confirmation import (
    PairedExternalDataset,
    paired_summary,
)
from scripts.experiment26_adapter_robustness_ablations import DEFAULT_CHECKPOINT
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment31_sae_adapter_mediation"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
FEATURE_SOURCE = (
    PROJECT_ROOT / "results" / "sae" / "experiment17_noise_bidirectional_repair"
    / "noise_bidirectional_development_gpu" / "summary.json"
)
BLOCK_INDEX = 10


def decoder_basis(sae, features):
    directions = sae.decoder.weight[:, features].float()
    basis, _ = torch.linalg.qr(directions, mode="reduced")
    return basis


def project(residual, basis):
    return (residual @ basis) @ basis.T


def safe_association(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.std() == 0 or second.std() == 0:
        return {"pearson_r": None, "pearson_p": None, "spearman_r": None, "spearman_p": None}
    pearson = pearsonr(first, second)
    spearman = spearmanr(first, second)
    return {
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }


def group_mean(values, mask):
    selected = np.asarray(values)[np.asarray(mask, dtype=bool)]
    return float(selected.mean()) if selected.size else None


def main():
    parser = argparse.ArgumentParser(description="Experiment 31: causal SAE mediation of adapter gains")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--feature-source", type=Path, default=FEATURE_SOURCE)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--feature-count", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if args.severity != 4:
        raise ValueError("This preregistered experiment uses the adapter's Noise-4 target only")

    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    feature_record = json.loads(args.feature_source.read_text())
    harmful_features = feature_record["selected"]["over_features"][:args.feature_count]
    if len(harmful_features) != args.feature_count:
        raise ValueError("Feature source contains fewer harmful features than requested")

    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    predictor = HiddenLinear().to(device)
    predictor.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    predictor.eval()
    sae = load_fixed_sae("clean", device)
    sae.eval()
    harmful_basis = decoder_basis(sae, harmful_features)
    generator = np.random.default_rng(args.seed)
    available = np.setdiff1d(np.arange(sae.latent_dim), np.asarray(harmful_features))
    random_features = generator.choice(available, args.feature_count, replace=False).tolist()
    random_basis = decoder_basis(sae, random_features)

    dataset = PairedExternalDataset(args.image_root, args.samples, args.start_index, args.noise_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    methods = ["full_adapter", "sae_aligned_only", "sae_orthogonal_only", "random_aligned_only", "random_orthogonal_only"]
    correct = {name: [] for name in methods}
    margin_change = {name: [] for name in methods}
    true_change = {name: [] for name in methods}
    baseline_correct = []
    clean_correct = []
    harmful_distance_before = []
    harmful_distance_after = []
    harmful_normalization = []
    residual_fraction_aligned = []
    random_fraction_aligned = []

    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="SAE-adapter causal mediation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True)
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            clean_patches = clean_hidden[:, 1:]
            noise_patches = noise_hidden[:, 1:]
            residual = args.alpha * predictor(noise_patches)
            aligned = project(residual, harmful_basis)
            random_aligned = project(residual, random_basis)
            variants = {
                "full_adapter": residual,
                "sae_aligned_only": aligned,
                "sae_orthogonal_only": residual - aligned,
                "random_aligned_only": random_aligned,
                "random_orthogonal_only": residual - random_aligned,
            }
            baseline_true, baseline_margin = classification_margin(noise_logits, labels)
            baseline_correct.extend((noise_logits.argmax(1) == labels).cpu().tolist())
            clean_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            for name, intervention in variants.items():
                candidate = torch.cat([noise_hidden[:, :1], noise_patches + intervention], dim=1)
                logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
                true_logit, margin = classification_margin(logits, labels)
                correct[name].extend((logits.argmax(1) == labels).cpu().tolist())
                margin_change[name].extend((margin - baseline_margin).cpu().tolist())
                true_change[name].extend((true_logit - baseline_true).cpu().tolist())

            selected = torch.as_tensor(harmful_features, device=device)
            clean_latent = sae.encode(clean_patches.flatten(0, 1)).reshape(batch, 196, -1)[..., selected]
            noise_latent = sae.encode(noise_patches.flatten(0, 1)).reshape(batch, 196, -1)[..., selected]
            adapted_latent = sae.encode((noise_patches + residual).flatten(0, 1)).reshape(batch, 196, -1)[..., selected]
            before = (noise_latent - clean_latent).abs().mean((1, 2))
            after = (adapted_latent - clean_latent).abs().mean((1, 2))
            harmful_distance_before.extend(before.cpu().tolist())
            harmful_distance_after.extend(after.cpu().tolist())
            harmful_normalization.extend((before - after).cpu().tolist())
            residual_norm = residual.square().sum((1, 2)).clamp_min(1e-12)
            residual_fraction_aligned.extend((aligned.square().sum((1, 2)) / residual_norm).cpu().tolist())
            random_fraction_aligned.extend((random_aligned.square().sum((1, 2)) / residual_norm).cpu().tolist())

    baseline_correct = np.asarray(baseline_correct, dtype=bool)
    clean_correct = np.asarray(clean_correct, dtype=bool)
    method_results = {}
    arrays = {
        "baseline_correct": baseline_correct,
        "clean_correct": clean_correct,
        "harmful_distance_before": np.asarray(harmful_distance_before),
        "harmful_distance_after": np.asarray(harmful_distance_after),
        "harmful_normalization": np.asarray(harmful_normalization),
        "residual_fraction_aligned": np.asarray(residual_fraction_aligned),
        "random_fraction_aligned": np.asarray(random_fraction_aligned),
    }
    for index, name in enumerate(methods):
        method_correct = np.asarray(correct[name], dtype=bool)
        method_margin = np.asarray(margin_change[name], dtype=np.float32)
        method_true = np.asarray(true_change[name], dtype=np.float32)
        method_results[name] = paired_summary(
            baseline_correct,
            method_correct,
            method_margin,
            method_true,
            args.seed + index,
            args.bootstrap_repetitions,
        )
        arrays[f"{name}_correct"] = method_correct
        arrays[f"{name}_margin_change"] = method_margin
        arrays[f"{name}_true_logit_change"] = method_true

    full_correct = arrays["full_adapter_correct"]
    recovered = ~baseline_correct & full_correct
    damaged = baseline_correct & ~full_correct
    unchanged = baseline_correct == full_correct
    normalization = arrays["harmful_normalization"]
    full_margin = arrays["full_adapter_margin_change"]
    full_gain = method_results["full_adapter"]["accuracy_gain"]
    orthogonal_gain = method_results["sae_orthogonal_only"]["accuracy_gain"]
    aligned_gain = method_results["sae_aligned_only"]["accuracy_gain"]
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "feature_source": str(args.feature_source.resolve()),
            "device": str(device),
            "vit_block": 11,
            "sae": "clean_base_vanilla_paper",
            "harmful_features_locked_before_evaluation": harmful_features,
            "random_control_features": random_features,
        },
        "baseline": {
            "noise_accuracy": float(baseline_correct.mean()),
            "clean_accuracy": float(clean_correct.mean()),
        },
        "interventions": method_results,
        "mediation": {
            "harmful_clean_distance_before": float(arrays["harmful_distance_before"].mean()),
            "harmful_clean_distance_after_full_adapter": float(arrays["harmful_distance_after"].mean()),
            "mean_harmful_normalization": float(normalization.mean()),
            "normalization_vs_margin_gain": safe_association(normalization, full_margin),
            "normalization_by_outcome": {
                "recovered": group_mean(normalization, recovered),
                "damaged": group_mean(normalization, damaged),
                "prediction_unchanged": group_mean(normalization, unchanged),
            },
            "mean_residual_energy_fraction_in_harmful_span": float(arrays["residual_fraction_aligned"].mean()),
            "mean_residual_energy_fraction_in_random_span": float(arrays["random_fraction_aligned"].mean()),
            "gain_removed_by_harmful_span_ablation": float(full_gain - orthogonal_gain),
            "gain_retained_by_harmful_span_ablation_fraction": float(orthogonal_gain / full_gain) if full_gain else None,
            "gain_explained_by_aligned_only": float(aligned_gain),
        },
        "decision_rules": {
            "supports_sae_mediation_if": [
                "full adapter moves harmful features closer to paired clean values",
                "normalization predicts margin improvement and recovery",
                "harmful-span removal reduces gain more than matched random-span removal",
                "harmful-aligned component alone retains non-trivial gain",
            ],
            "generic_corrector_if": "gain remains in the SAE-orthogonal component and matched random ablation behaves similarly",
        },
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **arrays)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"baseline": summary["baseline"], "interventions": method_results, "mediation": summary["mediation"]}, indent=2))
    print(f"Saved Experiment 31 to {output_dir}")


if __name__ == "__main__":
    main()
