import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment3_sae_causal_intervention import downstream_logits
from scripts.experiment4_non_oracle_sae_correction import discovery_statistics, feature_table, fit_affine
from scripts.experiment10_corruption_agnostic_sae_repair import encode, load_fixed_sae
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment13_patch_aware_residual_repair import make_patch_mask


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment17_noise_bidirectional_repair"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "configs" / "split_manifest_supervisor_v1.json"


def enforce_locked_feature_splits(args):
    manifest = json.loads(args.split_manifest.read_text())
    splits = {
        split["name"]: split
        for split in manifest[args.split_protocol]["splits"]
    }
    requested = {
        "harmful_feature_discovery": (args.discovery_start, args.discovery_samples),
        "harmful_feature_validation": (args.validation_start, args.validation_samples),
        "feature_protocol_dry_run": (args.evaluation_start, args.evaluation_samples),
    }
    for name, (start, samples) in requested.items():
        if name not in splits:
            raise ValueError(f"Locked split {name!r} is missing from {args.split_manifest}")
        split = splits[name]
        expected = (split["start"], split["end"] - split["start"])
        if (start, samples) != expected:
            raise ValueError(
                f"{name} requested start/samples={(start, samples)}, expected={expected}"
            )
        if split["isolation_group"] != "feature_development":
            raise ValueError(f"{name} is not assigned to feature_development")


class DiscoveryDataset(Dataset):
    def __init__(self, samples, start_index, seed):
        self.paired = PairedCorruptionDataset("noise", samples, start_index, seed)

    def __len__(self):
        return len(self.paired)

    def __getitem__(self, index):
        clean, noise, label = self.paired[index]
        return clean, noise, label, index, index


def signed_correction(latent, scale, intercept, config):
    feature_groups = []
    if config["mode"] in {"over", "bidirectional"}:
        feature_groups.append((config["over_features"], -1))
    if config["mode"] in {"under", "bidirectional"}:
        feature_groups.append((config["under_features"], 1))
    correction = torch.zeros_like(latent)
    patch_scores = latent.new_zeros(latent.shape[:2])
    for features, direction in feature_groups:
        selected = torch.as_tensor(features, dtype=torch.long, device=latent.device)
        values = latent[..., selected]
        affine_delta = values * scale[selected] + intercept[selected] - values
        if direction < 0 and config.get("correction") == "clip":
            directional_delta = (config["upper_threshold"][selected] - values).clamp_max(0)
        else:
            directional_delta = affine_delta.clamp_max(0) if direction < 0 else affine_delta.clamp_min(0)
        if direction < 0 and "upper_threshold" in config:
            directional_delta *= values > config["upper_threshold"][selected]
        if config.get("reverse", False):
            directional_delta = -directional_delta
        correction[..., selected] += directional_delta
        patch_scores += directional_delta.abs().sum(-1)
    patch_mask = make_patch_mask(patch_scores, config["patch_count"], config.get("strategy", "top"))
    correction *= patch_mask[..., None]
    effective = (correction != 0) & (config["alpha"] != 0)
    return latent + config["alpha"] * correction, effective


def evaluate(model, sae, loader, device, scale, intercept, configurations):
    scale = torch.as_tensor(scale, device=device)
    intercept = torch.as_tensor(intercept, device=device)
    totals = {
        name: {key: 0.0 for key in ["clean_correct", "noise_correct", "clean_margin", "noise_margin", "recovered", "damaged", "changed", "patches"]}
        for name in configurations
    }
    original = {key: 0.0 for key in ["clean_correct", "noise_correct", "clean_margin", "noise_margin"]}
    total = 0
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, leave=False):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True)
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[-2].split(batch)
            for kind, logits in [("clean", clean_logits), ("noise", noise_logits)]:
                original[f"{kind}_correct"] += int((logits.argmax(1) == labels).sum())
                original[f"{kind}_margin"] += float(classification_margin(logits, labels)[1].sum())
            for kind, hidden, original_logits in [("clean", clean_hidden, clean_logits), ("noise", noise_hidden, noise_logits)]:
                latent = sae.encode(hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)
                decoded = sae.decode(latent)
                for name, config in configurations.items():
                    candidate, changed = signed_correction(latent, scale, intercept, config)
                    patches = hidden[:, 1:] + sae.decode(candidate) - decoded
                    logits = downstream_logits(model, torch.cat([hidden[:, :1], patches], 1))
                    totals[name][f"{kind}_correct"] += int((logits.argmax(1) == labels).sum())
                    totals[name][f"{kind}_margin"] += float(classification_margin(logits, labels)[1].sum())
                    if kind == "noise":
                        original_prediction = original_logits.argmax(1)
                        prediction = logits.argmax(1)
                        totals[name]["recovered"] += int(((original_prediction != labels) & (prediction == labels)).sum())
                        totals[name]["damaged"] += int(((original_prediction == labels) & (prediction != labels)).sum())
                        totals[name]["changed"] += int(changed.sum())
                        totals[name]["patches"] += int(changed.any(-1).sum())
            total += batch
    results = {"original_vit": {key: value / total for key, value in original.items()}}
    for name, values in totals.items():
        results[name] = {
            "clean_accuracy": values["clean_correct"] / total,
            "noise4_accuracy": values["noise_correct"] / total,
            "clean_accuracy_gain": (values["clean_correct"] - original["clean_correct"]) / total,
            "noise4_accuracy_gain": (values["noise_correct"] - original["noise_correct"]) / total,
            "clean_margin_change": (values["clean_margin"] - original["clean_margin"]) / total,
            "noise4_margin_change": (values["noise_margin"] - original["noise_margin"]) / total,
            "predictions_recovered": int(values["recovered"]),
            "originally_correct_damaged": int(values["damaged"]),
            "mean_changed_feature_patch_pairs": values["changed"] / total,
            "mean_changed_patches": values["patches"] / total,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 17: Noise-specific bidirectional residual repair")
    parser.add_argument("--discovery-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--discovery-start", type=int, default=25000)
    parser.add_argument("--validation-start", type=int, default=30000)
    parser.add_argument("--evaluation-start", type=int, default=35000)
    parser.add_argument("--feature-counts", type=int, nargs="+", default=[8, 16, 24])
    parser.add_argument("--patch-counts", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--minimum-consistency", type=float, default=0.55)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--enforce-locked-splits", action="store_true")
    args = parser.parse_args()
    if args.enforce_locked_splits:
        enforce_locked_feature_splits(args)
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume or a new run name")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)

    discovery_path = output_dir / "discovery_candidates.json"
    affine_path = output_dir / "affine_parameters.npz"
    if args.resume and discovery_path.exists() and affine_path.exists():
        discovery = json.loads(discovery_path.read_text())
        over, under = discovery["over"], discovery["under"]
        affine = np.load(affine_path)
        scale, intercept = affine["scale"], affine["intercept"]
    else:
        discovery_loader = DataLoader(DiscoveryDataset(args.discovery_samples, args.discovery_start, args.seed), batch_size=args.batch_size)
        image_delta, delta_margin, clean_correct, noise_correct, sums = discovery_statistics(model, sae, discovery_loader, device)
        observations = args.discovery_samples * 196
        moments = fit_affine(sums, observations, args.ridge)
        _, candidates = feature_table(image_delta, delta_margin, clean_correct, noise_correct, moments, args.minimum_consistency)
        over = [row for row in candidates if row["mean_blur_minus_clean"] > 0 and row["failure_group_difference"] > 0]
        under = [row for row in candidates if row["mean_blur_minus_clean"] < 0 and row["failure_group_difference"] < 0]
        rank = lambda row: abs(row["pearson_delta_margin"] * row["standardized_change"] * row["failure_group_difference"] * row["direction_consistency"])
        over.sort(key=rank, reverse=True)
        under.sort(key=rank, reverse=True)
        scale, intercept = moments[:2]
        discovery_path.write_text(json.dumps({"over": over, "under": under}, indent=2))
        np.savez_compressed(affine_path, scale=scale, intercept=intercept)
    candidates_config = {}
    feasible = {"over": [], "under": [], "bidirectional": []}
    for count in args.feature_counts:
        base = {
            "over_features": [row["feature_index"] for row in over[:count]],
            "under_features": [row["feature_index"] for row in under[:count]],
        }
        for mode in ["over", "under", "bidirectional"]:
            required = (
                len(over) if mode == "over"
                else len(under) if mode == "under"
                else min(len(over), len(under))
            )
            if count > required:
                continue
            feasible[mode].append(count)
            for patches in args.patch_counts:
                for alpha in args.alphas:
                    name = f"{mode}_features{count}_patches{patches}_alpha{alpha:g}"
                    candidates_config[name] = base | {"mode": mode, "patch_count": patches, "alpha": alpha}
    if not candidates_config:
        raise RuntimeError(
            f"No requested feature count fits candidates: over={len(over)}, under={len(under)}"
        )
    print(f"Feasible counts by mode: {feasible}; candidates: over={len(over)}, under={len(under)}")
    validation_loader = DataLoader(PairedCorruptionDataset("noise", args.validation_samples, args.validation_start, args.seed), batch_size=args.batch_size)
    validation = evaluate(model, sae, validation_loader, device, scale, intercept, candidates_config)
    clean_base = validation["original_vit"]["clean_correct"]
    selected_name = max(candidates_config, key=lambda name: validation[name]["noise4_accuracy"] - 2 * max(0, clean_base - validation[name]["clean_accuracy"]))
    selected = candidates_config[selected_name]
    generator = np.random.default_rng(args.seed)
    random_count = max(len(selected["over_features"]), len(selected["under_features"]))
    random_features = generator.choice(sae.latent_dim, random_count * 2, replace=False).tolist()
    configurations = {
        "selected": selected,
        "over_only": selected | {"mode": "over"},
        "under_only": selected | {"mode": "under"},
        "wrong_direction_control": selected | {"reverse": True},
        "random_feature_control": selected | {
            "over_features": random_features[:random_count],
            "under_features": random_features[random_count:],
        },
        "residual_identity": selected | {"alpha": 0.0},
    }
    evaluation_loader = DataLoader(PairedCorruptionDataset("noise", args.evaluation_samples, args.evaluation_start, args.seed), batch_size=args.batch_size)
    evaluation = evaluate(model, sae, evaluation_loader, device, scale, intercept, configurations)
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "status": "exploratory feature-development experiment",
        },
        "candidate_counts": {"over": len(over), "under": len(under)},
        "feasible_feature_counts_by_mode": feasible,
        "selected": {"name": selected_name} | selected,
        "top_over_features": over[:max(args.feature_counts)],
        "top_under_features": under[:max(args.feature_counts)],
        "validation": validation,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"candidate_counts": summary["candidate_counts"], "selected": summary["selected"], "evaluation": evaluation}, indent=2))
    print(f"Saved Experiment 17 to {output_dir}")


if __name__ == "__main__":
    main()
