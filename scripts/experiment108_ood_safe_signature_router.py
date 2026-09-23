import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment99_attention_head_corruption_detector import attention_head_statistics
from scripts.experiment101_routed_batchtopk_sae_repairs import (
    BLOCK,
    ROUTER_ROOT,
    corrected_logits,
    load_repair,
)
from scripts.experiment107_fresh_sketch_defocus_confirmation import (
    SketchDefocusDataset,
)


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment108_ood_safe_signature_router"
SOURCE_RUN = (
    ACTIVE_ROOT
    / "results/sae/experiment107_fresh_sketch_defocus_confirmation/"
    "full_1000_disjoint_clean_defocus345_3seed_v1"
)
FAMILIES = {1: ("gaussian_noise", "shot_noise", "impulse_noise"), 2: ("gaussian_blur", "disk_blur", "motion_blur")}


def normalize(values):
    return values / values.norm(dim=1, keepdim=True).clamp_min(1e-8)


def fit_signature_router(seed):
    seed_dir = ROUTER_ROOT / f"seed_{seed}"
    calibration = torch.load(seed_dir / "clean_calibration_features.pt", weights_only=True)
    mean = calibration.mean(0)
    std = calibration.std(0).clamp_min(1e-6)
    train_clean = (torch.load(seed_dir / "train_clean_features.pt", weights_only=True) - mean) / std
    validation_clean = (torch.load(seed_dir / "validation_clean_features.pt", weights_only=True) - mean) / std
    signatures = []
    validation_family = {}
    for label in (1, 2):
        train = torch.cat([
            (torch.load(seed_dir / f"train_{name}_features.pt", weights_only=True) - mean) / std
            for name in FAMILIES[label]
        ])
        validation_family[label] = torch.cat([
            (torch.load(seed_dir / f"validation_{name}_features.pt", weights_only=True) - mean) / std
            for name in FAMILIES[label]
        ])
        signatures.append((train.mean(0) - train_clean.mean(0)))
    signatures = torch.stack(signatures)
    signatures = signatures / signatures.norm(dim=1, keepdim=True).clamp_min(1e-8)

    clean_scores = normalize(validation_clean) @ signatures.T
    family_scores = {label: normalize(values) @ signatures.T for label, values in validation_family.items()}
    best = None
    for noise_threshold in torch.linspace(0.0, 0.95, 192):
        for blur_threshold in torch.linspace(0.0, 0.95, 192):
            thresholds = torch.tensor([noise_threshold, blur_threshold])
            clean_routes = route_scores(clean_scores, thresholds)
            noise_routes = route_scores(family_scores[1], thresholds)
            blur_routes = route_scores(family_scores[2], thresholds)
            clean_accuracy = float((clean_routes == 0).float().mean())
            noise_accuracy = float((noise_routes == 1).float().mean())
            blur_accuracy = float((blur_routes == 2).float().mean())
            corruption_accuracy = (noise_accuracy + blur_accuracy) / 2
            if corruption_accuracy < 0.90:
                continue
            objective = (clean_accuracy + noise_accuracy + blur_accuracy) / 3
            candidate = (objective, clean_accuracy, corruption_accuracy, thresholds)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
    if best is None:
        raise RuntimeError("No signature thresholds retain 90% corruption routing")
    return mean, std, signatures, best[3], {
        "development_macro_accuracy": best[0],
        "development_clean_rejection_accuracy": best[1],
        "development_mean_corruption_routing_accuracy": best[2],
        "noise_blur_cosine_thresholds": best[3].tolist(),
    }


def route_scores(scores, thresholds):
    margins = scores - thresholds
    best_margin, best_index = margins.max(1)
    return torch.where(best_margin >= 0, best_index + 1, torch.zeros_like(best_index))


def paired(reference, candidate):
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    return {
        "baseline_accuracy": float(reference.mean()),
        "candidate_accuracy": float(candidate.mean()),
        "accuracy_difference": float(candidate.mean() - reference.mean()),
        "recovered": int((~reference & candidate).sum()),
        "damaged": int((reference & ~candidate).sum()),
    }


def evaluate(model, mean, std, signatures, thresholds, repairs, dataset, device, args):
    baseline, repaired, routes = [], [], []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            outputs = model(
                pixel_values=pixels,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            standardized = (attention_head_statistics(outputs.attentions).cpu() - mean) / std
            scores = normalize(standardized) @ signatures.T
            selected = route_scores(scores, thresholds).to(device)
            logits = corrected_logits(model, outputs.hidden_states[BLOCK], repairs, selected)
            baseline.extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            repaired.extend((logits.argmax(1) == labels).cpu().tolist())
            routes.extend(selected.cpu().tolist())
    result = paired(baseline, repaired)
    route_array = np.asarray(routes)
    result["routing_fractions_unknown_noise_blur"] = [float((route_array == i).mean()) for i in range(3)]
    return result


def main():
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[0, 3, 4, 5])
    parser.add_argument("--corruption-seed", type=int, default=107000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest = json.loads((args.source_run / "frozen_manifest.json").read_text())
    progress_path = output_dir / "progress.json"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    calibration = {}
    for seed in args.seeds:
        mean, std, signatures, thresholds, calibration[str(seed)] = fit_signature_router(seed)
        repairs = {family: load_repair(seed, family, device) for family in ("noise", "blur")}
        results.setdefault(str(seed), {})
        for severity in args.severities:
            key = "clean" if severity == 0 else f"defocus_blur_{severity}"
            if key in results[str(seed)]:
                continue
            print(f"Seed {seed} {key}", flush=True)
            results[str(seed)][key] = evaluate(
                model, mean, std, signatures, thresholds, repairs,
                SketchDefocusDataset(manifest, severity, args.corruption_seed),
                device, args,
            )
            progress_path.write_text(json.dumps(results, indent=2))
        del repairs
        torch.cuda.empty_cache()
    summary = {
        "configuration": vars(args) | {
            "source_run": str(args.source_run.resolve()),
            "model": BASE_MODEL,
            "router": "cosine match to clean-standardized Noise/Blur attention-head deviation signatures with unknown/no-repair rejection",
            "router_fitting_data": "Experiment 99 cached development features only",
            "repair_and_vit_frozen": True,
            "clean_counterpart_used_at_inference": False,
            "post_hoc_status": "designed after observing Experiment 107 Sketch routing failure; not an independent confirmation",
        },
        "calibration": calibration,
        "results": results,
        "limitations": [
            "Experiment 107 images are reused to diagnose whether the new rejection rule addresses its observed failure.",
            "A new untouched image set is required for confirmatory claims about this router.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
