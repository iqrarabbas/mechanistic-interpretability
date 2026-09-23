import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment41_disjoint_gate_development import paired_comparison
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment99_attention_head_corruption_detector import attention_head_statistics
from scripts.experiment101_routed_batchtopk_sae_repairs import BLOCK, ROUTER_ROOT, corrected_logits, load_repair
from scripts.experiment107_fresh_sketch_defocus_confirmation import SketchDefocusDataset


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment110_clean_head_anomaly_detector"
EXPERIMENT109 = ACTIVE_ROOT / "results/sae/experiment109_domain_safe_head_router/full_3seed_clean_defocus345_v1"
FRESH_RUN = ACTIVE_ROOT / "results/sae/experiment107_fresh_sketch_defocus_confirmation/full_1000_disjoint_clean_defocus345_3seed_v1"


def anomaly_score(features, mean, std, top_k):
    deviations = ((features - mean) / std).abs()
    return deviations.topk(top_k, dim=1).values.mean(1)


def fit_clean_detector(seed, sketch_features, top_k, clean_quantile):
    seed_dir = ROUTER_ROOT / f"seed_{seed}"
    calibration = torch.load(seed_dir / "clean_calibration_features.pt", weights_only=True)
    photo_train = torch.load(seed_dir / "train_clean_features.pt", weights_only=True)
    photo_validation = torch.load(seed_dir / "validation_clean_features.pt", weights_only=True)
    clean_train = torch.cat((calibration, photo_train, sketch_features[:2000]))
    clean_validation = torch.cat((photo_validation, sketch_features[2000:]))
    mean = clean_train.mean(0)
    std = clean_train.std(0).clamp_min(1e-6)
    validation_scores = anomaly_score(clean_validation, mean, std, top_k)
    threshold = torch.quantile(validation_scores, clean_quantile)
    return mean, std, threshold, {
        "clean_train_samples": len(clean_train),
        "clean_validation_samples": len(clean_validation),
        "threshold": float(threshold),
        "clean_validation_false_positive_rate": float((validation_scores > threshold).float().mean()),
        "clean_validation_score_mean": float(validation_scores.mean()),
        "clean_validation_score_std": float(validation_scores.std()),
    }


def evaluate(model, mean, std, threshold, repair, dataset, device, args, stat_seed):
    baseline, candidate, detected, scores = [], [], [], []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    repairs = {"blur": repair}
    with torch.no_grad():
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            outputs = model(
                pixel_values=pixels,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            batch_scores = anomaly_score(attention_head_statistics(outputs.attentions).cpu(), mean, std, args.top_k)
            anomaly = batch_scores > threshold
            routes = anomaly.long().to(device) * 2
            logits = corrected_logits(model, outputs.hidden_states[BLOCK], repairs, routes)
            baseline.extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            candidate.extend((logits.argmax(1) == labels).cpu().tolist())
            detected.extend(anomaly.tolist())
            scores.extend(batch_scores.tolist())
    baseline = np.asarray(baseline, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    detected = np.asarray(detected, dtype=bool)
    scores = np.asarray(scores)
    result = paired_comparison(baseline, candidate, stat_seed, args.bootstrap)
    result.update({
        "samples": len(baseline),
        "corruption_detection_fraction": float(detected.mean()),
        "anomaly_score_mean": float(scores.mean()),
        "anomaly_score_std": float(scores.std()),
    })
    return result, baseline, candidate, detected, scores


def main():
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[0, 3, 4, 5])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--clean-quantile", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    progress_path = output_dir / "progress.json"
    outcomes_path = output_dir / "paired_outcomes_partial.npz"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    outcomes = {}
    if outcomes_path.exists():
        with np.load(outcomes_path) as stored:
            outcomes = {name: stored[name] for name in stored.files}

    sketch_features = torch.load(EXPERIMENT109 / "old_sketch_head_features.pt", weights_only=True)
    manifest = json.loads((FRESH_RUN / "frozen_manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    calibration = {}
    for seed in args.seeds:
        mean, std, threshold, calibration[str(seed)] = fit_clean_detector(
            seed, sketch_features, args.top_k, args.clean_quantile
        )
        repair = load_repair(seed, "blur", device)
        results.setdefault(str(seed), {})
        for severity in args.severities:
            key = "clean" if severity == 0 else f"defocus_blur_{severity}"
            if key in results[str(seed)]:
                continue
            print(f"Seed {seed} {key}", flush=True)
            result, baseline, candidate, detected, scores = evaluate(
                model, mean, std, threshold, repair,
                SketchDefocusDataset(manifest, severity, 107000),
                device, args, 1100000 + seed * 100 + severity * 10,
            )
            results[str(seed)][key] = result
            outcomes[f"seed{seed}_{key}_baseline"] = baseline
            outcomes[f"seed{seed}_{key}_candidate"] = candidate
            outcomes[f"seed{seed}_{key}_detected"] = detected
            outcomes[f"seed{seed}_{key}_scores"] = scores
            progress_path.write_text(json.dumps(results, indent=2))
            np.savez_compressed(outcomes_path, **outcomes)
        del repair
        torch.cuda.empty_cache()
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "detector": "one-class clean attention-head anomaly detector",
            "features": "720 head statistics: entropy, maximum, CLS entropy, CLS maximum, CLS patch mass",
            "score": "mean of top-32 absolute per-feature z-scores",
            "threshold_selection": "99th percentile of clean-only validation scores",
            "corruption_examples_used_to_fit_detector": False,
            "vit_and_repair_frozen": True,
            "clean_counterpart_used_at_inference": False,
            "post_hoc_status": "designed after Experiment 107/109 results; requires future untouched confirmation",
        },
        "calibration": calibration,
        "results": results,
        "limitations": [
            "Only Defocus is evaluated here, so detection generality across unseen corruption families remains untested.",
            "The evaluation images were already inspected in Experiments 107 and 109; this is a post-hoc method-development result.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
