import argparse
import json
from pathlib import Path

import numpy as np
import torch
from imagecorruptions import get_corruption_names
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment111_leakage_free_head_anomaly_confirmation import freeze_manifest, sha256
from scripts.experiment113_all_corruption_head_detection import PREVIOUS_CONFIRMATIONS
from scripts.experiment114_multisignal_clean_detector import (
    EXPERIMENT113_MANIFEST,
    multisignal_features,
)
from scripts.experiment115_consistency_clean_detector import EXPERIMENT114_MANIFEST
from scripts.experiment116_multiview_clean_detector import (
    EXPERIMENT115_MANIFEST,
    MultiViewDataset,
    multiview_features,
)


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment117_paired_detector_fusion"
BASE_ROOT = ACTIVE_ROOT / "results/sae/experiment114_multisignal_clean_detector/full_level4_15corruptions_cleanonly_v1"
VIEW_ROOT = ACTIVE_ROOT / "results/sae/experiment116_multiview_clean_detector/full_level4_15corruptions_multiview_v1"
VIEW_MANIFEST = VIEW_ROOT / "frozen_manifest.json"
CALIBRATION_GROUPS = ("photo_train", "sketch_train", "photo_validation", "sketch_validation")


def load_features(root, condition):
    return torch.load(root / f"{condition}_features.pt", weights_only=True).numpy().astype(np.float64)


def fit_detector(root):
    photo = load_features(root, "photo_train")
    sketch = load_features(root, "sketch_train")
    training = np.concatenate((photo, sketch))
    location = training.mean(0)
    scale = training.std(0).clip(1e-5)
    models = (
        LedoitWolf().fit((photo - location) / scale),
        LedoitWolf().fit((sketch - location) / scale),
    )
    return location, scale, models


def distances(values, detector):
    location, scale, models = detector
    standardized = (values - location) / scale
    candidates = []
    for model in models:
        centered = standardized - model.location_
        candidates.append(np.einsum("ni,ij,nj->n", centered, model.precision_, centered))
    return np.minimum.reduce(candidates)


def percentile_scores(values, calibration):
    reference = np.sort(calibration)
    return np.searchsorted(reference, values, side="right") / len(reference)


def extract_pair(model, dataset, device, args, path):
    if path.exists():
        with np.load(path) as stored:
            base, views = stored["base"], stored["views"]
        if len(base) != len(dataset) or len(views) != len(dataset):
            raise ValueError(f"Incomplete cached features: {path}")
        return base, views
    base_parts, view_parts = [], []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for batch_index, views in enumerate(loader):
            views = views.to(device)
            original = model(
                pixel_values=views[:, 0],
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            others = [
                model(pixel_values=views[:, index], output_hidden_states=True, return_dict=True)
                for index in range(1, 4)
            ]
            base_parts.append(multisignal_features(views[:, 0], original).numpy())
            view_parts.append(multiview_features([original, *others]).numpy())
            if batch_index % 50 == 0:
                print(f"  batches {batch_index}/{len(loader)}", flush=True)
    base = np.concatenate(base_parts)
    multiview = np.concatenate(view_parts)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, base=base, views=multiview)
    temporary.replace(path)
    return base, multiview


def detection_metrics(clean, corrupt, threshold):
    targets = np.r_[np.zeros(len(clean)), np.ones(len(corrupt))]
    values = np.r_[clean, corrupt]
    return {
        "clean_false_positive_rate": float((clean > threshold).mean()),
        "corruption_recall": float((corrupt > threshold).mean()),
        "auroc": float(roc_auc_score(targets, values)),
        "samples": len(corrupt),
    }


def paired_recall_test(first, second, first_threshold, second_threshold, seed):
    first_detected = first > first_threshold
    second_detected = second > second_threshold
    first_only = int(np.count_nonzero(first_detected & ~second_detected))
    second_only = int(np.count_nonzero(second_detected & ~first_detected))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(first), size=(2000, len(first)))
    differences = first_detected[sampled].mean(1) - second_detected[sampled].mean(1)
    return {
        "recall_difference_pp": float(100 * (first_detected.mean() - second_detected.mean())),
        "first_only": first_only,
        "second_only": second_only,
        "exact_mcnemar_p": float(binomtest(min(first_only, second_only), first_only + second_only, 0.5).pvalue)
        if first_only + second_only else 1.0,
        "paired_bootstrap_95ci_pp": (100 * np.quantile(differences, [0.025, 0.975])).tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--corruption-seed", type=int, default=117000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    configuration_path = output_dir / "configuration.json"
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != vars(args) | {"resume": False}:
            raise ValueError("Resume arguments differ from the frozen run configuration")
    else:
        if args.resume and any(output_dir.iterdir()):
            raise ValueError("Cannot resume an unconfigured run")
        configuration_path.write_text(json.dumps(vars(args) | {"resume": False}, indent=2))
    manifest_path = output_dir / "frozen_manifest.json"
    excluded = (*PREVIOUS_CONFIRMATIONS, EXPERIMENT113_MANIFEST, EXPERIMENT114_MANIFEST,
                EXPERIMENT115_MANIFEST, VIEW_MANIFEST)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else freeze_manifest(manifest_path, excluded)
    print("Manifest", sha256(manifest_path), "images", manifest["samples"], flush=True)

    base_detector = fit_detector(BASE_ROOT)
    view_detector = fit_detector(VIEW_ROOT)
    base_validation = np.concatenate([load_features(BASE_ROOT, name) for name in CALIBRATION_GROUPS[2:]])
    view_validation = np.concatenate([load_features(VIEW_ROOT, name) for name in CALIBRATION_GROUPS[2:]])
    if len(base_validation) != len(view_validation):
        raise ValueError("Clean calibration orders differ")
    base_calibration = distances(base_validation, base_detector)
    view_calibration = distances(view_validation, view_detector)
    base_percentiles = percentile_scores(base_calibration, base_calibration)
    view_percentiles = percentile_scores(view_calibration, view_calibration)
    fusion_calibration = np.maximum(base_percentiles, view_percentiles)
    thresholds = {
        "single_view": float(np.quantile(base_percentiles, 0.95)),
        "multi_view": float(np.quantile(view_percentiles, 0.95)),
        "fusion": float(np.quantile(fusion_calibration, 0.95)),
    }
    print("Clean-only thresholds", thresholds, flush=True)

    np.float_ = np.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    conditions = ("clean", *get_corruption_names())
    scores = {}
    for name in conditions:
        print("Extracting", name, flush=True)
        base, views = extract_pair(
            model,
            MultiViewDataset(manifest, name, args.severity, args.corruption_seed),
            device, args, output_dir / f"{name}_paired_features.npz",
        )
        base_distance = distances(base, base_detector)
        view_distance = distances(views, view_detector)
        single = percentile_scores(base_distance, base_calibration)
        multiview = percentile_scores(view_distance, view_calibration)
        scores[name] = {
            "single_view": single,
            "multi_view": multiview,
            "fusion": np.maximum(single, multiview),
        }

    results = {}
    for method, threshold in thresholds.items():
        clean = scores["clean"][method]
        results[method] = {
            "clean_false_positive_rate": float((clean > threshold).mean()),
            "corruptions": {
                name: detection_metrics(clean, scores[name][method], threshold)
                for name in conditions[1:]
            },
        }
        rows = results[method]["corruptions"].values()
        results[method]["macro_recall"] = float(np.mean([row["corruption_recall"] for row in rows]))
        results[method]["macro_auroc"] = float(np.mean([row["auroc"] for row in rows]))

    paired_tests = {}
    for name_index, name in enumerate(conditions[1:]):
        paired_tests[name] = {
            "fusion_vs_single_view": paired_recall_test(
                scores[name]["fusion"], scores[name]["single_view"],
                thresholds["fusion"], thresholds["single_view"], 117000 + name_index,
            ),
            "fusion_vs_multi_view": paired_recall_test(
                scores[name]["fusion"], scores[name]["multi_view"],
                thresholds["fusion"], thresholds["multi_view"], 118000 + name_index,
            ),
        }

    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "manifest_sha256": sha256(manifest_path),
            "calibration": "same ordered clean photo/sketch validation images for both detectors",
            "fusion": "maximum of clean-calibrated empirical score percentiles",
            "threshold_selection": "95th percentile of clean-only validation fusion scores",
            "corruptions_used_for_fitting_or_threshold": False,
            "repair_applied": False,
            "benchmark_label": "paired detection on new hash-disjoint Sketch images with ImageNet-C severity-4 algorithms",
        },
        "thresholds": thresholds,
        "results": results,
        "paired_recall_tests": paired_tests,
        "limitations": [
            "The fusion is fixed before this evaluation, but its component choice was motivated by earlier corruption results.",
            "Four ViT views are computed for every image; this is not yet a conditional-compute implementation.",
            "No repair is applied in this detector comparison.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: {"clean_fpr": value["clean_false_positive_rate"],
                             "macro_recall": value["macro_recall"],
                             "macro_auroc": value["macro_auroc"]}
                      for key, value in results.items()}, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
