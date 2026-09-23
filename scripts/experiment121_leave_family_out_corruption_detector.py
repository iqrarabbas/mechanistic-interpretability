import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
from imagecorruptions import get_corruption_names
from scipy.stats import binomtest
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment111_leakage_free_head_anomaly_confirmation import OLD_MANIFEST, freeze_manifest, sha256
from scripts.experiment113_all_corruption_head_detection import PREVIOUS_CONFIRMATIONS
from scripts.experiment114_multisignal_clean_detector import (
    ImageDataset,
    multisignal_features,
    OUTPUT_ROOT as EXPERIMENT114_ROOT,
    EXPERIMENT113_MANIFEST,
)
from scripts.experiment115_consistency_clean_detector import EXPERIMENT114_MANIFEST
from scripts.experiment116_multiview_clean_detector import EXPERIMENT115_MANIFEST
from scripts.experiment117_paired_detector_fusion import VIEW_MANIFEST
from scripts.experiment118_frozen_utility_router_sketch import EXPERIMENT117_MANIFEST, OUTPUT_ROOT as EXPERIMENT118_ROOT
from scripts.experiment119_abstaining_repair_selector import OUTPUT_ROOT as EXPERIMENT119_ROOT


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment121_leave_family_out_corruption_detector"
EXPERIMENT118_MANIFEST = EXPERIMENT118_ROOT / "full_frozen_3seed_clean_noise_blur_ood_v1/frozen_manifest.json"
EXPERIMENT119_MANIFEST = EXPERIMENT119_ROOT / "full_3seed_clean_safe_abstaining_v1/frozen_test_manifest.json"
EXPERIMENT114_RUN = EXPERIMENT114_ROOT / "full_level4_15corruptions_cleanonly_v1"
FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur"),
    "weather": ("snow", "frost", "fog"),
    "appearance": ("brightness", "contrast"),
    "digital_geometry": ("elastic_transform", "pixelate", "jpeg_compression"),
}


def load_clean_development(old_items):
    train_path = EXPERIMENT114_RUN / "sketch_train_features.pt"
    validation_path = EXPERIMENT114_RUN / "sketch_validation_features.pt"
    ordered = np.concatenate([
        torch.load(path, map_location="cpu", weights_only=True).numpy()
        for path in (train_path, validation_path)
    ])
    if len(ordered) != len(old_items) or ordered.shape[1] != 152:
        raise ValueError("Cached clean features do not match the old Sketch manifest")
    return ordered[0::3], ordered[1::3]


def extract_features(model, manifest, corruption, corruption_seed, args, path):
    if path.exists():
        with np.load(path) as stored:
            features = stored["features"]
        if len(features) != len(manifest["items"]):
            raise ValueError(f"Incomplete feature cache: {path}")
        return features
    dataset = ImageDataset(manifest, corruption, args.severity, corruption_seed)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    parts = []
    with torch.no_grad():
        for batch_index, pixels in enumerate(loader):
            pixels = pixels.to(model.device)
            outputs = model(
                pixel_values=pixels, output_attentions=True,
                output_hidden_states=True, return_dict=True,
            )
            parts.append(multisignal_features(pixels, outputs).numpy())
            if batch_index % 50 == 0:
                print(f"  {corruption}: {batch_index}/{len(loader)} batches", flush=True)
    features = np.concatenate(parts).astype(np.float32)
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, features=features)
    temporary.replace(path)
    return features


def fit_supervised(clean_features, corrupted_features, held_out, path):
    if path.exists():
        return joblib.load(path)
    if held_out == "core_noise_blur":
        training_names = [name for name in corrupted_features if name in (*FAMILIES["noise"], *FAMILIES["blur"])]
    else:
        training_names = [name for name in corrupted_features if held_out == "all_seen" or name not in FAMILIES[held_out]]
    positive = np.concatenate([corrupted_features[name] for name in training_names])
    features = np.concatenate((clean_features, positive))
    labels = np.r_[np.zeros(len(clean_features), dtype=np.int8), np.ones(len(positive), dtype=np.int8)]
    weights = np.r_[
        np.full(len(clean_features), len(features) / (2 * len(clean_features))),
        np.full(len(positive), len(features) / (2 * len(positive))),
    ]
    model = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=30, l2_regularization=10.0,
        early_stopping=False, random_state=121,
    )
    model.fit(features, labels, sample_weight=weights)
    joblib.dump(model, path)
    return model


def fit_clean_only(clean_features):
    location = clean_features.mean(axis=0).astype(np.float64)
    scale = clean_features.std(axis=0).clip(1e-5).astype(np.float64)
    model = LedoitWolf().fit((clean_features - location) / scale)
    return location, scale, model


def clean_only_scores(features, detector):
    location, scale, model = detector
    centered = (features - location) / scale - model.location_
    return np.einsum("ni,ij,nj->n", centered, model.precision_, centered)


def paired_recall(first, second, seed, bootstrap):
    first_only = int((first & ~second).sum())
    second_only = int((~first & second).sum())
    generator = np.random.default_rng(seed)
    sampled = generator.integers(0, len(first), size=(bootstrap, len(first)))
    differences = first[sampled].mean(axis=1) - second[sampled].mean(axis=1)
    return {
        "first_only": first_only,
        "second_only": second_only,
        "recall_difference_pp": float(100 * (first.mean() - second.mean())),
        "paired_bootstrap_95ci_pp": (100 * np.quantile(differences, [0.025, 0.975])).tolist(),
        "exact_mcnemar_p": float(binomtest(first_only, first_only + second_only, 0.5).pvalue)
        if first_only + second_only else 1.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--corruption-seed", type=int, default=121000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.severity != 4:
        raise ValueError("This experiment is restricted to severity 4")
    corruptions = tuple(get_corruption_names())
    if set(corruptions) != {name for names in FAMILIES.values() for name in names}:
        raise ValueError("Corruption families do not cover the available 15 algorithms exactly")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    configuration = vars(args) | {"resume": False}
    config_path = output_dir / "configuration.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError("Resume arguments differ from the frozen configuration")
    else:
        if args.resume and any(output_dir.iterdir()):
            raise ValueError("Cannot resume without frozen configuration")
        config_path.write_text(json.dumps(configuration, indent=2))
    old_items = json.loads(OLD_MANIFEST.read_text())["items"]
    train_items, validation_items = old_items[0::3], old_items[1::3]
    if len(train_items) != 1000 or len(validation_items) != 1000:
        raise ValueError("Expected one clean training and one validation image per class")
    train_manifest = {"items": train_items}
    test_manifest_path = output_dir / "frozen_test_manifest.json"
    exclusions = (
        *PREVIOUS_CONFIRMATIONS, EXPERIMENT113_MANIFEST, EXPERIMENT114_MANIFEST,
        EXPERIMENT115_MANIFEST, VIEW_MANIFEST, EXPERIMENT117_MANIFEST,
        EXPERIMENT118_MANIFEST, EXPERIMENT119_MANIFEST,
    )
    test_manifest = (
        json.loads(test_manifest_path.read_text()) if test_manifest_path.exists()
        else freeze_manifest(test_manifest_path, exclusions)
    )
    development_hashes = {item["sha256"] for item in train_items + validation_items}
    if development_hashes & {item["sha256"] for item in test_manifest["items"]}:
        raise ValueError("Development/test image hash overlap")
    print("Frozen test manifest", sha256(test_manifest_path), "images", test_manifest["samples"], flush=True)
    clean_train, clean_validation = load_clean_development(old_items)
    np.float_ = np.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    model.requires_grad_(False)
    train_features = {}
    for corruption in corruptions:
        print("Training features", corruption, flush=True)
        train_features[corruption] = extract_features(
            model, train_manifest, corruption, args.corruption_seed,
            args, output_dir / f"train_{corruption}.npz",
        )
    clean_detector = fit_clean_only(clean_train)
    clean_only_validation = clean_only_scores(clean_validation, clean_detector)
    clean_only_threshold = float(np.quantile(clean_only_validation, 0.95))
    models = {}
    thresholds = {}
    for family in (*FAMILIES, "all_seen", "core_noise_blur"):
        print("Fitting detector", family, flush=True)
        classifier = fit_supervised(clean_train, train_features, family, output_dir / f"detector_{family}.joblib")
        models[family] = classifier
        thresholds[family] = float(np.quantile(classifier.predict_proba(clean_validation)[:, 1], 0.95))
    test_features = {}
    for corruption in ("clean", *corruptions):
        print("Test features", corruption, flush=True)
        test_features[corruption] = extract_features(
            model, test_manifest, corruption, args.corruption_seed + 200000,
            args, output_dir / f"test_{corruption}.npz",
        )
    clean_scores = {
        family: model.predict_proba(test_features["clean"])[:, 1]
        for family, model in models.items()
    }
    clean_only_test = clean_only_scores(test_features["clean"], clean_detector)
    results = {}
    detection_arrays = {}
    for condition_index, corruption in enumerate(corruptions):
        family = next(family for family, members in FAMILIES.items() if corruption in members)
        score = models[family].predict_proba(test_features[corruption])[:, 1]
        all_seen_score = models["all_seen"].predict_proba(test_features[corruption])[:, 1]
        core_score = models["core_noise_blur"].predict_proba(test_features[corruption])[:, 1]
        clean_only_score = clean_only_scores(test_features[corruption], clean_detector)
        held_out_detected = score > thresholds[family]
        all_seen_detected = all_seen_score > thresholds["all_seen"]
        core_detected = core_score > thresholds["core_noise_blur"]
        clean_only_detected = clean_only_score > clean_only_threshold
        detection_arrays[f"{corruption}_held_out"] = held_out_detected
        detection_arrays[f"{corruption}_clean_only"] = clean_only_detected
        detection_arrays[f"{corruption}_core"] = core_detected
        results[corruption] = {
            "family": family,
            "held_out_recall": float(held_out_detected.mean()),
            "held_out_auroc": float(roc_auc_score(
                np.r_[np.zeros(len(clean_scores[family])), np.ones(len(score))],
                np.r_[clean_scores[family], score],
            )),
            "all_seen_recall": float(all_seen_detected.mean()),
            "core_recall": float(core_detected.mean()),
            "core_auroc": float(roc_auc_score(
                np.r_[np.zeros(len(clean_scores["core_noise_blur"])), np.ones(len(core_score))],
                np.r_[clean_scores["core_noise_blur"], core_score],
            )),
            "clean_only_recall": float(clean_only_detected.mean()),
            "held_out_vs_clean_only": paired_recall(
                held_out_detected, clean_only_detected, 121000 + condition_index, args.bootstrap,
            ),
            "core_vs_clean_only": paired_recall(
                core_detected, clean_only_detected, 122000 + condition_index, args.bootstrap,
            ),
        }
    clean_fpr = {family: float((scores > thresholds[family]).mean()) for family, scores in clean_scores.items()}
    clean_fpr["clean_only"] = float((clean_only_test > clean_only_threshold).mean())
    summary = {
        "configuration": configuration | {
            "model": BASE_MODEL,
            "feature_dimension": int(clean_train.shape[1]),
            "test_manifest_sha256": sha256(test_manifest_path),
            "train_images": 1000,
            "clean_validation_images": 1000,
            "test_images": test_manifest["samples"],
            "family_map": FAMILIES,
            "threshold_rule": "95th percentile of disjoint clean validation scores for each detector",
            "vit_frozen": True,
            "clean_counterpart_at_inference": False,
            "core_model_uses_corruption_name_at_inference": False,
            "leave_one_family_out_model_selection": "offline evaluation protocol only; different frozen model per held-out family, not a deployable oracle router",
            "benchmark": "online ImageNet-C algorithms applied to Sketch images at severity 4, not official ImageNet-C",
        },
        "clean_fpr": clean_fpr,
        "thresholds": thresholds | {"clean_only": clean_only_threshold},
        "results": results,
        "macro_held_out_recall": float(np.mean([row["held_out_recall"] for row in results.values()])),
        "macro_all_seen_recall": float(np.mean([row["all_seen_recall"] for row in results.values()])),
        "macro_clean_only_recall": float(np.mean([row["clean_only_recall"] for row in results.values()])),
        "macro_core_unseen_recall": float(np.mean([
            row["core_recall"] for row in results.values() if row["family"] not in ("noise", "blur")
        ])),
        "macro_core_seen_recall": float(np.mean([
            row["core_recall"] for row in results.values() if row["family"] in ("noise", "blur")
        ])),
        "limitations": [
            "Previously studied Sketch development images are reused for fitting, but all test image hashes are excluded from prior frozen manifests.",
            "Leave-one-family-out results test unknown corruption algorithms, not a guarantee for all possible image changes.",
            "Only the core Noise+Blur model is a single fixed detector for all corruption families; LOFO rows use distinct offline-frozen models.",
            "Only severity 4 is evaluated; clean images from different natural domains may have a different false-positive rate.",
            "All-seen results are a supervised known-corruption ceiling, not an unknown-family test.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    np.savez_compressed(output_dir / "paired_detection_outcomes.npz", **detection_arrays)
    print(json.dumps({
        "clean_fpr": clean_fpr,
        "macro_held_out_recall": summary["macro_held_out_recall"],
        "macro_all_seen_recall": summary["macro_all_seen_recall"],
        "macro_clean_only_recall": summary["macro_clean_only_recall"],
        "macro_core_unseen_recall": summary["macro_core_unseen_recall"],
    }, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
