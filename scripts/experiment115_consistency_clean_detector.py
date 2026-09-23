import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from imagecorruptions import get_corruption_names
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment107_fresh_sketch_defocus_confirmation import DATA_ROOT
from scripts.experiment111_leakage_free_head_anomaly_confirmation import freeze_manifest, sha256
from scripts.experiment113_all_corruption_head_detection import PREVIOUS_CONFIRMATIONS
from scripts.experiment114_multisignal_clean_detector import (
    EXPERIMENT113_MANIFEST,
    ImageDataset,
    multisignal_features,
)


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment115_consistency_clean_detector"
OLD_SKETCH_MANIFEST = Path(
    "/media/dr-yougart/Iqrar/datasets/imagenet_sketch/frozen_subset_3_per_class_unique_manifest.json"
)
EXPERIMENT114_MANIFEST = (
    ACTIVE_ROOT / "results/sae/experiment114_multisignal_clean_detector/"
    "full_level4_15corruptions_cleanonly_v1/frozen_manifest.json"
)


def entropy(probabilities):
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)


def consistency_features(pixels, original, flipped):
    original_probabilities = original.logits.softmax(1)
    flipped_probabilities = flipped.logits.softmax(1)
    mean_probabilities = (original_probabilities + flipped_probabilities) / 2
    js_divergence = (
        functional.kl_div(mean_probabilities.log(), original_probabilities, reduction="none").sum(1)
        + functional.kl_div(mean_probabilities.log(), flipped_probabilities, reduction="none").sum(1)
    ) / 2
    original_top = original_probabilities.topk(2, dim=1).values
    flipped_top = flipped_probabilities.topk(2, dim=1).values
    values = [
        original_top[:, 0],
        original_top[:, 0] - original_top[:, 1],
        entropy(original_probabilities),
        flipped_top[:, 0],
        flipped_top[:, 0] - flipped_top[:, 1],
        entropy(flipped_probabilities),
        js_divergence,
        (original_probabilities.argmax(1) != flipped_probabilities.argmax(1)).float(),
    ]
    for block in (3, 6, 9, 12):
        original_cls = original.hidden_states[block][:, 0].float()
        flipped_cls = flipped.hidden_states[block][:, 0].float()
        original_patch = original.hidden_states[block][:, 1:].float().mean(1)
        flipped_patch = flipped.hidden_states[block][:, 1:].float().mean(1)
        values.extend((
            functional.cosine_similarity(original_cls, flipped_cls),
            (original_cls - flipped_cls).norm(dim=1) / original_cls.norm(dim=1).clamp_min(1e-6),
            functional.cosine_similarity(original_patch, flipped_patch),
            (original_patch - flipped_patch).norm(dim=1) / original_patch.norm(dim=1).clamp_min(1e-6),
        ))
    base = multisignal_features(pixels, original)
    return torch.cat((base, torch.stack(values, dim=1).cpu()), dim=1)


def extract(model, dataset, device, args, path):
    if path.exists():
        return torch.load(path, weights_only=True)
    features = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for batch_index, pixels in enumerate(loader):
            pixels = pixels.to(device)
            original = model(
                pixel_values=pixels,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            flipped = model(
                pixel_values=torch.flip(pixels, dims=(3,)),
                output_hidden_states=True,
                return_dict=True,
            )
            features.append(consistency_features(pixels, original, flipped))
            if batch_index % 50 == 0:
                print(f"  batches {batch_index}/{len(loader)}", flush=True)
    values = torch.cat(features)
    torch.save(values, path)
    return values


def density_scores(values, location, scale, models):
    standardized = (values - location) / scale
    distances = []
    for model in models:
        centered = standardized - model.location_
        distances.append(np.einsum("ni,ij,nj->n", centered, model.precision_, centered))
    return np.minimum.reduce(distances)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--corruption-seed", type=int, default=115000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output_dir / "frozen_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists()
        else freeze_manifest(
            manifest_path,
            (*PREVIOUS_CONFIRMATIONS, EXPERIMENT113_MANIFEST, EXPERIMENT114_MANIFEST),
        )
    )
    print("Manifest", sha256(manifest_path), "images", manifest["samples"], flush=True)
    old_sketch = json.loads(OLD_SKETCH_MANIFEST.read_text())
    photo_train = ImageNetDataset(Path("Dataset"), 1000, 0)
    photo_validation = ImageNetDataset(Path("Dataset"), 500, 10000)
    groups = {
        "photo_train": ImageDataset(photo_train, "clean", args.severity, args.corruption_seed),
        "photo_validation": ImageDataset(photo_validation, "clean", args.severity, args.corruption_seed),
        "sketch_train": ImageDataset({"items": old_sketch["items"][:2000]}, "clean", args.severity, args.corruption_seed),
        "sketch_validation": ImageDataset({"items": old_sketch["items"][2000:]}, "clean", args.severity, args.corruption_seed),
        "clean": ImageDataset(manifest, "clean", args.severity, args.corruption_seed),
    }
    for name in get_corruption_names():
        groups[name] = ImageDataset(manifest, name, args.severity, args.corruption_seed)

    np.float_ = np.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    for name, dataset in groups.items():
        print("Extracting", name, flush=True)
        extract(model, dataset, device, args, output_dir / f"{name}_features.pt")

    def loaded(name):
        return torch.load(output_dir / f"{name}_features.pt", weights_only=True).numpy().astype(np.float64)

    photo, sketch = loaded("photo_train"), loaded("sketch_train")
    training = np.concatenate((photo, sketch))
    location = training.mean(0)
    scale = training.std(0).clip(1e-5)
    models = (
        LedoitWolf().fit((photo - location) / scale),
        LedoitWolf().fit((sketch - location) / scale),
    )
    validation = np.concatenate((loaded("photo_validation"), loaded("sketch_validation")))
    threshold = float(np.quantile(density_scores(validation, location, scale, models), 0.95))
    clean_scores = density_scores(loaded("clean"), location, scale, models)
    results = {
        "clean": {
            "false_positive_rate": float((clean_scores > threshold).mean()),
            "samples": len(clean_scores),
        }
    }
    for name in get_corruption_names():
        scores = density_scores(loaded(name), location, scale, models)
        results[name] = {
            "recall": float((scores > threshold).mean()),
            "auroc": float(roc_auc_score(
                np.r_[np.zeros(len(clean_scores)), np.ones(len(scores))],
                np.r_[clean_scores, scores],
            )),
            "samples": len(scores),
        }
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "manifest_sha256": sha256(manifest_path),
            "feature_dimension": len(location),
            "features": "Experiment 114 multi-signal features plus flip-based logits/CLS/patch consistency",
            "detector": "minimum photo/sketch Ledoit-Wolf Mahalanobis distance",
            "threshold": "95th percentile of clean-only validation distances",
            "corruptions_used_for_fitting_or_threshold": False,
            "repair_applied": False,
            "benchmark_label": "ImageNet-C algorithms on a new hash-disjoint ImageNet-Sketch slice",
        },
        "threshold": threshold,
        "results": results,
        "limitations": [
            "Horizontal flip approximately doubles ViT inference cost.",
            "No method can guarantee detection of every possible attack; this tests the 15 standard severity-4 corruptions.",
            "No repair is applied in this detection experiment.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
