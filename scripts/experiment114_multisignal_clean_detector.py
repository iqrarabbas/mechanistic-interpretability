import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment99_attention_head_corruption_detector import attention_head_statistics
from scripts.experiment107_fresh_sketch_defocus_confirmation import DATA_ROOT
from scripts.experiment111_leakage_free_head_anomaly_confirmation import freeze_manifest, sha256
from scripts.experiment113_all_corruption_head_detection import PREVIOUS_CONFIRMATIONS


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment114_multisignal_clean_detector"
OLD_SKETCH_MANIFEST = Path(
    "/media/dr-yougart/Iqrar/datasets/imagenet_sketch/frozen_subset_3_per_class_unique_manifest.json"
)
EXPERIMENT113_MANIFEST = (
    ACTIVE_ROOT / "results/sae/experiment113_all_corruption_head_detection/"
    "full_15corruptions_level4_q95_v1/frozen_manifest.json"
)


class ImageDataset(Dataset):
    def __init__(self, source, name, severity, seed):
        self.source = source
        self.name = name
        self.severity = severity
        self.seed = seed
        if isinstance(source, dict):
            self.items = source["items"]
        else:
            self.items = None

    def __len__(self):
        return len(self.items) if self.items is not None else len(self.source)

    def __getitem__(self, index):
        if self.items is not None:
            row = self.items[index]
            path = DATA_ROOT / row["relative_path"]
        else:
            path = self.source.image_paths[index]
        image = Image.open(path).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.uint8)
        if self.name != "clean":
            np.random.seed(self.seed + index)
            values = corrupt(values, corruption_name=self.name, severity=self.severity)
        pixels = torch.from_numpy(np.asarray(values).copy()).permute(2, 0, 1).float()
        return pixels.div_(127.5).sub_(1.0)


def multisignal_features(pixels, outputs):
    heads = attention_head_statistics(outputs.attentions)
    heads = heads.reshape(len(pixels), 12, 12, 5)
    head_summary = torch.cat((heads.mean(2).flatten(1), heads.std(2).flatten(1)), dim=1)

    patch_summary = []
    for block in (3, 6, 9, 12):
        patches = outputs.hidden_states[block][:, 1:].float()
        norms = patches.norm(dim=-1)
        patch_summary.extend((
            patches.mean(dim=(1, 2)),
            patches.std(dim=(1, 2)),
            norms.mean(dim=1),
            norms.std(dim=1),
            outputs.hidden_states[block][:, 0].float().norm(dim=1),
        ))

    image = (pixels.float() + 1.0) * 0.5
    gray = image.mean(dim=1, keepdim=True)
    pixels_summary = [
        image.mean(dim=(2, 3))[:, channel] for channel in range(3)
    ] + [
        image.std(dim=(2, 3))[:, channel] for channel in range(3)
    ]
    pixels_summary.extend((
        (gray[:, :, 1:] - gray[:, :, :-1]).abs().mean(dim=(1, 2, 3)),
        (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs().mean(dim=(1, 2, 3)),
    ))
    for width in (3, 9, 17):
        high_frequency = gray - functional.avg_pool2d(gray, width, stride=1, padding=width // 2)
        pixels_summary.append(high_frequency.abs().mean(dim=(1, 2, 3)))
    pixels_summary.append(
        (gray[:, :, :, 8::8] - gray[:, :, :, 7:-1:8]).abs().mean(dim=(1, 2, 3))
    )
    return torch.cat((head_summary, torch.stack(patch_summary + pixels_summary, dim=1)), dim=1).cpu()


def extract(model, dataset, device, args, path):
    if path.exists():
        return torch.load(path, weights_only=True)
    features = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for batch_index, pixels in enumerate(loader):
            pixels = pixels.to(device)
            outputs = model(
                pixel_values=pixels,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )
            features.append(multisignal_features(pixels, outputs))
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
    parser.add_argument("--corruption-seed", type=int, default=114000)
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
        else freeze_manifest(manifest_path, (*PREVIOUS_CONFIRMATIONS, EXPERIMENT113_MANIFEST))
    )
    print("Manifest", sha256(manifest_path), "images", manifest["samples"], flush=True)
    old_sketch = json.loads(OLD_SKETCH_MANIFEST.read_text())
    photo_train = ImageNetDataset(Path("Dataset"), 1000, 0)
    photo_validation = ImageNetDataset(Path("Dataset"), 500, 10000)

    np.float_ = np.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    groups = {
        "photo_train": ImageDataset(photo_train, "clean", args.severity, args.corruption_seed),
        "photo_validation": ImageDataset(photo_validation, "clean", args.severity, args.corruption_seed),
        "sketch_train": ImageDataset({"items": old_sketch["items"][:2000]}, "clean", args.severity, args.corruption_seed),
        "sketch_validation": ImageDataset({"items": old_sketch["items"][2000:]}, "clean", args.severity, args.corruption_seed),
        "clean": ImageDataset(manifest, "clean", args.severity, args.corruption_seed),
    }
    for name in get_corruption_names():
        groups[name] = ImageDataset(manifest, name, args.severity, args.corruption_seed)
    for name, dataset in groups.items():
        print("Extracting", name, flush=True)
        extract(model, dataset, device, args, output_dir / f"{name}_features.pt")

    def loaded(name):
        return torch.load(output_dir / f"{name}_features.pt", weights_only=True).numpy().astype(np.float64)

    photo = loaded("photo_train")
    sketch = loaded("sketch_train")
    clean_training = np.concatenate((photo, sketch))
    location = clean_training.mean(0)
    scale = clean_training.std(0).clip(1e-5)
    photo_model = LedoitWolf().fit((photo - location) / scale)
    sketch_model = LedoitWolf().fit((sketch - location) / scale)
    models = (photo_model, sketch_model)
    clean_validation = np.concatenate((loaded("photo_validation"), loaded("sketch_validation")))
    validation_scores = density_scores(clean_validation, location, scale, models)
    threshold = float(np.quantile(validation_scores, 0.95))
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
            "features": "signed per-block head means/dispersion, patch-token moments, image gradients/frequencies",
            "detector": "minimum of photo/sketch Ledoit-Wolf Mahalanobis distances",
            "threshold": "95th percentile on clean-only photo and sketch validation",
            "corruptions_used_for_fitting_or_threshold": False,
            "repair_applied": False,
            "benchmark_label": "ImageNet-C algorithms on hash-disjoint ImageNet-Sketch, not official ImageNet-C JPEGs",
        },
        "threshold": threshold,
        "results": results,
        "limitations": [
            "One deterministic detector, not three independently trained adapters.",
            "ImageNet-Sketch is a previously studied domain, though this image set excludes prior frozen manifests by SHA-256.",
            "No repair is applied; this tests detection only.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
