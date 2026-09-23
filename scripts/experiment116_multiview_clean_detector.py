import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image, ImageEnhance
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment107_fresh_sketch_defocus_confirmation import DATA_ROOT
from scripts.experiment111_leakage_free_head_anomaly_confirmation import freeze_manifest, sha256
from scripts.experiment113_all_corruption_head_detection import PREVIOUS_CONFIRMATIONS
from scripts.experiment114_multisignal_clean_detector import EXPERIMENT113_MANIFEST
from scripts.experiment115_consistency_clean_detector import EXPERIMENT114_MANIFEST


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment116_multiview_clean_detector"
OLD_SKETCH_MANIFEST = Path(
    "/media/dr-yougart/Iqrar/datasets/imagenet_sketch/frozen_subset_3_per_class_unique_manifest.json"
)
EXPERIMENT115_MANIFEST = (
    ACTIVE_ROOT / "results/sae/experiment115_consistency_clean_detector/"
    "full_level4_15corruptions_flipconsistency_v1/frozen_manifest.json"
)
BLOCKS = (3, 6, 9, 12)


def normalize(image):
    values = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float()
    return values.div_(127.5).sub_(1.0)


def jpeg_view(image):
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def make_views(image):
    resize = image.resize((180, 180), Image.Resampling.BILINEAR).resize(
        (224, 224), Image.Resampling.BILINEAR
    )
    brightness = ImageEnhance.Brightness(image).enhance(1.05)
    return image, resize, brightness, jpeg_view(image)


class MultiViewDataset(Dataset):
    def __init__(self, source, name, severity, seed):
        self.source = source
        self.items = source["items"] if isinstance(source, dict) else None
        self.name = name
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.items) if self.items is not None else len(self.source)

    def __getitem__(self, index):
        if self.items is not None:
            path = DATA_ROOT / self.items[index]["relative_path"]
        else:
            path = self.source.image_paths[index]
        image = Image.open(path).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        if self.name != "clean":
            np.random.seed(self.seed + index)
            image = Image.fromarray(
                corrupt(np.asarray(image, dtype=np.uint8), corruption_name=self.name, severity=self.severity)
            )
        return torch.stack([normalize(view) for view in make_views(image)])


def distribution_features(output):
    probabilities = output.logits.softmax(1)
    top = probabilities.topk(2, dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
    values = [top[:, 0], top[:, 0] - top[:, 1], entropy]
    for block in BLOCKS:
        hidden = output.hidden_states[block].float()
        values.extend((hidden[:, 0].norm(dim=1), hidden[:, 1:].norm(dim=2).mean(1)))
    return probabilities, torch.stack(values, dim=1)


def multiview_features(outputs):
    base_probabilities, base_distribution = distribution_features(outputs[0])
    features = [base_distribution]
    for output in outputs[1:]:
        probabilities, distribution = distribution_features(output)
        mean_probability = (base_probabilities + probabilities) / 2
        js = (
            functional.kl_div(mean_probability.log(), base_probabilities, reduction="none").sum(1)
            + functional.kl_div(mean_probability.log(), probabilities, reduction="none").sum(1)
        ) / 2
        consistency = [
            js,
            (base_probabilities.argmax(1) != probabilities.argmax(1)).float(),
        ]
        for block in BLOCKS:
            base_hidden = outputs[0].hidden_states[block].float()
            view_hidden = output.hidden_states[block].float()
            for token_slice in (slice(0, 1), slice(1, None)):
                base_token = base_hidden[:, token_slice].mean(1)
                view_token = view_hidden[:, token_slice].mean(1)
                consistency.extend((
                    functional.cosine_similarity(base_token, view_token),
                    (base_token - view_token).norm(dim=1) / base_token.norm(dim=1).clamp_min(1e-6),
                ))
        features.extend((distribution, torch.stack(consistency, dim=1)))
    return torch.cat(features, dim=1).cpu()


def extract(model, dataset, device, args, path):
    if path.exists():
        return torch.load(path, weights_only=True)
    features = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for batch_index, views in enumerate(loader):
            views = views.to(device)
            outputs = [
                model(pixel_values=views[:, index], output_hidden_states=True, return_dict=True)
                for index in range(views.shape[1])
            ]
            features.append(multiview_features(outputs))
            if batch_index % 50 == 0:
                print(f"  batches {batch_index}/{len(loader)}", flush=True)
    values = torch.cat(features)
    torch.save(values, path)
    return values


def scores(values, location, scale, models):
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
    parser.add_argument("--corruption-seed", type=int, default=116000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output_dir / "frozen_manifest.json"
    excluded = (*PREVIOUS_CONFIRMATIONS, EXPERIMENT113_MANIFEST, EXPERIMENT114_MANIFEST, EXPERIMENT115_MANIFEST)
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else freeze_manifest(manifest_path, excluded)
    print("Manifest", sha256(manifest_path), "images", manifest["samples"], flush=True)

    old_sketch = json.loads(OLD_SKETCH_MANIFEST.read_text())
    photo_train = ImageNetDataset(Path("Dataset"), 1000, 0)
    photo_validation = ImageNetDataset(Path("Dataset"), 500, 10000)
    groups = {
        "photo_train": MultiViewDataset(photo_train, "clean", args.severity, args.corruption_seed),
        "photo_validation": MultiViewDataset(photo_validation, "clean", args.severity, args.corruption_seed),
        "sketch_train": MultiViewDataset({"items": old_sketch["items"][:2000]}, "clean", args.severity, args.corruption_seed),
        "sketch_validation": MultiViewDataset({"items": old_sketch["items"][2000:]}, "clean", args.severity, args.corruption_seed),
        "clean": MultiViewDataset(manifest, "clean", args.severity, args.corruption_seed),
    }
    for name in get_corruption_names():
        groups[name] = MultiViewDataset(manifest, name, args.severity, args.corruption_seed)

    np.float_ = np.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL, local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    for name, dataset in groups.items():
        print("Extracting", name, flush=True)
        extract(model, dataset, device, args, output_dir / f"{name}_features.pt")

    def loaded(name):
        return torch.load(output_dir / f"{name}_features.pt", weights_only=True).numpy().astype(np.float64)

    photo, sketch = loaded("photo_train"), loaded("sketch_train")
    training = np.concatenate((photo, sketch))
    location, scale = training.mean(0), training.std(0).clip(1e-5)
    models = (
        LedoitWolf().fit((photo - location) / scale),
        LedoitWolf().fit((sketch - location) / scale),
    )
    validation = np.concatenate((loaded("photo_validation"), loaded("sketch_validation")))
    threshold = float(np.quantile(scores(validation, location, scale, models), 0.95))
    clean_scores = scores(loaded("clean"), location, scale, models)
    results = {"clean": {"false_positive_rate": float((clean_scores > threshold).mean())}}
    for name in get_corruption_names():
        corruption_scores = scores(loaded(name), location, scale, models)
        results[name] = {
            "recall": float((corruption_scores > threshold).mean()),
            "auroc": float(roc_auc_score(
                np.r_[np.zeros(len(clean_scores)), np.ones(len(corruption_scores))],
                np.r_[clean_scores, corruption_scores],
            )),
        }
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "manifest_sha256": sha256(manifest_path),
            "feature_dimension": len(location),
            "views": ["original", "resize_180_roundtrip", "brightness_1.05", "jpeg_quality_85"],
            "detector": "clean-only photo/sketch multiview consistency Mahalanobis ensemble",
            "threshold": "95th percentile on clean-only validation",
            "corruptions_used_for_fit_or_threshold": False,
            "repair_applied": False,
        },
        "threshold": threshold,
        "results": results,
        "limitations": [
            "Four ViT passes per image make this substantially more expensive than Experiment 114.",
            "No detector can guarantee recognition of every possible attack.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
