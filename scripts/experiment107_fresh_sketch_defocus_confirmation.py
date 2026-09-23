import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from imagecorruptions import corrupt
from PIL import Image
from torch.utils.data import Dataset
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment101_routed_batchtopk_sae_repairs import (
    BLOCK,
    THRESHOLD_SUMMARY,
    load_repair,
    load_router,
)
from scripts.experiment106_imagecorruptions_blur_benchmark import evaluate


DATA_ROOT = Path("/media/dr-yougart/Iqrar/datasets/imagenet_sketch/extracted/sketch")
PRIOR_MANIFEST = Path(
    "/media/dr-yougart/Iqrar/datasets/imagenet_sketch/"
    "frozen_subset_3_per_class_unique_manifest.json"
)
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment107_fresh_sketch_defocus_confirmation"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_manifest(path, images_per_class):
    prior = json.loads(PRIOR_MANIFEST.read_text())
    used_paths = {row["relative_path"] for row in prior["items"]}
    used_hashes = {row["sha256"] for row in prior["items"]}
    selected_hashes = set()
    items = []
    classes = sorted(folder for folder in DATA_ROOT.iterdir() if folder.is_dir())
    if len(classes) != 1000:
        raise ValueError(f"Expected 1000 classes, found {len(classes)}")
    for label, folder in enumerate(classes):
        selected = 0
        for image_path in sorted(folder.glob("*.JPEG")):
            relative_path = str(image_path.relative_to(DATA_ROOT))
            if relative_path in used_paths:
                continue
            image_hash = sha256(image_path)
            if image_hash in used_hashes or image_hash in selected_hashes:
                continue
            items.append({
                "relative_path": relative_path,
                "label": label,
                "sha256": image_hash,
            })
            selected_hashes.add(image_hash)
            selected += 1
            if selected == images_per_class:
                break
        if selected != images_per_class:
            raise ValueError(f"Only {selected} eligible images in {folder.name}")
    manifest = {
        "selection_rule": "first SHA-unique images per class, excluding the prior frozen Sketch manifest; fixed before inference",
        "source_root": str(DATA_ROOT),
        "prior_manifest_sha256": sha256(PRIOR_MANIFEST),
        "images_per_class": images_per_class,
        "classes": len(classes),
        "samples": len(items),
        "items": items,
    }
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


class SketchDefocusDataset(Dataset):
    def __init__(self, manifest, severity, corruption_seed):
        self.items = manifest["items"]
        self.severity = severity
        self.corruption_seed = corruption_seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        image = Image.open(DATA_ROOT / row["relative_path"]).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.uint8)
        if self.severity:
            np.random.seed(self.corruption_seed + index)
            values = corrupt(values, corruption_name="defocus_blur", severity=self.severity)
        pixels = torch.from_numpy(np.asarray(values).copy()).permute(2, 0, 1).float()
        pixels = pixels.div_(255.0).sub_(0.5).div_(0.5)
        return pixels, row["label"]


def main():
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--images-per-class", type=int, default=3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[0, 3, 4, 5])
    parser.add_argument("--corruption-seed", type=int, default=107000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output_dir / "frozen_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest["images_per_class"] != args.images_per_class:
            raise ValueError("Existing manifest uses a different images-per-class value")
    else:
        manifest = freeze_manifest(manifest_path, args.images_per_class)
    print("Frozen manifest:", manifest_path, "SHA256:", sha256(manifest_path), flush=True)

    progress_path = output_dir / "progress.json"
    outcome_path = output_dir / "paired_outcomes_partial.npz"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    outcomes = {}
    if outcome_path.exists():
        with np.load(outcome_path) as stored:
            outcomes = {name: stored[name] for name in stored.files}

    thresholds = json.loads(THRESHOLD_SUMMARY.read_text())["results"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    for seed in args.seeds:
        router, mean, std = load_router(seed, device)
        repairs = {family: load_repair(seed, family, device) for family in ("noise", "blur")}
        threshold = thresholds[str(seed)]["selected_clean_probability_threshold"]
        results.setdefault(str(seed), {})
        for severity in args.severities:
            key = "clean" if severity == 0 else f"defocus_blur_{severity}"
            if key in results[str(seed)]:
                print(f"Seed {seed} {key}: already complete", flush=True)
                continue
            print(f"Seed {seed} {key}", flush=True)
            dataset = SketchDefocusDataset(manifest, severity, args.corruption_seed)
            result, arrays = evaluate(
                model, router, mean, std, repairs, threshold, dataset, device,
                args, 1070000 + seed * 100 + severity * 10,
            )
            results[str(seed)][key] = result
            for method, values in arrays.items():
                outcomes[f"seed{seed}_{key}_{method}"] = values
            progress_path.write_text(json.dumps(results, indent=2))
            np.savez_compressed(outcome_path, **outcomes)
        del router, repairs
        torch.cuda.empty_cache()

    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "manifest_sha256": sha256(manifest_path),
            "prior_manifest_sha256": sha256(PRIOR_MANIFEST),
            "data_root": str(DATA_ROOT),
            "benchmark_label": "fresh disjoint ImageNet-Sketch images; ImageNet-C-compatible defocus transform, not official ImageNet-C JPEGs",
            "all_components_frozen": True,
            "clean_counterpart_used_at_inference": False,
        },
        "results": results,
        "limitations": [
            "The images are disjoint from the earlier 3000-image Sketch manifest by path and SHA-256, but the Sketch dataset/domain has been studied earlier.",
            "This is independent image-level confirmation, not a pristine new-dataset or official ImageNet-C benchmark.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
