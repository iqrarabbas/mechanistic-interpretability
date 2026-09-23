import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment68_unseen_online_corruptions import (
    BLOCK,
    CORRUPTIONS,
    TABLES,
    corrupt,
    evaluate,
    load_group,
)


ROOT = Path(__file__).parent.parent
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment69_imagenet_sketch_generalization"
DEFAULT_DATA_ROOT = Path(
    "/media/dr-yougart/Iqrar/datasets/imagenet_sketch/extracted/sketch"
)
DEFAULT_MANIFEST = Path(
    "/media/dr-yougart/Iqrar/datasets/imagenet_sketch/"
    "frozen_subset_3_per_class_unique_manifest.json"
)


class ManifestDataset(Dataset):
    def __init__(self, data_root, manifest, corruption_name, severity, seed):
        self.data_root = Path(data_root)
        self.items = manifest["items"]
        self.corruption_name = corruption_name
        self.severity = severity
        self.seed = seed
        self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(self.data_root / item["relative_path"]).convert("RGB")
        if self.corruption_name is not None:
            image = corrupt(
                image,
                self.corruption_name,
                self.severity,
                self.seed + index,
            )
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"]
        return pixels.squeeze(0), item["label"]


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=2068)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest = json.loads(args.manifest.read_text())
    if manifest["samples"] != 3000 or manifest["classes"] != 1000:
        raise ValueError("The locked protocol requires 3,000 images and 1,000 classes")
    if len({item["sha256"] for item in manifest["items"]}) != 3000:
        raise ValueError("The locked manifest contains duplicate image hashes")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    groups = {
        "mixed": load_group(
            ACTIVE_ROOT
            / "results/sae/experiment67_mixed_noise_blur_block6/"
            "full_3seed_noise_blur_identity0p2_v1",
            "mixed_identity0p2.pt",
            device,
        ),
        "noise_only": load_group(
            ACTIVE_ROOT
            / "results/sae/experiment50_block6_clean_preservation/"
            "full_3seed_identity_sweep_v1",
            "identity_0p2.pt",
            device,
        ),
    }

    results = {}
    outcomes = {}
    conditions = [("clean", None, 0)] + [
        (f"{name}_{severity}", name, severity)
        for name in CORRUPTIONS
        for severity in range(1, 6)
    ]
    for condition_index, (key, name, severity) in enumerate(conditions):
        condition_path = output_dir / f"{key}.json"
        if args.resume and condition_path.exists():
            results[key] = json.loads(condition_path.read_text())
            print(f"Resume: keeping completed condition {key}")
            continue
        dataset = ManifestDataset(
            args.data_root, manifest, name, severity, args.corruption_seed
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        results[key], arrays = evaluate(
            model,
            groups,
            loader,
            device,
            args,
            args.corruption_seed + condition_index * 20,
        )
        for method, values in arrays.items():
            outcomes[f"{key}__{method}"] = values
        condition_path.write_text(json.dumps(results[key], indent=2))

    aggregate = {}
    corruption_keys = [key for key, name, _ in conditions if name is not None]
    for group in groups:
        gains = [
            results[key]["methods"][f"{group}_seed{seed}"]["accuracy_difference"]
            for key in corruption_keys
            for seed in groups[group]
        ]
        aggregate[group] = {
            "mean_gain_all_corruptions_severities": float(np.mean(gains)),
            "per_seed_mean_gain": [
                float(
                    np.mean(
                        [
                            results[key]["methods"][f"{group}_seed{seed}"][
                                "accuracy_difference"
                            ]
                            for key in corruption_keys
                        ]
                    )
                )
                for seed in groups[group]
            ],
        }

    summary = {
        "configuration": vars(args)
        | {
            "data_root": str(args.data_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": file_sha256(args.manifest),
            "samples": manifest["samples"],
            "classes": manifest["classes"],
            "corruptions": CORRUPTIONS,
            "severity_tables": TABLES,
            "model": BASE_MODEL,
            "block": BLOCK,
            "vit_frozen": True,
            "adapters_frozen": True,
            "training_or_tuning": False,
            "dataset_status": "previously untouched before this frozen evaluation",
            "benchmark_label": "ImageNet-Sketch cross-domain corruption generalization; not ImageNet-C",
        },
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "ImageNet-Sketch is a domain-shift benchmark, not standard ImageNet validation.",
            "The online corruptions approximate common families and are not official ImageNet-C.",
            "The frozen evaluation must not be used to tune the adapters or protocol.",
        ],
    }
    outcome_filename = (
        "paired_outcomes_resumed_conditions.npz" if args.resume else "paired_outcomes.npz"
    )
    np.savez_compressed(output_dir / outcome_filename, **outcomes)
    summary["configuration"]["paired_outcome_coverage"] = sorted(
        {key.split("__", 1)[0] for key in outcomes}
    )
    summary["configuration"]["paired_outcome_limitation"] = (
        "Conditions completed before interruption retain paired statistics in their JSON "
        "files but lack saved image-level outcome arrays."
        if args.resume
        else None
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
