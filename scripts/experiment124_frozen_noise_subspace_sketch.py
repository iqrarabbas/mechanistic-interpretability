import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification

from corruption.gaussian_noise import apply_gaussian_noise
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment72_downstream_circuit_necessity import load_adapter


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results/sae/experiment124_frozen_noise_subspace_sketch"
DATA_ROOT = Path("/media/dr-yougart/Iqrar/datasets/imagenet_sketch/extracted/sketch")
SOURCE_ROOT = ROOT / "results/sae/experiment123_noise_adapter_delta_subspace/full_3seed_ranks4_256_v1"
BLOCK = 6
RANK = 128


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prior_manifest_paths():
    roots = [
        Path("/media/dr-yougart/Iqrar/datasets/imagenet_sketch"),
        Path("/media/dr-yougart/Iqrar/vit_mi/results/sae"),
        ROOT / "results/sae",
    ]
    paths = set()
    for root in roots:
        if root.exists():
            paths.update(root.glob("**/*manifest*.json"))
    return sorted(paths)


def freeze_manifest(path, images_per_class):
    excluded_paths, excluded_hashes, source_manifests = set(), set(), []
    for manifest_path in prior_manifest_paths():
        try:
            document = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        items = document.get("items")
        if not isinstance(items, list):
            continue
        summary_path = manifest_path.parent / "summary.json"
        evaluated_items = items
        if summary_path.exists():
            try:
                maximum = json.loads(summary_path.read_text()).get("configuration", {}).get("max_samples")
                if maximum is not None:
                    evaluated_items = items[:maximum]
            except (OSError, json.JSONDecodeError):
                pass
        source_manifests.append({"path": str(manifest_path), "sha256": sha256(manifest_path)})
        for item in evaluated_items:
            if "relative_path" in item:
                excluded_paths.add(item["relative_path"])
            if "sha256" in item:
                excluded_hashes.add(item["sha256"])
    selected_hashes, items = set(), []
    classes = sorted(folder for folder in DATA_ROOT.iterdir() if folder.is_dir())
    if len(classes) != 1000:
        raise ValueError(f"Expected 1000 ImageNet-Sketch classes, found {len(classes)}")
    for label, folder in enumerate(classes):
        selected = 0
        for image_path in sorted(folder.glob("*.JPEG")):
            relative_path = str(image_path.relative_to(DATA_ROOT))
            if relative_path in excluded_paths:
                continue
            image_hash = sha256(image_path)
            if image_hash in excluded_hashes or image_hash in selected_hashes:
                continue
            items.append({"relative_path": relative_path, "label": label, "sha256": image_hash})
            selected_hashes.add(image_hash)
            selected += 1
            if selected == images_per_class:
                break
    eligible_classes = len({item["label"] for item in items})
    if not items:
        raise ValueError("No path-and-SHA-disjoint ImageNet-Sketch images remain")
    document = {
        "selection_rule": "first path-and-SHA-disjoint images per sorted class, excluding every discoverable prior Sketch manifest; frozen before inference",
        "source_root": str(DATA_ROOT),
        "images_per_class": images_per_class,
        "classes": len(classes),
        "eligible_classes": eligible_classes,
        "samples": len(items),
        "excluded_manifest_count": len(source_manifests),
        "excluded_manifests": source_manifests,
        "items": items,
    }
    path.write_text(json.dumps(document, indent=2))
    return document


class FreshSketchNoise(Dataset):
    def __init__(self, manifest, noise_seed, max_samples=None):
        self.items = manifest["items"][:max_samples]
        self.noise_seed = noise_seed
        self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL, local_files_only=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        image = Image.open(DATA_ROOT / row["relative_path"]).convert("RGB")
        noise = apply_gaussian_noise(image, severity=4, seed=self.noise_seed + index)
        clean_pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        noise_pixels = self.processor(images=noise, return_tensors="pt")["pixel_values"].squeeze(0)
        return clean_pixels, noise_pixels, row["label"]


def folded_delta(patches, factors):
    positions = torch.arange(patches.shape[1], device=patches.device)
    compressed = (
        patches @ factors["input_to_rank"]
        + factors["bias_rank"]
        + factors["position_rank"][positions][None]
    )
    return compressed @ factors["rank_to_output"]


def load_factors(seed, kind, device):
    path = SOURCE_ROOT / f"seed_{seed}" / f"{kind}_folded_factors.pt"
    all_ranks = torch.load(path, map_location=device, weights_only=True)
    factors = all_ranks[RANK]
    return {name: value.to(device) for name, value in factors.items()}, path


def evaluate(model, adapter, factor_groups, loader, device, bootstrap, statistical_seed):
    methods = ("baseline", "full", "delta_pca", "hidden_pca", "random")
    outcomes = {condition: {method: [] for method in methods} for condition in ("clean", "noise")}
    with torch.inference_mode():
        for clean, noise, labels in tqdm(loader, desc="Frozen Sketch confirmation"):
            labels = labels.to(device)
            batch = labels.shape[0]
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK].split(batch)
            for condition, logits, hidden in (
                ("clean", clean_logits, clean_hidden), ("noise", noise_logits, noise_hidden)
            ):
                patches = hidden[:, 1:]
                outcomes[condition]["baseline"].extend(
                    (logits.argmax(1) == labels).cpu().tolist()
                )
                full_hidden = torch.cat([hidden[:, :1], patches + adapter(patches)], dim=1)
                full_logits = downstream_from_layer(model, full_hidden, BLOCK - 1)
                outcomes[condition]["full"].extend(
                    (full_logits.argmax(1) == labels).cpu().tolist()
                )
                for kind, factors in factor_groups.items():
                    candidate = torch.cat(
                        [hidden[:, :1], patches + folded_delta(patches, factors)], dim=1
                    )
                    candidate_logits = downstream_from_layer(model, candidate, BLOCK - 1)
                    outcomes[condition][kind].extend(
                        (candidate_logits.argmax(1) == labels).cpu().tolist()
                    )
    outcomes = {
        condition: {method: np.asarray(values, dtype=bool) for method, values in methods.items()}
        for condition, methods in outcomes.items()
    }
    results = {}
    for condition_index, condition in enumerate(("clean", "noise")):
        results[condition] = {}
        baseline = outcomes[condition]["baseline"]
        for method_index, method in enumerate(methods[1:]):
            results[condition][f"{method}_vs_baseline"] = paired_comparison(
                baseline, outcomes[condition][method],
                statistical_seed + condition_index * 100 + method_index, bootstrap,
            )
        results[condition]["delta_pca_vs_full"] = paired_comparison(
            outcomes[condition]["full"], outcomes[condition]["delta_pca"],
            statistical_seed + condition_index * 100 + 50, bootstrap,
        )
        for control_index, control in enumerate(("hidden_pca", "random")):
            results[condition][f"delta_pca_vs_{control}"] = paired_comparison(
                outcomes[condition][control], outcomes[condition]["delta_pca"],
                statistical_seed + condition_index * 100 + 60 + control_index, bootstrap,
            )
    return results, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--images-per-class", type=int, default=2)
    parser.add_argument("--noise-seed", type=int, default=124000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("Independent confirmation is locked to adapter seeds [0,1,2]")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output_dir / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else freeze_manifest(
        manifest_path, args.images_per_class
    )
    if manifest["images_per_class"] != args.images_per_class:
        raise ValueError("Cannot change images-per-class after freezing the manifest")
    print("Frozen manifest", manifest_path, sha256(manifest_path), flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    dataset = FreshSketchNoise(manifest, args.noise_seed, args.max_samples)
    all_results, all_outcomes, paths = {}, {}, {}
    for seed in args.seeds:
        adapter, adapter_path = load_adapter(seed, device)
        adapter.requires_grad_(False)
        factor_groups, factor_paths = {}, {}
        for kind in ("delta_pca", "hidden_pca", "random"):
            factor_groups[kind], factor_path = load_factors(seed, kind, device)
            factor_paths[kind] = str(factor_path)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )
        result, outcomes = evaluate(
            model, adapter, factor_groups, loader, device,
            args.bootstrap, 1240000 + seed * 1000,
        )
        all_results[str(seed)] = result
        paths[str(seed)] = {"adapter": str(adapter_path), "factors": factor_paths}
        for condition, methods in outcomes.items():
            for method, values in methods.items():
                all_outcomes[f"seed{seed}_{condition}_{method}_correct"] = values
        print(f"Completed seed {seed}", flush=True)
    pooled = {}
    for condition_index, condition in enumerate(("clean", "noise")):
        pooled[condition] = {}
        arrays = {
            method: np.concatenate([
                all_outcomes[f"seed{seed}_{condition}_{method}_correct"] for seed in args.seeds
            ])
            for method in ("baseline", "full", "delta_pca", "hidden_pca", "random")
        }
        for method_index, method in enumerate(("full", "delta_pca", "hidden_pca", "random")):
            pooled[condition][f"{method}_vs_baseline"] = paired_comparison(
                arrays["baseline"], arrays[method],
                1249000 + condition_index * 100 + method_index, args.bootstrap,
            )
        for control_index, control in enumerate(("full", "hidden_pca", "random")):
            pooled[condition][f"delta_pca_vs_{control}"] = paired_comparison(
                arrays[control], arrays["delta_pca"],
                1249000 + condition_index * 100 + 50 + control_index, args.bootstrap,
            )
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "rank": RANK,
            "source_experiment": str(SOURCE_ROOT),
            "data_root": str(DATA_ROOT),
            "manifest_sha256": sha256(manifest_path),
            "all_components_frozen": True,
            "rank_scale_threshold_tuning_on_sketch": False,
            "clean_counterpart_used_for_noise_repair": False,
            "benchmark_label": "new path-and-SHA-disjoint ImageNet-Sketch images with online project Gaussian Noise-4; previously studied domain, not official ImageNet-C",
        },
        "artifact_paths": paths,
        "results": all_results,
        "pooled_disjoint_images": pooled,
        "limitations": [
            "ImageNet-Sketch is a previously studied domain; only the individual images are newly excluded by path and SHA-256.",
            "The project Gaussian Noise-4 transform is applied online and is not an official pre-generated ImageNet-C benchmark.",
            "The three adapter seeds evaluate the same images, so pooled seed-image outcomes are descriptive and not three independent image datasets.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **all_outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(pooled, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
