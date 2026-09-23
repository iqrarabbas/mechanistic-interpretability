import argparse
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
from scripts.experiment124_frozen_noise_subspace_sketch import (
    BLOCK,
    DATA_ROOT,
    RANK,
    SOURCE_ROOT,
    folded_delta,
    freeze_manifest,
    load_factors,
    sha256,
)


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results/sae/experiment125_frozen_noise_severity_sweep"


class FreshSketchSeverity(Dataset):
    def __init__(self, manifest, severity, noise_seed, max_samples=None):
        self.items = manifest["items"][:max_samples]
        self.severity = severity
        self.noise_seed = noise_seed
        self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL, local_files_only=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        image = Image.open(DATA_ROOT / row["relative_path"]).convert("RGB")
        if self.severity:
            image = apply_gaussian_noise(
                image, severity=self.severity, seed=self.noise_seed + index
            )
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, row["label"]


def evaluate(model, adapter, factors, loader, device, bootstrap, statistical_seed):
    methods = ("baseline", "full", "delta_pca", "hidden_pca", "random")
    outcomes = {method: [] for method in methods}
    with torch.inference_mode():
        for pixels, labels in tqdm(loader, desc="Frozen severity evaluation", leave=False):
            labels = labels.to(device)
            outputs = model(pixel_values=pixels.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            patches = hidden[:, 1:]
            outcomes["baseline"].extend(
                (outputs.logits.argmax(1) == labels).cpu().tolist()
            )
            full_hidden = torch.cat([hidden[:, :1], patches + adapter(patches)], dim=1)
            full_logits = downstream_from_layer(model, full_hidden, BLOCK - 1)
            outcomes["full"].extend((full_logits.argmax(1) == labels).cpu().tolist())
            for kind in ("delta_pca", "hidden_pca", "random"):
                candidate = torch.cat(
                    [hidden[:, :1], patches + folded_delta(patches, factors[kind])], dim=1
                )
                logits = downstream_from_layer(model, candidate, BLOCK - 1)
                outcomes[kind].extend((logits.argmax(1) == labels).cpu().tolist())
    outcomes = {name: np.asarray(values, dtype=bool) for name, values in outcomes.items()}
    results = {}
    for method_index, method in enumerate(methods[1:]):
        results[f"{method}_vs_baseline"] = paired_comparison(
            outcomes["baseline"], outcomes[method],
            statistical_seed + method_index, bootstrap,
        )
    for control_index, control in enumerate(("full", "hidden_pca", "random")):
        results[f"delta_pca_vs_{control}"] = paired_comparison(
            outcomes[control], outcomes["delta_pca"],
            statistical_seed + 50 + control_index, bootstrap,
        )
    return results, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    parser.add_argument("--images-per-class", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--noise-seed", type=int, default=125000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("Frozen confirmation is locked to seeds [0,1,2]")
    if args.severities != [0, 1, 2, 3, 4, 5]:
        raise ValueError("Frozen severity protocol is locked to [0,1,2,3,4,5]")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output_dir / "frozen_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    elif args.manifest is not None:
        manifest = json.loads(args.manifest.read_text())
        manifest_path.write_text(json.dumps(manifest, indent=2))
    else:
        manifest = freeze_manifest(manifest_path, args.images_per_class)
    print(
        "Frozen manifest", manifest_path, sha256(manifest_path),
        "samples", manifest["samples"], "eligible classes", manifest["eligible_classes"],
        flush=True,
    )
    progress_path = output_dir / "progress.json"
    outcome_path = output_dir / "paired_outcomes_partial.npz"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    outcomes = {}
    if outcome_path.exists():
        with np.load(outcome_path) as saved:
            outcomes = {name: saved[name] for name in saved.files}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    artifact_paths = {}
    for seed in args.seeds:
        adapter, adapter_path = load_adapter(seed, device)
        adapter.requires_grad_(False)
        factors, factor_paths = {}, {}
        for kind in ("delta_pca", "hidden_pca", "random"):
            factors[kind], path = load_factors(seed, kind, device)
            factor_paths[kind] = str(path)
        artifact_paths[str(seed)] = {"adapter": str(adapter_path), "factors": factor_paths}
        results.setdefault(str(seed), {})
        for severity in args.severities:
            key = "clean" if severity == 0 else f"noise_{severity}"
            if key in results[str(seed)]:
                print(f"Skipping completed seed {seed} {key}", flush=True)
                continue
            loader = DataLoader(
                FreshSketchSeverity(manifest, severity, args.noise_seed, args.max_samples),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            condition, arrays = evaluate(
                model, adapter, factors, loader, device, args.bootstrap,
                1250000 + seed * 1000 + severity * 100,
            )
            results[str(seed)][key] = condition
            for method, values in arrays.items():
                outcomes[f"seed{seed}_{key}_{method}_correct"] = values
            progress_path.write_text(json.dumps(results, indent=2))
            np.savez_compressed(outcome_path, **outcomes)
            print(f"Completed seed {seed} {key}", flush=True)
    aggregate = {}
    for severity in args.severities:
        key = "clean" if severity == 0 else f"noise_{severity}"
        aggregate[key] = {}
        for method in ("full", "delta_pca", "hidden_pca", "random"):
            gains = [
                results[str(seed)][key][f"{method}_vs_baseline"]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[key][method] = {
                "gains_by_seed": gains,
                "mean_gain": float(np.mean(gains)),
            }
        comparisons = {}
        for control in ("full", "hidden_pca", "random"):
            differences = [
                results[str(seed)][key][f"delta_pca_vs_{control}"]["accuracy_difference"]
                for seed in args.seeds
            ]
            comparisons[f"delta_pca_vs_{control}"] = {
                "differences_by_seed": differences,
                "mean_difference": float(np.mean(differences)),
            }
        aggregate[key]["comparisons"] = comparisons
    summary = {
        "configuration": vars(args) | {
            "manifest": str(args.manifest.resolve()) if args.manifest is not None else None,
            "model": BASE_MODEL,
            "rank": RANK,
            "source_experiment": str(SOURCE_ROOT),
            "manifest_sha256": sha256(manifest_path),
            "data_root": str(DATA_ROOT),
            "all_components_frozen": True,
            "selection_or_tuning_on_this_manifest": False,
            "clean_counterpart_used_at_inference": False,
            "benchmark_label": "second path-and-SHA-disjoint ImageNet-Sketch slice with online project Gaussian Noise severities; previously studied domain, not official ImageNet-C",
        },
        "manifest": {
            "samples": manifest["samples"],
            "eligible_classes": manifest["eligible_classes"],
            "excluded_manifest_count": manifest["excluded_manifest_count"],
        },
        "artifact_paths": artifact_paths,
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "The ImageNet-Sketch domain is previously studied; this manifest provides only new path-and-SHA-disjoint images.",
            "Noise is the project's online Gaussian implementation, not official pre-generated ImageNet-C.",
            "The same images are evaluated for all three adapter seeds; seed means are not independent dataset replications.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
