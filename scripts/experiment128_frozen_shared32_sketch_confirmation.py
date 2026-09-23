import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment68_unseen_online_corruptions import corrupt
from scripts.experiment72_downstream_circuit_necessity import load_adapter
from scripts.experiment123_noise_adapter_delta_subspace import folded_delta
from scripts.experiment124_frozen_noise_subspace_sketch import DATA_ROOT, freeze_manifest, sha256


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results/sae/experiment128_frozen_shared32_sketch"
SOURCE_ROOT = ROOT / "results/sae/experiment127_shared_ood_repair_subspace/full_3seed_shared32_familyrank128_v1"
BLOCK = 6
RANK = 32
CONDITIONS = ("clean", "impulse_noise", "pixelate", "jpeg", "brightness")


class FreshSketchFamily(Dataset):
    def __init__(self, manifest, family, corruption_seed, max_samples=None):
        self.items = manifest["items"][:max_samples]
        self.family = family
        self.corruption_seed = corruption_seed
        self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL, local_files_only=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        image = Image.open(DATA_ROOT / row["relative_path"]).convert("RGB")
        if self.family != "clean":
            image = corrupt(
                image, self.family, 4, self.corruption_seed + index
            )
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, row["label"]


def load_factor_groups(seed, device):
    path = SOURCE_ROOT / f"seed_{seed}" / "folded_factors.pt"
    groups = torch.load(path, map_location=device, weights_only=True)
    selected = {
        name: {key: value.to(device) for key, value in groups[name].items()}
        for name in ("shared", "hidden", "random")
    }
    return selected, path


def evaluate(model, adapter, factors, loader, device, bootstrap, statistical_seed):
    methods = ("baseline", "full", "shared", "hidden", "random")
    outcomes = {method: [] for method in methods}
    with torch.inference_mode():
        for pixels, labels in tqdm(loader, desc="Frozen shared-32 confirmation", leave=False):
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
            for name, current_factors in factors.items():
                candidate = torch.cat(
                    [hidden[:, :1], patches + folded_delta(patches, current_factors)], dim=1
                )
                logits = downstream_from_layer(model, candidate, BLOCK - 1)
                outcomes[name].extend((logits.argmax(1) == labels).cpu().tolist())
    outcomes = {name: np.asarray(values, dtype=bool) for name, values in outcomes.items()}
    results = {"baseline_accuracy": float(outcomes["baseline"].mean()), "methods": {}}
    for method_index, method in enumerate(methods[1:]):
        results["methods"][method] = paired_comparison(
            outcomes["baseline"], outcomes[method],
            statistical_seed + method_index, bootstrap,
        )
    for control_index, control in enumerate(("full", "hidden", "random")):
        results[f"shared_vs_{control}"] = paired_comparison(
            outcomes[control], outcomes["shared"],
            statistical_seed + 50 + control_index, bootstrap,
        )
    return results, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--images-per-class", type=int, default=1)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--corruption-seed", type=int, default=128000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("Frozen confirmation is locked to seeds [0,1,2]")
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
        "Frozen manifest", sha256(manifest_path),
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
        BASE_MODEL, local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    artifact_paths = {}
    for seed in args.seeds:
        adapter, adapter_path = load_adapter(seed, device)
        adapter.requires_grad_(False)
        factors, factor_path = load_factor_groups(seed, device)
        artifact_paths[str(seed)] = {
            "adapter": str(adapter_path), "rank32_factors": str(factor_path)
        }
        results.setdefault(str(seed), {})
        for condition_index, condition in enumerate(CONDITIONS):
            if condition in results[str(seed)]:
                print(f"Skipping completed seed {seed} {condition}", flush=True)
                continue
            loader = DataLoader(
                FreshSketchFamily(
                    manifest, condition, args.corruption_seed, args.max_samples
                ),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True,
            )
            result, arrays = evaluate(
                model, adapter, factors, loader, device, args.bootstrap,
                1280000 + seed * 1000 + condition_index * 100,
            )
            results[str(seed)][condition] = result
            for method, values in arrays.items():
                outcomes[f"seed{seed}_{condition}_{method}_correct"] = values
            progress_path.write_text(json.dumps(results, indent=2))
            np.savez_compressed(outcome_path, **outcomes)
            print(f"Completed seed {seed} {condition}", flush=True)
    aggregate = {}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for method in ("full", "shared", "hidden", "random"):
            gains = [
                results[str(seed)][condition]["methods"][method]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[condition][method] = {
                "gains_by_seed": gains,
                "mean_gain": float(np.mean(gains)),
            }
        aggregate[condition]["shared_comparisons"] = {
            f"vs_{control}": {
                "differences_by_seed": [
                    results[str(seed)][condition][f"shared_vs_{control}"]["accuracy_difference"]
                    for seed in args.seeds
                ]
            }
            for control in ("full", "hidden", "random")
        }
    summary = {
        "configuration": vars(args) | {
            "manifest": str(args.manifest.resolve()) if args.manifest is not None else None,
            "model": BASE_MODEL,
            "rank": RANK,
            "source_experiment": str(SOURCE_ROOT),
            "conditions": CONDITIONS,
            "severity": 4,
            "manifest_sha256": sha256(manifest_path),
            "all_components_frozen": True,
            "selection_or_tuning_on_this_manifest": False,
            "clean_counterpart_used_at_inference": False,
            "benchmark_label": "new path-and-SHA-disjoint ImageNet-Sketch images with online approximate corruptions; previously studied domain, not official ImageNet-C",
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
            "ImageNet-Sketch is a previously studied domain; only image paths and hashes are newly excluded.",
            "The corruptions are deterministic online approximations, not official ImageNet-C files.",
            "All three seed-specific maps evaluate the same images; seed outcomes are not independent image datasets.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
