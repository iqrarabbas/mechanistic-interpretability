import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from corruption.gaussian_blur import apply_gaussian_blur
from corruption.gaussian_noise import apply_gaussian_noise
from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment68_unseen_online_corruptions import corrupt
from scripts.experiment72_downstream_circuit_necessity import load_adapter
from scripts.experiment123_noise_adapter_delta_subspace import (
    folded_delta,
    folded_factors,
    random_basis,
    top_basis,
)


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results/sae/experiment127_shared_ood_repair_subspace"
BLOCK = 6
WIDTH = 768
DISCOVERY_FAMILIES = ("gaussian_noise", "gaussian_blur", "shot_noise", "defocus", "contrast")
HELDOUT_FAMILIES = ("impulse_noise", "pixelate", "jpeg", "brightness")


class FamilyDataset(ImageNetDataset):
    def __init__(self, *args, family=None, severity=4, seed=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.family = family
        self.severity = severity
        self.seed = seed

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index]).convert("RGB")
        if self.family == "gaussian_noise":
            image = apply_gaussian_noise(
                image, severity=self.severity, seed=self.seed + self.start_index + index
            )
        elif self.family == "gaussian_blur":
            image = apply_gaussian_blur(image, severity=self.severity)
        elif self.family is not None:
            image = corrupt(
                image, self.family, self.severity, self.seed + self.start_index + index
            )
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, self.labels[index]


def make_loader(family, split, seed, samples, batch_size, workers):
    count = min(samples, split["end"] - split["start"])
    dataset = FamilyDataset(
        ROOT / "Dataset", max_samples=count, start_index=split["start"],
        family=family, severity=4, seed=seed,
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
    )


def accumulate(model, adapter, loader, device):
    delta_second = torch.zeros(WIDTH, WIDTH, dtype=torch.float64, device=device)
    hidden_sum = torch.zeros(WIDTH, dtype=torch.float64, device=device)
    hidden_second = torch.zeros(WIDTH, WIDTH, dtype=torch.float64, device=device)
    count = 0
    with torch.inference_mode():
        for pixels, _ in tqdm(loader, desc="Family subspace discovery", leave=False):
            hidden = model(
                pixel_values=pixels.to(device), output_hidden_states=True
            ).hidden_states[BLOCK][:, 1:]
            delta = adapter(hidden)
            flat_hidden = hidden.flatten(0, 1).double()
            flat_delta = delta.flatten(0, 1).double()
            delta_second.addmm_(flat_delta.T, flat_delta)
            hidden_sum += flat_hidden.sum(0)
            hidden_second.addmm_(flat_hidden.T, flat_hidden)
            count += flat_hidden.shape[0]
    mean = hidden_sum / count
    hidden_covariance = hidden_second / count - torch.outer(mean, mean)
    return delta_second / count, hidden_covariance, count


def discover_bases(model, adapter, split, seed, args, device):
    family_bases, family_values = {}, {}
    pooled_delta = torch.zeros(WIDTH, WIDTH, dtype=torch.float64, device=device)
    pooled_hidden = torch.zeros(WIDTH, WIDTH, dtype=torch.float64, device=device)
    counts = {}
    for family in DISCOVERY_FAMILIES:
        delta_second, hidden_covariance, count = accumulate(
            model, adapter,
            make_loader(
                family, split, seed, args.discovery_samples,
                args.batch_size, args.workers,
            ),
            device,
        )
        basis, values = top_basis(delta_second, args.family_rank)
        family_bases[family] = basis
        family_values[family] = values
        pooled_delta += delta_second
        pooled_hidden += hidden_covariance
        counts[family] = count
    shared_projector = sum(
        basis @ basis.T for basis in family_bases.values()
    ) / len(family_bases)
    shared_basis, shared_values = top_basis(shared_projector.double(), args.rank)
    pooled_basis, pooled_values = top_basis(pooled_delta, args.rank)
    hidden_basis, hidden_values = top_basis(pooled_hidden, args.rank)
    bases = {
        "shared": shared_basis,
        "pooled": pooled_basis,
        "hidden": hidden_basis,
        "random": random_basis(127000 + seed, args.rank, device),
        "noise_specific": family_bases["gaussian_noise"][:, :args.rank],
        "blur_specific": family_bases["gaussian_blur"][:, :args.rank],
    }
    diagnostics = {
        "patch_counts": counts,
        "sharedness_eigenvalues": shared_values[: args.rank].cpu().tolist(),
        "pooled_delta_eigenvalues": pooled_values[: args.rank].cpu().tolist(),
        "hidden_eigenvalues": hidden_values[: args.rank].cpu().tolist(),
        "family_delta_energy_captured_by_family_rank": {
            family: float(values[: args.family_rank].clamp_min(0).sum() / values.clamp_min(0).sum())
            for family, values in family_values.items()
        },
        "mean_projection_overlap": {
            family: float(((basis.T @ shared_basis).square().sum() / args.rank).item())
            for family, basis in family_bases.items()
        },
    }
    return bases, diagnostics


def evaluate(model, adapter, factors, loader, device, bootstrap, statistical_seed):
    methods = ("baseline", "full", *factors.keys())
    outcomes = {method: [] for method in methods}
    with torch.inference_mode():
        for pixels, labels in tqdm(loader, desc="Held-out family evaluation", leave=False):
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
    for control_index, control in enumerate(("full", "pooled", "hidden", "random", "noise_specific", "blur_specific")):
        results[f"shared_vs_{control}"] = paired_comparison(
            outcomes[control], outcomes["shared"],
            statistical_seed + 50 + control_index, bootstrap,
        )
    return results, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--family-rank", type=int, default=128)
    parser.add_argument("--discovery-samples", type=int, default=1000)
    parser.add_argument("--evaluation-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("Confirmation is locked to seeds [0,1,2]")
    if args.rank != 32 or args.family_rank != 128:
        raise ValueError("Preregistered shared rank is 32 and family rank is 128")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
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
    artifact_paths, diagnostics = {}, {}
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        result_path = seed_dir / "summary.json"
        if args.resume and result_path.exists():
            seed_document = json.loads(result_path.read_text())
            results[str(seed)] = seed_document["results"]
            diagnostics[str(seed)] = seed_document["diagnostics"]
            artifact_paths[str(seed)] = seed_document["artifact_paths"]
            print(f"Skipping completed seed {seed}", flush=True)
            continue
        adapter, adapter_path = load_adapter(seed, device)
        adapter.requires_grad_(False)
        bases, seed_diagnostics = discover_bases(
            model, adapter, splits[seed]["train"], seed, args, device
        )
        factors = {name: folded_factors(adapter, basis) for name, basis in bases.items()}
        factor_path = seed_dir / "folded_factors.pt"
        torch.save(
            {name: {key: value.cpu() for key, value in group.items()} for name, group in factors.items()},
            factor_path,
        )
        seed_results = {}
        for condition_index, family in enumerate((None, *HELDOUT_FAMILIES)):
            key = "clean" if family is None else family
            condition, arrays = evaluate(
                model, adapter, factors,
                make_loader(
                    family, splits[seed]["validation"], seed,
                    args.evaluation_samples, args.batch_size, args.workers,
                ),
                device, args.bootstrap,
                1270000 + seed * 1000 + condition_index * 100,
            )
            seed_results[key] = condition
            for method, values in arrays.items():
                outcomes[f"seed{seed}_{key}_{method}_correct"] = values
        seed_document = {
            "results": seed_results,
            "diagnostics": seed_diagnostics,
            "artifact_paths": {"adapter": str(adapter_path), "factors": str(factor_path)},
        }
        result_path.write_text(json.dumps(seed_document, indent=2))
        results[str(seed)] = seed_results
        diagnostics[str(seed)] = seed_diagnostics
        artifact_paths[str(seed)] = seed_document["artifact_paths"]
        progress_path.write_text(json.dumps(results, indent=2))
        np.savez_compressed(outcome_path, **outcomes)
        print(f"Completed seed {seed}", flush=True)
    aggregate = {}
    for family in ("clean", *HELDOUT_FAMILIES):
        aggregate[family] = {}
        for method in ("full", "shared", "pooled", "hidden", "random", "noise_specific", "blur_specific"):
            gains = [
                results[str(seed)][family]["methods"][method]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[family][method] = {
                "gains_by_seed": gains,
                "mean_gain": float(np.mean(gains)),
            }
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "block": BLOCK,
            "discovery_families": DISCOVERY_FAMILIES,
            "heldout_families": HELDOUT_FAMILIES,
            "severity": 4,
            "vit_frozen": True,
            "teacher_adapters_frozen": True,
            "labels_used_for_subspace_discovery": False,
            "imageNetV2_accessed": False,
            "status": "family-held-out development evaluation",
        },
        "splits": {str(seed): splits[seed] for seed in args.seeds},
        "artifact_paths": artifact_paths,
        "diagnostics": diagnostics,
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "The shared map is distilled from a Noise+Blur-trained teacher and is not adapter-free discovery.",
            "Evaluation families are absent from subspace discovery, but the teacher architecture and intervention point were selected earlier.",
            "The validation partitions selected the teacher checkpoints, so this is OOD-family development evidence rather than untouched final evaluation.",
            "Online corruptions approximate common corruption families and are not official pre-generated ImageNet-C.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
