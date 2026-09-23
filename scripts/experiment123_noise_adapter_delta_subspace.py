import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment52_block6_repair_propagation import make_loader
from scripts.experiment72_downstream_circuit_necessity import load_adapter


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results/sae/experiment123_noise_adapter_delta_subspace"
WIDTH = 768
PATCHES = 196
BLOCK = 6


def accumulate_second_moments(model, adapter, loader, device):
    delta_second = torch.zeros(WIDTH, WIDTH, dtype=torch.float64, device=device)
    hidden_sum = torch.zeros(WIDTH, dtype=torch.float64, device=device)
    hidden_second = torch.zeros(WIDTH, WIDTH, dtype=torch.float64, device=device)
    count = 0
    with torch.inference_mode():
        for _, noise, _ in tqdm(loader, desc="Discovering repair geometry", leave=False):
            hidden = model(pixel_values=noise.to(device), output_hidden_states=True).hidden_states[BLOCK][:, 1:]
            delta = adapter(hidden)
            flat_hidden = hidden.flatten(0, 1).double()
            flat_delta = delta.flatten(0, 1).double()
            delta_second.addmm_(flat_delta.T, flat_delta)
            hidden_sum += flat_hidden.sum(0)
            hidden_second.addmm_(flat_hidden.T, flat_hidden)
            count += flat_hidden.shape[0]
    hidden_covariance = hidden_second / count
    hidden_mean = hidden_sum / count
    hidden_covariance -= torch.outer(hidden_mean, hidden_mean)
    return delta_second / count, hidden_covariance, count


def top_basis(second_moment, maximum_rank):
    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    order = torch.argsort(eigenvalues, descending=True)
    return eigenvectors[:, order[:maximum_rank]].float(), eigenvalues[order].float()


def random_basis(seed, maximum_rank, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    matrix = torch.randn(WIDTH, maximum_rank, generator=generator, device=device)
    return torch.linalg.qr(matrix, mode="reduced").Q


def projected_delta(adapter, patches, basis):
    # This expression is used for numerical evaluation. The saved folded
    # factors below implement the same map without requiring the full adapter.
    delta = adapter(patches)
    return (delta @ basis) @ basis.T


def folded_factors(adapter, basis):
    weight = adapter.linear.weight.detach()
    bias = adapter.linear.bias.detach()
    positions = adapter.position.weight.detach()
    return {
        "input_to_rank": weight.T @ basis,
        "bias_rank": bias @ basis,
        "position_rank": positions @ basis,
        "rank_to_output": basis.T,
    }


def folded_delta(patches, factors):
    positions = torch.arange(PATCHES, device=patches.device)
    compressed = (
        patches @ factors["input_to_rank"]
        + factors["bias_rank"]
        + factors["position_rank"][positions][None]
    )
    return compressed @ factors["rank_to_output"]


def evaluate(model, adapter, bases, ranks, loader, device, bootstrap, statistical_seed):
    names = ["baseline", "full"] + [f"{kind}_rank{rank}" for kind in bases for rank in ranks]
    correct = {name: [] for name in names}
    max_fold_error = 0.0
    factors = {
        kind: {rank: folded_factors(adapter, basis[:, :rank]) for rank in ranks}
        for kind, basis in bases.items()
    }
    with torch.inference_mode():
        for _, noise, labels in tqdm(loader, desc="Evaluating compressed repairs", leave=False):
            labels = labels.to(device)
            output = model(pixel_values=noise.to(device), output_hidden_states=True)
            hidden = output.hidden_states[BLOCK]
            patches = hidden[:, 1:]
            correct["baseline"].extend((output.logits.argmax(1) == labels).cpu().tolist())
            full_hidden = torch.cat([hidden[:, :1], patches + adapter(patches)], dim=1)
            full_logits = downstream_from_layer(model, full_hidden, BLOCK - 1)
            correct["full"].extend((full_logits.argmax(1) == labels).cpu().tolist())
            for kind, basis in bases.items():
                for rank in ranks:
                    current_basis = basis[:, :rank]
                    direct = projected_delta(adapter, patches, current_basis)
                    folded = folded_delta(patches, factors[kind][rank])
                    max_fold_error = max(max_fold_error, float((direct - folded).abs().max()))
                    candidate = torch.cat([hidden[:, :1], patches + folded], dim=1)
                    logits = downstream_from_layer(model, candidate, BLOCK - 1)
                    correct[f"{kind}_rank{rank}"].extend(
                        (logits.argmax(1) == labels).cpu().tolist()
                    )
    if max_fold_error > 1e-3:
        raise RuntimeError(f"Folded low-rank map mismatch: {max_fold_error}")
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    results = {
        "full_vs_baseline": paired_comparison(
            correct["baseline"], correct["full"], statistical_seed, bootstrap
        ),
        "compressed": {},
        "max_folded_delta_absolute_error": max_fold_error,
    }
    full_gain = results["full_vs_baseline"]["accuracy_difference"]
    for offset, name in enumerate(names[2:]):
        versus_baseline = paired_comparison(
            correct["baseline"], correct[name], statistical_seed + offset + 1, bootstrap
        )
        versus_full = paired_comparison(
            correct["full"], correct[name], statistical_seed + 100 + offset, bootstrap
        )
        results["compressed"][name] = {
            "vs_baseline": versus_baseline,
            "vs_full": versus_full,
            "retained_full_gain_fraction": (
                versus_baseline["accuracy_difference"] / full_gain if full_gain else None
            ),
        }
    return results, correct, factors


def coefficient_count(rank):
    return WIDTH * rank + rank + PATCHES * rank + rank * WIDTH


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
    parser.add_argument("--discovery-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if len(set(args.ranks)) != len(args.ranks) or any(rank < 1 or rank > WIDTH for rank in args.ranks):
        raise ValueError("Ranks must be unique integers in [1,768]")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in (0, 1, 2) for seed in args.seeds):
        raise ValueError("Seeds must be unique and drawn from {0,1,2}")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    maximum_rank = max(args.ranks)
    completed = {}
    outcomes_path = output_dir / "paired_outcomes.npz"
    outcomes = {}
    if args.resume and outcomes_path.exists():
        with np.load(outcomes_path) as saved:
            outcomes = {name: saved[name] for name in saved.files}
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        seed_summary_path = seed_dir / "summary.json"
        if args.resume and seed_summary_path.exists():
            completed[str(seed)] = json.loads(seed_summary_path.read_text())
            print(f"Skipping completed seed {seed}", flush=True)
            continue
        adapter, adapter_path = load_adapter(seed, device)
        adapter.requires_grad_(False)
        train_split = splits[seed]["train"]
        validation_split = splits[seed]["validation"]
        discovery_count = min(args.discovery_samples, train_split["end"] - train_split["start"])
        evaluation_count = min(args.evaluation_samples, validation_split["end"] - validation_split["start"])
        discovery_loader = make_loader(
            "noise", {"start": train_split["start"], "end": train_split["start"] + discovery_count},
            seed, args.batch_size, args.workers, discovery_count,
        )
        delta_second, hidden_covariance, patch_count = accumulate_second_moments(
            model, adapter, discovery_loader, device
        )
        delta_basis, delta_values = top_basis(delta_second, maximum_rank)
        hidden_basis, hidden_values = top_basis(hidden_covariance, maximum_rank)
        bases = {
            "delta_pca": delta_basis,
            "hidden_pca": hidden_basis,
            "random": random_basis(123000 + seed, maximum_rank, device),
        }
        condition_results = {}
        factors = None
        for condition_index, condition in enumerate(("noise", "clean")):
            evaluation_loader = make_loader(
                "noise", {"start": validation_split["start"], "end": validation_split["start"] + evaluation_count},
                seed, args.batch_size, args.workers, evaluation_count,
            )
            if condition == "clean":
                evaluation_loader = ((clean, clean, labels) for clean, _, labels in evaluation_loader)
            result, correct, current_factors = evaluate(
                model, adapter, bases, args.ranks, evaluation_loader, device,
                args.bootstrap, 123000 + seed * 1000 + condition_index * 500,
            )
            condition_results[condition] = result
            factors = current_factors
            for name, values in correct.items():
                outcomes[f"seed{seed}_{condition}_{name}_correct"] = values
        for kind in bases:
            torch.save(
                {rank: {name: value.cpu() for name, value in factors[kind][rank].items()} for rank in args.ranks},
                seed_dir / f"{kind}_folded_factors.pt",
            )
        total_delta_energy = float(delta_values.clamp_min(0).sum())
        seed_summary = {
            "seed": seed,
            "adapter_path": str(adapter_path),
            "discovery_split": train_split | {"used_start": train_split["start"], "used_end": train_split["start"] + discovery_count},
            "evaluation_split": validation_split | {"used_start": validation_split["start"], "used_end": validation_split["start"] + evaluation_count},
            "discovery_patch_count": patch_count,
            "delta_energy_explained": {
                str(rank): float(delta_values[:rank].clamp_min(0).sum() / total_delta_energy)
                for rank in args.ranks
            },
            "hidden_variance_explained": {
                str(rank): float(hidden_values[:rank].clamp_min(0).sum() / hidden_values.clamp_min(0).sum())
                for rank in args.ranks
            },
            "results": condition_results,
        }
        seed_summary_path.write_text(json.dumps(seed_summary, indent=2))
        completed[str(seed)] = seed_summary
        print(f"Completed seed {seed}", flush=True)
    aggregate = {}
    for kind in ("delta_pca", "hidden_pca", "random"):
        aggregate[kind] = {}
        for rank in args.ranks:
            name = f"{kind}_rank{rank}"
            gains = [completed[str(seed)]["results"]["noise"]["compressed"][name]["vs_baseline"]["accuracy_difference"] for seed in args.seeds]
            clean_changes = [completed[str(seed)]["results"]["clean"]["compressed"][name]["vs_baseline"]["accuracy_difference"] for seed in args.seeds]
            retention = [completed[str(seed)]["results"]["noise"]["compressed"][name]["retained_full_gain_fraction"] for seed in args.seeds]
            aggregate[kind][str(rank)] = {
                "standalone_coefficients": coefficient_count(rank),
                "gains_by_seed": gains,
                "mean_gain": float(np.mean(gains)),
                "clean_changes_by_seed": clean_changes,
                "mean_clean_change": float(np.mean(clean_changes)),
                "retained_gain_fractions_by_seed": retention,
                "mean_retained_gain_fraction": float(np.mean(retention)),
            }
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "block": BLOCK,
            "vit_frozen": True,
            "teacher_adapters_frozen": True,
            "teacher_absent_from_deployed_folded_map": True,
            "uses_labels_for_subspace_discovery": False,
            "imageNetV2_accessed": False,
            "status": "development teacher-subspace distillation",
        },
        "splits": {str(seed): splits[seed] for seed in args.seeds},
        "seeds": completed,
        "aggregate": aggregate,
        "controls": {
            "delta_pca": "Top uncentered second-moment directions of teacher adapter patch-token corrections.",
            "hidden_pca": "Top centered variance directions of noisy Block-6 patch states.",
            "random": "Seeded random orthonormal output directions, rank matched.",
        },
        "limitations": [
            "The compact maps are distilled from the successful full adapter, so this tests mechanism compression rather than adapter-free discovery.",
            "Subspace discovery uses only each adapter seed's training partition; evaluation uses its disjoint validation partition.",
            "The validation splits were previously used to select the teacher checkpoints, so this is development evidence, not final untouched evaluation.",
            "No ImageNetV2 result is used for selection or evaluation in this experiment.",
        ],
    }
    pooled = {condition: {} for condition in ("noise", "clean")}
    for condition_index, condition in enumerate(("noise", "clean")):
        for rank in args.ranks:
            candidate = np.concatenate([
                outcomes[f"seed{seed}_{condition}_delta_pca_rank{rank}_correct"]
                for seed in args.seeds
            ])
            pooled[condition][str(rank)] = {}
            for control_index, control in enumerate(("hidden_pca", "random", "full")):
                control_name = control if control == "full" else f"{control}_rank{rank}"
                reference = np.concatenate([
                    outcomes[f"seed{seed}_{condition}_{control_name}_correct"]
                    for seed in args.seeds
                ])
                pooled[condition][str(rank)][f"delta_pca_vs_{control}"] = paired_comparison(
                    reference, candidate,
                    123900 + condition_index * 10000 + rank * 10 + control_index,
                    args.bootstrap,
                )
    summary["pooled_disjoint_validation_comparisons"] = pooled
    np.savez_compressed(outcomes_path, **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
