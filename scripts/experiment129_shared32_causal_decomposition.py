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
from scripts.experiment72_downstream_circuit_necessity import load_adapter
from scripts.experiment123_noise_adapter_delta_subspace import folded_delta, folded_factors, random_basis
from scripts.experiment127_shared_ood_repair_subspace import HELDOUT_FAMILIES, make_loader


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results/sae/experiment129_shared32_causal_decomposition"
SOURCE_ROOT = ROOT / "results/sae/experiment127_shared_ood_repair_subspace/full_3seed_shared32_familyrank128_v1"
BLOCK = 6
RANK = 32
CONDITIONS = ("clean", *HELDOUT_FAMILIES)


def load_source_groups(seed, device):
    path = SOURCE_ROOT / f"seed_{seed}" / "folded_factors.pt"
    groups = torch.load(path, map_location=device, weights_only=True)
    selected = {
        name: {key: value.to(device) for key, value in groups[name].items()}
        for name in ("shared", "hidden", "random")
    }
    return selected, path


def correction_logits(model, hidden, patches, correction):
    candidate = torch.cat([hidden[:, :1], patches + correction], dim=1)
    return downstream_from_layer(model, candidate, BLOCK - 1)


def energy_match(reference, candidate):
    reference_norm = reference.flatten(1).norm(dim=1, keepdim=True)
    candidate_norm = candidate.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = (reference_norm / candidate_norm).view(-1, 1, 1)
    return candidate * scale


def evaluate(
    model, adapter, factors, random_factors, loader, device, bootstrap, statistical_seed
):
    fixed_methods = (
        "baseline",
        "full",
        "shared_only",
        "full_minus_shared",
        "hidden_only",
        "full_minus_hidden",
        "random_only",
        "full_minus_random",
    )
    outcomes = {method: [] for method in fixed_methods}
    for control_index in range(len(random_factors)):
        outcomes[f"full_minus_energy_random_{control_index}"] = []
    energy = {"shared_fraction": [], "hidden_fraction": [], "random_fraction": []}

    with torch.inference_mode():
        for pixels, labels in tqdm(loader, desc="Causal rank-32 decomposition", leave=False):
            labels = labels.to(device)
            outputs = model(pixel_values=pixels.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            patches = hidden[:, 1:]
            full_delta = adapter(patches)
            shared_delta = folded_delta(patches, factors["shared"])
            hidden_delta = folded_delta(patches, factors["hidden"])
            random_delta = folded_delta(patches, factors["random"])

            outcomes["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            corrections = {
                "full": full_delta,
                "shared_only": shared_delta,
                "full_minus_shared": full_delta - shared_delta,
                "hidden_only": hidden_delta,
                "full_minus_hidden": full_delta - hidden_delta,
                "random_only": random_delta,
                "full_minus_random": full_delta - random_delta,
            }
            for method, correction in corrections.items():
                logits = correction_logits(model, hidden, patches, correction)
                outcomes[method].extend((logits.argmax(1) == labels).cpu().tolist())

            full_energy = full_delta.flatten(1).square().sum(1).clamp_min(1e-12)
            for name, delta in (
                ("shared_fraction", shared_delta),
                ("hidden_fraction", hidden_delta),
                ("random_fraction", random_delta),
            ):
                energy[name].extend(
                    (delta.flatten(1).square().sum(1) / full_energy).cpu().tolist()
                )

            for control_index, control_factors in enumerate(random_factors):
                control_delta = folded_delta(patches, control_factors)
                matched_delta = energy_match(shared_delta, control_delta)
                logits = correction_logits(
                    model, hidden, patches, full_delta - matched_delta
                )
                outcomes[f"full_minus_energy_random_{control_index}"].extend(
                    (logits.argmax(1) == labels).cpu().tolist()
                )

    outcomes = {name: np.asarray(values, dtype=bool) for name, values in outcomes.items()}
    results = {
        "baseline_accuracy": float(outcomes["baseline"].mean()),
        "methods": {},
        "causal_costs": {},
        "energy": {name: float(np.mean(values)) for name, values in energy.items()},
    }
    for method_index, method in enumerate(fixed_methods[1:]):
        results["methods"][method] = paired_comparison(
            outcomes["baseline"], outcomes[method],
            statistical_seed + method_index, bootstrap,
        )

    for basis_index, basis_name in enumerate(("shared", "hidden", "random")):
        comparison = paired_comparison(
            outcomes[f"full_minus_{basis_name}"], outcomes["full"],
            statistical_seed + 100 + basis_index, bootstrap,
        )
        results["causal_costs"][basis_name] = comparison

    random_costs = []
    for control_index in range(len(random_factors)):
        comparison = paired_comparison(
            outcomes[f"full_minus_energy_random_{control_index}"], outcomes["full"],
            statistical_seed + 1000 + control_index, bootstrap,
        )
        random_costs.append(comparison["accuracy_difference"])
    shared_cost = results["causal_costs"]["shared"]["accuracy_difference"]
    results["energy_matched_random_controls"] = {
        "count": len(random_costs),
        "costs": random_costs,
        "mean_cost": float(np.mean(random_costs)),
        "maximum_cost": float(np.max(random_costs)),
        "shared_cost": shared_cost,
        "shared_percentile": float(100 * np.mean(np.asarray(random_costs) <= shared_cost)),
        "empirical_pvalue": float(
            (1 + np.sum(np.asarray(random_costs) >= shared_cost)) / (1 + len(random_costs))
        ),
    }
    return results, outcomes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--evaluation-samples", type=int, default=1000)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("Confirmation is locked to seeds [0,1,2]")
    if args.random_controls < 1:
        raise ValueError("At least one energy-matched random control is required")

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    progress_path = output_dir / "progress.json"
    partial_outcomes_path = output_dir / "paired_outcomes_partial.npz"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    outcomes = {}
    if partial_outcomes_path.exists():
        with np.load(partial_outcomes_path) as saved:
            outcomes = {name: saved[name] for name in saved.files}

    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    artifacts = {}

    for seed in args.seeds:
        adapter, adapter_path = load_adapter(seed, device)
        adapter.requires_grad_(False)
        factors, factor_path = load_source_groups(seed, device)
        random_factors = []
        for control_index in range(args.random_controls):
            basis = random_basis(129000 + seed * 1000 + control_index, RANK, device)
            random_factors.append(folded_factors(adapter, basis))
        artifacts[str(seed)] = {
            "adapter": str(adapter_path),
            "source_factors": str(factor_path),
        }
        results.setdefault(str(seed), {})
        for condition_index, condition in enumerate(CONDITIONS):
            if condition in results[str(seed)]:
                print(f"Skipping completed seed {seed} {condition}", flush=True)
                continue
            family = None if condition == "clean" else condition
            loader = make_loader(
                family, splits[seed]["validation"], seed,
                args.evaluation_samples, args.batch_size, args.workers,
            )
            condition_results, arrays = evaluate(
                model, adapter, factors, random_factors, loader, device,
                args.bootstrap, 1290000 + seed * 10000 + condition_index * 1000,
            )
            results[str(seed)][condition] = condition_results
            for method, values in arrays.items():
                outcomes[f"seed{seed}_{condition}_{method}_correct"] = values
            progress_path.write_text(json.dumps(results, indent=2))
            np.savez_compressed(partial_outcomes_path, **outcomes)
            print(f"Completed seed {seed} {condition}", flush=True)

    aggregate = {}
    for condition in CONDITIONS:
        aggregate[condition] = {}
        for method in (
            "full", "shared_only", "full_minus_shared", "hidden_only",
            "full_minus_hidden", "random_only", "full_minus_random",
        ):
            gains = [
                results[str(seed)][condition]["methods"][method]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[condition][method] = {
                "gains_by_seed": gains,
                "mean_gain": float(np.mean(gains)),
            }
        for basis_name in ("shared", "hidden", "random"):
            costs = [
                results[str(seed)][condition]["causal_costs"][basis_name]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[condition][f"{basis_name}_ablation_cost"] = {
                "costs_by_seed": costs,
                "mean_cost": float(np.mean(costs)),
            }
        aggregate[condition]["energy_matched_random"] = {
            "shared_percentiles_by_seed": [
                results[str(seed)][condition]["energy_matched_random_controls"]["shared_percentile"]
                for seed in args.seeds
            ],
            "empirical_pvalues_by_seed": [
                results[str(seed)][condition]["energy_matched_random_controls"]["empirical_pvalue"]
                for seed in args.seeds
            ],
        }

    serialized_args = vars(args).copy()
    serialized_args["split_manifest"] = str(args.split_manifest)
    summary = {
        "configuration": serialized_args | {
            "model": BASE_MODEL,
            "block": BLOCK,
            "rank": RANK,
            "conditions": CONDITIONS,
            "source_experiment": str(SOURCE_ROOT),
            "vit_frozen": True,
            "adapter_frozen": True,
            "subspaces_frozen": True,
            "training_or_tuning_performed": False,
            "clean_counterpart_used_at_inference": False,
            "status": "development causal decomposition; not independent final confirmation",
        },
        "development_splits": {
            str(seed): splits[seed]["validation"] for seed in args.seeds
        },
        "artifacts": artifacts,
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "The validation conditions were previously used in Experiment 127; this is a mechanistic decomposition, not an untouched confirmation.",
            "Twenty random controls give a minimum attainable empirical p-value of 1/21; stronger significance requires more controls.",
            "Gaussian corruptions are deterministic online approximations rather than official ImageNet-C files.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print("Saved", output_dir / "summary.json", flush=True)


if __name__ == "__main__":
    main()
