import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

import scripts.experiment72_downstream_circuit_necessity as circuit
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment52_block6_repair_propagation import make_loader
from scripts.experiment73_seed_specific_attention_circuits import evaluate


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment74_attention_circuit_size_sweep"
DISCOVERY_SUMMARY = (
    ACTIVE_ROOT
    / "results/sae/experiment71_downstream_circuit_recovery/"
    "full_3seed_noise_blur_2000_v1/summary.json"
)
HEAD_METRICS = ("routing_recovery", "cls_routing_recovery", "content_recovery")


def rank_heads(discovery):
    ranking = []
    for block in range(7, 13):
        for head in range(12):
            metric_means = {}
            for metric in HEAD_METRICS:
                values = [
                    discovery[str(seed)][corruption]["blocks"][str(block)][metric][head]
                    for seed in range(3)
                    for corruption in ["noise", "blur"]
                ]
                metric_means[metric] = float(np.mean(values))
            best_metric = max(metric_means, key=metric_means.get)
            ranking.append(
                {
                    "block": block,
                    "head": head,
                    "score": metric_means[best_metric],
                    "best_metric": best_metric,
                    "metric_means": metric_means,
                }
            )
    return sorted(ranking, key=lambda row: row["score"], reverse=True)


def grouped_heads(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["block"], []).append(row["head"])
    return {block: tuple(sorted(heads)) for block, heads in grouped.items()}


def random_sets(target, controls, seed):
    generator = np.random.default_rng(seed)
    sets = []
    for _ in range(controls):
        sampled = {}
        for block, heads in target.items():
            population = np.arange(12)
            sampled[block] = tuple(
                sorted(generator.choice(population, len(heads), replace=False).tolist())
            )
        sets.append(sampled)
    return sets


def variants_for(target, controls, seed):
    variants = [
        {"name": "full", "head": None, "neuron": None},
        {"name": "seed_specific", "head": target, "neuron": None},
        {"name": "common", "head": target, "neuron": None},
    ]
    variants.extend(
        {"name": f"random_{index}", "head": heads, "neuron": None}
        for index, heads in enumerate(random_sets(target, controls, seed))
    )
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-summary", type=Path, default=DISCOVERY_SUMMARY)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corruptions", nargs="+", choices=["noise", "blur"], default=["noise", "blur"])
    parser.add_argument("--k-values", type=int, nargs="+", default=[1, 3, 6, 12])
    parser.add_argument("--start-index", type=int, default=48000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start_index < 48000 or args.start_index + args.samples > 49000:
        raise ValueError("Size-sweep confirmation is locked to [48000,49000)")
    if any(k < 1 or k > 12 for k in args.k_values):
        raise ValueError("k-values must be between 1 and 12")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    discovery_document = json.loads(args.discovery_summary.read_text())
    ranking = rank_heads(discovery_document["results"])
    target_sets = {k: grouped_heads(ranking[:k]) for k in args.k_values}
    print("Frozen ranking:", [(row["block"], row["head"]) for row in ranking[: max(args.k_values)]])
    print("Target sets:", target_sets)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)
    split = {"start": args.start_index, "end": args.start_index + args.samples}
    results = {}
    outcomes = {}
    adapter_paths = {}
    for seed in args.seeds:
        adapter, path = circuit.load_adapter(seed, device)
        adapter_paths[str(seed)] = str(path)
        results[str(seed)] = {}
        for corruption_index, corruption in enumerate(args.corruptions):
            results[str(seed)][corruption] = {}
            for k_index, k in enumerate(args.k_values):
                target = target_sets[k]
                circuit.TARGET_HEADS = target
                circuit.TARGET_NEURONS = {}
                variants = variants_for(
                    target, args.random_controls, 2074 + seed * 100 + k
                )
                loader = make_loader(
                    corruption, split, seed, args.batch_size, args.workers, args.samples
                )
                condition, arrays = evaluate(
                    model,
                    adapter,
                    loader,
                    variants,
                    device,
                    args,
                    2074 + seed * 10000 + corruption_index * 1000 + k_index * 100,
                )
                condition.pop("common_ablation")
                condition.pop("common_removal_cost")
                condition.pop("seed_specific_minus_common_cost")
                condition.pop("common_exceeds_random")
                results[str(seed)][corruption][str(k)] = condition
                for name, values in arrays.items():
                    if name != "common":
                        outcomes[f"seed{seed}_{corruption}_k{k}_{name}"] = values

    summary = {
        "configuration": vars(args)
        | {
            "discovery_summary": str(args.discovery_summary.resolve()),
            "model": BASE_MODEL,
            "adapter_paths": adapter_paths,
            "head_ranking": ranking,
            "target_sets": target_sets,
            "selection_rule": "Rank each unique head by its best mean recovery metric across three adapter seeds and Noise-4/Blur-4; use cumulative top-k sets.",
            "candidate_source_range": [36000, 38000],
            "confirmation_range": [args.start_index, args.start_index + args.samples],
            "random_matching": "same head count per block and per-token removed projected-output norm",
            "vit_frozen": True,
            "adapters_frozen": True,
            "uses_clean_counterpart_for_intervention": False,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "disjoint causal attention-circuit size sweep",
        },
        "results": results,
        "guardrails": [
            "The complete ranking and all k-values were frozen before confirmation inference.",
            "The confirmation split is disjoint from Experiments 71, 72, and 73.",
            "Random controls match block distribution and removed output energy, but may overlap targeted heads by chance.",
            "This tests distributed necessity, not sufficiency or completeness.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
