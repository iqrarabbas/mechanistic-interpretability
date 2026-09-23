import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
DEFAULT_RUN = (
    ACTIVE_ROOT
    / "results/sae/experiment77_learned_moe_router/"
    "full_3seed_linear_router_seen_unseen_v1"
)


def exact_mcnemar(reference, candidate):
    gained = int((~reference & candidate).sum())
    damaged = int((reference & ~candidate).sum())
    discordant = gained + damaged
    if discordant == 0:
        return gained, damaged, 1.0
    tail = sum(comb(discordant, index) for index in range(min(gained, damaged) + 1))
    return gained, damaged, min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_ci(differences, repetitions, seed):
    rng = np.random.default_rng(seed)
    samples = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        chosen = rng.integers(0, differences.shape[-1], differences.shape[-1])
        samples[index] = differences[..., chosen].mean()
    return np.quantile(samples, [0.025, 0.975]).tolist()


def condition_files(run_dir):
    sections = {
        "development_seen": "development_",
        "reserve_unseen": "reserve_",
        "sketch_unseen": "sketch_",
    }
    result = {}
    for section, prefix in sections.items():
        result[section] = {}
        for seed in range(3):
            files = sorted((run_dir / f"seed_{seed}").glob(f"{prefix}*.npz"))
            result[section][seed] = {
                path.stem.removeprefix(prefix): path for path in files
            }
    return result


def load_pair(path, candidate):
    with np.load(path) as values:
        return values["mixed"].astype(bool), values[candidate].astype(bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=7800)
    args = parser.parse_args()
    output_dir = args.run_dir / "paired_statistics"
    output_dir.mkdir(exist_ok=False)
    files = condition_files(args.run_dir)
    candidates = ("router_soft", "router_hard", "uniform")
    rows = []
    aggregate = {}

    for section, seed_files in files.items():
        common = sorted(set.intersection(*(set(seed_files[seed]) for seed in range(3))))
        if section == "development_seen":
            corruption_conditions = [condition for condition in common if condition != "clean"]
        else:
            corruption_conditions = common
        aggregate[section] = {}
        for candidate_index, candidate in enumerate(candidates):
            aggregate[section][candidate] = {}
            for seed in range(3):
                condition_differences = []
                for condition in common:
                    reference, prediction = load_pair(seed_files[seed][condition], candidate)
                    gained, damaged, pvalue = exact_mcnemar(reference, prediction)
                    difference = prediction.astype(float) - reference.astype(float)
                    ci = bootstrap_ci(
                        difference[None], args.bootstrap, args.seed + candidate_index * 1000 + seed * 100 + len(rows)
                    )
                    rows.append(
                        {
                            "section": section,
                            "seed": seed,
                            "condition": condition,
                            "candidate": candidate,
                            "samples": len(reference),
                            "mixed_accuracy": float(reference.mean()),
                            "candidate_accuracy": float(prediction.mean()),
                            "difference": float(difference.mean()),
                            "ci_low": ci[0],
                            "ci_high": ci[1],
                            "gained": gained,
                            "damaged": damaged,
                            "mcnemar_exact_pvalue": pvalue,
                        }
                    )
                    if condition in corruption_conditions:
                        condition_differences.append(difference)
                matrix = np.stack(condition_differences)
                per_image = matrix.mean(axis=0)
                ci = bootstrap_ci(
                    per_image[None], args.bootstrap, args.seed + candidate_index * 1000 + seed * 100
                )
                aggregate[section][candidate][str(seed)] = {
                    "conditions": corruption_conditions,
                    "images": matrix.shape[1],
                    "mean_difference": float(matrix.mean()),
                    "image_clustered_bootstrap_95ci": ci,
                }

            seed_matrices = []
            for seed in range(3):
                seed_matrices.append(
                    np.stack(
                        [
                            load_pair(seed_files[seed][condition], candidate)[1].astype(float)
                            - load_pair(seed_files[seed][condition], candidate)[0].astype(float)
                            for condition in corruption_conditions
                        ]
                    ).mean(axis=0)
                )
            if section == "development_seen":
                pooled = np.concatenate(seed_matrices)
                ci = bootstrap_ci(
                    pooled[None], args.bootstrap, args.seed + candidate_index * 1000 + 77
                )
                bootstrap_label = "disjoint_image_bootstrap_95ci"
            else:
                matrix = np.stack(seed_matrices)
                ci = bootstrap_ci(
                    matrix, args.bootstrap, args.seed + candidate_index * 1000 + 77
                )
                bootstrap_label = "image_clustered_across_adapter_seeds_95ci"
            aggregate[section][candidate]["three_seed_mean"] = {
                "mean_difference": float(np.mean(seed_matrices)),
                bootstrap_label: ci,
                "note": (
                    "Development seed validation ranges are disjoint."
                    if section == "development_seen"
                    else "The same images are evaluated across adapter seeds, so bootstrap resamples images and keeps seed predictions clustered."
                ),
            }

    with (output_dir / "condition_tests.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "configuration": {
            "source_run": str(args.run_dir.resolve()),
            "bootstrap_repetitions": args.bootstrap,
            "bootstrap_seed": args.seed,
            "inference_rerun": False,
            "primary_comparison": "learned soft router versus monolithic mixed adapter",
        },
        "aggregate": aggregate,
        "condition_tests_csv": str((output_dir / "condition_tests.csv").resolve()),
        "statistical_rules": [
            "McNemar exact tests use paired image outcomes within each seed and condition.",
            "Aggregate confidence intervals cluster repeated corruption conditions by image.",
            "Reserve and ImageNet-Sketch intervals additionally keep all three adapter-seed predictions for an image in the same bootstrap cluster.",
            "No multiplicity-adjusted condition-level claims are made; aggregate soft-router comparisons are primary.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
