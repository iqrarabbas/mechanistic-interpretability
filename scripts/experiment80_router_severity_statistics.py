import argparse
import csv
import json
from math import comb
from pathlib import Path

import numpy as np


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
DEFAULT_RUN = (
    ACTIVE_ROOT
    / "results/sae/experiment79_frozen_router_severity_sweep/"
    "full_3seed_10families_severity1_5_v1"
)


def exact_mcnemar(reference, candidate):
    gained = int((~reference & candidate).sum())
    damaged = int((reference & ~candidate).sum())
    discordant = gained + damaged
    if discordant == 0:
        return gained, damaged, 1.0
    tail = sum(comb(discordant, index) for index in range(min(gained, damaged) + 1))
    return gained, damaged, min(1.0, 2.0 * tail / (2**discordant))


def bootstrap_ci(values, repetitions, seed):
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        selected = rng.integers(0, values.shape[-1], values.shape[-1])
        draws[index] = values[..., selected].mean()
    return np.quantile(draws, [0.025, 0.975]).tolist()


def holm_adjust(pvalues):
    count = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(count, dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * pvalues[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def load_pair(path, candidate):
    with np.load(path) as data:
        return data["mixed"].astype(bool), data[candidate].astype(bool)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=8000)
    parser.add_argument("--output-name", default="paired_statistics_v2")
    args = parser.parse_args()
    output_dir = args.run_dir / args.output_name
    output_dir.mkdir(exist_ok=False)
    section_prefixes = {"seen_development": "seen_", "unseen_reserve": "unseen_"}
    candidates = ("router_soft", "router_hard", "uniform")
    rows = []
    aggregate = {}

    for section_index, (section, prefix) in enumerate(section_prefixes.items()):
        files = {
            seed: {
                path.stem.removeprefix(prefix): path
                for path in sorted((args.run_dir / f"seed_{seed}").glob(f"{prefix}*.npz"))
            }
            for seed in range(3)
        }
        conditions = sorted(set.intersection(*(set(files[seed]) for seed in range(3))))
        if len(conditions) != (30 if section == "seen_development" else 20):
            raise RuntimeError(f"Incomplete {section}: {len(conditions)} conditions")
        aggregate[section] = {}

        for candidate_index, candidate in enumerate(candidates):
            aggregate[section][candidate] = {"by_seed": {}, "by_severity": {}}
            seed_image_differences = []
            for seed in range(3):
                condition_differences = []
                for condition_index, condition in enumerate(conditions):
                    reference, prediction = load_pair(files[seed][condition], candidate)
                    difference = prediction.astype(float) - reference.astype(float)
                    gained, damaged, pvalue = exact_mcnemar(reference, prediction)
                    ci = bootstrap_ci(
                        difference[None],
                        args.bootstrap,
                        args.seed + section_index * 10000 + candidate_index * 1000 + seed * 100 + condition_index,
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
                    condition_differences.append(difference)
                matrix = np.stack(condition_differences)
                per_image = matrix.mean(axis=0)
                seed_image_differences.append(per_image)
                aggregate[section][candidate]["by_seed"][str(seed)] = {
                    "mean_difference": float(matrix.mean()),
                    "image_clustered_bootstrap_95ci": bootstrap_ci(
                        per_image[None], args.bootstrap, args.seed + section_index * 10000 + candidate_index * 1000 + seed
                    ),
                }

            for severity in range(1, 6):
                severity_conditions = [condition for condition in conditions if condition.endswith(f"_{severity}")]
                per_seed_image = []
                for seed in range(3):
                    per_seed_image.append(
                        np.stack(
                            [
                                load_pair(files[seed][condition], candidate)[1].astype(float)
                                - load_pair(files[seed][condition], candidate)[0].astype(float)
                                for condition in severity_conditions
                            ]
                        ).mean(axis=0)
                    )
                severity_matrix = np.stack(per_seed_image)
                values = np.concatenate(per_seed_image) if section == "seen_development" else severity_matrix
                aggregate[section][candidate]["by_severity"][str(severity)] = {
                    "mean_difference": float(severity_matrix.mean()),
                    "clustered_bootstrap_95ci": bootstrap_ci(
                        values if values.ndim == 2 else values[None],
                        args.bootstrap,
                        args.seed + section_index * 10000 + candidate_index * 1000 + 50 + severity,
                    ),
                }

            seed_matrix = np.stack(seed_image_differences)
            values = np.concatenate(seed_image_differences) if section == "seen_development" else seed_matrix
            aggregate[section][candidate]["three_seed_mean"] = {
                "mean_difference": float(seed_matrix.mean()),
                "clustered_bootstrap_95ci": bootstrap_ci(
                    values if values.ndim == 2 else values[None],
                    args.bootstrap,
                    args.seed + section_index * 10000 + candidate_index * 1000 + 99,
                ),
                "clustering": (
                    "Seed validation ranges are disjoint; images are clustered across corruption families and severities."
                    if section == "seen_development"
                    else "The same reserve images occur across seeds; all seed, family, and severity outcomes remain clustered by image."
                ),
            }

    for section in section_prefixes:
        primary_indices = [
            index
            for index, row in enumerate(rows)
            if row["section"] == section and row["candidate"] == "router_soft"
        ]
        adjusted = holm_adjust([rows[index]["mcnemar_exact_pvalue"] for index in primary_indices])
        for index, value in zip(primary_indices, adjusted):
            rows[index]["holm_pvalue_within_section"] = value
    for row in rows:
        row.setdefault("holm_pvalue_within_section", "")

    csv_path = output_dir / "condition_tests.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "configuration": {
            "source_run": str(args.run_dir.resolve()),
            "bootstrap_repetitions": args.bootstrap,
            "bootstrap_seed": args.seed,
            "inference_rerun": False,
            "primary_comparison": "frozen soft router versus frozen monolithic mixed adapter",
        },
        "aggregate": aggregate,
        "condition_tests_csv": str(csv_path.resolve()),
        "statistical_rules": [
            "Exact McNemar tests preserve paired image outcomes within each seed-condition.",
            "Bootstrap intervals cluster all corruption families and severities belonging to an image.",
            "Reserve intervals also keep all three adapter-seed predictions for each image clustered.",
            "Soft-router condition tests use Holm correction separately within seen and unseen sections.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
