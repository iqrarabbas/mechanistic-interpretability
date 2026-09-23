import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest


PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "results" / "sae" / "experiment36_sae_abnormality_gate"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment44_gate_replication_aggregation"
RUN_PATTERN = re.compile(r"replication_adapter(?P<adapter>\d+)_gate(?P<gate>\d+)_random(?P<random>\d+)")


def bootstrap_interval(differences, generator, repetitions):
    sample_count = differences.size
    negative = int((differences < 0).sum())
    zero = int((differences == 0).sum())
    positive = int((differences > 0).sum())
    counts = generator.multinomial(
        sample_count,
        np.asarray([negative, zero, positive], dtype=np.float64) / sample_count,
        size=repetitions,
    )
    values = (counts[:, 2] - counts[:, 0]) / sample_count
    return [float(value) for value in np.quantile(values, [0.025, 0.975])]


def paired_statistics(reference, candidate, generator, repetitions):
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    recovered = int((~reference & candidate).sum())
    damaged = int((reference & ~candidate).sum())
    differences = candidate.astype(np.float64) - reference.astype(np.float64)
    confidence_interval = bootstrap_interval(differences, generator, repetitions)
    return {
        "reference_accuracy": float(reference.mean()),
        "candidate_accuracy": float(candidate.mean()),
        "accuracy_difference": float(differences.mean()),
        "accuracy_difference_pp": float(100 * differences.mean()),
        "accuracy_difference_95ci": confidence_interval,
        "accuracy_difference_95ci_pp": [
            float(100 * value) for value in confidence_interval
        ],
        "recovered": recovered,
        "damaged": damaged,
        "mcnemar_exact_pvalue": (
            float(binomtest(recovered, recovered + damaged, 0.5).pvalue)
            if recovered + damaged else 1.0
        ),
    }


def run_metadata(path):
    match = RUN_PATTERN.fullmatch(path.parent.name)
    if match is None:
        raise ValueError(f"Unexpected run name: {path.parent.name}")
    return {key: int(value) for key, value in match.groupdict().items()}


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows, path):
    conditions = ["noise4", "blur4"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for axis, condition in zip(axes, conditions):
        selected = [row for row in rows if row["condition"] == condition]
        x = np.arange(len(selected))
        y = np.asarray([row["difference_pp"] for row in selected])
        lower = y - np.asarray([row["ci_low_pp"] for row in selected])
        upper = np.asarray([row["ci_high_pp"] for row in selected]) - y
        for index, row in enumerate(selected):
            color = f"C{row['adapter_seed']}"
            axis.errorbar(
                x[index], y[index], yerr=[[lower[index]], [upper[index]]],
                fmt="o", color=color, capsize=3, markersize=6,
            )
        axis.axhline(0, color="black", linewidth=1, linestyle="--")
        axis.set_xticks(x)
        axis.set_xticklabels([f"A{row['adapter_seed']}/G{row['gate_seed']}" for row in selected], rotation=45, ha="right")
        axis.set_title("Noise-4" if condition == "noise4" else "Blur-4")
        axis.set_xlabel("Adapter / gate seed")
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Harmful gate − ungated adapter (pp)")
    fig.suptitle("Paired ImageNetV2 gate improvements across nine replications")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Aggregate nine completed Experiment 36 gate replications")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=4400)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    summaries = sorted(args.input_root.glob("replication_*/summary.json"))
    if len(summaries) != 9:
        raise RuntimeError(f"Expected nine replication summaries, found {len(summaries)}")
    generator = np.random.default_rng(args.seed)
    records = []
    csv_rows = []
    random_records = []

    for summary_path in summaries:
        metadata = run_metadata(summary_path)
        for condition in ["noise4", "blur4"]:
            arrays = np.load(summary_path.parent / f"{condition}_outcomes.npz")
            paired = paired_statistics(
                arrays["existing_adapter_correct"],
                arrays["harmful_sae_gate_correct"],
                generator,
                args.bootstrap_repetitions,
            )
            record = metadata | {"run_name": summary_path.parent.name, "condition": condition, "paired": paired}
            records.append(record)
            csv_rows.append({
                "run_name": record["run_name"],
                "adapter_seed": metadata["adapter"],
                "gate_seed": metadata["gate"],
                "condition": condition,
                "adapter_accuracy": paired["reference_accuracy"],
                "harmful_gate_accuracy": paired["candidate_accuracy"],
                "difference_pp": paired["accuracy_difference_pp"],
                "ci_low_pp": paired["accuracy_difference_95ci_pp"][0],
                "ci_high_pp": paired["accuracy_difference_95ci_pp"][1],
                "recovered": paired["recovered"],
                "damaged": paired["damaged"],
                "mcnemar_p": paired["mcnemar_exact_pvalue"],
            })
            random_keys = sorted(key for key in arrays.files if re.fullmatch(r"random_sae_gate_\d+_correct", key))
            for key in random_keys:
                random_difference = (
                    arrays[key].astype(np.float64)
                    - arrays["existing_adapter_correct"].astype(np.float64)
                ).mean()
                random_records.append({
                    "adapter_seed": metadata["adapter"],
                    "gate_seed": metadata["gate"],
                    "condition": condition,
                    "control": key,
                    "difference_pp": float(100 * random_difference),
                    "harmful_difference_pp": paired["accuracy_difference_pp"],
                    "harmful_beats_control": paired["accuracy_difference"] > random_difference,
                })

    aggregate = {}
    for condition in ["noise4", "blur4"]:
        condition_records = [record for record in records if record["condition"] == condition]
        differences = np.asarray([record["paired"]["accuracy_difference"] for record in condition_records])
        wins = int((differences > 0).sum())
        controls = [record for record in random_records if record["condition"] == condition]
        control_wins = int(sum(record["harmful_beats_control"] for record in controls))
        aggregate[condition] = {
            "runs": len(condition_records),
            "positive_runs": wins,
            "mean_difference_pp": float(100 * differences.mean()),
            "range_difference_pp": [float(100 * differences.min()), float(100 * differences.max())],
            "exact_sign_test_pvalue_one_sided": float(0.5 ** len(differences)) if wins == len(differences) else None,
            "exact_sign_test_pvalue_two_sided": float(binomtest(wins, len(differences), 0.5).pvalue),
            "random_controls": len(controls),
            "harmful_beats_random_controls": control_wins,
            "random_control_sign_test_pvalue_two_sided": (
                float(binomtest(control_wins, len(controls), 0.5).pvalue) if controls else None
            ),
            "minimum_attainable_paired_empirical_p_with_20_controls_per_adapter": 1 / 21,
            "warning": "Gate replications reuse images; do not treat nine runs as independent image-level samples.",
        }

    write_csv(output_dir / "per_run_paired_statistics.csv", csv_rows)
    if random_records:
        write_csv(output_dir / "random_control_comparisons.csv", random_records)
    make_plot(csv_rows, output_dir / "gate_replication_paired_differences.png")
    summary = {
        "configuration": {
            "input_root": str(args.input_root.resolve()),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "inference_rerun": False,
            "source_runs": [path.parent.name for path in summaries],
        },
        "per_run": records,
        "aggregate": aggregate,
        "limitations": [
            "These historical Experiment 36 runs predate the leakage-free split correction.",
            "The same 10,000 ImageNetV2 images are reused across gate replications.",
            "Only one of three gate seeds per adapter has 20 random controls.",
            "One thousand random controls requested by the supervisor have not been run.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))
    print(f"Saved aggregation to {output_dir}")


if __name__ == "__main__":
    main()
