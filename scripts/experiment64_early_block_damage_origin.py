#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_summary(root: Path, corruption: str, block: int) -> dict:
    run = f"full_{corruption}4_block{block}_3000_v1" if block < 6 else f"full_{corruption}4_3000_v1"
    path = root / "results/sae/experiment63_block6_residual_mlp" / run / "summary.json"
    with path.open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    rows = []
    for corruption in ("noise", "blur"):
        previous_x = None
        for block in range(1, 7):
            summary = load_summary(args.root, corruption, block)
            results = summary["results"]
            shapley = results["factorial_shapley_accuracy_contributions"]
            full_gain = results["conditions"]["factorial_clean_xam"]["accuracy_difference"]
            x_gain = results["conditions"]["factorial_clean_x"]["accuracy_difference"]
            rows.append({
                "corruption": corruption,
                "block": block,
                "baseline_accuracy": results["corrupted_baseline_accuracy"],
                "full_oracle_gain_pp": 100 * full_gain,
                "incoming_residual_gain_pp": 100 * x_gain,
                "incoming_residual_increment_pp": None if previous_x is None else 100 * (x_gain - previous_x),
                "attention_shapley_pp": 100 * shapley["a"],
                "mlp_shapley_pp": 100 * shapley["m"],
                "incoming_residual_shapley_pp": 100 * shapley["x"],
            })
            previous_x = x_gain

    with (args.output_dir / "blockwise_damage_origin.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    conclusions = {}
    for corruption in ("noise", "blur"):
        subset = [row for row in rows if row["corruption"] == corruption]
        first = subset[0]
        conclusions[corruption] = {
            "earliest_dominant_source": "Block 1 MLP",
            "block1_mlp_shapley_pp": first["mlp_shapley_pp"],
            "block1_full_oracle_gain_pp": first["full_oracle_gain_pp"],
            "block1_mlp_share_of_block1_full_percent": 100 * first["mlp_shapley_pp"] / first["full_oracle_gain_pp"],
            "largest_later_incoming_residual_jump_block": max(
                subset[1:], key=lambda row: row["incoming_residual_increment_pp"]
            )["block"],
        }

    payload = {
        "experiment": 64,
        "question": "Which of Blocks 1-5 first creates most corruption damage?",
        "method": "Aggregate the matched 3,000-image factorial clean-component interventions from Experiment 63 across Blocks 1-6.",
        "split": {"start_index": 47000, "samples": 3000, "status": "mechanistic analysis only"},
        "training_or_tuning": False,
        "deployment_method": False,
        "rows": rows,
        "conclusions": conclusions,
        "limitations": [
            "This is paired-clean oracle diagnosis, not a deployable inference procedure.",
            "Component swaps can be off-manifold; Shapley values average all factorial interactions.",
            "The split was reused for mechanistic analysis and must not be presented as untouched final evaluation.",
        ],
    }
    with (args.output_dir / "summary.json").open("w") as handle:
        json.dump(payload, handle, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, corruption in zip(axes, ("noise", "blur")):
        subset = [row for row in rows if row["corruption"] == corruption]
        blocks = [row["block"] for row in subset]
        axis.plot(blocks, [row["incoming_residual_shapley_pp"] for row in subset], marker="o", label="Incoming residual")
        axis.plot(blocks, [row["attention_shapley_pp"] for row in subset], marker="o", label="Attention")
        axis.plot(blocks, [row["mlp_shapley_pp"] for row in subset], marker="o", label="MLP")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(f"{corruption.title()}-4")
        axis.set_xlabel("Transformer block")
        axis.set_xticks(blocks)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Shapley accuracy contribution (pp)")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(args.output_dir / "blockwise_damage_origin.png", dpi=200)


if __name__ == "__main__":
    main()
