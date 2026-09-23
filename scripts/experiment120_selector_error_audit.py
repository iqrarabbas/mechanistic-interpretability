import argparse
import csv
import json
from pathlib import Path

import numpy as np

from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment111_leakage_free_head_anomaly_confirmation import sha256
from scripts.experiment118_frozen_utility_router_sketch import CONDITIONS
from scripts.experiment119_abstaining_repair_selector import (
    ACTIONS,
    OUTPUT_ROOT as SELECTOR_ROOT,
    action_scores,
    fit_selector,
    select_actions,
)


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment120_selector_error_audit"
DEFAULT_SOURCE = SELECTOR_ROOT / "full_3seed_clean_safe_abstaining_v1"


def load_cache(path):
    with np.load(path) as cache:
        return {key: cache[key] for key in cache.files}


def fraction(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def paired_interval(per_image_difference, seed, repetitions):
    generator = np.random.default_rng(seed)
    samples = generator.integers(0, len(per_image_difference), size=(repetitions, len(per_image_difference)))
    return [float(value) for value in np.quantile(per_image_difference[samples].mean(axis=1), [0.025, 0.975])]


def audit_condition(data, choices, seed, condition_index, bootstrap):
    correct = data["correct"]
    baseline = correct[:, 0]
    selected = correct[np.arange(len(choices)), choices]
    mixed = data["mixed_correct"]
    no_repair = choices == 0
    any_expert_recovery = ~baseline & (correct[:, 1] | correct[:, 2])
    mixed_recovery = ~baseline & mixed
    mixed_damage = baseline & ~mixed
    counts = {
        "samples": len(choices),
        "no_repair": int(no_repair.sum()),
        "potential_recovery_by_either_expert": int(any_expert_recovery.sum()),
        "missed_expert_recovery_due_to_no_repair": int((any_expert_recovery & no_repair).sum()),
        "mixed_recoveries": int(mixed_recovery.sum()),
        "mixed_recoveries_skipped": int((mixed_recovery & no_repair).sum()),
        "mixed_damages": int(mixed_damage.sum()),
        "mixed_damages_avoided_by_no_repair": int((mixed_damage & no_repair).sum()),
        "selected_recoveries": int((~baseline & selected).sum()),
        "selected_damages": int((baseline & ~selected).sum()),
        "selected_better_than_mixed": int((selected & ~mixed).sum()),
        "selected_worse_than_mixed": int((~selected & mixed).sum()),
        "selected_wrong_action_when_expert_could_recover": int((any_expert_recovery & ~no_repair & ~selected).sum()),
    }
    counts["fraction_expert_recoveries_missed_by_abstention"] = fraction(
        counts["missed_expert_recovery_due_to_no_repair"], counts["potential_recovery_by_either_expert"]
    )
    counts["fraction_mixed_recoveries_skipped"] = fraction(
        counts["mixed_recoveries_skipped"], counts["mixed_recoveries"]
    )
    counts["fraction_mixed_damages_avoided"] = fraction(
        counts["mixed_damages_avoided_by_no_repair"], counts["mixed_damages"]
    )
    return {
        "counts": counts,
        "selected_vs_baseline": paired_comparison(baseline, selected, 120000 + seed * 100 + condition_index, bootstrap),
        "mixed_vs_baseline": paired_comparison(baseline, mixed, 121000 + seed * 100 + condition_index, bootstrap),
        "selected_vs_mixed": paired_comparison(mixed, selected, 122000 + seed * 100 + condition_index, bootstrap),
        "oracle_expert_accuracy_upper_bound": float((correct[:, 0] | correct[:, 1] | correct[:, 2]).mean()),
    }, selected.astype(float) - mixed.astype(float)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    source_summary_path = args.source / "summary.json"
    source_summary = json.loads(source_summary_path.read_text())
    frozen_manifest = args.source / "frozen_test_manifest.json"
    if sha256(frozen_manifest) != source_summary["configuration"]["test_manifest_sha256"]:
        raise ValueError("Frozen test manifest changed since Experiment 119")
    results = {}
    differences = {condition: [] for condition in CONDITIONS}
    for seed in (0, 1, 2):
        seed_dir = args.source / f"seed_{seed}"
        train = {condition: load_cache(seed_dir / f"train_{condition}.npz") for condition in CONDITIONS}
        scaler, models, _ = fit_selector(train)
        threshold = json.loads((seed_dir / "frozen_selector.json").read_text())["threshold"]
        results[str(seed)] = {}
        for condition_index, condition in enumerate(CONDITIONS):
            data = load_cache(seed_dir / f"test_{condition}.npz")
            choices = select_actions(action_scores(data["features"], scaler, models), threshold)
            audit, difference = audit_condition(data, choices, seed, condition_index, args.bootstrap)
            saved = source_summary["results"][str(seed)][condition]
            if abs(audit["selected_vs_baseline"]["candidate_accuracy"] - saved["selected_vs_base"]["candidate_accuracy"]) > 1e-12:
                raise ValueError(f"Reconstructed selector does not match frozen result: seed {seed} {condition}")
            if abs(audit["mixed_vs_baseline"]["candidate_accuracy"] - saved["mixed_vs_base"]["candidate_accuracy"]) > 1e-12:
                raise ValueError(f"Mixed-adapter correctness differs from saved result: seed {seed} {condition}")
            results[str(seed)][condition] = audit
            differences[condition].append(difference)

    aggregate = {}
    for condition_index, condition in enumerate(CONDITIONS):
        rows = [results[str(seed)][condition] for seed in (0, 1, 2)]
        by_image = np.stack(differences[condition]).mean(axis=0)
        aggregate[condition] = {
            "samples_per_seed": len(by_image),
            "mean_selected_minus_mixed_pp": float(100 * by_image.mean()),
            "image_clustered_95ci_pp": [100 * value for value in paired_interval(by_image, 123000 + condition_index, args.bootstrap)],
            "mean_no_repair_fraction": float(np.mean([row["counts"]["no_repair"] / row["counts"]["samples"] for row in rows])),
            "mean_mixed_recoveries_skipped": float(np.mean([row["counts"]["mixed_recoveries_skipped"] for row in rows])),
            "mean_mixed_damages_avoided": float(np.mean([row["counts"]["mixed_damages_avoided_by_no_repair"] for row in rows])),
            "mean_potential_expert_recoveries_missed": float(np.mean([row["counts"]["missed_expert_recovery_due_to_no_repair"] for row in rows])),
        }
    macro_difference = np.stack([np.stack(differences[name]).mean(axis=0) for name in CONDITIONS[1:]]).mean(axis=0)
    aggregate["macro_corruption"] = {
        "mean_selected_minus_mixed_pp": float(100 * macro_difference.mean()),
        "image_clustered_95ci_pp": [100 * value for value in paired_interval(macro_difference, 124000, args.bootstrap)],
    }
    summary = {
        "configuration": {
            "source_summary": str(source_summary_path),
            "source_summary_sha256": sha256(source_summary_path),
            "source_test_manifest_sha256": sha256(frozen_manifest),
            "bootstrap": args.bootstrap,
            "new_vit_inference": False,
            "new_training_or_threshold_selection": False,
            "three_seeds_are_not_treated_as_independent_image_samples": True,
        },
        "per_seed": results,
        "aggregate": aggregate,
        "definitions": {
            "missed_expert_recovery_due_to_no_repair": "Base wrong, at least one specialist correct, but selector chose no repair.",
            "mixed_recoveries_skipped": "Base wrong and mixed adapter correct, but selector chose no repair; counterfactual to always-on mixed.",
            "mixed_damages_avoided_by_no_repair": "Base correct and mixed adapter wrong, but selector chose no repair; counterfactual to always-on mixed.",
            "oracle_expert_accuracy_upper_bound": "Uses true labels to choose best of no-repair/Noise/Blur; not deployable.",
        },
        "limitations": [
            "This is post-hoc analysis of the already-used Experiment 119 test set; no model or threshold may be tuned on it.",
            "Per-seed exact McNemar tests use paired predictions; cross-seed intervals cluster by image index, not by treating seeds as independent samples.",
            "Correctness caches cannot show whether two incorrect actions predict the same wrong class.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    with (output_dir / "aggregate.csv").open("w", newline="") as destination:
        writer = csv.writer(destination)
        writer.writerow(["condition", "selected_minus_mixed_pp", "ci_low_pp", "ci_high_pp", "no_repair_fraction", "mixed_recoveries_skipped", "mixed_damages_avoided"])
        for condition in CONDITIONS:
            row = aggregate[condition]
            writer.writerow([
                condition, row["mean_selected_minus_mixed_pp"], *row["image_clustered_95ci_pp"],
                row["mean_no_repair_fraction"], row["mean_mixed_recoveries_skipped"], row["mean_mixed_damages_avoided"],
            ])
    for condition in CONDITIONS:
        row = aggregate[condition]
        print(condition, "selected-minus-mixed pp", round(row["mean_selected_minus_mixed_pp"], 2),
              "CI", [round(value, 2) for value in row["image_clustered_95ci_pp"]],
              "missed mixed recoveries", round(row["mean_mixed_recoveries_skipped"], 1),
              "avoided mixed damage", round(row["mean_mixed_damages_avoided"], 1))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
