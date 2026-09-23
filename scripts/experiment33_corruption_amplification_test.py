import argparse
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).parent.parent
INPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment32_multirandom_subspaces"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment33_corruption_amplification"
DEFAULT_RUNS = {
    "clean": INPUT_ROOT / "clean_noise_features_3seeds_20controls" / "paired_outcomes.npz",
    "noise4": INPUT_ROOT / "full_3seeds_20controls" / "paired_outcomes.npz",
    "blur4": INPUT_ROOT / "blur4_noise_features_3seeds_20controls" / "paired_outcomes.npz",
}


def removal_damage(full, removed):
    return full.astype(np.int8) - removed.astype(np.int8)


def bootstrap_interval(values, repetitions, seed):
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    for index in range(repetitions):
        sample = generator.integers(0, values.size, values.size)
        means[index] = values[sample].mean()
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def transition_counts(full, removed):
    return {
        "full_correct_to_removed_wrong": int((full & ~removed).sum()),
        "full_wrong_to_removed_correct": int((~full & removed).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 33: paired corruption-amplification test")
    parser.add_argument("--clean", type=Path, default=DEFAULT_RUNS["clean"])
    parser.add_argument("--noise", type=Path, default=DEFAULT_RUNS["noise4"])
    parser.add_argument("--blur", type=Path, default=DEFAULT_RUNS["blur4"])
    parser.add_argument("--adapter-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    paths = {"clean": args.clean, "noise4": args.noise, "blur4": args.blur}
    arrays = {name: np.load(path) for name, path in paths.items()}
    sample_counts = {
        arrays[condition][f"seed{seed}_full"].size
        for condition in arrays for seed in args.adapter_seeds
    }
    if len(sample_counts) != 1:
        raise ValueError(f"Outcome arrays are not aligned in length: {sample_counts}")

    results = {}
    saved = {}
    for condition in ["noise4", "blur4"]:
        per_seed = {}
        harmful_interactions = []
        null_interactions = []
        for adapter_seed in args.adapter_seeds:
            clean_full = arrays["clean"][f"seed{adapter_seed}_full"].astype(bool)
            corrupt_full = arrays[condition][f"seed{adapter_seed}_full"].astype(bool)
            clean_harmful = arrays["clean"][f"seed{adapter_seed}_harmful_removed"].astype(bool)
            corrupt_harmful = arrays[condition][f"seed{adapter_seed}_harmful_removed"].astype(bool)
            clean_damage = removal_damage(clean_full, clean_harmful)
            corrupt_damage = removal_damage(corrupt_full, corrupt_harmful)
            interaction = corrupt_damage - clean_damage
            harmful_interactions.append(interaction)

            clean_random = arrays["clean"][f"seed{adapter_seed}_energy_random"].astype(bool)
            corrupt_random = arrays[condition][f"seed{adapter_seed}_energy_random"].astype(bool)
            seed_null = []
            for corrupt_index in range(corrupt_random.shape[0]):
                corrupt_random_damage = removal_damage(corrupt_full, corrupt_random[corrupt_index])
                for clean_index in range(clean_random.shape[0]):
                    clean_random_damage = removal_damage(clean_full, clean_random[clean_index])
                    seed_null.append(float((corrupt_random_damage - clean_random_damage).mean()))
            seed_null = np.asarray(seed_null)
            null_interactions.extend(seed_null.tolist())
            harmful_mean = float(interaction.mean())
            per_seed[str(adapter_seed)] = {
                "clean_harmful_removal_cost": float(clean_damage.mean()),
                "corrupt_harmful_removal_cost": float(corrupt_damage.mean()),
                "corruption_amplification": harmful_mean,
                "amplification_95ci": bootstrap_interval(
                    interaction, args.bootstrap_repetitions, args.seed + adapter_seed
                ),
                "random_all_pairs_null_count": int(seed_null.size),
                "random_null_mean": float(seed_null.mean()),
                "random_null_95percent_range": [
                    float(value) for value in np.quantile(seed_null, [0.025, 0.975])
                ],
                "empirical_one_sided_pvalue": float(
                    (1 + np.sum(seed_null >= harmful_mean)) / (seed_null.size + 1)
                ),
                "clean_transitions": transition_counts(clean_full, clean_harmful),
                "corrupt_transitions": transition_counts(corrupt_full, corrupt_harmful),
            }

        harmful_interactions = np.stack(harmful_interactions)
        pooled_per_image = harmful_interactions.mean(0)
        null_interactions = np.asarray(null_interactions)
        pooled_mean = float(pooled_per_image.mean())
        results[condition] = {
            "per_seed": per_seed,
            "pooled": {
                "mean_corruption_amplification": pooled_mean,
                "amplification_95ci": bootstrap_interval(
                    pooled_per_image, args.bootstrap_repetitions, args.seed + 100
                ),
                "positive_for_every_adapter_seed": bool(
                    all(record["corruption_amplification"] > 0 for record in per_seed.values())
                ),
                "all_pairs_random_null_count": int(null_interactions.size),
                "all_pairs_random_null_mean": float(null_interactions.mean()),
                "all_pairs_random_null_95percent_range": [
                    float(value) for value in np.quantile(null_interactions, [0.025, 0.975])
                ],
                "empirical_one_sided_pvalue": float(
                    (1 + np.sum(null_interactions >= pooled_mean)) / (null_interactions.size + 1)
                ),
                "fraction_random_null_below_harmful": float(
                    np.mean(null_interactions < pooled_mean)
                ),
            },
        }
        saved[f"{condition}_harmful_interactions"] = harmful_interactions
        saved[f"{condition}_random_null_means"] = null_interactions

    summary = {
        "configuration": {
            "inputs": {name: str(path.resolve()) for name, path in paths.items()},
            "adapter_seeds": args.adapter_seeds,
            "samples": sample_counts.pop(),
            "bootstrap_repetitions": args.bootstrap_repetitions,
            "seed": args.seed,
            "random_null": "all 20x20 pairs of independently sampled clean and corrupt energy-matched random subspaces per adapter seed",
        },
        "results": results,
        "decision_rule": "A positive paired interaction with CI excluding zero and an extreme random-null percentile supports disproportionate corruption dependence.",
    }
    np.savez_compressed(output_dir / "interaction_arrays.npz", **saved)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved Experiment 33 to {output_dir}")


if __name__ == "__main__":
    main()
