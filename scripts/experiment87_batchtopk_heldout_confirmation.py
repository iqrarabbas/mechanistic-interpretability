import argparse
import json
from pathlib import Path

import numpy as np


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
DIAGNOSTIC_ROOT = ACTIVE_ROOT / "results/sae/experiment85_block6_batchtopk_diagnostics"
STABILITY_SUMMARY = ACTIVE_ROOT / "results/sae/experiment86_batchtopk_seed_stability/full_top100_1000controls_v1/summary.json"
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment87_batchtopk_heldout_confirmation"
LATENT_DIM = 24576


def atomic_json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def empirical_test(observed, controls):
    controls = np.asarray(controls, dtype=np.float64)
    return {
        "observed": float(observed),
        "random_mean": float(controls.mean()),
        "random_ci95": [float(np.quantile(controls, 0.025)), float(np.quantile(controls, 0.975))],
        "empirical_p_greater_equal": float((1 + np.sum(controls >= observed)) / (len(controls) + 1)),
        "controls": int(len(controls)),
    }


def diagnostic(seed, split):
    name = f"full_seed{seed}_noise4_blur4_corrected_v1"
    if split == "confirmation":
        name = f"confirmation_seed{seed}_noise4_blur4_frozen_v1"
    return DIAGNOSTIC_ROOT / name / "summary.json"


def main():
    parser = argparse.ArgumentParser(description="Experiment 87: frozen SAE-direction confirmation")
    parser.add_argument("--controls", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8700)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    stability = json.loads(STABILITY_SUMMARY.read_text())
    confirmation = [json.loads(diagnostic(seed, "confirmation").read_text())["results"] for seed in range(3)]
    generator = np.random.default_rng(args.seed)
    results = {}
    for corruption, categories in stability["results"].items():
        results[corruption] = {}
        for category, development in categories.items():
            seed1_matches = development["pairwise"]["seed0_to_seed1"]["matches"]
            seed2_matches = development["pairwise"]["seed0_to_seed2"]["matches"]
            triplets = []
            for first, second in zip(seed1_matches, seed2_matches):
                if first["target_category_rank"] is None or second["target_category_rank"] is None:
                    continue
                if first["seed0_feature"] != second["seed0_feature"]:
                    raise RuntimeError("Experiment 86 match ordering differs")
                triplets.append(
                    (
                        first["seed0_feature"],
                        first["target_feature"],
                        second["target_feature"],
                    )
                )
            confirmation_sets = [
                {row["feature"] for row in confirmation[seed][corruption][category]}
                for seed in range(3)
            ]
            memberships = np.asarray(
                [[feature in confirmation_sets[seed] for seed, feature in enumerate(triplet)] for triplet in triplets],
                dtype=bool,
            )
            observed_all_three = int(memberships.all(1).sum()) if len(triplets) else 0
            observed_any = int(memberships.any(1).sum()) if len(triplets) else 0
            random_all_three = []
            random_any = []
            count = len(triplets)
            for _ in range(args.controls):
                random_memberships = np.column_stack(
                    [
                        np.isin(generator.integers(0, LATENT_DIM, size=count), list(confirmation_sets[seed]))
                        for seed in range(3)
                    ]
                )
                random_all_three.append(int(random_memberships.all(1).sum()))
                random_any.append(int(random_memberships.any(1).sum()))
            results[corruption][category] = {
                "frozen_development_triplets": count,
                "heldout_reappearance_per_seed": memberships.sum(0).tolist() if count else [0, 0, 0],
                "heldout_reappearance_per_seed_fraction": memberships.mean(0).tolist() if count else [0.0, 0.0, 0.0],
                "heldout_any_seed": empirical_test(observed_any, random_any),
                "heldout_all_three_seeds": empirical_test(observed_all_three, random_all_three),
                "heldout_all_three_fraction": float(observed_all_three / max(count, 1)),
                "frozen_triplets": [list(map(int, triplet)) for triplet in triplets],
            }
            print(corruption, category, f"all-three {observed_all_three}/{count}")
    summary = {
        "configuration": vars(args) | {
            "development_stability_summary": str(STABILITY_SUMMARY),
            "confirmation_summaries": [str(diagnostic(seed, "confirmation")) for seed in range(3)],
            "development_range": [11000, 12000],
            "confirmation_range": [12000, 13000],
            "feature_selection_frozen": True,
            "top_n": 100,
        },
        "results": results,
        "guardrails": [
            "Only direction triplets selected by Experiment 86 on [11000,12000) are tested.",
            "No feature IDs, thresholds, or categories are reselected on confirmation images.",
            "The confirmation images [12000,13000) were not used for SAE training, calibration, clean-reference statistics, or development ranking.",
            "Random controls use the same number of directions and the frozen confirmation top-100 set sizes.",
            "ImageNetV2 and ImageNet-Sketch are not accessed.",
        ],
    }
    atomic_json_write(output_dir / "summary.json", summary)
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
