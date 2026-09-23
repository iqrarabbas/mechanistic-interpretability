import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
TRAINING_ROOT = ACTIVE_ROOT / "results/sae/experiment84_block6_clean_sae_calibration"
DIAGNOSTIC_ROOT = ACTIVE_ROOT / "results/sae/experiment85_block6_batchtopk_diagnostics"
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment86_batchtopk_seed_stability"
CATEGORIES = (
    "top_strengthened",
    "top_weakened",
    "top_absolute_change",
    "top_standardized_abnormality",
    "top_entering_dominant_set",
    "top_leaving_dominant_set",
    "top_failure_specific_change",
    "top_margin_harm_correlation",
)


def atomic_json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def checkpoint(seed):
    return TRAINING_ROOT / f"full_seed{seed}_1000train_200val_3epoch_v1/batchtopk_lambda_1em03/model.pt"


def diagnostic(seed):
    return DIAGNOSTIC_ROOT / f"full_seed{seed}_noise4_blur4_corrected_v1/summary.json"


def load_decoder(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    decoder = state["decoder.weight"].float()
    return F.normalize(decoder, dim=0).to(device)


def nearest_directions(reference_decoder, target_decoder, indices, chunk_size=32):
    matched_indices = []
    similarities = []
    for chunk in torch.as_tensor(indices, device=reference_decoder.device).split(chunk_size):
        similarity = reference_decoder[:, chunk].T @ target_decoder
        values, matches = similarity.max(dim=1)
        matched_indices.extend(matches.cpu().tolist())
        similarities.extend(values.cpu().tolist())
    return np.asarray(matched_indices), np.asarray(similarities)


def bootstrap_mean(values, seed, replicates=10000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def empirical_test(observed, controls):
    observed = float(observed)
    controls = np.asarray(controls, dtype=np.float64)
    return {
        "observed": observed,
        "random_mean": float(controls.mean()),
        "random_ci95": [float(np.quantile(controls, 0.025)), float(np.quantile(controls, 0.975))],
        "empirical_p_greater_equal": float((1 + np.sum(controls >= observed)) / (len(controls) + 1)),
        "controls": int(len(controls)),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 86: cross-seed BatchTopK decoder-direction stability")
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--random-controls", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=8600)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    checkpoints = [checkpoint(seed) for seed in range(3)]
    diagnostics = [json.loads(diagnostic(seed).read_text())["results"] for seed in range(3)]
    decoders = [load_decoder(path, device) for path in checkpoints]
    latent_dim = decoders[0].shape[1]
    generator = np.random.default_rng(args.seed)
    random_reference_features = generator.choice(
        latent_dim, size=args.random_controls, replace=False
    )
    random_match_pool = {}
    for target_seed in (1, 2):
        random_match_pool[target_seed] = nearest_directions(
            decoders[0], decoders[target_seed], random_reference_features
        )
    results = {}
    for corruption in ("noise", "blur"):
        results[corruption] = {}
        for category_index, category in enumerate(CATEGORIES):
            reference_rows = diagnostics[0][corruption][category][: args.top_n]
            reference_features = np.asarray([row["feature"] for row in reference_rows])
            seed_results = {}
            matches_by_seed = {}
            for target_seed in (1, 2):
                matches, similarities = nearest_directions(
                    decoders[0], decoders[target_seed], reference_features
                )
                target_rows = diagnostics[target_seed][corruption][category][: args.top_n]
                target_rank = {row["feature"]: rank + 1 for rank, row in enumerate(target_rows)}
                membership = np.asarray([int(feature) in target_rank for feature in matches])
                matched_ranks = [target_rank.get(int(feature)) for feature in matches]
                pool_matches, pool_similarities = random_match_pool[target_seed]
                pool_membership = np.asarray(
                    [int(feature) in target_rank for feature in pool_matches],
                    dtype=np.float64,
                )
                control_indices = generator.integers(
                    0,
                    args.random_controls,
                    size=(args.random_controls, args.top_n),
                )
                random_membership_rates = pool_membership[control_indices].mean(1)
                random_similarity_means = pool_similarities[control_indices].mean(1)
                seed_results[f"seed0_to_seed{target_seed}"] = {
                    "decoder_cosine": bootstrap_mean(similarities, args.seed + target_seed + category_index * 10),
                    "matched_top_n_fraction": empirical_test(membership.mean(), random_membership_rates),
                    "matched_top_n_count": int(membership.sum()),
                    "matched_target_ranks": matched_ranks,
                    "decoder_cosine_vs_random_features": empirical_test(similarities.mean(), random_similarity_means),
                    "matches": [
                        {
                            "seed0_feature": int(source),
                            "target_feature": int(target),
                            "decoder_cosine": float(similarity),
                            "target_category_rank": target_rank.get(int(target)),
                        }
                        for source, target, similarity in zip(reference_features, matches, similarities)
                    ],
                }
                matches_by_seed[target_seed] = set(matches[membership.astype(bool)].tolist())
            seed1_match = nearest_directions(decoders[0], decoders[1], reference_features)[0]
            seed2_match = nearest_directions(decoders[0], decoders[2], reference_features)[0]
            seed1_top = {row["feature"] for row in diagnostics[1][corruption][category][: args.top_n]}
            seed2_top = {row["feature"] for row in diagnostics[2][corruption][category][: args.top_n]}
            triple = np.asarray([
                int(first) in seed1_top and int(second) in seed2_top
                for first, second in zip(seed1_match, seed2_match)
            ])
            chance = (args.top_n / latent_dim) ** 2
            results[corruption][category] = {
                "top_n": args.top_n,
                "reference_seed": 0,
                "pairwise": seed_results,
                "three_seed_reappearance_count": int(triple.sum()),
                "three_seed_reappearance_fraction": float(triple.mean()),
                "independent_uniform_chance_fraction": float(chance),
            }
            print(
                corruption,
                category,
                "three-seed=",
                int(triple.sum()),
                "/",
                args.top_n,
            )
    summary = {
        "configuration": vars(args) | {
            "device": str(device),
            "latent_dimension": latent_dim,
            "sae_checkpoints": [str(path) for path in checkpoints],
            "diagnostic_summaries": [str(diagnostic(seed)) for seed in range(3)],
            "matching": "maximum positive cosine similarity of normalized decoder columns; seed 0 to every target-seed feature",
        },
        "results": results,
        "interpretation_guardrails": [
            "SAE feature IDs are never compared directly across seeds.",
            "Direction matching searches the complete target dictionary, not only selected features.",
            "Matched-direction reappearance in the same diagnostic top-N is compared with 1000 random seed-0 feature controls.",
            "This is a development stability analysis on [11000,12000), not held-out confirmation.",
            "Nearest-neighbor matching is not one-to-one; a future Hungarian assignment can be restricted to the reproducible candidate pool.",
        ],
    }
    atomic_json_write(output_dir / "summary.json", summary)
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
