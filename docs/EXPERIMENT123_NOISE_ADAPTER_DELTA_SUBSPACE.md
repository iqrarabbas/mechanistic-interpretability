# Experiment 123: Compact Noise Repair from Adapter-Delta Geometry

## Question

Can mechanistic analysis compress the successful full Block-6 adapter into a much smaller, standalone Noise-repair target? The full adapter is used only as a development-time teacher. It is not needed by the saved low-rank maps at inference.

## Protocol

- Frozen `google/vit-base-patch16-224` and three independently trained mixed Noise+Blur Block-6 adapter seeds.
- For each seed, use 2,000 images from that seed's adapter-training partition to collect Noise-4 patch-token corrections `delta = adapter(h_noise)`.
- Compute the uncentered second moment of those corrections and its top PCA/SVD directions.
- Fold each output projection into a standalone map: `delta_rank(h) = (h WU + bU + position U) U^T`. This uses two low-rank matrix multiplications and does **not** execute the full adapter.
- Evaluate ranks 4, 8, 16, 32, 64, 128, and 256 on that seed's disjoint 2,000-image validation partition.
- Compare against rank-matched noisy-hidden-state PCA directions and seeded random orthonormal directions.
- Report clean accuracy, paired bootstrap intervals, and exact McNemar tests.

No labels are used to discover the subspace. No ImageNetV2 data are accessed.

## Main results

The full teacher gained `+3.55`, `+2.90`, and `+3.20 pp` Noise-4 across seeds (mean `+3.22 pp`).

| Method | Rank | Stored coefficients | Mean Noise-4 gain | Teacher gain retained | Mean clean change |
|---|---:|---:|---:|---:|---:|
| Delta PCA | 4 | 6,932 | +1.28 pp | 41.0% | -0.02 pp |
| Delta PCA | 8 | 13,864 | +1.52 pp | 47.9% | -0.03 pp |
| Delta PCA | 16 | 27,728 | +2.18 pp | 68.5% | +0.33 pp |
| Delta PCA | 32 | 55,456 | +2.35 pp | 73.7% | +0.27 pp |
| Delta PCA | 64 | 110,912 | +2.87 pp | 90.1% | +0.27 pp |
| **Delta PCA** | **128** | **221,824** | **+3.52 pp** | **110.0%** | **+0.37 pp** |
| Delta PCA | 256 | 443,648 | +3.20 pp | 100.0% | +0.32 pp |

Rank 128 gained `+3.75`, `+3.65`, and `+3.15 pp`; every seed was individually significant against baseline. Its pooled difference from the full teacher was `+0.30 pp`, but its paired CI included zero and McNemar `p=0.173`; therefore, claim **matching**, not statistically outperforming, the teacher.

At rank 128, Delta PCA beat hidden-state PCA by `+2.10 pp` (paired 95% CI `+1.47` to `+2.75 pp`, exact `p=8.65e-11`) and random directions by `+2.73 pp` (CI `+2.03` to `+3.43 pp`, `p=3.47e-14`) across the 6,000 disjoint validation images.

The teacher correction energy is highly concentrated: the first 4 directions contain approximately 93.6–93.7% of squared delta energy, but preserve only about 41% of accuracy gain. Rank 16 contains about 95.9–96.0% of energy yet preserves 68.5% of gain. Thus **correction energy alone does not determine causal classification utility**; lower-energy directions remain important.

## Interpretation

This supports a compact, distributed Noise mechanism:

> The useful Noise correction is not well described by a few sparse SAE features, but the full adapter's patch-token repair can be distilled into a rank-64/128 raw-hidden subspace that preserves most or all of its robustness benefit.

Rank 128 uses about 70% fewer stored coefficients than the 741,120-parameter full adapter and matches its mean development gain. This is also stronger than Experiment 54's independently trained rank-128 adapter (`+2.25 pp`, clean `-0.40 pp`), showing that extracting the teacher's causal correction geometry is more effective than retraining the same nominal rank from scratch.

## Leakage and limitations

- Discovery and evaluation images are disjoint for every seed.
- The three seed partitions are mutually disjoint under `configs/split_manifest_supervisor_v1.json`.
- However, the validation partitions selected the original teacher checkpoints. This is valid development/distillation evidence, but not an untouched final confirmation.
- The subspace is teacher-derived, so the result demonstrates **teacher compression and causal mechanism isolation**, not discovery without an adapter.
- Rank 128 was selected after viewing this sweep. It must now be frozen and evaluated once on an independent dataset/range; ImageNetV2 must not be reused for selection.

## Artifacts

- Script: `scripts/experiment123_noise_adapter_delta_subspace.py`
- Summary: `results/sae/experiment123_noise_adapter_delta_subspace/full_3seed_ranks4_256_v1/summary.json`
- Paired outcomes: `results/sae/experiment123_noise_adapter_delta_subspace/full_3seed_ranks4_256_v1/paired_outcomes.npz`
- Deployable factors: each `seed_N/delta_pca_folded_factors.pt`
