# Experiment 120: Why the abstaining selector misses repairs

Experiment 119 tested a frozen three-way decision: no repair, Noise expert, or Blur expert. This audit uses only its saved per-image correctness and feature caches. It performs no new ViT inference, expert training, threshold selection, or test-set tuning. The original thresholds are reconstructed and the original accuracies are verified before counting errors.

## Main result

The selector's modest gains over always-on mixed repair on Clean, JPEG, and Contrast are **not statistically established** by this paired test. Every per-condition image-clustered 95% confidence interval for selector minus mixed includes zero. Across the seven corruptions, the mean difference is **−0.13 percentage points**, with an image-clustered 95% interval of **[−0.74, +0.46] pp**. The three adapter seeds share the same images and are not counted as independent image samples.

| Condition | Selector − mixed (pp) | 95% CI (pp) | Mixed recoveries skipped by no repair | Mixed damages avoided by no repair |
| --- | ---: | ---: | ---: | ---: |
| Clean | +0.34 | [−0.94, +1.55] | 20.7 | 21.7 |
| Shot noise | −0.94 | [−2.15, +0.20] | 19.7 | 19.0 |
| Impulse noise | −0.10 | [−1.35, +1.11] | 25.3 | 25.3 |
| Defocus blur | −0.61 | [−1.75, +0.54] | 16.0 | 12.0 |
| JPEG | +0.98 | [−0.34, +2.22] | 23.0 | 27.3 |
| Pixelate | −0.17 | [−1.28, +0.91] | 20.3 | 17.3 |
| Brightness | −0.87 | [−2.02, +0.24] | 22.3 | 17.7 |
| Contrast | +0.81 | [−0.17, +1.85] | 15.7 | 17.0 |

Counts are **mean images per seed out of 991**, not percentages. A “mixed recovery skipped” is a base-wrong image that the mixed adapter gets right but the selector leaves unchanged. An “avoided mixed damage” is a base-correct image that mixed repair gets wrong but the selector leaves unchanged. These are counterfactual comparisons to the *separate mixed adapter*; they do not imply that either specialist expert would have made the same prediction.

The selector often abstains, but that single decision both prevents damage and forfeits useful corrections. The two effects are similar in size, and choosing the wrong specialist sometimes adds further errors. This is why high abstention alone is not evidence of a useful router.

## Decision

Do not retune a threshold or claim a selector improvement on these 991 already-used test images. The current data support a **negative/uncertain selector result**, not a superior routing method. Keep the always-on mixed adapter as the simple reference. A new selector should be developed on separate data and frozen before any further confirmation set; otherwise stop adding routing complexity.

## Reproduction and caveats

- Script: `scripts/experiment120_selector_error_audit.py`.
- Saved audit: `/media/dr-yougart/Iqrar/vit_mi/results/sae/experiment120_selector_error_audit/full_3seed_paired_error_audit_v1/summary.json` and `aggregate.csv`.
- Source: Experiment 119's frozen three-seed run and its 991-image SHA-disjoint Sketch test manifest.
- Exact paired McNemar p-values are reported per seed and condition in `summary.json`; bootstrap intervals resample image indices after averaging differences across seeds. The seven corruptions reuse the same source images, so the macro interval also resamples image indices rather than treating conditions as independent.
- Cached correctness cannot reveal whether two incorrect methods predicted the *same* wrong class. The “best of three” expert accuracy in `summary.json` is a label-dependent diagnostic upper bound, not an inference-time method.
