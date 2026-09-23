# Experiment 64: Audit of the Repeated 74.6% Block-1 MLP Share

## Reason for the audit

Experiment 64 reported that the Block-1 MLP accounted for 74.6% of the full clean-block oracle recovery for both Noise-4 and Blur-4. Because identical rounded values can indicate accidental result reuse, the source code, source summaries, and per-image outcome arrays were checked independently.

## What was checked

- The Experiment 64 aggregation reads a separate Experiment 63 `summary.json` for every corruption and block.
- The Shapley implementation uses the standard three-component factorial formula over incoming residual (`x`), attention output (`a`), and MLP output (`m`).
- Accuracy and Shapley values were recomputed directly from each corruption's `paired_outcomes.npz`, without using the stored aggregate values.
- Outcome-array hashes and elementwise equality were compared between Noise-4 and Blur-4.

## Independent recomputation

| Quantity | Noise-4 | Blur-4 |
|---|---:|---:|
| Images | 3,000 | 3,000 |
| Corrupted baseline accuracy | 72.5333% | 65.5667% |
| Full clean Block-1 restoration | 81.1333% | 81.1333% |
| Full oracle gain | 8.6000 pp | 15.5667 pp |
| Block-1 MLP Shapley contribution | 6.4167 pp | 11.6167 pp |
| MLP share of oracle gain | **74.6124%** | **74.6253%** |

The values are therefore not exactly equal. They become the same only when rounded to one decimal place.

## Array-independence check

| Outcome array | Noise/Blur identical? | Differing image outcomes |
|---|---:|---:|
| Corrupted baseline | No | 605 / 3,000 |
| Clean-MLP intervention | No | 342 / 3,000 |
| Full clean `x+a+m` restoration | Yes | 0 / 3,000 |

The identical full-restoration array is expected: both conditions replace the complete Block-1 decomposition with the same clean image's components, so both reproduce the same clean continuation and accuracy. The corrupted baselines and MLP-only intervention arrays are distinct and have different hashes, ruling out accidental reuse of those results.

## Conclusion

There is no evidence of a caching, denominator, aggregation, or outcome-reuse bug. The repeated **74.6%** is a rounding coincidence arising from two independently computed values: **74.6124% for Noise-4** and **74.6253% for Blur-4**.

The result should nevertheless be stated cautiously:

> Under the three-component factorial oracle decomposition, the Block-1 MLP accounts for approximately 74.6% of recoverable Block-1 corruption damage for both Noise-4 and Blur-4.

This does **not** prove that the Block-1 MLP is the first source of all corruption damage. The incoming residual already contains the patch-embedding output, which was not decomposed in Experiment 64. The defensible claim is that Block 1's MLP is the earliest dominant **Transformer-sublayer amplifier among the components tested**. A patch-embedding attribution experiment is still required for a complete origin claim.

## Source artifacts

- `/media/dr-yougart/Iqrar/vit_mi/results/sae/experiment63_block6_residual_mlp/full_noise4_block1_3000_v1/summary.json`
- `/media/dr-yougart/Iqrar/vit_mi/results/sae/experiment63_block6_residual_mlp/full_noise4_block1_3000_v1/paired_outcomes.npz`
- `/media/dr-yougart/Iqrar/vit_mi/results/sae/experiment63_block6_residual_mlp/full_blur4_block1_3000_v1/summary.json`
- `/media/dr-yougart/Iqrar/vit_mi/results/sae/experiment63_block6_residual_mlp/full_blur4_block1_3000_v1/paired_outcomes.npz`
