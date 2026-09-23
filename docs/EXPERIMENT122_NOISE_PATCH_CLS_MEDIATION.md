# Experiment 122: Where the Block-6 Noise Repair Travels

## Question and protocol

Does the successful **full mixed Noise+Blur Block-6 adapter** send its Noise-4 benefit through patch tokens, the CLS token, or both? This is not a test of the weaker Block-6 SAE-selected subspace and does not train a new adapter.

The frozen `google/vit-base-patch16-224` and three independently trained, frozen mixed adapters were evaluated on the same 1,000 ImageNet validation images `[38000,39000)` with online Noise-4 and a clean control. Each adapter seed used its own deterministic noise seed. At every stage after Blocks 6–12, the experiment formed two same-image hybrids: corrected representation with the **baseline CLS** restored, and corrected representation with the **baseline patch tokens** restored. It ran the remaining frozen blocks and compared final top-1 outcomes against the fully adapted and baseline passes. No clean counterpart is used to repair a noisy image. The clean run is a separate control condition.

The code is `scripts/experiment122_noise_patch_cls_mediation.py`; paired outcomes and exact McNemar/paired-bootstrap comparisons are in `results/sae/experiment122_noise_patch_cls_mediation/full_3seed_noise_clean_1000_v1/`. Manual replay and Block-6 identity checks had zero maximum logit/state error in all six seed/condition runs.

## Results

| Adapter seed | Noise-4 baseline | Noise-4 adapter | Gain | Clean baseline | Clean adapter | Change |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 71.1% | 74.7% | +3.6 pp | 81.1% | 80.7% | -0.4 pp |
| 1 | 70.1% | 72.9% | +2.8 pp | 81.1% | 80.6% | -0.5 pp |
| 2 | 71.3% | 73.9% | +2.6 pp | 81.1% | 80.7% | -0.4 pp |

Mean Noise-4 gain was **+3.0 pp**. Exact paired McNemar p-values for the full adapter versus baseline were `8.7e-5`, `0.0046`, and `0.0103` across seeds.

| Stage | Mean gain lost when CLS reverted | Mean gain lost when patches reverted |
|---|---:|---:|
| 6 | 0.00 pp | 3.00 pp |
| 7 | 0.03 pp | 3.00 pp |
| 8 | 0.20 pp | 2.80 pp |
| 9 | 0.30 pp | 2.67 pp |
| 10 | 0.40 pp | 2.87 pp |
| 11 | 0.57 pp | 1.73 pp |
| 12 | 3.00 pp | 0.00 pp |

At Block 10, reverting the patches removed `3.2`, `3.2`, and `2.2 pp` across seeds (exact paired p-values `0.00053`, `0.00085`, and `0.028`). Block-11 patch-reversion costs were `2.0`, `2.3`, and `0.9 pp`, so the last seed was not individually significant. Block-11 CLS-reversion costs were `1.4`, `-0.2`, and `0.5 pp`, also inconsistent across seeds.

## Interpretation and limits

The correction is injected into patch tokens after Block 6 and its useful effect remains substantially patch-carried through Block 10. Later computation transfers the effect toward the final CLS prediction. Stage 12 is **not** independent evidence of a special circuit: the classifier reads only CLS, so CLS reversion must erase the final gain and patch reversion cannot change the already-computed logits. Patch and CLS reversion effects are not additive because the Transformer is nonlinear and the hybrids can be off-manifold. This identifies conditional dependence of the adapter benefit, **not** a unique head/neuron circuit or proof that Block 6 is optimal for a narrow SAE repair.

The 1,000-image range was used in prior mechanistic analyses; these are exploratory/replication findings, not pristine final-test claims. Adapter training/validation splits are disjoint from this range under `configs/split_manifest_supervisor_v1.json`, but use of this range for previous research decisions limits independent-confirmation language. Do not tune architecture on these results and then report the same range as final evaluation.
