# Experiment 128: Frozen Shared Rank-32 Confirmation

## Protocol

Experiment 127 selected a shared rank-32 Block-6 repair map using ImageNet development data. Experiment 128 froze the rank, factors, scale, corruption severity, methods, and seeds before evaluation.

A new ImageNet-Sketch manifest was created after excluding paths and SHA-256 hashes from 19 prior manifests. It contains 990 images from 990 still-eligible classes. The ImageNet-Sketch domain has been studied previously; only the individual images are newly excluded.

The following frozen methods were evaluated on clean, impulse noise, pixelation, JPEG, and brightness:

- Base ViT
- Full mixed adapter
- Shared rank-32 repair
- Hidden-PCA rank-32 control
- Random rank-32 control

No clean counterpart, corruption label, parameter update, rank selection, or threshold tuning was used at inference.

## Results

Mean changes across three seed-specific maps:

| Condition | Base accuracy | Full adapter | Shared rank-32 | Hidden-PCA | Random |
|---|---:|---:|---:|---:|---:|
| Clean | 35.86% | -0.10 pp | -0.37 pp | +0.14 pp | -0.27 pp |
| Impulse noise | 31.92% | +1.55 pp | **+0.88 pp** | +1.18 pp | -0.10 pp |
| Pixelation | 34.14% | +1.41 pp | **+0.51 pp** | +0.44 pp | -0.10 pp |
| JPEG | 35.15% | -0.44 pp | **+0.47 pp** | +0.40 pp | -0.13 pp |
| Brightness | 35.45% | +0.47 pp | **+0.84 pp** | +0.81 pp | +0.10 pp |

Across the four held-out families, shared rank-32 gained an average `+0.67 pp`, approximately 90% of the full adapter's `+0.75 pp` average. Random directions averaged approximately `-0.06 pp`. Hidden-PCA averaged `+0.71 pp`, essentially matching shared rank-32.

Shared rank-32 improved every held-out family for every seed. Impulse-noise seed 0 was individually significant (`+1.41 pp`, exact `p=0.0201`); other per-seed confidence intervals generally crossed zero because only 990 images remained.

Clean changes were `-0.20`, `-0.61`, and `-0.30 pp`; all were nonsignificant. There is no statistically established clean degradation, though the consistent negative direction should be reported rather than described as clean improvement.

## Replication decision

The broad-transfer claim replicated:

> A frozen shared rank-32 map remains positive across four held-out corruption families on newly excluded images and retains most of the full adapter's average OOD gain while strongly outperforming random directions on average.

The stronger pixelation-specific claim did **not** replicate. Pixelation gains remained positive (`+0.40`, `+0.30`, and `+0.81 pp`) but were not individually significant. Differences from hidden-PCA were `+0.10`, `-0.30`, and `+0.40 pp`, all nonsignificant. Therefore, Experiment 127's significant pixelation advantage over hidden-PCA should be treated as development evidence, not a confirmed independent effect.

The independent result also confirms that shared rank-32 is not uniquely superior to a generic hidden-PCA repair. Their average OOD performance was nearly identical. The correct scientific conclusion is compact common repair, not uniqueness of the shared estimator.

## Final interpretation

The most defensible statement is:

> A compact rank-32 Block-6 correction map transfers positively across multiple unseen corruption families and retains approximately 90% of a much larger adapter's average OOD benefit. However, an equally sized high-variance hidden-space map performs similarly, so the current evidence supports a broad low-rank repair geometry rather than a uniquely adapter-derived causal basis.

This remains useful mechanistically: Block 6 is the repair location, patch tokens are the intervention target, rank 32 is sufficient for broad OOD transfer, and Experiment 126 identifies late Blocks 11–12 attention as the downstream integration bottleneck.

## Artifacts

- Script: `scripts/experiment128_frozen_shared32_sketch_confirmation.py`
- Summary: `results/sae/experiment128_frozen_shared32_sketch/full_fresh_shared32_4heldout_3seed_v1/summary.json`
- Frozen manifest: `results/sae/experiment128_frozen_shared32_sketch/full_fresh_shared32_4heldout_3seed_v1/frozen_manifest.json`
- Paired outcomes: `results/sae/experiment128_frozen_shared32_sketch/full_fresh_shared32_4heldout_3seed_v1/paired_outcomes.npz`
