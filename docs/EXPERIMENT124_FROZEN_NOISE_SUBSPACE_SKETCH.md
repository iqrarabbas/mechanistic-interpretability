# Experiment 124: Frozen Rank-128 Noise Repair on Fresh Sketch Images

## Protocol

Experiment 123 selected rank 128 on ImageNet development data. This experiment froze the rank, scale, factors, corruption severity, and all model components before independent-image evaluation.

A new manifest was constructed from ImageNet-Sketch by excluding every image path and SHA-256 hash recorded in 13 prior Sketch manifests. It contains 1,982 images from 991 still-eligible classes, with up to two images per class. Nine classes had no unused images remaining. The dataset/domain has been studied previously, so this is **fresh image-level confirmation**, not a pristine new-domain benchmark.

For each of three independently trained teacher seeds, the same images were evaluated with:

1. Frozen base ViT
2. Full frozen Block-6 adapter
3. Frozen standalone rank-128 adapter-delta PCA map
4. Rank-128 noisy-hidden PCA control
5. Rank-128 random-direction control

Both clean images and online project Gaussian Noise-4 were tested. The noisy image's clean counterpart was never used for repair.

## Results

### Noise-4

| Seed | Base | Full adapter | Rank-128 Delta PCA | Hidden-PCA control | Random control |
|---|---:|---:|---:|---:|---:|
| 0 | 27.19% | 29.11% (+1.92) | **29.36% (+2.17)** | 29.01% (+1.82) | 28.10% (+0.91) |
| 1 | 27.19% | 28.96% (+1.77) | **29.01% (+1.82)** | 28.76% (+1.56) | 27.40% (+0.20) |
| 2 | 27.19% | 28.76% (+1.56) | **29.06% (+1.87)** | 28.71% (+1.51) | 27.35% (+0.15) |
| Mean | 27.19% | 28.94% (+1.75) | **29.14% (+1.95)** | 28.82% (+1.63) | 27.61% (+0.42) |

Every Delta-PCA gain over baseline was individually significant: exact paired McNemar `p=0.000637`, `0.00296`, and `0.00416`. Its differences from the full adapter were only `+0.25`, `+0.05`, and `+0.30 pp`; all paired confidence intervals included zero. Therefore, the standalone rank-128 map **matches the full adapter on fresh images**.

Delta PCA exceeded random directions by `+1.26`, `+1.61`, and `+1.72 pp`; each seed was significant. It exceeded hidden PCA by only `+0.35`, `+0.25`, and `+0.35 pp`; none was significant. Thus fresh-image confirmation supports compression and non-randomness, but does not independently establish superiority over a generic high-variance hidden subspace.

### Clean control

| Seed | Base | Full adapter change | Rank-128 Delta-PCA change |
|---|---:|---:|---:|
| 0 | 35.72% | -0.10 pp | +0.05 pp |
| 1 | 35.72% | -0.25 pp | +0.25 pp |
| 2 | 35.72% | +0.71 pp | +0.71 pp |
| Mean | 35.72% | +0.12 pp | +0.34 pp |

No Delta-PCA clean change was individually significant. There is no evidence of clean-accuracy damage.

## Conclusion

The primary claim replicates:

> A standalone rank-128 patch-token map, distilled from the full Block-6 adapter's correction geometry, reproduces the adapter's Noise-4 robustness gain on newly excluded images without clean degradation.

The stronger mechanism-specific claim must remain qualified: Delta PCA clearly beats random directions, but hidden-PCA repair is also effective on this OOD domain and is not significantly worse. The result therefore validates compact repair, while the uniqueness of adapter-delta geometry requires a larger or different independent benchmark.

## Artifacts

- Script: `scripts/experiment124_frozen_noise_subspace_sketch.py`
- Summary: `results/sae/experiment124_frozen_noise_subspace_sketch/full_fresh_rank128_clean_noise4_3seed_v1/summary.json`
- Frozen manifest: `results/sae/experiment124_frozen_noise_subspace_sketch/full_fresh_rank128_clean_noise4_3seed_v1/frozen_manifest.json`
- Paired outcomes: `results/sae/experiment124_frozen_noise_subspace_sketch/full_fresh_rank128_clean_noise4_3seed_v1/paired_outcomes.npz`
