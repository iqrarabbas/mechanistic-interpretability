# Experiment 125: Frozen Rank-128 Noise Severity Sweep

## Audit and protocol

An initial run was stopped after detecting that the manifest scanner excluded mounted historical results but omitted this repository's local result manifests. That interrupted run is invalid and is not reported. The scanner was corrected before the valid run to exclude both locations by path and SHA-256.

The valid manifest contains 990 newly excluded ImageNet-Sketch images from 990 eligible classes and excludes 17 prior manifests. Rank 128, all factors, scales, seeds, and severities were frozen before inference. No method selection or tuning used these images.

The frozen base ViT, full Block-6 adapter, rank-128 adapter-delta PCA map, rank-128 hidden-PCA control, and rank-128 random control were evaluated on clean images and project Gaussian Noise severities 1–5 across three adapter seeds.

## Mean accuracy changes

| Condition | Base accuracy | Full adapter | Delta PCA | Hidden PCA | Random | Delta minus hidden |
|---|---:|---:|---:|---:|---:|---:|
| Clean | 36.46% | +0.14 pp | -0.10 pp | +0.77 pp | +0.37 pp | -0.88 pp |
| Noise-1 | 33.23% | +0.54 pp | +0.84 pp | +1.28 pp | +0.20 pp | -0.44 pp |
| Noise-2 | 31.62% | +0.81 pp | +0.88 pp | +1.11 pp | +0.40 pp | -0.24 pp |
| Noise-3 | 30.81% | +0.77 pp | +0.51 pp | +1.15 pp | +0.27 pp | -0.64 pp |
| Noise-4 | 29.70% | +0.91 pp | +0.88 pp | +0.51 pp | -0.14 pp | +0.37 pp |
| Noise-5 | 27.47% | +1.99 pp | +1.68 pp | +1.45 pp | +0.88 pp | +0.24 pp |

Delta-PCA gains were positive at every Noise severity and increased to a mean `+1.68 pp` at Noise-5. The three Noise-5 gains were `+1.41`, `+2.12`, and `+1.52 pp`; only seed 1 was individually significant on this 990-image set. Clean changes were `-0.30`, `+0.20`, and `-0.20 pp`, with every confidence interval crossing zero.

## Interpretation

The preregistered specificity question has a negative answer:

> Adapter-delta PCA does not consistently outperform rank-matched hidden-state PCA as Noise severity increases.

Delta PCA was below hidden PCA at severities 1–3 and above it at severities 4–5, but no seed-level Delta-versus-hidden comparison was significant at any severity. Therefore, current evidence does not justify claiming that the adapter teacher exposes a uniquely superior Noise subspace on OOD Sketch data.

The defensible result is narrower:

> Block-6 Noise repair is compressible into a frozen rank-128 patch-space map that generalizes across severity and remains clean-safe, but a generic high-variance Block-6 subspace can be similarly effective.

This differs from Blur, where SAE-selected directions beat matched controls. Noise appears to admit a broader distributed low-rank correction rather than a uniquely sparse or teacher-specific mechanism.

## Artifacts

- Script: `scripts/experiment125_frozen_noise_severity_sweep.py`
- Valid summary: `results/sae/experiment125_frozen_noise_severity_sweep/full_fresh_rank128_severity1_5_3seed_v2_strict_local_exclusion/summary.json`
- Valid manifest: `results/sae/experiment125_frozen_noise_severity_sweep/full_fresh_rank128_severity1_5_3seed_v2_strict_local_exclusion/frozen_manifest.json`
- Interrupted invalid run: `results/sae/experiment125_frozen_noise_severity_sweep/full_fresh_rank128_severity1_5_3seed_v1/` (must not be reported)
