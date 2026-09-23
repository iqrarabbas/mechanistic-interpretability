# Experiment 127: Shared OOD Repair Subspace

## Question

Can several corruption families reveal one compact Block-6 repair subspace that transfers without retraining to entirely held-out corruption families?

## Protocol

For each of three frozen mixed-adapter seeds, five severity-4 development families were used without labels:

- Gaussian noise
- Gaussian blur
- Shot noise
- Defocus blur
- Contrast

For each family, the top-128 output directions of the adapter's Block-6 patch-token correction were estimated on 1,000 images from that seed's training partition. The shared score was the average of the five rank-128 projection matrices. Its top 32 directions defined one standalone rank-32 repair map.

The map was frozen and evaluated on 2,000 disjoint validation images from four families absent from subspace discovery:

- Impulse noise
- Pixelation
- JPEG compression
- Brightness

Controls were the full adapter, pooled correction PCA, noisy-hidden PCA, random directions, Noise-specific correction PCA, and Blur-specific correction PCA. Clean accuracy was included. No ImageNetV2 or ImageNet-Sketch data were accessed.

Gaussian blur and the custom defocus implementation both reduce to PIL Gaussian blur with radius 4 at this severity, so they are not fully independent discovery transforms and effectively give blur-family behavior additional weight. This must be stated when reporting the protocol.

## Shared geometry

The top sharedness eigenvalues were approximately `0.993–1.000` across all three seeds. The rank-32 shared basis had normalized projection overlap:

- Gaussian noise: approximately 0.995
- Gaussian blur: approximately 0.997
- Shot noise: approximately 0.997
- Defocus blur: approximately 0.997
- Contrast: approximately 0.994

Therefore, the mixed adapter does not generate unrelated correction spaces for each corruption. Its useful outputs occupy an almost identical common low-rank geometry.

## Held-out-family results

Mean accuracy changes across three seeds:

| Condition | Full adapter | Shared rank-32 | Pooled PCA | Hidden PCA | Random | Noise-specific | Blur-specific |
|---|---:|---:|---:|---:|---:|---:|---:|
| Clean | +0.25 pp | +0.50 pp | +0.55 pp | +0.08 pp | +0.02 pp | +0.20 pp | +0.57 pp |
| Impulse noise | +0.85 pp | **+0.90 pp** | +0.67 pp | +0.10 pp | -0.10 pp | +0.55 pp | +0.75 pp |
| Pixelation | **+1.47 pp** | +1.20 pp | +1.22 pp | +0.05 pp | +0.17 pp | +0.88 pp | +1.45 pp |
| JPEG | **+1.05 pp** | +0.65 pp | +0.98 pp | +0.27 pp | +0.05 pp | +0.77 pp | +0.88 pp |
| Brightness | +0.22 pp | **+0.40 pp** | +0.40 pp | +0.37 pp | -0.07 pp | +0.30 pp | +0.52 pp |

Across the four held-out families, the shared rank-32 map gained an average `+0.79 pp`, approximately 88% of the full adapter's `+0.90 pp` average gain. The folded rank-32 map uses 55,456 stored coefficients, about 92.5% fewer than the 741,120-parameter full adapter.

Shared rank-32 improved every held-out family and clean accuracy on average. Pixelation was the strongest confirmation: gains were `+1.30`, `+1.10`, and `+1.20 pp`, each individually significant. Shared rank-32 also beat hidden-PCA by `+1.15`, `+1.30`, and `+1.00 pp`, and random by `+1.10`, `+1.15`, and `+0.85 pp`; every seed-level paired comparison was significant.

Impulse-noise, JPEG, and brightness gains were positive but less consistently significant at the seed level. The shared basis did not consistently beat pooled PCA, the full adapter, or the Blur-specific basis. This is expected from the measured near-identity of the adapter's family-specific output subspaces and means the shared construction is compact and transferable, but not uniquely optimal.

## Mechanistic interpretation

The experiment supports:

> Different corruption families produce different inputs, but the frozen mixed adapter expresses their Block-6 corrections through an almost common low-rank output geometry.

This geometry is not simply the highest-variance hidden-state space: shared rank-32 substantially outperformed hidden-PCA and random directions on pixelation and generally did better on impulse noise and JPEG.

The emerging picture is:

1. Corruptions damage Block-6 patch representations differently.
2. A common adapter correction basis can redirect several kinds of damaged representations.
3. Later blocks—especially the Blocks 11–12 attention bottleneck identified in Experiment 126—convert this shared patch correction into classification evidence.

## Valid claim and limitations

The valid claim is:

> A rank-32 Block-6 correction subspace shared across development corruption families transfers zero-shot to four held-out families and retains most of the full adapter's average OOD gain.

Do not claim that shared-subspace extraction significantly beats every corruption-specific or pooled alternative. The family spaces are themselves nearly identical, the teacher was trained on Noise and Blur, and evaluation uses teacher-validation partitions. This is family-held-out development evidence, not untouched final evaluation or official ImageNet-C.

## Artifacts

- Script: `scripts/experiment127_shared_ood_repair_subspace.py`
- Summary: `results/sae/experiment127_shared_ood_repair_subspace/full_3seed_shared32_familyrank128_v1/summary.json`
- Paired outcomes: `results/sae/experiment127_shared_ood_repair_subspace/full_3seed_shared32_familyrank128_v1/paired_outcomes.npz`
- Deployable maps: `results/sae/experiment127_shared_ood_repair_subspace/full_3seed_shared32_familyrank128_v1/seed_N/folded_factors.pt`
