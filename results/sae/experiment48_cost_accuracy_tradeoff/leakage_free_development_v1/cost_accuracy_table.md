# Leakage-Free Development Cost–Accuracy Trade-off

| Method | Noise-4 accuracy | Noise gain | Clean change | Positive seeds | Trainable params | Added GMAC | Evidence |
|---|---:|---:|---:|---:|---:|---:|---|
| Frozen ViT baseline | 69.75% | +0.00 pp | +0.00 pp | 0/3 | 0 | 0.0000 | Reference |
| Full 768D residual adapter | 70.77% | +1.02 pp | -0.28 pp | 3/3 | 741,120 | 0.1156 | Supported development method |
| 16D raw-hidden repair | 70.18% | +0.43 pp | +0.07 pp | 3/3 | 3,408 | 0.0048 | Promising; paired CIs cross zero |
| 16D SAE-discovered hybrid | 70.02% | +0.27 pp | +0.07 pp | 3/3 | 15,440 | 0.0048 | Suggestive; not best matched basis |
| 16D high-variance raw subspace | 70.32% | +0.57 pp | -0.05 pp | 2/3 | 15,440 | 0.0048 | Highest small mean; fails one seed |
| 16-feature SAE direct repair | 69.87% | +0.12 pp | -0.12 pp | 2/3 | 3,408 | 0.0048 | Not supported |
| Leakage-free top-16 SAE gate | 70.57% | +0.82 pp | -0.28 pp | 3/3 | 17 | 3.6994 | Below ungated adapter; not supported |

## Interpretation

- The full adapter has the largest stable gain and remains the primary engineering method.
- The 16D raw repair has the strongest efficiency/clean-preservation trade-off, but its per-seed paired tests were not individually significant.
- The SAE hybrid is positive in every seed and beats random SAE directions, but does not beat the strongest raw-hidden control.
- The leakage-free SAE gate remains below the ungated full adapter and adds substantial frozen-SAE inference cost.
- These are development results, not final ImageNetV2 claims. Exact CUDA latency remains pending.
