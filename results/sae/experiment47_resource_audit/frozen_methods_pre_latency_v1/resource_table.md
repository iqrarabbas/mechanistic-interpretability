# Frozen ViT Robustness Resource Audit

Frozen backbone: `google/vit-base-patch16-224` with 86,567,656 parameters.
Analytical backbone compute: 17.564 GMAC/image (35.128 GFLOP using two FLOPs per MAC).

| Method | Trainable params | Added params vs ViT | Added GMAC/image | Checkpoint MiB | SAE at inference |
|---|---:|---:|---:|---:|:---:|
| Frozen ViT baseline | 0 | 0.0000% | 0.0000 | 2.830 | No |
| Full 768D residual adapter | 741,120 | 0.8561% | 0.1156 | 2.830 | No |
| 16D raw-hidden repair | 3,408 | 0.0039% | 0.0048 | 2.830 | No |
| 16D shared-input subspace repair | 15,440 | 0.0178% | 0.0048 | 2.830 | No |
| 17-parameter SAE gate | 17 | 0.0000% | 3.6994 | 144.099 | Yes |

## Conventions and limitations

- A MAC is one multiply-accumulate. The FLOP column in JSON/CSV uses two FLOPs per MAC.
- Analytical compute counts dominant dense matrix multiplications and attention products; normalization, activations, additions, and data preprocessing are omitted.
- Adapter compute includes the 768-by-768 patch projection for 196 patch tokens; positional additions are omitted as non-MAC operations.
- The 16-coordinate methods include raw-input projection and fixed-basis decoding.
- SAE-gate compute includes a full 768-to-24,576 encoder because the existing implementation materializes the full SAE latent vector.
- Exact CUDA latency is deliberately pending until Experiment 46 releases the GPU.
- No test-time parameter updates are performed by these frozen correction methods.
