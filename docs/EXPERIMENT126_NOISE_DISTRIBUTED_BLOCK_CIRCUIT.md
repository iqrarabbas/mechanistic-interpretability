# Experiment 126: Distributed Block-Level Circuit for Noise Repair

## Question and method

Where does the frozen rank-128 Block-6 Noise repair become causally necessary downstream?

For 1,000 Noise-4 images and a separate clean control, the experiment injected each seed's frozen rank-128 Delta-PCA correction after Block 6. It then propagated the corrected stream through Blocks 7–12 while replacing selected attention and/or MLP outputs with the corresponding outputs from the same image's uncorrected baseline stream.

The experiment tested every downstream block independently, groups 7–8, 9–10, and 11–12, and all Blocks 7–12. Three independently derived rank-128 maps were evaluated. Exact replay checks reproduced standard baseline and full-repair logits within the required tolerance.

No clean counterpart was used to repair Noise. Clean images formed a separate specificity control.

## Full repair

| Seed | Baseline-to-rank-128 Noise gain | Exact paired p-value |
|---|---:|---:|
| 0 | +2.50 pp | 0.0073 |
| 1 | +3.30 pp | 0.00038 |
| 2 | +2.60 pp | 0.0103 |

Mean gain was `+2.80 pp`. Clean changes were `-0.30`, `0.00`, and `-0.10 pp`, all nonsignificant.

## Block-level mediation

Mean repair gain lost after reverting both attention and MLP outputs:

| Reverted location | Noise gain lost | Clean gain lost |
|---|---:|---:|
| Block 7 | 0.87 pp | -0.63 pp |
| Block 8 | 1.23 pp | -0.10 pp |
| Block 9 | 0.47 pp | -0.47 pp |
| Block 10 | 0.40 pp | -0.43 pp |
| Block 11 | 1.43 pp | -0.20 pp |
| Block 12 | 1.83 pp | -0.57 pp |
| Blocks 7–8 | 1.07 pp | +0.17 pp |
| Blocks 9–10 | 1.43 pp | -0.60 pp |
| **Blocks 11–12** | **2.73 pp** | **-0.23 pp** |
| Blocks 7–12 | 2.80 pp | -0.13 pp |

Reverting Blocks 11–12 jointly removed `2.60`, `3.30`, and `2.30 pp` across seeds. Every seed was significant, and this accounts for nearly the entire `+2.80 pp` mean repair gain.

## Attention versus MLP

| Reverted pathway | Mean Noise gain lost |
|---|---:|
| Blocks 11–12 attention | **2.43 pp** |
| Blocks 11–12 MLP | 0.93 pp |
| All Blocks 7–12 attention | **2.80 pp** |
| All Blocks 7–12 MLP | 1.90 pp |

Blocks 11–12 attention reversion removed `2.30`, `3.10`, and `1.90 pp`; the first two seeds were significant and seed 2 was borderline by exact McNemar (`p=0.0558`) while its bootstrap interval remained positive. Blocks 11–12 MLP reversion was smaller and nonsignificant for every seed.

The clean controls showed no analogous positive loss of benefit. Reversions often slightly improved clean predictions because the rank-128 repair is not needed on clean inputs. This supports corruption-specific mediation rather than general classifier fragility.

## Mechanistic conclusion

The supported coarse circuit is:

> **Block-6 patch-space correction → distributed transformation through Blocks 7–10 → dominant late attention integration in Blocks 11–12 → final CLS decision.**

Blocks 7–10 contribute, but no individual early/middle block is consistently necessary across seeds. The late pair is reproducibly necessary as a group. This is consistent with redundancy: several earlier pathways can carry the repair forward, after which late attention integrates it into classification evidence.

This is not a uniquely complete head/neuron circuit. Component reversion creates hybrid states, attention and MLP effects are nonlinear, and prior small-head ablations did not identify a stable tiny Noise circuit. The defensible claim is a reproducible **distributed block-level causal circuit** with a late-attention bottleneck.

## Artifacts

- Script: `scripts/experiment126_noise_distributed_block_circuit.py`
- Summary: `results/sae/experiment126_noise_distributed_block_circuit/full_3seed_noise_clean_1000_v1/summary.json`
- Paired outcomes: `results/sae/experiment126_noise_distributed_block_circuit/full_3seed_noise_clean_1000_v1/paired_outcomes.npz`
