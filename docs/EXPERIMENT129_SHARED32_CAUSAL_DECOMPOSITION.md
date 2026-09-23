# Experiment 129: Shared Rank-32 Causal Decomposition

## Question

Does the shared rank-32 Block-6 space causally carry the frozen mixed adapter's repair, rather than merely providing a useful standalone correction?

## Protocol

- Frozen ViT, three frozen mixed-adapter seeds, and frozen Experiment 127 rank-32 factors.
- 1,000 images from each seed's disjoint validation split.
- Clean, impulse noise, pixelation, JPEG, and brightness.
- Compared baseline, full adapter, shared-only, full-minus-shared, hidden-PCA-only, full-minus-hidden, random-only, and full-minus-random.
- Included 20 per-seed energy-matched random rank-32 ablations.
- No training, tuning, corruption label, or clean counterpart was used at inference.
- This is a development causal decomposition because these validation conditions were already examined in Experiment 127.

## Results

Mean changes across three seeds:

| Condition | Full adapter | Shared-only | Full minus shared | Hidden-only | Random-only |
|---|---:|---:|---:|---:|---:|
| Clean | -0.03 pp | +0.30 pp | +0.57 pp | +0.10 pp | +0.07 pp |
| Impulse noise | +1.10 pp | **+1.00 pp** | +0.80 pp | -0.07 pp | -0.23 pp |
| Pixelation | +1.17 pp | **+1.03 pp** | +1.07 pp | +0.20 pp | +0.13 pp |
| JPEG | +0.73 pp | **+0.53 pp** | +0.60 pp | +0.17 pp | +0.03 pp |
| Brightness | +0.07 pp | **+0.37 pp** | -0.03 pp | +0.07 pp | +0.03 pp |

Across the four corruptions, the full adapter averaged `+0.77 pp`, shared-only averaged `+0.73 pp`, full-minus-shared averaged `+0.61 pp`, hidden-only averaged `+0.09 pp`, and random-only averaged approximately `-0.01 pp`. Shared-only therefore retained about 95.7% of the full adapter's average gain in this development analysis.

## Causal interpretation

The sufficiency result is strong: shared-only remained positive for all four families and nearly matched the full adapter on average, while hidden-PCA-only and random-only corrections were much weaker on these images.

The necessity result is weak: removing the shared component reduced the full adapter by only about `0.16 pp` on average across corruptions. Every seed-level paired confidence interval for the shared-ablation cost crossed zero. Shared ablation was also not unusually damaging relative to the 20 energy-matched random controls; empirical p-values were generally large, except one isolated JPEG seed at the minimum attainable `1/21 = 0.0476`.

Therefore the rank-32 space is a compact sufficient repair route, but it is not established as a uniquely necessary route. The full adapter retains redundant useful correction outside this space.

## Valid conclusion

> A frozen shared rank-32 Block-6 map is sufficient to reproduce most of the mixed adapter's average cross-corruption benefit, but removing that component does not reliably destroy the full adapter's repair. The adapter therefore contains redundant repair pathways rather than one uniquely necessary rank-32 mechanism.

## Artifacts

- Script: `scripts/experiment129_shared32_causal_decomposition.py`
- Summary: `results/sae/experiment129_shared32_causal_decomposition/full_3seed_1000_20controls_v1/summary.json`
- Outcomes: `results/sae/experiment129_shared32_causal_decomposition/full_3seed_1000_20controls_v1/paired_outcomes.npz`
- Log: `logs/experiment129/full_3seed_1000_20controls_v1.log`
