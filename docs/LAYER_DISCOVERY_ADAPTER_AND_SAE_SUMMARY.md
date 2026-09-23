# How We Found the Repair Layer and Built the Adapter

## Layer Discovery, Adapter Training, Propagation, SAE Evidence, and Data-Leakage Controls

**Model:** `google/vit-base-patch16-224`  
**Primary corruption:** Gaussian Noise, severity 4  
**Transfer corruption:** Gaussian Blur, severity 4  
**Backbone policy:** the ViT remains frozen  
**Deployment:** the adapter uses only the current image; no clean counterpart, SAE, label, or online optimization is required

---

## 1. Main Question

We wanted to answer three connected questions:

1. **Where** in the ViT is corruption damage most recoverable?
2. Can a small module repair that layer using only a corrupted image at inference?
3. Does the repair remain useful as the representation passes through later Transformer blocks?

We answered the first question in two different ways:

- an **oracle clean-restoration test** measuring theoretical recoverability;
- a **deployable adapter comparison** measuring practical learnability.

These tests answer different questions and produced complementary results.

---

## 2. Method 1: Oracle Layer-Recovery Test

### 2.1 What we did

For a paired clean image and Noise-4 image, we extracted the hidden state at a candidate layer:

`h_clean(layer)` and `h_noise(layer)`

We calculated the clean correction:

`delta(layer) = h_clean(layer) - h_noise(layer)`

We then partially restored the noisy representation:

`h_test(layer) = h_noise(layer) + alpha × delta(layer)`

Finally, we passed `h_test` through the remaining frozen ViT blocks and measured:

- recovered accuracy;
- true-class margin change;
- true-class logit change;
- recovered and damaged predictions;
- exact paired McNemar significance.

### 2.2 Why this is an oracle test

The correction uses the matching clean representation. Therefore, it cannot be used at deployment. Its purpose is only to answer:

> If this layer could be repaired correctly, would the later ViT blocks recover the prediction?

### 2.3 Oracle result

The base Noise-4 accuracy on the 2,000-image diagnostic set was **68.70%**. At partial restoration `alpha = 0.25`:

| Restored layer | Accuracy | Gain | Recovered | Damaged | Mean margin change |
|---:|---:|---:|---:|---:|---:|
| Layer 1 | 73.30% | +4.60 pp | 124 | 32 | +0.485 |
| Layer 3 | 73.60% | +4.90 pp | 130 | 32 | +0.530 |
| Layer 6 | 74.75% | +6.05 pp | 143 | 22 | +0.677 |
| Layer 9 | 74.70% | +6.00 pp | 134 | 14 | +0.661 |
| **Layer 11** | **75.00%** | **+6.30 pp** | **135** | **9** | **+0.691** |
| Layer 12 | 75.00% | +6.30 pp | 132 | 6 | +0.636 |

Layer 11 gave the strongest late-layer partial restoration under the chosen diagnostic criteria, including a larger margin and true-class-logit improvement than Layer 12. It also left one Transformer block capable of processing repaired patch information.

This motivated the first Block-11 adapter. However, oracle recoverability does not guarantee that a correction can be predicted from corrupted activations alone.

---

## 3. Method 2: Deployable Adapter Layer Comparison

### 3.1 Why we needed a second test

The oracle gives the correct clean direction directly. A deployable adapter must infer a useful correction without seeing the clean image.

Therefore, we trained identical residual adapters after Blocks:

`6, 8, 9, 10, 11, and 12`

Every layer used:

- the same adapter architecture;
- 741,120 trainable parameters;
- the same Noise-4 generation;
- the same losses and hyperparameters;
- zero initialization;
- the same seed-specific train/validation protocol;
- three independently trained seeds;
- a frozen ViT backbone.

### 3.2 Deployable layer results

Mean three-seed development results:

| Adapter location | Noise-4 change | Clean change | Result |
|---:|---:|---:|---|
| **After Block 6** | **+2.20 pp** | −1.28 pp | Highest practical robustness |
| After Block 8 | −4.85 pp | −4.77 pp | Harmful |
| After Block 9 | −8.38 pp | −6.23 pp | Harmful |
| After Block 10 | −0.13 pp | −2.03 pp | Not useful |
| After Block 11 | +1.05 pp | −0.45 pp | Useful but weaker |
| After Block 12 | 0.00 pp | 0.00 pp | Negative control |

Block 12 cannot influence the final prediction when only patch tokens are edited because no later attention block remains to transfer patch information into the CLS token.

### 3.3 Final layer conclusion

- **Oracle result:** Layer 11 was highly recoverable when given the true clean correction.
- **Deployable result:** Block 6 was the best location for a learned correction inferred from corrupted activations.

The correct conclusion is:

> Layer 11 has strong theoretical recoverability, but Block 6 has the highest practical repairability. At Block 6, enough useful information remains for the adapter to estimate a correction, and six later blocks remain to propagate it into the CLS token.

---

## 4. How the Block-6 Adapter Was Built

### 4.1 Full adapter architecture

The adapter operates on the 196 patch tokens after Block 6. The hidden dimension is 768.

For corrupted patch tokens `h`, it predicts a residual:

`delta_h = Linear(h) + PositionEmbedding`

The corrected patch tokens are:

`h_corrected = h + delta_h`

The CLS token is not directly edited.

| Component | Shape / role |
|---|---|
| Linear weight | `768 × 768` |
| Linear bias | `768` |
| Patch-position embedding | `196 × 768` |
| Total trainable parameters | **741,120** |
| FP32 checkpoint size | **2.83 MiB** |
| Added analytical compute | **0.1156 GMAC/image** |

### 4.2 Training data

During training only, each image produced:

- a clean Block-6 hidden representation;
- a Noise-4 Block-6 hidden representation.

The clean representation provided a target for learning the repair. At inference, the clean counterpart is not required.

### 4.3 Training objective

The selected adapter combined four objectives:

1. **Residual reconstruction:** predict the clean-minus-noisy patch residual.
2. **Classification:** improve the corrupted-image class prediction.
3. **Prediction preservation:** avoid reducing the margin of already-correct noisy predictions.
4. **Clean identity:** make the adapter output approximately zero on clean patch tokens.

Conceptually:

`Loss = residual_loss + 0.05 × classification_loss + 0.05 × preservation_loss + identity_weight × clean_identity_loss`

### 4.4 Training hyperparameters

| Setting | Value |
|---|---:|
| Epochs | 5 |
| Early-stopping patience | 2 |
| Initial learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Smooth-L1 beta | 1.0 |
| Classification weight | 0.05 |
| Prediction-preservation weight | 0.05 |
| Training correction scale | 0.5 |
| Evaluation correction scale | 1.0 |
| Selected clean-identity weight | **0.2** |
| Adapter seeds | 3 |

### 4.5 Why clean-identity loss mattered

The original Block-6 adapter improved Noise but unnecessarily changed clean activations. We swept clean-identity weights:

| Identity weight | Mean Noise-4 gain | Mean clean change |
|---:|---:|---:|
| 0 | +2.20 pp | −1.28 pp |
| **0.2** | **+3.13 pp** | **+0.03 pp** |
| 1 | +2.90 pp | +0.13 pp |
| 5 | +2.17 pp | +0.18 pp |
| 20 | +1.68 pp | +0.07 pp |

Weight 0.2 was selected using development data before independent evaluation.

---

## 5. Final Full-Adapter Results

Three frozen adapters were evaluated once on 10,000 independent ImageNetV2 images.

| Condition | Baseline | Corrected mean | Mean change | Changes by seed |
|---|---:|---:|---:|---|
| Clean | 68.37% | 68.28% | −0.09 pp | −0.16, −0.07, −0.03 pp |
| Noise-4 | 57.25% | 59.81% | **+2.56 pp** | +2.72, +2.61, +2.36 pp |
| Blur-4 | 50.82% | 52.14% | **+1.32 pp** | +1.12, +1.40, +1.44 pp |

All Noise and Blur gains were significant using paired tests. Every clean confidence interval crossed zero, so there was no evidence of meaningful clean damage.

---

## 6. How the Block-6 Repair Improved Downstream

The adapter edits patch tokens only at Block 6. We tracked the corrected representations through Blocks 7–12.

### 6.1 Full-adapter propagation

On Noise-4 development data:

- Block-6 patch cosine to clean increased by approximately **+0.0565**.
- The CLS token was unchanged at Block 6 because it was not edited.
- Margin improvement grew as later blocks processed the repaired patches.
- By Block 11, mean margin gain was approximately **+0.178**.
- By Block 12, mean margin gain reached approximately **+0.312**.
- Final Noise-4 gain was **+3.13 pp** on development data.

### 6.2 Rank-8 propagation on held-out reserve data

| Stage | Patch cosine gain | CLS cosine gain | True-class margin gain |
|---:|---:|---:|---:|
| Block 6 | +0.0261 | 0.0000 | 0.0000 |
| Block 7 | +0.0277 | −0.0023 | +0.0056 |
| Block 10 | +0.0225 | −0.0032 | +0.0381 |
| Block 11 | +0.0222 | +0.0031 | +0.1266 |
| Block 12 | +0.0238 | **+0.0223** | **+0.2227** |

The downstream mechanism is:

1. Block 6 receives improved patch information.
2. Blocks 7–12 preserve and transform this information.
3. Attention transfers the useful patch correction into the CLS token.
4. The final classifier receives a representation with a larger true-class margin.

Later blocks do not independently run another adapter. They progressively convert the original Block-6 patch repair into a better decision.

---

## 7. Adapter Sizes and Results

### 7.1 Low-rank architecture

The compressed adapter predicts:

`delta_h = Up(Down(h) + LowRankPositionEmbedding)`

where `Down: 768 → rank` and `Up: rank → 768`.

### 7.2 Development sweep

| Adapter | Parameters | Reduction vs full | Noise-4 gain | Clean change | Blur-4 gain |
|---|---:|---:|---:|---:|---:|
| Rank 8 | 14,632 | 98.0% | +1.53 pp | −0.25 pp | +0.35 pp |
| Rank 16 | 28,496 | 96.2% | +1.65 pp | −0.37 pp | +0.65 pp |
| Rank 32 | 56,224 | 92.4% | +1.55 pp | −0.28 pp | +0.73 pp |
| Rank 64 | 111,680 | 84.9% | +1.97 pp | −0.18 pp | +1.18 pp |
| Rank 128 | 222,592 | 70.0% | +2.25 pp | −0.40 pp | +1.25 pp |
| Full | 741,120 | — | **+3.13 pp** | +0.03 pp | **+1.33 pp** |

No compressed rank retained the preregistered 90% target on development Noise-4. Rank 8 was examined further because it offered the largest parameter reduction.

### 7.3 Held-out rank-8 results

| Condition | Mean change | Seed changes | Conclusion |
|---|---:|---|---|
| Noise-4 | **+1.91 pp** | +1.83, +1.91, +1.98 pp | Stable improvement |
| Clean | **−0.07 pp** | −0.12, −0.04, −0.05 pp | No significant damage |
| Blur-4 | approximately **0.00 pp** | +0.03, −0.01, −0.03 pp | No replicated transfer |

Rank 8 used 98% fewer parameters and retained approximately 86.7% of the full adapter's Noise gain on the same reserve evaluation. It is currently a compact Noise-specific adapter, not a corruption-general adapter.

---

## 8. Latest SAE Causal Experiment

### 8.1 Why we repeated it

Historical experiments claimed that 16 harmful SAE features were essential to both Noise repair and Noise-to-Blur transfer. The supervisor correctly questioned whether historical data overlap inflated this result.

We repeated the experiment with:

- a leakage-free SAE/feature-development split;
- newly rediscovered harmful features;
- three independently trained, zero-initialized adapters;
- 10,000 reserve images for paired causal evaluation;
- a separate 1,000-image reserve subset;
- **1,000 energy-matched random 16-feature controls** per adapter and condition;
- no ImageNetV2 access.

### 8.2 What was removed

The adapter residual was projected into the decoder subspace of the 16 SAE-discovered directions. We compared:

- the full residual;
- the residual with the harmful 16-direction component removed;
- 1,000 residuals with equal-energy random 16-direction components removed.

### 8.3 Results

| Condition | Full-adapter gain | Cost of removing SAE-16 | Random-control conclusion |
|---|---:|---:|---|
| Clean | −0.14 pp | 0.19 pp | Not special |
| Noise-4 | +0.74 pp | 0.33 pp | Not consistently beyond random |
| Blur-4 | +0.52 pp | **1.18 pp** | **99.9–100th percentile** |

Blur empirical p-values across the three adapters were:

`0.000999, 0.000999, 0.001998`

Removing the 16-direction subspace reduced Blur accuracy to an average **0.65 pp below the uncorrected baseline**.

### 8.4 Correct SAE conclusion

The latest leakage-free result does **not** support saying that the 16 SAE directions uniquely perform Noise repair. Noise repair appears distributed across a broader hidden-space mechanism.

It does support a strong cross-corruption claim:

> The Noise-trained full adapter's unexpected Blur benefit depends specifically on the SAE-discovered shared subspace.

This gives SAE a clear mechanistic role even though direct SAE editing was not the strongest accuracy method.

---

## 9. Data-Leakage Safeguards

### 9.1 Historical problem

The original harmful-feature discovery data overlapped with data used by the initialization adapter. ImageNetV2 itself was independent, but the mechanism-selection process had circular ancestry.

### 9.2 Corrected split protocol

| Purpose | Data allocation |
|---|---:|
| SAE/feature development | ImageNet validation `[0,15000)` |
| Adapter seed 0 train/validation | `[15000,22000)` |
| Adapter seed 1 train/validation | `[22000,29000)` |
| Adapter seed 2 train/validation | `[29000,36000)` |
| Reserve evaluation | `[36000,50000)` |
| Full-adapter independent final test | ImageNetV2, 10,000 images |

### 9.3 Additional controls

- Adapters were trained from zero initialization.
- The ViT was frozen.
- Three adapter seeds were retained.
- Paired sample ordering and corruption seeds were fixed.
- Recent experiments saved per-image outcomes.
- McNemar tests and paired bootstrap intervals replaced unpaired accuracy claims.
- The 1,000 random controls set the minimum empirical p-value near `0.001` rather than `0.0476`.
- Clean counterparts were used only for training targets or mechanistic analysis, never by the deployed correction.

### 9.4 Current limitation

The reserve split has now supported several fixed-method analyses. It remains free from training leakage, but it is no longer an untouched final set. Any future corruption-general adapter must be frozen before evaluation on a new independent final dataset.

---

## 10. Final Takeaway

The project followed a clear mechanistic-to-engineering path:

1. SAE analysis showed meaningful corruption-related feature-strength changes.
2. Oracle restoration identified highly recoverable late representations.
3. Matched deployable tests showed that Block 6 is the best learnable intervention point.
4. A clean-identity objective produced a strong, clean-preserving Block-6 adapter.
5. Downstream analysis showed that repaired patches are progressively converted into improved CLS evidence.
6. Low-rank compression produced a 14,632-parameter Noise adapter with stable held-out gain.
7. Leakage-free SAE ablation revealed that Noise repair is distributed, while Blur transfer depends on a small interpretable shared subspace.

The strongest current claims are:

- **Full adapter:** generalizes from Noise training to independent Noise-4 and Blur-4 improvement with clean preservation.
- **Rank 8:** highly efficient and reliable for Noise-4, but not yet corruption-general.
- **SAE:** valuable for mechanism discovery and causal explanation, rather than as the best direct intervention space.

---

## Main Artifacts

- Oracle layer localization: `results/sae/experiment19_noise_layer_localization/noise_layer_localization_gpu/summary.json`
- Deployable layer comparison: `results/sae/experiment46_layer_specific_adapters/full_3seed_blocks6_8_9_10_11_12_v1/summary.json`
- Block-6 clean-preservation sweep: `results/sae/experiment50_block6_clean_preservation/full_3seed_identity_sweep_v1/summary.json`
- Independent full-adapter result: `results/sae/experiment53_block6_independent_confirmation/full_10000_frozen_3seed_v1/summary.json`
- Low-rank sweep: `results/sae/experiment54_lowrank_block6_adapter/full_3seed_ranks8_128_v1/summary.json`
- Leakage-free SAE causal replication: `results/sae/experiment56_leakage_free_causal_subspace/full_disjoint_section67_replication_v1/summary.json`
- Rank-8 Noise propagation: `results/sae/experiment57_rank8_repair_propagation/full_noise4_reserve_3seed_v1/summary.json`
- Rank-8 clean/Blur evaluation: `results/sae/experiment58_rank8_clean_blur_reserve/full_3seed_reserve_v1/summary.json`

All reported gains and losses are absolute percentage-point changes.
