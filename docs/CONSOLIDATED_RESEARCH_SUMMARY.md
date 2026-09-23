# Mechanistically Guided Lightweight ViT Corruption Repair

## Consolidated Research Summary and Supervisor-Comment Response

**Backbone:** `google/vit-base-patch16-224`  
**Primary corruptions:** Gaussian Noise and Gaussian Blur, severity 4  
**Core constraint:** the ViT backbone remains frozen  
**Deployment:** one corrupted image only; no clean counterpart, SAE, labels, or online gradients are required by the final adapters

---

## 1. Executive Summary

This project began by reproducing the SAE representation analysis of Gao et al. and then asked a broader question:

> Can mechanistic interpretability reveal where corruption damages a ViT and guide a small, deployable repair module?

The main findings are:

1. Corruption does not simply replace the entire SAE feature dictionary. The strongest feature identities remain substantially more stable for a clean/corrupted image pair than for unrelated images, while feature magnitudes change more strongly in classification-failure cases.
2. Directly editing a few SAE activations is not the best repair method. SAE is most defensible as a mechanism-discovery and diagnostic tool; correction works better in the original ViT hidden space.
3. A causal layer study and matched deployable-adapter study identified **Block 6** as the most practically repairable location for Noise-4.
4. A clean-identity loss removed the original Block-6 clean-accuracy penalty and improved robustness.
5. The full Block-6 adapter improved independent ImageNetV2 accuracy by **+2.56 pp on Noise-4** and **+1.32 pp on Blur-4**, with no significant clean change.
6. A rank-8 adapter uses only **14,632 parameters** and achieved a held-out **+1.91 pp Noise-4 gain** with only **−0.07 pp clean change**. Its Blur-4 gain did not replicate on held-out reserve data.
7. Leakage-free causal ablation with 1,000 energy-matched controls showed that the SAE-discovered 16-direction subspace is specifically important for the Noise-trained full adapter's **Blur transfer**, but is not uniquely responsible for its Noise repair.

The strongest current engineering result is the full Block-6 adapter. The strongest efficiency result is rank-8 for Noise robustness. The strongest mechanistic result is that cross-corruption Blur transfer depends on a small SAE-discovered subspace, whereas Noise repair is distributed more broadly.

---

## 2. How Mechanistic Interpretability Helped

Mechanistic interpretability was not used merely to visualize the model. It changed the design of the repair method.

### 2.1 SAE diagnosis

Initial SAE analysis showed that classification failures were associated with larger activation-strength changes. The original positive-support Jaccard of approximately 0.99 was later found to be saturated: unrelated images also scored around 0.98. A stronger control used dominant features:

| Identity metric | Clean/Blur pair | Unrelated controls |
|---|---:|---:|
| Positive-support Jaccard | 0.9900 | approximately 0.980 |
| Top-16 Jaccard | **0.3308** | approximately 0.050–0.054 |
| SAE cosine | **0.9286** | approximately 0.808–0.812 |

The corrected conclusion is:

> Corruption preserves a meaningful portion of the dominant feature set, but changes feature strengths and the relative importance of those features.

### 2.2 SAE intervention lesson

Replacing a ViT hidden state with an SAE reconstruction introduces reconstruction error. We therefore used residual decoder edits:

`h_repaired = h + Decode(z_corrected) - Decode(z_original)`

Even with this correction, direct SAE-space repair remained weak. Parameter-matched raw-hidden repair was stronger. This established a clean division of roles:

- **SAE:** mechanism discovery, diagnosis, feature/subspace identification.
- **Raw ViT hidden state:** more effective intervention space.

### 2.3 Layer localization

An oracle clean-restoration test measured the theoretical recovery available at each layer. A second, more important deployable test trained identical residual adapters after Blocks 6, 8, 9, 10, 11, and 12. This distinguished:

- **Oracle recoverability:** what could recover if the clean representation were available.
- **Practical repairability:** what a learned adapter can recover from the corrupted representation alone.

---

## 3. Data-Leakage Correction

The supervisor correctly identified circularity in the historical pipeline: harmful-feature discovery overlapped with data used by the initialization adapter. There was no direct ImageNetV2 test-image leakage, but the mechanism-selection process was not fully independent.

We corrected this with `configs/split_manifest_supervisor_v1.json`:

| Purpose | ImageNet validation indices |
|---|---:|
| SAE/feature development | `[0,15000)` |
| Adapter seed 0 train/validation | `[15000,22000)` |
| Adapter seed 1 train/validation | `[22000,29000)` |
| Adapter seed 2 train/validation | `[29000,36000)` |
| Reserve evaluation | `[36000,50000)` |
| Independent final evaluation | ImageNetV2, 10,000 images |

Additional safeguards:

- Adapters were initialized from zero rather than a historically overlapping checkpoint.
- Feature ranking, adapter fitting, and reserve evaluation were separated.
- Three independently trained adapter seeds were preserved.
- Recent evaluations saved paired per-image outcomes.
- ImageNetV2 was not used to select the low-rank architecture or tune the recent reserve experiments.

**Important present limitation:** the reserve split has now been examined repeatedly and must be treated as a confirmation/development set. A future corruption-general method needs a new independent final dataset after all choices are frozen.

---

## 4. Supervisor Requests and Current Status

| Supervisor request | What we did | Status |
|---|---|---|
| Precisely document feature-selection splits and remove circularity | Added split manifest and audit; rediscovered features; trained zero-initialized adapters on disjoint splits; evaluated on reserve data | **Completed** |
| Replace 20-control significance floor with 1,000 controls | Experiment 56 used 1,000 energy-matched random 16-feature subspaces per adapter/condition | **Completed for causal subspace claim** |
| Use paired tests rather than raw accuracy differences | Recent experiments report recovered/damaged predictions, exact McNemar tests, and paired bootstrap CIs | **Completed for recent claims** |
| Report sign tests | Historical 9/9 sign test is `p=0.00195`, but nested runs are not fully independent | **Answered with caveat** |
| Add parameters/FLOPs/latency | Parameters, checkpoint size, and analytical GMACs were measured | **Latency still pending** |
| Train adapters at Blocks 6, 8, 9, 10, 11, 12 | Identical three-seed deployable layer comparison completed | **Completed** |
| Compare non-SAE gates | Leakage-free SAE gate failed to beat the ungated adapter; full requested PCA/mean/variance/probe comparison remains | **Partial** |
| Test clean-importance features | Clean controls and corruption-amplification tests were added, but the exact requested matched clean-importance gate remains | **Partial** |
| Parameter-matched LoRA/MLP/capacity baselines | Experiment 59 compared five approximately 15K-parameter Block-6 methods across three disjoint seeds; Q/V LoRA gave the strongest Noise result | **Completed on development validation** |
| TENT, SAR, MEMO, CoTTA, pixel denoiser | Controlled CFA comparison completed; remaining TTA methods not yet run | **Partial** |
| Feature-count sweep `k=1,4,8,16,32,64,128` | Earlier gate sweep covered 16–128; low-rank adapter sweep covered ranks 8–128, which is a different question | **Feature-k sweep incomplete** |
| SAE width/type/seed stability | Leakage-free top-16 ranking overlapped historical ranking by 13/16, but full multi-SAE seed/type matching remains | **Open and important** |

---

## 5. Adapter Results

### 5.1 Identical deployable adapters across layers

All adapters had 741,120 parameters and used matched training settings. Mean three-seed development changes were:

| Intervention block | Noise-4 change | Clean change | Interpretation |
|---:|---:|---:|---|
| **6** | **+2.20 pp** | −1.28 pp | Highest robustness; initial clean penalty |
| 8 | −4.85 pp | −4.77 pp | Harmful |
| 9 | −8.38 pp | −6.23 pp | Harmful |
| 10 | −0.13 pp | −2.03 pp | Not useful |
| 11 | +1.05 pp | −0.45 pp | Smaller but cleaner repair |
| 12 | 0.00 pp | 0.00 pp | Negative control; no later attention remains |

This answered the objection that Block 11 was uniquely special. Block 6 was the most practically repairable location.

### 5.2 Clean-preserving full Block-6 adapters

Adding a clean-identity penalty encouraged the adapter to output approximately zero on clean activations.

| Identity weight | Mean Noise-4 gain | Mean clean change |
|---:|---:|---:|
| 0 | +2.20 pp | −1.28 pp |
| **0.2** | **+3.13 pp** | **+0.03 pp** |
| 1 | +2.90 pp | +0.13 pp |
| 5 | +2.17 pp | +0.18 pp |
| 20 | +1.68 pp | +0.07 pp |

Identity weight 0.2 was selected on development data before independent evaluation.

### 5.3 Independent ImageNetV2 confirmation of the full adapter

The three frozen Block-6 adapters were evaluated once on 10,000 ImageNetV2 images.

| Condition | Baseline | Adapter mean | Mean change | Seed changes |
|---|---:|---:|---:|---|
| Clean | 68.37% | 68.28% | −0.09 pp | −0.16, −0.07, −0.03 pp |
| Noise-4 | 57.25% | 59.81% | **+2.56 pp** | +2.72, +2.61, +2.36 pp |
| Blur-4 | 50.82% | 52.14% | **+1.32 pp** | +1.12, +1.40, +1.44 pp |

All Noise and Blur gains were paired-significant. Every clean confidence interval included zero.

### 5.4 Low-rank adapter sweep

All low-rank adapters used the same Block-6 identity-preserving training recipe.

| Adapter | Parameters | Reduction vs full | Development Noise-4 gain | Development clean change | Development Blur-4 gain |
|---|---:|---:|---:|---:|---:|
| Rank 8 | 14,632 | 98.0% | +1.53 pp | −0.25 pp | +0.35 pp |
| Rank 16 | 28,496 | 96.2% | +1.65 pp | −0.37 pp | +0.65 pp |
| Rank 32 | 56,224 | 92.4% | +1.55 pp | −0.28 pp | +0.73 pp |
| Rank 64 | 111,680 | 84.9% | +1.97 pp | −0.18 pp | +1.18 pp |
| Rank 128 | 222,592 | 70.0% | +2.25 pp | −0.40 pp | +1.25 pp |
| Full | 741,120 | — | +3.13 pp | +0.03 pp | +1.33 pp |

No compressed rank met the preregistered target of retaining 90% of the full Noise gain. Rank 64 offered the best development efficiency/transfer compromise. Rank 8 was subsequently examined as the minimum-cost design, not declared the formal sweep winner.

### 5.5 Held-out reserve evaluation of rank 8

| Condition | Seed changes | Mean change | Conclusion |
|---|---|---:|---|
| Noise-4 | +1.83, +1.91, +1.98 pp | **+1.91 pp** | Stable positive gain |
| Clean | −0.12, −0.04, −0.05 pp | **−0.07 pp** | No significant damage |
| Blur-4 | +0.03, −0.01, −0.03 pp | **approximately 0.00 pp** | Development transfer did not replicate |

Rank 8 retained approximately 86.7% of the full adapter's Noise gain on the same reserve evaluation while using 98% fewer parameters. It should currently be described as a compact **Noise-specific adapter**, not a general corruption adapter.

### 5.6 Other tested repair designs

Leakage-free development comparison:

| Method | Parameters | Noise-4 gain | Clean change | Conclusion |
|---|---:|---:|---:|---|
| Full 768D residual adapter | 741,120 | **+1.02 pp** | −0.28 pp | Largest stable gain in that comparison |
| 16D raw-hidden repair | 3,408 | +0.43 pp | +0.07 pp | Best tiny clean/efficiency trade-off; CIs crossed zero |
| 16D SAE-discovered hidden repair | 15,440 | +0.27 pp | +0.07 pp | Suggestive, not best matched basis |
| High-variance raw subspace | 15,440 | +0.57 pp | −0.05 pp | Larger mean, but failed one seed |
| Direct 16-feature SAE repair | 3,408 | +0.12 pp | −0.12 pp | Not supported |
| Leakage-free SAE gate plus full adapter | 17 trainable gate parameters | +0.82 pp | −0.28 pp | Below ungated adapter; requires expensive SAE |

These experiments support using SAE for diagnosis rather than claiming that sparse SAE coordinates are the best intervention basis.

---

## 6. Causal 16-Feature Subspace Replication

Historical causal-ablation magnitudes were potentially inflated by overlapping development provenance. Experiment 56 repeated the test using:

- leakage-free feature rediscovery;
- three zero-initialized disjoint adapters;
- 10,000 reserve images for paired causal accuracy;
- a separate 1,000-image subset for 1,000 energy-matched controls;
- no ImageNetV2 access.

| Condition | Full-adapter gain | Cost of removing 16 directions | Random-null result |
|---|---:|---:|---|
| Clean | −0.14 pp | 0.19 pp | Not special |
| Noise-4 | +0.74 pp | 0.33 pp | Not consistently beyond random |
| Blur-4 | +0.52 pp | **1.18 pp** | **99.9–100th percentile** |

For Blur, empirical p-values across adapters were `0.000999`, `0.000999`, and `0.001998`. Removing the 16 directions reduced Blur accuracy to an average **0.65 pp below the uncorrected baseline**.

The corrected mechanistic claim is:

> Noise repair is distributed across a broader hidden-space mechanism. The Noise-trained adapter's transfer to Blur, however, depends strongly and specifically on the SAE-discovered shared subspace.

We should not repeat the older claim that the same 16 directions uniquely mediate the adapter's Noise benefit.

---

## 7. How Block-6 Repair Propagates

Rank-8 propagation was evaluated through Blocks 6–12 on 10,000 reserve Noise-4 images.

| Stage | Patch cosine gain | CLS cosine gain | True-class margin gain |
|---:|---:|---:|---:|
| Block 6 | +0.0261 | 0.0000 | 0.0000 |
| Block 7 | +0.0277 | −0.0023 | +0.0056 |
| Block 10 | +0.0225 | −0.0032 | +0.0381 |
| Block 11 | +0.0222 | +0.0031 | +0.1266 |
| Block 12 | +0.0238 | **+0.0223** | **+0.2227** |

The interpretation is simple:

1. The adapter directly repairs patch-token information after Block 6.
2. It does not directly modify the CLS token.
3. Blocks 7–12 propagate and transform the patch correction.
4. The useful signal increasingly appears in the CLS token and classification margin near the output.

Thus later blocks do not independently “repair again.” They convert the improved patch trajectory into better final evidence.

---

## 8. Computational Cost

| Method | Trainable parameters | Checkpoint | Added GMAC/image | Requires SAE at inference | Online gradients |
|---|---:|---:|---:|---|---|
| Full residual adapter | 741,120 | 2.83 MiB | 0.1156 | No | No |
| Rank-8 adapter | 14,632 | approximately 0.056 MiB FP32 | small; below full adapter | No | No |
| 16D raw repair | 3,408 | 0.015 MiB | 0.0048 | No | No |
| SAE gate plus adapter | 17 gate parameters plus full adapter | SAE alone 144.10 MiB | SAE encoder approximately 3.699 | **Yes** | No |

The frozen ViT has 86.57M parameters and approximately 17.56 GMAC/image under the project's analytical convention. Exact synchronized CUDA latency, p95 latency, and peak memory remain to be measured.

---

## 9. Controlled CFA Comparison

The earlier fixed Block-11 adapter was compared with CFA on the same ImageNetV2 images, online corruptions, and severities 1–5.

| Mean across 10 Noise/Blur conditions | Accuracy | Gain over baseline |
|---|---:|---:|
| Baseline | 58.09% | — |
| Fixed adapter | 59.22% | +1.13 pp |
| CFA | **59.84%** | **+1.75 pp** |

- CFA was stronger on Blur and overall.
- The fixed adapter was stronger at every tested Noise severity.
- CFA performs online test-time optimization and depends on batches.
- Our adapter performs a fixed forward correction without test-time gradients.

This was a controlled ImageNetV2 comparison, **not an official ImageNet-C reproduction**. We must not claim general state-of-the-art superiority.

---

## 10. What We Can Claim Now

### Supported claims

1. Mechanistic analysis identified corruption-sensitive feature-strength changes and a practically repairable Block-6 intervention point.
2. A frozen-backbone Block-6 residual adapter improves independent Noise-4 and Blur-4 accuracy while preserving clean accuracy.
3. Clean-identity regularization is essential for preventing unnecessary clean corrections.
4. Rank-8 provides a stable, highly parameter-efficient Noise-4 improvement with negligible clean change.
5. Noise repair is distributed, while Noise-to-Blur transfer in the full adapter depends specifically on a small SAE-discovered subspace.
6. Block-6 patch repair is progressively converted into CLS-token and margin improvement by downstream blocks.

### Claims we should not make

1. Rank-8 is currently corruption-general—it did not improve held-out Blur-4.
2. The 16 SAE features uniquely explain Noise repair—they did not beat random controls consistently for Noise.
3. SAE-space intervention is the best correction method—raw hidden-space repair was stronger.
4. The method generally beats CFA or state of the art—the official ImageNet-C/TTA benchmark is incomplete.
5. The exact SAE feature IDs are universally reproducible—multi-SAE seed/type stability remains untested.

---

## 11. Recommended Next Work

Before training a larger corruption-general adapter, complete the strongest reviewer-facing baselines:

1. **Capacity-matched architectures:** rank/parameter-matched LoRA, bottleneck MLP, linear residual, and random low-rank controls at Block 6.
2. **Exact latency table:** synchronized CUDA mean/median/p95 latency and peak memory.
3. **TTA baselines:** TENT, SAR, MEMO, CoTTA, and a pixel-space denoiser under a matched protocol.
4. **SAE reproducibility:** multiple SAE seeds and architectures with activation-pattern/Hungarian feature matching.
5. **Corruption-general adapter study:** train one shared adapter on multiple corruption families and severities, then use leave-one-family-out evaluation. Sweep rank 8/16/32/64 rather than assuming rank 8 has enough capacity.

The final corruption-general design must be frozen before touching a new independent test set.

---

## 12. Main Reproducibility Artifacts

- Split protocol: `configs/split_manifest_supervisor_v1.json`
- Supervisor log: `docs/SUPERVISOR_RESPONSE_LOG.md`
- Full Block-6 independent confirmation: `results/sae/experiment53_block6_independent_confirmation/full_10000_frozen_3seed_v1/summary.json`
- Low-rank sweep: `results/sae/experiment54_lowrank_block6_adapter/full_3seed_ranks8_128_v1/summary.json`
- Leakage-free causal subspace: `results/sae/experiment56_leakage_free_causal_subspace/full_disjoint_section67_replication_v1/summary.json`
- Rank-8 Noise propagation: `results/sae/experiment57_rank8_repair_propagation/full_noise4_reserve_3seed_v1/summary.json`
- Rank-8 clean/Blur reserve test: `results/sae/experiment58_rank8_clean_blur_reserve/full_3seed_reserve_v1/summary.json`

All percentages labeled “change” are absolute percentage-point changes, not relative percentages.
