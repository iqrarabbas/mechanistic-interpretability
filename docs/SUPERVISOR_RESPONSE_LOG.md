# Supervisor Comment Response Log

Last updated: 24 August 2026

Purpose: this is the canonical running record for supervisor feedback, our technical assessment, decisions, analyses, experiments, results, and final response text. Update it after every discussion or run so the final thesis/paper response can be compiled without reconstructing history.

## Status Vocabulary

- **Open**: not yet answered or tested.
- **Audited**: existing code/results checked; response not finalized.
- **Planned**: protocol agreed but not implemented.
- **Running**: experiment currently active.
- **Answered**: resolved from existing evidence.
- **Completed**: new analysis/experiment finished and documented.
- **Blocked**: cannot proceed without data, compute, or a decision.

## Executive Assessment

The supervisor's comments are technically strong and should be treated as a reviewer-readiness plan. The most urgent issue is data provenance. The final 10,000-image ImageNetV2 evaluation is independent of the ImageNet development data, but the current 16-feature discovery range overlaps the development range used by the initialization adapter in Experiments 23–24. Therefore the present work supports a promising result, but the strongest confirmatory claim requires a new, explicitly disjoint pipeline.

The second priority is statistical aggregation. Existing Experiment 36 NPZ files are sufficient to compute gate-versus-ungated paired McNemar tests and image-level bootstrap confidence intervals without rerunning inference. Existing summaries already contain paired tests versus the uncorrected baseline, but not the decisive gate-versus-adapter comparison.

## Verified Current Split Provenance

### SAE training

- Checkpoint: `checkpoints/sae/blur4_base_vanilla_paper`
- Recorded samples: 10,000 training and 1,000 validation images.
- The checkpoint metadata does not record explicit source indices or split boundaries.
- Status: **provenance incomplete**. The exact image IDs/indices must be reconstructed from `scripts/train_sae_level4.py` and the dataset implementation before a paper claim.

### Harmful 16-feature discovery

- Source: `results/sae/experiment17_noise_bidirectional_repair/noise_bidirectional_development_gpu/summary.json`
- Discovery: ImageNet development indices `[25000, 30000)`.
- Validation: `[30000, 32000)`.
- Exploratory evaluation: `[35000, 40000)`.
- Features: `14024, 14852, 11406, 4889, 8139, 17892, 18955, 3893, 6757, 17227, 12743, 12135, 8903, 3867, 22608, 8779`.

### Initial adapter development

- Experiment 23 training: `[25000, 30000)`.
- Experiment 23 validation: `[30000, 32000)`.
- Experiment 23 exploratory evaluation: `[35000, 40000)`.
- Experiment 24 reused the Experiment 23 checkpoint and the same ranges.
- This overlaps feature discovery/validation and is the primary circularity concern.

### Three adapter/gate training splits

Experiment 25 used three disjoint ImageNet development partitions:

| Adapter seed | Training indices | Validation indices |
|---:|---:|---:|
| 0 | `[0, 5000)` | `[5000, 7000)` |
| 1 | `[7000, 12000)` | `[12000, 14000)` |
| 2 | `[14000, 19000)` | `[19000, 21000)` |

However, all three were initialized from the Experiment 23 adapter trained on `[25000, 30000)` and validated on `[30000, 32000)`. The gate then trained on the corresponding Experiment 25 train cache and monitored the corresponding validation cache.

### Final evaluation

- Dataset: ImageNetV2 matched-frequency, not the ImageNet development set.
- Range: all 10,000 images starting at index 0.
- Conditions: paired Clean, Noise-4, and Blur-4.
- This final evaluation set is dataset-level disjoint from feature/adapter/gate development.

### Split conclusion

- No evidence of direct ImageNetV2 test leakage was found.
- There is overlap between harmful-feature discovery and the initialization adapter's training/validation data.
- The exact SAE-training image provenance is not yet documented.
- A clean confirmatory rerun should enforce three explicit groups: SAE/feature-selection development, adapter+gate training/validation, and untouched final evaluation.

## Comment 1 — Feature-Selection Protocol and Circularity

**Supervisor comment:** Precisely document which split selected the 16 features and which evaluated them. Enforce three disjoint splits.

**Status:** Protocol implemented and audited; clean confirmatory training remains pending.

**Existing answer:** ImageNetV2 final evaluation is independent. The gate-training caches are separate from Experiment 17 feature discovery. However, Experiment 17 and the Experiment 23/24 initialization adapter use overlapping ImageNet development ranges. The SAE checkpoint's exact image IDs are also absent from metadata.

**Required action:**

1. Add a machine-readable split manifest with dataset, image IDs/indices, corruption seed, and purpose.
2. Define one immutable feature-development split.
3. Define disjoint adapter/gate train and validation splits.
4. Retain ImageNetV2 as untouched final evaluation.
5. Retrain adapters from a neutral/random initialization rather than the overlapping Experiment 23 checkpoint, or retrain the initialization checkpoint on the adapter split only.
6. Rediscover the 16-feature rule using only the feature-development split.
7. Freeze every choice before final evaluation.

**Implemented artifacts:**

- Canonical manifest: `configs/split_manifest_supervisor_v1.json`
- Validator: `scripts/experiment38_split_protocol_audit.py`
- Audit output: `results/protocol/split_audit_supervisor_v1.json`

The validator found six forbidden historical intersections: four between SAE train/validation and adapter seeds 0/1, plus feature-discovery overlap with initialization-adapter training and feature-validation overlap with initialization-adapter validation. The proposed protocol has zero forbidden intersections and passes automatic isolation checks.

**Locked proposed allocation:**

- Feature development: ImageNet validation `[0,15000)`.
- Adapter/gate seed 0: train `[15000,20000)`, validation `[20000,22000)`.
- Adapter/gate seed 1: train `[22000,27000)`, validation `[27000,29000)`.
- Adapter/gate seed 2: train `[29000,34000)`, validation `[34000,36000)`.
- Unused reserve: `[36000,50000)`.
- Final evaluation: all 10,000 ImageNetV2 matched-frequency images.
- Adapter initialization must be random/neutral or trained only inside the corresponding adapter split.

**Proposed response wording:**

> The final ImageNetV2 evaluation was independent of ImageNet development, so there was no direct test-set leakage. However, our audit found that the exploratory feature-discovery range overlapped the data used by the initialization adapter. We therefore treat the current result as preliminary and will rerun the confirmatory pipeline with an explicit manifest enforcing disjoint feature-development, adapter/gate-training, and final-evaluation sets.

## Comment 2 — 1,000 Random/Permutation Controls

**Supervisor comment:** Twenty controls impose a minimum empirical p-value of `1/21 = 0.0476`; run 1,000 controls.

**Status:** Planned, but implementation must be redesigned.

**Assessment:** Correct. The existing “beat all 20” result establishes direction but is underpowered for a strong empirical-tail claim. Across three adapters there are 60 controls, but they are grouped by adapter and share data/model structure; pooling them naively does not replace 1,000 controls per locked comparison.

**Efficiency warning:** In Experiment 36, each random gate is separately trained and evaluated, so 1,000 controls are not cheap under the current implementation. The control calculation should be vectorized or moved to cached activations/logits before scaling to 1,000.

**Required action:**

- Define the null precisely: random 16-feature sets, matched for activation frequency/variance where possible.
- Precompute SAE feature statistics and adapter outputs.
- Train/evaluate random gates in vectorized batches or use a justified fixed fitting procedure.
- Use at least 1,000 controls for each locked adapter/corruption comparison.
- Report empirical p as `(1 + number(null >= observed)) / (1 + N)`.
- Preserve random seeds and feature sets in output files.

## Comment 3 — Paired McNemar Tests and Bootstrap CIs

**Supervisor comment:** Replace raw accuracy-difference claims with paired tests.

**Status:** Partly answerable now; aggregation script still required.

**Existing assets:** Every completed Experiment 36 corruption run saves 10,000-element Boolean arrays for uncorrected baseline, ungated adapter, harmful gate, and controls. Therefore no inference rerun is needed for corruption paired tests.

**Important distinction:** Existing `summary.json` McNemar values compare each method with the uncorrected baseline. The critical new test is harmful gate versus the corresponding ungated adapter.

**Current gate-versus-adapter discordance audit:**

| Adapter/gate | Noise gain | Noise b/c | Exact two-sided p | Blur gain | Blur b/c | Exact two-sided p |
|---|---:|---:|---:|---:|---:|---:|
| A0/G101 | +0.08 pp | 33/25 | 0.358 | +0.27 pp | 59/32 | 0.0061 |
| A0/G102 | +0.08 pp | 33/25 | 0.358 | +0.27 pp | 59/32 | 0.0061 |
| A0/G103 | +0.08 pp | 32/24 | 0.350 | +0.28 pp | 59/31 | 0.0042 |
| A1/G201 | +0.22 pp | 47/25 | 0.0128 | +0.53 pp | 83/30 | `6.31e-7` |
| A1/G202 | +0.19 pp | 46/27 | 0.0344 | +0.55 pp | 85/30 | `2.86e-7` |
| A1/G203 | +0.20 pp | 46/26 | 0.0245 | +0.54 pp | 84/30 | `4.25e-7` |
| A2/G301 | +0.20 pp | 40/20 | 0.0135 | +0.61 pp | 79/18 | `2.72e-10` |
| A2/G302 | +0.20 pp | 40/20 | 0.0135 | +0.60 pp | 79/19 | `7.13e-10` |
| A2/G303 | +0.20 pp | 40/20 | 0.0135 | +0.61 pp | 80/19 | `4.39e-10` |

Here `b` means adapter wrong/gate correct and `c` means adapter correct/gate wrong.

**Interpretation:** Blur's incremental gate benefit is individually significant in all nine runs. Noise is significant for adapters 1 and 2 but not adapter 0. The honest claim is therefore stronger for Blur; Noise is directionally consistent but smaller and seed-dependent in paired significance.

**Required action:** Build the planned aggregator to add paired bootstrap CIs and correctly aggregate repeated gate seeds without pretending that 90,000 evaluations are independent.

**Clean limitation:** Experiment 36 NPZ files do not save per-image clean predictions for every method, only clean summary accuracies. A future confirmatory run should save paired clean outcomes for gate versus adapter.

## Comment 4 — Sign Tests

**Supervisor comment:** Quote the sign test for 9/9 and 60/60.

**Status:** Answered mathematically; reporting caveat required.

- Nine positive gate-versus-adapter runs: one-sided exact sign-test `p = 1/2^9 = 0.001953125`.
- Sixty of sixty random controls beaten: a naive one-sided sign-test gives `p = 1/2^60 = 8.67e-19`.

**Caveat:** The 9 gate runs are nested within 3 adapter checkpoints and reuse the same 10,000 images. The 60 random controls also share adapters, data, and the same harmful comparator. These are not 9 or 60 fully independent Bernoulli trials. Report the sign tests as descriptive robustness checks, not the sole inferential evidence. The paired image-level tests and 1,000-control empirical null are more defensible.

## Comment 5 — Parameters, FLOPs, and Latency

**Supervisor comment:** Add a computational-cost table.

**Status:** Open.

**Known now:**

- Frozen ViT backbone.
- Residual adapter: 741,120 trainable parameters, approximately 2.83 MiB in FP32.
- SAE abnormality gate: 17 trainable parameters.
- No test-time gradient updates.

**Must measure, not estimate:**

- Total and trainable parameters for baseline, adapter, and gate.
- Additional FLOPs/MACs at batch 1 and the evaluation batch size.
- GPU latency with warm-up, synchronization, fixed batch sizes, mean/median/p95, and repeated trials.
- Peak GPU memory.
- Compare with CFA/TTA online-update cost under the same hardware and batch protocol.

## Requested Experiment 1 — Adapter Layer Ablation

**Objection:** “Layer 11 is not special.”

**Status:** Open; high priority after split repair.

Train the same parameter-matched adapter at Blocks 6, 8, 9, 10, 11, and 12 using identical disjoint data, initialization policy, optimization, and three seeds. Evaluate Clean, Noise-4, and Blur-4 with paired tests. This tests whether oracle restoration at Block 11 predicts the best learned repair location.

## Requested Experiment 2 — Non-SAE Gate Baselines

**Objection:** “You do not need an SAE.”

**Status:** Open; high priority.

Compare the locked SAE gate against parameter/computation-matched gates derived from:

- corruption delta difference-of-means;
- PCA-16 directions;
- top-variance hidden neurons;
- supervised linear-probe directions.

All direction selection must use the same feature-development split. Gate fitting must use only the adapter/gate split. Compare paired accuracy, clean preservation, latency, and parameter count.

## Requested Experiment 3 — Clean-Importance Feature Gate

**Objection:** “You only found generally important directions.”

**Status:** Open; essential mechanistic control.

Select the top 16 features using clean-only importance criteria with no corruption labels or deltas, then fit the same gate. Match feature activation frequency/scale if possible. Compare against harmful features on clean, Noise-4, and Blur-4. This complements the existing clean-subspace ablation and difference-in-differences evidence.

## Requested Experiment 4 — Capacity-Matched Architectures

**Objection:** “It is just capacity.”

**Status:** Open.

Compare:

- parameter-matched LoRA at Block 11;
- parameter-matched no-SAE residual adapter;
- MLP-only adapter;
- current adapter plus SAE gate.

Keep the backbone frozen, training data identical, parameter budget reported, and three seeds preserved. “No-SAE adapter” must be defined carefully because the current base adapter already does not use the SAE; the meaningful comparison is the ungated adapter versus SAE-gated adapter and matched alternative gates.

## Requested Experiment 5 — TTA and Pixel Baselines

**Objection:** “Existing test-time adaptation does this better.”

**Status:** Partly addressed, otherwise open.

Existing controlled evidence includes CFA across Blur/Noise severities 1–5. CFA is stronger on Blur and overall; the fixed adapter is stronger at each tested Noise severity and avoids online optimization. Add TENT, SAR, MEMO, CoTTA, and a pixel-space denoiser only under an identical backbone, data order, batch/reset protocol, and corruption implementation. Report accuracy together with latency/memory because operating cost is part of the contribution.

## Requested Experiment 6 — Feature-Count Sweep

**Objection:** “Sixteen is arbitrary.”

**Status:** Partly addressed, incomplete.

Experiment 37 tested feature counts 16, 32, 64, and 128 across scales; 64/1.5 was selected on validation but did not beat the original 16/2 gate on Blur. The requested full sweep adds `k = 1, 4, 8` and should use the repaired disjoint protocol. Do not select on final ImageNetV2 outcomes.

## Requested Experiment 7 — SAE Hyperparameter and Seed Stability

**Objection:** “SAE features are not reproducible.”

**Status:** Open; highest scientific priority after leakage repair.

Required dimensions:

- width/expansion sensitivity;
- sparsity strength;
- Vanilla L1, TopK, BatchTopK, and JumpReLU where implementation is validated;
- at least 3 SAE seeds per chosen family;
- feature matching across SAEs using activation-pattern similarity on a held-out matching set plus Hungarian assignment;
- decoder-direction cosine as a secondary matching criterion;
- stability of harmful ranking, sign of corruption change, margin association, and downstream gated benefit.

Do not expect identical feature indices across independently trained SAEs. The valid question is whether an equivalent matched subspace or feature family re-emerges. Matching data must be separate from final evaluation. Report dead-feature rates, reconstruction error, sparsity, matched-feature similarity, subspace principal angles, and performance after transferring/reselecting matched features.

This experiment can produce either outcome:

- Stable matched harmful features/subspaces would strongly strengthen the mechanistic claim.
- Instability would require reframing the claim at the subspace/function level rather than around 16 specific latent IDs.

## Recommended Order of Work

1. **Freeze and document split protocol.** Do not launch more confirmatory claims before resolving leakage/provenance.
2. **Build the non-expensive paired-statistics aggregator.** Use existing NPZ files; produce McNemar b/c, bootstrap CIs, sign tests, and tables.
3. **Measure params/FLOPs/latency.** Cheap and immediately useful.
4. **Implement the fully disjoint confirmatory pipeline.** Rediscover features and retrain adapter/gate without overlapping development ranges.
5. **Run the full 1,000-control empirical null efficiently.** Prefer vectorized cached computation.
6. **Run SAE seed/hyperparameter stability.** Highest-value mechanistic experiment.
7. **Run Block-location and non-SAE/clean-importance controls.** These test specificity.
8. **Run capacity-matched alternatives and k-sweep.** Preserve three seeds.
9. **Expand TTA baselines.** Do this after the core causal claim is secure.

## Response Tracker

| ID | Item | Status | Existing evidence | Next artifact |
|---|---|---|---|---|
| C1 | Disjoint feature/adapter/eval splits | Planned | Validator proves old overlap and proposed isolation | Clean confirmatory rerun under locked manifest |
| C2 | 1,000 controls | Planned | 20 per adapter currently | Vectorized null experiment |
| C3 | Paired tests and CIs | Audited | Per-image NPZ available | Aggregator JSON/CSV/plot |
| C4 | Sign tests | Answered | 9/9 and 60/60 counts | Include caveat in report |
| C5 | Params/FLOPs/latency | Open | Adapter/gate params known | Benchmark table |
| E1 | Layer ablation | Open | Oracle Layer 11 localization only | Three-seed layer study |
| E2 | Non-SAE gates | Open | Hidden projection and constant controls only | Matched baseline study |
| E3 | Clean-importance gate | Open | Clean ablation/difference-in-differences | Clean-only feature control |
| E4 | Capacity controls | Open | Ungated adapter exists | LoRA/MLP/param-matched study |
| E5 | TTA baselines | Partial | CFA severity sweep complete | TENT/SAR/MEMO/CoTTA/denoiser |
| E6 | k sweep | Partial | 16/32/64/128 tested | Add 1/4/8 with clean split |
| E7 | SAE stability | Open | One principal SAE seed/type | Multi-SAE matching study |

## Running Session Log

### 24 August 2026 — Initial supervisor-comment audit

- Read all supervisor comments and mapped them to completed Experiments 1–37.
- Verified ImageNetV2 final evaluation is independent.
- Identified overlap between Experiment 17 feature discovery and Experiment 23/24 initialization-adapter development.
- Identified missing exact image provenance in SAE checkpoint metadata.
- Confirmed existing Experiment 36 NPZ files support gate-versus-adapter paired tests without rerunning inference.
- Computed preliminary gate-versus-adapter McNemar b/c counts and exact p-values.
- Computed exact one-sided sign-test values for 9/9 and 60/60, with dependence caveats.
- No new model training or evaluation was launched.

### 24 August 2026 — Split protocol implementation

- Traced SAE training to ImageNet validation `[0,10000)` and validation to `[10000,11000)`.
- Added `configs/split_manifest_supervisor_v1.json` with historical and proposed protocols.
- Added `scripts/experiment38_split_protocol_audit.py`.
- Historical audit found six forbidden feature-development versus adapter/gate intersections.
- Proposed three-group protocol found zero forbidden intersections and passed validation.
- Saved machine-readable report to `results/protocol/split_audit_supervisor_v1.json`.
- No training or final evaluation was run; the next decision is how to implement clean feature rediscovery and from-scratch three-seed adapter/gate training under this manifest.

### 24 August 2026 — Leakage-free harmful-feature rediscovery

- Run: `supervisor_disjoint_feature_dev_v1`
- Output: `results/sae/experiment17_noise_bidirectional_repair/supervisor_disjoint_feature_dev_v1/summary.json`
- Discovery: ImageNet validation `[0,5000)`.
- Validation: `[11000,13000)`.
- Feature-development dry-run: `[13000,15000)`.
- Adapter/gate and ImageNetV2 data were untouched.
- Found 51 over-activated and 15 under-activated candidates.
- Fixed a mode-eligibility bug that had incorrectly required enough under-features before testing an over-only feature count. Cached discovery was reused; discovery was not repeated.
- Feasible candidates after the fix: over-only `k = 8,16,24`; under-only `k = 8`; bidirectional `k = 8`.
- Validation selected `over_features8_patches32_alpha0.5`, not 16 features.
- Selected eight: `14024, 14852, 4889, 11406, 18955, 22608, 6757, 8139`.
- All 8/8 selected features belonged to the historical harmful-16 set.
- The leakage-free top-16 ranking shared 13/16 features with the historical harmful-16 set. New top-16-only entries were `9977, 2664, 23000`; historical entries not in the new top 16 were `17892, 3867, 12135`.
- Validation direct residual repair: Noise-4 +0.05 pp, Clean +0.05 pp, 2 recovered and 1 damaged.
- Feature-development dry-run: Noise-4 0.00 pp, Clean 0.00 pp, 4 recovered and 4 damaged; margins improved slightly.
- Interpretation: feature ranking is strikingly stable across disjoint development data, but direct SAE correction remains weak. The next adapter/gate experiment must distinguish the validation-selected top 8 from the preregistered top-16 gate and must not state that validation selected 16.

### 24 August 2026 — Disjoint adapter trainer launched

- Added explicit `checkpoint`, `zero`, and `random` initialization support to `train_variant` in Experiment 24; legacy behavior remains checkpoint initialization.
- Added `scripts/experiment39_disjoint_adapter_training.py`.
- The new script reads exact seed splits from `configs/split_manifest_supervisor_v1.json`, trains only on ImageNet validation development partitions, and contains no ImageNetV2 evaluation path.
- Zero initialization makes the initial residual adapter an identity intervention and removes dependence on the overlapping Experiment 23 checkpoint.
- Smoke run `smoke_zero_init_seed0_4` completed successfully.
- Full run launched as `full_disjoint_zero_init_3seed_v1` with adapter seeds 0, 1, and 2, five epochs, classification/preservation weights 0.05, and Block 11 fixed.
- Active process session at launch: `33054`.
- Final ImageNetV2 remains untouched.

### 24 August 2026 — Disjoint adapter training completed

- Full output: `results/sae/experiment39_disjoint_adapter_training/full_disjoint_zero_init_3seed_v1`.
- All three zero-initialized Block-11 adapter checkpoints were saved successfully.
- Seed 0 validation: 68.85% to 70.30%, +1.45 pp; 57 recovered, 28 damaged; McNemar `p = 0.00219`.
- Seed 1 validation: 69.65% to 70.90%, +1.25 pp; 52 recovered, 27 damaged; McNemar `p = 0.00655`.
- Seed 2 validation: 70.75% to 71.10%, +0.35 pp; 50 recovered, 43 damaged; McNemar `p = 0.534`.
- Mean validation gain across seeds: +1.02 pp.
- Interpretation: all three gains are positive, but seed 2 is weak and not individually significant. This is honest development evidence, not final evaluation.
- The current validation function did not apply each adapter to clean hidden states, so adapter clean-preservation must be measured before freezing the final method.
- ImageNetV2 was not accessed.

### 24 August 2026 — Paired clean-preservation audit completed

- Script: `scripts/experiment40_disjoint_adapter_clean_audit.py`.
- Output: `results/sae/experiment40_disjoint_adapter_clean_audit/full_disjoint_clean_audit_v1`.
- Used only each seed's 2,000-image locked validation cache; ImageNetV2 was not accessed.
- Seed 0: 80.45% to 80.30%, -0.15 pp; 95% CI [-0.90, +0.60] pp; 28 recovered, 31 damaged; McNemar `p = 0.795`.
- Seed 1: 80.05% to 79.80%, -0.25 pp; CI [-1.05, +0.60] pp; 31 recovered, 36 damaged; `p = 0.625`.
- Seed 2: 80.80% to 80.35%, -0.45 pp; CI [-1.15, +0.25] pp; 22 recovered, 31 damaged; `p = 0.272`.
- Mean clean change: -0.283 pp.
- Interpretation: all three point estimates are slightly negative, but every paired CI includes zero and no McNemar test is significant. Clean preservation is therefore plausible but not proven as exact identity. The gate should be judged partly by whether it reduces this clean trend while retaining corruption gain.

### 24 August 2026 — Disjoint gate development launched

- Added `scripts/experiment41_disjoint_gate_development.py`.
- Candidate gates are top 8 (selected by leakage-free direct-repair validation) and top 16 (the preregistered historical gate size), both drawn from the leakage-free over-feature ranking.
- Each gate trains only on its adapter seed's train cache and is assessed on that seed's validation cache.
- Selection score is mean Noise-4 accuracy minus twice any mean clean-accuracy loss relative to the original ViT.
- The script saves paired clean/noise outcomes for baseline, ungated adapter, and each gate.
- Smoke run `smoke_seed0_top8_4` passed.
- Full run launched as `full_top8_top16_3seed_v1` in process session `34040`.
- ViT, clean SAE, and all three residual adapters are frozen. ImageNetV2 remains untouched.

### 24 August 2026 — Disjoint gate development completed

- Output: `results/sae/experiment41_disjoint_gate_development/full_top8_top16_3seed_v1`.
- The preregistered selection score chose top-16 over top-8, but neither gate improved over the corresponding ungated adapters.
- Mean validation accuracies:
  - Baseline Noise-4: 69.75%.
  - Ungated adapter Noise-4: 70.77%, +1.02 pp.
  - Top-8 gate Noise-4: 70.52%, +0.77 pp baseline and -0.25 pp adapter.
  - Top-16 gate Noise-4: 70.57%, +0.82 pp baseline and -0.20 pp adapter.
  - Baseline Clean: 80.43%.
  - Ungated adapter Clean: 80.15%, -0.28 pp.
  - Top-8 gate Clean: 80.12%, -0.32 pp.
  - Top-16 gate Clean: 80.15%, -0.28 pp.
- Top-16 gate versus adapter Noise differences by seed: -0.10, -0.25, and -0.25 pp; all paired confidence intervals include zero.
- Top-16 gate versus baseline Noise gains by seed: +1.35, +1.00, and +0.10 pp.
- The top-16 gate did not improve mean clean preservation relative to the adapter.
- Interpretation: under the leakage-free development protocol, the adapters replicate a positive Noise benefit, but the previous incremental SAE-gate advantage does not replicate. The SAE feature ranking remains stable and mechanistically informative, yet the current gate-training rule should not be promoted as the final accuracy method.
- Decision: do not access ImageNetV2 with this gate as though it were successful. The ungated three-seed adapter is currently the stronger frozen development choice. Any revised gate must be motivated and selected entirely within development data.

### 24 August 2026 — Proposed SAE-guided patch-weighted adapter training

- Motivation: the linear SAE output gate failed because 8/16 feature values are too narrow a signal for scaling an already conditional 768-dimensional adapter.
- Proposed role for SAE: use leakage-free harmful-feature abnormality during adapter training to weight which images/patches receive more residual/classification learning pressure. The adapter still receives the full hidden state and decides the repair direction.
- At inference, the trained adapter can run without the SAE, avoiding gate failure and SAE latency.
- Do not repeat Experiment 35's simple harmful-feature auxiliary loss: it previously failed to beat random SAE guidance. The new mechanism must be patch/example weighting or curriculum, not another feature reconstruction term.
- Required matched controls across three seeds:
  1. existing uniform-loss adapter;
  2. harmful-SAE patch-weighted adapter;
  3. random-feature patch-weighted adapter;
  4. top-variance-hidden-direction weighted adapter.
- Freeze weighting formula and strength on development validation only. Report clean/noise paired outcomes, parameters, and inference cost. Do not access ImageNetV2 unless harmful-SAE weighting beats uniform and both matched controls consistently.

### 24 August 2026 — SAE patch-weighted adapter experiment launched

- Added `scripts/experiment42_sae_patch_weighted_adapter.py`.
- Harmful SAE abnormality is used only to weight per-patch residual loss during training; the adapter still receives the full 768-dimensional hidden state.
- All adapters use identical architecture, zero initialization, optimizer, data, and training schedule.
- Variants: uniform, harmful top-16 SAE weighting, random 16-feature SAE weighting, and top-16 hidden-variance-dimension weighting.
- Patch weights are normalized to mean one per image, so weighting redistributes training emphasis rather than changing average residual-loss scale.
- Deployed adapters do not require the SAE.
- Smoke run `smoke_seed0_all_controls_4` passed.
- Full three-seed development run launched as `full_3seed_patch_weighting_v1`, process session `45020`.
- ImageNetV2 remains untouched.

### 24 August 2026 — SAE patch-weighted adapter result

- Output: `results/sae/experiment42_sae_patch_weighted_adapter/full_3seed_patch_weighting_v1`.
- Uniform adapter mean Noise-4 gain over baseline: +1.02 pp across the three disjoint development splits.
- Harmful-SAE weighting changed Noise-4 accuracy by -0.25, -0.05, and -0.10 pp versus the matched uniform adapter; mean -0.13 pp.
- Random-SAE weighting changed Noise-4 accuracy by -0.05, 0.00, and 0.00 pp; mean -0.02 pp.
- Top-variance hidden weighting changed Noise-4 accuracy by +0.35, +0.15, and -0.05 pp; mean +0.15 pp, but did not beat uniform in all seeds.
- Harmful-SAE weighting changed mean clean accuracy by only +0.02 pp versus uniform, so it did not provide a meaningful clean-preservation advantage.
- Paired harmful-SAE-versus-uniform differences were nonsignificant in every seed; all bootstrap confidence intervals included zero.
- Conclusion: this specific SAE patch-weighting objective is not supported. The harmful SAE features remain useful for diagnosis and stable mechanism discovery, but emphasizing their abnormal patches during residual-loss training does not improve the adapter and performs worse than a generic hidden-variance control.
- Decision: do not evaluate this variant on ImageNetV2 and do not claim an accuracy advantage from SAE-guided patch weighting.

### 24 August 2026 — Matched direct latent repair launched

- Added `scripts/experiment43_direct_latent_repair.py` to test whether SAE coordinates are a better intervention space than equally sized raw hidden coordinates.
- Every correction method has exactly 3,408 trainable parameters at 16 coordinates: a 16-by-16 linear map, 16 biases, and 196-by-16 patch-position parameters.
- Harmful-SAE method predicts the clean-minus-Noise change in the 16 leakage-free harmful SAE coordinates and applies only the corresponding fixed SAE decoder delta residually at Block 11.
- Raw-hidden control predicts the same paired residual in the 16 highest-variance raw hidden dimensions and applies it residually at Block 11.
- Random-SAE control uses 16 non-harmful randomly selected SAE coordinates with the identical architecture.
- Coordinate inputs and residual targets are standardized using training-split statistics, preventing raw/SAE scale differences from changing loss balance.
- ViT, SAE, and coordinate bases remain frozen. Clean counterparts are used only to form development training targets; inference uses the corrupted hidden state alone.
- The existing full 768-dimensional adapter is included as a performance reference but is not parameter matched.
- Evaluation uses three disjoint adapter train/validation splits, paired bootstrap confidence intervals, McNemar tests, recovered/damaged counts, and clean accuracy. ImageNetV2 is untouched.
- Normalized smoke run `smoke_seed0_normalized_v2` passed.
- Full CUDA run: `results/sae/experiment43_direct_latent_repair/full_3seed_matched_direct_repair_v1`; session `81520`.
- Decision rule: SAE direct repair is supported only if harmful-SAE repair improves Noise in every seed and has a larger mean gain than both raw-hidden and random-SAE controls without meaningful clean damage.
- The first full run stopped before its first checkpoint, consistent with excessive memory from constructing a dense 24,576-dimensional latent delta for every patch.
- The implementation now computes the exact same residual directly with the 16 selected SAE decoder columns, eliminating the dense latent allocation without changing the intervention mathematically.
- Optimized 16-coordinate smoke run `smoke_seed0_optimized_decoder_v3` passed.
- Restarted CUDA run as `full_3seed_matched_direct_repair_optimized_v2`, session `98817`; the incomplete original directory remains preserved.

### 24 August 2026 — Historical nine-run gate aggregation

- Added `scripts/experiment44_aggregate_gate_replications.py`; it performs no inference and aggregates the paired NPZ outcomes from the nine completed Experiment 36 replications.
- Canonical output: `results/sae/experiment44_gate_replication_aggregation/historical_nine_replications_v2` with JSON, two CSV tables, and a 300-DPI paired-difference plot.
- Noise-4 harmful gate versus ungated adapter was positive in 9/9 runs: mean +0.161 pp, range +0.08 to +0.22 pp. Exact sign test: one-sided p=0.001953 and two-sided p=0.003906.
- Noise-4 paired image-level tests were individually significant in 6/9 runs; the three adapter-0 runs had confidence intervals crossing zero and McNemar p approximately 0.35.
- Blur-4 harmful gate versus ungated adapter was positive in 9/9 runs: mean +0.473 pp, range +0.27 to +0.61 pp. Exact sign test: one-sided p=0.001953 and two-sided p=0.003906.
- Blur-4 paired bootstrap intervals excluded zero and McNemar tests were significant in all 9/9 runs.
- Across the three runs containing controls, the harmful gate beat all 60/60 random-feature controls for both Noise-4 and Blur-4. This is a control-wise sign result, not a 1,000-control empirical permutation test.
- With only 20 controls per adapter, the minimum conventional rank-based empirical p-value remains 1/21=0.0476; this aggregation does not satisfy the supervisor's 1,000-control request.
- Critical limitation: the nine historical runs reuse the same 10,000 ImageNetV2 images, so gate seeds are not independent image samples. They also predate the corrected leakage-free protocol. These results quantify historical consistency but cannot rescue the failed leakage-free gate result from Experiment 41.

### 24 August 2026 — Matched direct latent repair result

- Output: `results/sae/experiment43_direct_latent_repair/full_3seed_matched_direct_repair_optimized_v2`.
- All matched methods used 3,408 trainable parameters and the same disjoint development splits.
- Harmful-SAE direct repair Noise-4 gains by seed were -0.05, +0.35, and +0.05 pp; mean +0.117 pp. Mean clean change was -0.117 pp.
- Raw-hidden direct repair Noise-4 gains were +0.50, +0.65, and +0.15 pp; mean +0.433 pp. Mean clean change was +0.067 pp.
- Random-SAE repair Noise-4 gains were -0.05, 0.00, and 0.00 pp; mean -0.017 pp.
- Harmful-SAE paired confidence intervals included zero and McNemar tests were nonsignificant in all three seeds. Raw-hidden intervals also included zero in all seeds, although its gain was positive in 3/3 seeds.
- The full 768-dimensional adapter reference gained +1.45, +1.25, and +0.35 pp; mean +1.02 pp, substantially exceeding both 16-coordinate repairs.
- Decision: direct SAE repair is not supported as the best intervention space. SAE beats random coordinates on average but loses clearly to the exactly parameter-matched raw-hidden repair. Current evidence supports SAE as a mechanism-discovery/interpretability tool for Noise, while raw hidden state is the better correction space.
- Do not access ImageNetV2 with this SAE direct-repair method and do not claim a direct SAE classification advantage.

### 24 August 2026 — SAE-discovered hidden-subspace hybrid launched

- Added `scripts/experiment45_sae_discovered_hidden_subspace.py` to disentangle predictor input from correction output basis.
- Every matched method receives the same standardized full 768-dimensional noisy Block-11 patch as input and predicts 16 correction coefficients.
- Only the frozen output basis changes: harmful-SAE decoder span, random-SAE decoder span, top-variance raw coordinate span, or clean-minus-Noise residual-PCA span.
- SAE decoder columns are orthonormalized, so the test concerns their 16-dimensional span rather than arbitrary decoder scaling.
- Projected residual targets are standardized separately per basis using only the corresponding training split, ensuring equal loss scaling.
- Every matched method has 15,440 trainable parameters: 768-by-16 linear weights, 16 biases, and 196-by-16 positional coefficients.
- ViT, clean SAE, and all bases remain frozen. Interventions are residual Block-11 hidden-state deltas; no SAE reconstruction replaces the hidden state.
- The existing full adapter is retained as a non-parameter-matched performance reference.
- Uses the three corrected disjoint development train/validation splits. ImageNetV2 is untouched.
- Decision rule: the harmful-SAE decoder subspace is supported only if it improves Noise in all three seeds and has a larger mean gain than random-SAE, top-variance, and residual-PCA controls without meaningful clean damage.
- Smoke `smoke_seed0_rank4_normalized_v2` passed.
- Full CUDA run: `results/sae/experiment45_sae_hidden_subspace/full_3seed_rank16_normalized_v1`, session `49549`.

### 24 August 2026 — SAE-discovered hidden-subspace hybrid result

- Output: `results/sae/experiment45_sae_hidden_subspace/full_3seed_rank16_normalized_v1`.
- Harmful-SAE decoder subspace Noise-4 gains were +0.15, +0.25, and +0.40 pp; mean +0.267 pp. Mean clean change was +0.067 pp.
- Random-SAE decoder subspace gains were +0.05, +0.10, and +0.25 pp; mean +0.133 pp. Harmful SAE exceeded random SAE by +0.10, +0.15, and +0.15 pp in the three seeds, but none of these direct paired differences was individually significant.
- High-variance raw-hidden subspace gains were +1.10, +0.80, and -0.20 pp; mean +0.567 pp, with mean clean change -0.05 pp. It had the highest mean but failed the all-seed consistency criterion.
- Residual-PCA gains were +1.55, -0.25, and -0.25 pp; mean +0.35 pp, with mean clean change -0.217 pp. It was strongly seed-dependent.
- Harmful-SAE repair was positive in 3/3 seeds and preserved clean accuracy well, but its paired baseline comparisons were nonsignificant in every seed.
- The preregistered SAE-support decision is false because the harmful-SAE basis did not have the largest mean gain among matched raw controls.
- Interpretation: SAE feature selection identifies a non-random, relatively stable correction span, but current evidence does not establish it as the best intervention basis. Generic raw-hidden bases can produce larger gains, although less consistently.
- Decision: do not access ImageNetV2 with this hybrid and stop additional SAE accuracy-optimization experiments. Retain SAE's defensible role as mechanism discovery/diagnosis, with suggestive but not confirmatory evidence for selecting intervention directions.

### 24 August 2026 — Deployable adapter layer comparison launched

- Added `scripts/experiment46_layer_specific_adapters.py` to answer the supervisor objection that Block 11 may not be special.
- Trains identical zero-initialized patch-residual adapters after Blocks 6, 8, 9, 10, 11, and 12 with the ViT frozen.
- Uses the same architecture, 741,120 trainable parameters, loss weights, optimizer, schedule, Noise-4 generation, disjoint development splits, and three adapter seeds at every block.
- All six adapters for a seed share one clean/Noise image forward per batch, avoiding redundant backbone extraction.
- No persistent hidden caches are created because only 4.9 GB disk is free and a six-layer cache would be unsafe. Training is performed online from images.
- Block numbering means intervention after the numbered Transformer block. The Block-12 patch-only adapter cannot affect the final CLS token because no attention block remains; it is intentionally retained as a negative control.
- Reports clean and Noise-4 paired outcomes, recovered/damaged predictions, bootstrap confidence intervals, and McNemar tests. ImageNetV2 remains untouched.
- CUDA smoke tests passed at batch sizes 2 and 4.
- Full run: `results/sae/experiment46_layer_specific_adapters/full_3seed_blocks6_8_9_10_11_12_v1`, session `55458`.
- Electricity interrupted the original process during Seed 0 before `training_record.json` was finalized. Six partial best checkpoints existed, but optimizer/scheduler state was unavailable.
- To preserve matched training rather than resume from inconsistent optimizer state, Seed 0 was restarted deterministically from zero with `--resume`; the run directory and name remain unchanged. Restarted CUDA session: `75013`.
- The restarted run completed all five Seed 0 training epochs and saved all six checkpoints plus `training_record.json`, but failed at paired evaluation because the filesystem reached 100% and multiprocessing could not create `/tmp/pymp-*` directories.
- No scientific artifacts were deleted. The disposable pip download cache was purged with `pip cache purge`, releasing approximately 5.0 GB; datasets, checkpoints, results, and Git objects were untouched.
- Experiment 46 resumed from the complete Seed 0 checkpoints with `--num-workers 0` to prevent multiprocessing temporary-file recurrence. Current CUDA session: `38493`.
- User requested a manual stop before moving the repository to a larger filesystem. Experiment 46 was stopped gracefully with SIGINT; no process remains.
- Resume state at stop: Seed 0 has all six final block checkpoints and `training_record.json`, so `--resume` will load it without retraining. Seed 1 has six partial best checkpoints but no `training_record.json` or optimizer state, so the script will restart Seed 1 deterministically from zero. Seed 2 has not started.
- Required resume command after path migration is the same Experiment 46 command with `--resume --num-workers 0`. Do not copy only checkpoints; copy the complete repository/results tree so split manifests, code, Seed 0 record, and paired outputs remain aligned.
- Repository migration verified at `/media/dr-yougart/Iqrar/vit_mi` on `/dev/sda1`, with approximately 722 GB free at verification time.
- The moved copy contains today's Experiments 46–48, supervisor log, split/configuration files, and complete Seed 0 Experiment 46 checkpoints plus `training_record.json`. Seed 1 partial checkpoints were not copied, which is acceptable because they lacked optimizer/training state and required a clean restart.
- Latest supervisor log was synchronized into the moved repository. Experiment 46 resumed from the new working directory with `--resume --num-workers 0`; CUDA confirmed. New session: `44431`.

### 24 August 2026 — Parameter and analytical compute audit

- Added CPU-only `scripts/experiment47_resource_audit.py`; no inference was rerun and Experiment 46 retained the GPU.
- Canonical pre-latency output: `results/sae/experiment47_resource_audit/frozen_methods_pre_latency_v2` with JSON, CSV, and Markdown tables.
- Frozen ViT-B/16 has 86,567,656 parameters and approximately 17.564 GMAC/image under the stated dominant-dense-operation convention.
- Full residual adapter: 741,120 trainable parameters (0.8561% of ViT), 2.830 MiB checkpoint, and 0.1156 added GMAC/image (approximately 0.66% of backbone MACs). No SAE or test-time optimization is required.
- Matched 16-dimensional raw repair: 3,408 parameters, 0.015 MiB checkpoint, and 0.0048 added GMAC/image.
- Shared-input 16-dimensional subspace repair: 15,440 parameters, 0.115 MiB checkpoint, and 0.0048 added GMAC/image.
- Historical 17-parameter SAE gate appears tiny by trainable count but requires the frozen 37.77M-parameter SAE, a 144.10 MiB SAE checkpoint, and approximately 3.699 GMAC/image for the full SAE encoder. Therefore trainable parameters alone materially understate its deployment cost.
- Exact same-hardware CUDA latency remains pending until Experiment 46 releases the GPU; it must include warmup, synchronized timing, fixed batch size, and mean/median/tail latency.

### 24 August 2026 — Leakage-free cost–accuracy trade-off

- Added `scripts/experiment48_cost_accuracy_tradeoff.py`; it joins existing disjoint-development accuracies with Experiment 47 resources without rerunning inference.
- Output: `results/sae/experiment48_cost_accuracy_tradeoff/leakage_free_development_v1` with JSON, CSV, and Markdown tables.
- Shared development baseline: 69.75% Noise-4 and 80.43% clean.
- Full adapter: 70.77% Noise-4 (+1.02 pp), clean -0.28 pp, positive 3/3 seeds, 741,120 parameters, 0.1156 added GMAC. This is the largest stable gain and current primary method.
- 16D raw repair: 70.18% (+0.43 pp), clean +0.07 pp, positive 3/3 seeds, 3,408 parameters, 0.0048 GMAC. Best current efficiency/clean trade-off, but individual paired CIs crossed zero.
- 16D SAE-discovered hybrid: 70.02% (+0.27 pp), clean +0.07 pp, positive 3/3 seeds, 15,440 parameters, 0.0048 GMAC. Suggestive but not the best matched basis.
- High-variance raw subspace: 70.32% (+0.57 pp), clean -0.05 pp, positive 2/3 seeds. Higher mean than the SAE hybrid but inconsistent.
- SAE direct repair: 69.87% (+0.12 pp), clean -0.12 pp, positive 2/3 seeds; not supported.
- Leakage-free SAE gate: 70.57% (+0.82 pp versus baseline) but remains 0.20 pp below the ungated full adapter, has the same clean loss, and requires 3.699 added GMAC plus the frozen SAE. It is not supported as an incremental method.
- Conclusion: the full adapter currently gives the best reliable accuracy; the tiny raw repair offers the best resource/clean-preservation compromise; SAE methods do not currently justify their intervention/inference cost through superior accuracy.

### 25 August 2026 — Deployable adapter layer comparison completed

- Experiment 46 completed at `results/sae/experiment46_layer_specific_adapters/full_3seed_blocks6_8_9_10_11_12_v1` using three independently trained adapters per layer.
- Mean Noise-4 changes were: Block 6 +2.20 pp, Block 8 -4.85 pp, Block 9 -8.38 pp, Block 10 -0.13 pp, Block 11 +1.05 pp, and Block 12 0.00 pp.
- Mean clean changes were: Block 6 -1.28 pp, Block 8 -4.77 pp, Block 9 -6.23 pp, Block 10 -2.03 pp, Block 11 -0.45 pp, and Block 12 0.00 pp.
- Block 6 produced the greatest corruption gain in all three seeds, but its clean loss was significant in two seeds. Block 11 produced a smaller gain with substantially better clean preservation; its clean change was nonsignificant in every seed.
- Conclusion: Block 11 is not uniquely optimal for maximum Noise-4 accuracy. Block 6 maximizes corruption accuracy, while Block 11 remains a better robustness/clean-preservation trade-off. A Block-6 clean-preservation study is warranted after the SAE-overlap control.

### 25 August 2026 — SAE Jaccard baseline control launched

- **Question/objection:** The earlier approximately 0.99 active-feature Jaccard has no unrelated-image baseline and may simply reflect a dense Vanilla SAE dictionary.
- **Decision:** Added `scripts/experiment49_sae_jaccard_controls.py` to compare each clean image with its own Blur-4 image, an unrelated clean image, an unrelated Blur-4 image, and an independently shuffled Blur-4 image.
- **Protocol:** Frozen base ViT, paper-style Blur-4 Vanilla SAE, Block-11 output, 196 corresponding patch positions, 1,000 images beginning at index 11,000, and seeded cyclic derangements with no self-pairs.
- **Metrics:** Original `z > 0` Jaccard, activation-weighted Jaccard, SAE cosine, and top-k Jaccard for k=16, 32, 64, and 128. Paired differences use 10,000 bootstrap replicates.
- **Smoke result:** On eight images, positive Jaccard was 0.989 for true clean/Blur pairs but still 0.975-0.977 for unrelated controls, confirming substantial saturation. Top-16 Jaccard was 0.357 for true pairs versus approximately 0.049-0.058 for unrelated controls, indicating that strongest-feature identity may nevertheless be meaningfully preserved.
- **Full run:** `results/sae/experiment49_sae_jaccard_controls/full_1000_jaccard_controls_v1`; completed successfully on CUDA.
- **Primary result:** The original positive Jaccard was 0.9900 for true clean/Blur pairs but 0.9794 for unrelated clean pairs and approximately 0.9801 for both unrelated/shuffled Blur controls. The paired advantage was only approximately 0.010, despite a narrow bootstrap interval excluding zero, so the raw 0.99 value is substantially saturated and must not be interpreted by itself.
- **Strong-feature control:** Top-16 Jaccard was 0.3308 for true pairs versus 0.0498 for unrelated clean, 0.0538 for unrelated Blur, and 0.0540 for shuffled Blur. True-pair minus control differences were approximately 0.277-0.281 with all 95% bootstrap intervals around 0.270-0.287 and true pairs greater for 100% of images.
- **Other top-k results:** True-pair top-32/64/128 Jaccards were 0.3232/0.3299/0.3645 versus approximately 0.051-0.096 for controls. Every paired difference had a strictly positive bootstrap interval.
- **Magnitude controls:** Weighted Jaccard was 0.7510 for true pairs versus approximately 0.607-0.608 for controls; cosine was 0.9286 versus approximately 0.808-0.812. Their paired differences were also positive with narrow bootstrap intervals.
- **Interpretation:** The old `z > 0` Jaccard is not a reliable standalone identity measure because unrelated images also overlap near 0.98. However, the strongest SAE feature identities are substantially more stable for a clean image and its own Blur-4 counterpart than for unrelated images. The corrected claim is that corruption preserves a meaningful portion of the dominant feature set while changing activation strengths—not that all positive-support identities are uniquely preserved.
- **Status/next action:** Complete. Use top-k/weighted overlap and unrelated controls in future reporting; retire the unsupported standalone “0.99 proves identity preservation” wording.

### 25 August 2026 — Block-6 clean-preservation sweep launched

- **Question:** Can the +2.20 pp mean Noise-4 gain at Block 6 be retained while reducing its -1.28 pp mean clean-accuracy loss?
- **Diagnosis:** Experiment 46 trained the adapter to predict the clean-minus-noisy residual and improve noisy classification, but it imposed no loss requiring the adapter to leave already-clean hidden states unchanged.
- **Protocol:** Added `scripts/experiment50_block6_clean_preservation.py`. The ViT remains frozen; the adapter remains patch-only, residual, and exactly 741,120 trainable parameters. The same disjoint development splits and three independently trained seeds are preserved; ImageNetV2 is untouched.
- **Intervention:** Add `SmoothL1(adapter(clean_patch), 0)` to the existing residual, Noise classification, and Noise prediction-preservation losses.
- **Sweep:** Clean-identity weights 0, 0.2, 1, 5, and 20. A smoke calibration showed identity loss around 0.017-0.019, motivating this logarithmic range rather than ineffective very small weights.
- **Selection rule:** Among variants whose mean clean loss is no worse than -0.50 pp, select the one with the highest mean Noise-4 gain. This is a development-only hyperparameter decision.
- **Validation:** Syntax check and CUDA smoke `smoke_seed0_8_v1` passed.
- **Full run:** `results/sae/experiment50_block6_clean_preservation/full_3seed_identity_sweep_v1`, CUDA session `23114`.
- **Status/next action:** Running five epochs for each of five matched variants across three seeds. Report paired bootstrap intervals, McNemar recovered/damaged counts, and clean/Noise Pareto trade-offs after completion.

### 25 August 2026 — Noise-4 SAE Jaccard control prepared

- **Question:** Does the Blur conclusion that dominant SAE feature identities are preserved relative to unrelated images also hold for Gaussian Noise-4?
- Extended `scripts/experiment49_sae_jaccard_controls.py` to support matched Blur or Noise corruption and the corresponding paper-style base SAE checkpoint.
- The Noise run will use the frozen base ViT, `checkpoints/sae/noise4_base_vanilla_paper`, Block-11 patch activations, 1,000 clean/Noise-4 pairs, unrelated-clean, unrelated-Noise, and shuffled-Noise controls.
- Metrics and inference protocol remain matched to the completed Blur control: positive-support, weighted, cosine, and top-16/32/64/128 Jaccard with 10,000 paired bootstrap replicates.
- Syntax validation passed. GPU smoke/full execution is intentionally deferred until Experiment 50 releases CUDA, avoiding concurrent-model memory pressure and changes to Experiment 50 timing.
- Planned run name: `full_noise4_1000_jaccard_controls_v1`.

### 25 August 2026 — Block-6 clean-preservation sweep completed

- **Output:** `results/sae/experiment50_block6_clean_preservation/full_3seed_identity_sweep_v1`.
- The no-identity baseline exactly reproduced Experiment 46: mean Noise-4 gain +2.20 pp and mean clean change -1.283 pp.
- Identity weight 0.2 was selected by the preregistered development rule. Its Noise-4 gains were +3.80, +2.60, and +3.00 pp across seeds; mean +3.133 pp. Clean changes were +0.45, -0.35, and 0.00 pp; mean +0.033 pp.
- Every identity-0.2 Noise gain had a strictly positive paired bootstrap interval and significant McNemar test: Seed 0 CI [+2.40,+5.20] pp, p=1.69e-7; Seed 1 CI [+1.15,+4.00] pp, p=3.56e-4; Seed 2 CI [+1.60,+4.40] pp, p=5.38e-5.
- Identity-0.2 clean changes were nonsignificant in every seed, with every paired confidence interval including zero. Thus there is no evidence of clean degradation on these development splits.
- Identity weight 1 also performed well: mean Noise gain +2.90 pp and mean clean change +0.133 pp. Stronger weights progressively reduced Noise benefit: weight 5 gave +2.167 pp and weight 20 gave +1.683 pp, while keeping clean approximately neutral.
- **Interpretation:** The original Block-6 clean loss was not an unavoidable layer trade-off. It arose because the adapter had no objective requiring zero correction on clean activations. A modest clean-identity constraint both regularized the repair and improved corruption accuracy.
- **Status/next action:** Treat identity weight 0.2 as the frozen development winner. Evaluate its checkpoints on Blur-4 as a transfer-only test without using Blur for selection, then perform a final untouched-dataset evaluation only after the method and protocol are frozen.

### 25 August 2026 — Frozen Block-6 Noise-to-Blur transfer completed

- Added `scripts/experiment51_block6_blur_transfer.py`; output: `results/sae/experiment51_block6_blur_transfer/full_3seed_frozen_identity0p2_v1`.
- The Experiment 50 identity-0.2 winner was frozen before Blur evaluation. No Blur images, labels, or metrics were used for adapter training, weight selection, or retuning.
- Blur-4 gains for identity-0.2 were +1.10, +1.50, and +1.40 pp; mean +1.333 pp. Mean clean change remained +0.033 pp.
- Seed 1 Blur CI was [+0.35,+2.65] pp with McNemar p=0.0133; Seed 2 CI [+0.20,+2.60] pp with p=0.0282. Seed 0 was directionally positive but borderline: CI [-0.05,+2.30] pp, p=0.0798.
- The matched no-identity Block-6 control transferred only +0.20 pp on average (+0.20,+0.35,+0.05 pp), with nonsignificant paired tests in all seeds, while retaining its -1.283 pp mean clean loss.
- **Interpretation:** Clean-identity regularization does more than protect clean accuracy. It substantially improves cross-corruption generalization, suggesting that it suppresses corruption-specific/overaggressive edits and encourages a reusable hidden-state repair rule.
- **Status/next action:** Identity-0.2 remains the frozen development winner. Complete the prepared Noise Jaccard control, then decide whether the full protocol is sufficiently frozen for a one-time independent final evaluation.

### 25 August 2026 — Block-6 repair propagation completed

- **Question:** Does repairing noisy patch tokens at Block 6 remain useful through the later Transformer blocks, or is the initial correction subsequently erased?
- **Protocol:** Added `scripts/experiment52_block6_repair_propagation.py`. It freezes the ViT and the three independently trained identity-0.2 adapters, intervenes only on Block-6 patch tokens, then tracks clean/noisy/corrected representations through Blocks 6-12 on each seed's disjoint 2,000-image Noise-4 validation split. ImageNetV2 was not accessed.
- **Validity control:** Replaying the unmodified Block-6 noisy state through Blocks 7-12 reproduced normal ViT logits with maximum absolute errors of `5.72e-6`, `6.20e-6`, and `7.63e-6`; all were below the preregistered `1e-4` abort threshold.
- **Output:** `results/sae/experiment52_block6_repair_propagation/full_3seed_identity0p2_propagation_v1`. Per-image paired outcomes are saved separately for every seed.
- **Representation trajectory:** At Block 6, patch cosine to clean increased by 0.0565 and relative patch L2 error fell by 0.1676 on average. The CLS token is unchanged at the intervention itself because only patches are edited. By Block 12, the corrected CLS cosine to clean increased by 0.0309 and relative CLS L2 error fell by 0.0448.
- **Decision trajectory:** Mean diagnostic true-class margin improvement emerged after Block 6 and grew from +0.0279 at Block 7 to +0.1784 at Block 11 and +0.3123 at Block 12. Mean true-class-logit improvement at Block 12 was +0.2477.
- **Final classification:** Noise-4 gains were +3.80, +2.60, and +3.00 pp; mean +3.133 pp. Paired 95% CIs were [+2.35,+5.20], [+1.20,+4.00], and [+1.55,+4.40] pp. McNemar p-values were `1.69e-7`, `3.56e-4`, and `5.38e-5`.
- **Interpretation:** The Block-6 adapter does not make every later block independently perform another explicit repair. Instead, it places patch representations on a cleaner trajectory; downstream attention progressively transfers that information into the CLS token, and the final classifier receives a representation with a substantially improved true-class margin.
- **Limitation:** Intermediate margins use the final layer norm and classifier as a diagnostic logit lens, not native classifiers trained at each block. This is a mechanistic development analysis, not a new independent test-set claim.
- **Status/next action:** Complete. The experiment directly supports the propagation hypothesis. The prepared Noise SAE Jaccard control remains the next inexpensive analysis before freezing the complete method for final evaluation.

### 25 August 2026 — Frozen Block-6 Blur propagation completed

- **Question:** Does the Noise-trained Block-6 adapter's previously observed Blur-4 transfer follow the same downstream patch-to-CLS-to-decision propagation mechanism?
- **Protocol:** Reused the validated Experiment 52 evaluator with `--corruption blur`. The same three frozen identity-0.2 Noise-trained adapters and their disjoint 2,000-image validation splits were used. Blur was not used for training, adapter selection, or tuning; ImageNetV2 was not accessed.
- **Validity control:** Manual downstream replay reproduced standard Blur inference with maximum absolute logit errors between `4.77e-6` and `5.72e-6`, below the `1e-4` abort threshold.
- **Output:** `results/sae/experiment52_block6_repair_propagation/full_blur4_3seed_identity0p2_propagation_v1`.
- **Final classification:** Blur-4 gains were +1.10, +1.50, and +1.40 pp; mean +1.333 pp. The paired 95% CIs were [-0.05,+2.30], [+0.35,+2.65], and [+0.20,+2.60] pp; McNemar p-values were 0.0798, 0.0133, and 0.0282.
- **Trajectory:** At Block 6, the Noise-trained adapter did not move Blur patches closer to paired clean patches under simple global geometry: mean cosine change was -0.00193 and relative-L2 “reduction” was -0.1018, meaning L2 distance initially increased. Nevertheless, downstream blocks transformed this edit into decision-relevant recovery. By Block 11, mean CLS relative-L2 reduction was +0.0276, true-class-logit gain +0.1877, and margin gain +0.0971. By Block 12, CLS cosine gain was +0.0110, true-class-logit gain +0.2338, and margin gain +0.1695.
- **Interpretation:** Blur transfer is real but is not accurately described as uniformly reconstructing the paired clean hidden state at Block 6. The Noise-trained adapter introduces a task-useful perturbation that later blocks convert into improved CLS evidence and classification. This supports downstream propagation while showing that global clean-state similarity is not a sufficient measure of a useful repair.
- **Limitation:** Seed 0's final gain is positive but not individually significant. Intermediate margins remain diagnostic logit-lens measurements. The experiment is a development transfer analysis rather than an untouched final-test claim.
- **Status/next action:** Complete. Report the Noise and Blur trajectories together: Noise shows direct geometric and decision recovery, whereas Blur shows cross-corruption decision recovery despite no immediate global geometric restoration.

### 25 August 2026 — Independent frozen Block-6 confirmation completed

- **Question:** Does the development-selected identity-0.2 Block-6 repair generalize to unseen clean, Noise-4, and Blur-4 images?
- **Protocol:** Added `scripts/experiment53_block6_independent_confirmation.py`. Evaluated the three frozen adapters once on all 10,000 ImageNetV2 matched-frequency images. The backbone and adapters were frozen; no ImageNetV2 result was used for training, feature selection, hyperparameter selection, early stopping, or retuning. Inference used only the current image, never its clean counterpart.
- **Output:** `results/sae/experiment53_block6_independent_confirmation/full_10000_frozen_3seed_v1`; paired per-image outcomes are saved in `paired_outcomes.npz`.
- **Base ViT:** Clean 68.37%, Noise-4 57.25%, and Blur-4 50.82%.
- **Noise-4:** Corrected accuracies were 59.97%, 59.86%, and 59.61%; gains +2.72, +2.61, and +2.36 pp; mean +2.563 pp. All paired CIs excluded zero and McNemar p-values ranged from `2.80e-13` to `1.38e-16`.
- **Blur-4:** Corrected accuracies were 51.94%, 52.22%, and 52.26%; gains +1.12, +1.40, and +1.44 pp; mean +1.320 pp. All paired CIs excluded zero and McNemar p-values ranged from `2.29e-5` to `8.25e-8`.
- **Clean control:** Corrected accuracies were 68.21%, 68.30%, and 68.34%; changes -0.16, -0.07, and -0.03 pp; mean -0.087 pp. Every paired CI included zero and McNemar p-values were 0.503, 0.786, and 0.930, providing no evidence of a clean-accuracy change.
- **Interpretation:** The frozen Block-6 method generalizes beyond its ImageNet development data. It reliably improves both the trained corruption (Noise-4) and unseen-corruption transfer (Blur-4), while the small clean changes are statistically indistinguishable from zero.
- **Limitation:** ImageNetV2 has appeared in earlier historical project experiments, but these Block-6 checkpoints and identity-weight selection did not use it. This run must now remain the single frozen final evaluation for this method; do not tune against these results or rerun variants on ImageNetV2.
- **Status/next action:** Complete and confirmatory. Freeze the Block-6 identity-0.2 method and report these ImageNetV2 numbers as the primary independent result.

### 25 August 2026 — Historical Section 6/7 leakage audit

- **Question:** Were the 16-feature causal-ablation and Noise-to-Blur mediation results produced after the corrected disjoint-split protocol?
- **Finding:** No. Experiments 31-33 predate the split repair and use the historical harmful-16 feature source from Experiment 17 plus the historical Experiment 25 adapters.
- **Exact circularity:** Experiment 17 selected features on ImageNet validation `[25000,30000)` and validated them on `[30000,32000)`. Experiment 25's adapters were initialized from the Experiment 23/24 adapter trained and validated on those same respective ranges. Consequently, feature selection and adapter ancestry overlap.
- **Evaluation distinction:** Experiments 31-32 evaluated on all 10,000 ImageNetV2 images, which are dataset-level disjoint from ImageNet validation. Therefore there is no direct test-image leakage. However, independent evaluation does not eliminate the circular mechanism-selection concern created before evaluation.
- **Section 7 dependency:** Experiment 33 is an aggregation of Experiment 32's clean, Noise-4, and Blur-4 paired outcomes, so it inherits the same historical feature/adapter provenance rather than providing an independent leakage-free replication.
- **Decision:** Treat the published Section 6/7 numbers as historical/preliminary, not definitive. Rerun the causal-ablation and cross-corruption amplification tests using leakage-free rediscovered features and independently zero-initialized disjoint adapters. Use a development reserve split first and freeze the analysis before any additional final-set claim.
- **Supervisor-response wording:** “The causal-ablation evaluation images were independent ImageNetV2 images, so there was no direct test-set leakage. Nevertheless, the ablated feature set and adapter initialization share historical development data. We therefore agree that the original mechanism magnitudes may be inflated and will not present them as confirmatory until they replicate under the corrected disjoint protocol.”

### 27 August 2026 — Low-rank Block-6 adapter sweep completed

- **Question:** Can the 741,120-parameter identity-preserving Block-6 adapter be compressed while retaining most of its Noise-4 benefit?
- **Protocol:** Added `scripts/experiment54_lowrank_block6_adapter.py`. Replaced the full 768-to-768 map and full positional embedding with `up(down(h)+position_rank)`. Tested ranks 8, 16, 32, 64, and 128 using identity weight 0.2, the same three disjoint adapter train/validation splits, five epochs with early stopping, and a frozen ViT. ImageNetV2 was not accessed.
- **Preregistered selection:** Smallest rank retaining at least 90% of the full adapter's +3.133 pp mean development Noise-4 gain while keeping mean clean loss no worse than -0.50 pp.
- **Output:** `results/sae/experiment54_lowrank_block6_adapter/full_3seed_ranks8_128_v1`.
- **Results:** Rank 8 used 14,632 parameters (98.0% reduction), gained +1.533 pp Noise-4, and changed clean by -0.250 pp. Rank 16 used 28,496 parameters (96.2% reduction), gained +1.650 pp, and changed clean by -0.367 pp. Rank 32 used 56,224 parameters (92.4% reduction), gained +1.550 pp, and changed clean by -0.283 pp. Rank 64 used 111,680 parameters (84.9% reduction), gained +1.967 pp, and changed clean by -0.183 pp. Rank 128 used 222,592 parameters (70.0% reduction), gained +2.250 pp, and changed clean by -0.400 pp.
- **Paired statistics:** Rank-64 Noise gains were +2.10, +2.20, and +1.60 pp; every paired CI excluded zero and McNemar p-values were 0.00139, 0.00113, and 0.0152. Rank-128 gains were +2.30, +2.55, and +1.90 pp; every CI excluded zero and p-values were 0.000561, 0.000236, and 0.00594.
- **Clean caveat:** Mean clean tolerance passed for every rank, but Seed 1 showed significant clean losses for ranks 16, 32, and 128. Rank 64's Seed-1 loss was -0.85 pp with CI ending at zero and McNemar p=0.0675. Thus mean clean preservation alone hides seed-specific instability.
- **Selection outcome:** No rank retained the preregistered 90% target, so no formal winner was selected. Rank 128 retained 71.8% of the full gain with 70.0% fewer parameters. Rank 64 is the more attractive efficiency/clean trade-off, retaining 62.8% of the gain with 84.9% fewer parameters and the smallest mean clean loss among the larger ranks.
- **Interpretation:** The repair mechanism is compressible but not losslessly low-rank under this architecture/training recipe. Very small adapters preserve roughly half the full gain; ranks 64-128 recover more, but the full 741K adapter still provides materially higher robustness and better three-seed clean stability.
- **Status/next action:** Complete development result. Do not use ImageNetV2 to choose between ranks. If compression remains a priority, first test an intermediate rank 256 or distill the frozen full adapter into a low-rank student on development data.

### 27 August 2026 — Frozen low-rank Noise-to-Blur transfer completed

- **Question:** Do compressed Noise-trained Block-6 adapters preserve cross-corruption transfer to Blur-4?
- **Protocol:** Added `scripts/experiment55_lowrank_blur_transfer.py`. Evaluated the already-trained ranks 8, 16, 32, 64, and 128 without parameter updates on the same three disjoint 2,000-image Blur-4 validation splits. Blur was not used for training or retuning; ImageNetV2 was not accessed.
- **Output:** `results/sae/experiment55_lowrank_blur_transfer/full_3seed_ranks8_128_v1`.
- **Mean Blur-4 gains:** Rank 8 +0.35 pp, rank 16 +0.65 pp, rank 32 +0.733 pp, rank 64 +1.183 pp, and rank 128 +1.250 pp. The full 741K adapter reference was +1.333 pp.
- **Transfer retention:** Rank 64 retains approximately 88.8% of the full Blur transfer while using 84.9% fewer parameters. Rank 128 retains approximately 93.8% while using 70.0% fewer parameters.
- **Paired statistics:** Rank-64 Blur gains were +1.50, +0.90, and +1.15 pp. Seed 0 was significant; Seed 1 was borderline with CI [-0.05,+1.85] pp; Seed 2's CI excluded zero while McNemar p=0.0523. Rank-128 gains were +1.15, +1.05, and +1.55 pp; every paired bootstrap CI excluded zero, while Seed 1's McNemar p=0.0572 was borderline.
- **Clean trade-off:** Rank 64 has mean clean change -0.183 pp; rank 128 -0.400 pp. The full adapter reference is approximately clean-neutral.
- **Interpretation:** Cross-corruption transfer is more compressible than the trained-corruption gain. Rank 64 nearly preserves the full Blur benefit at roughly one-sixth the parameter count and offers the best overall efficiency/clean trade-off. Rank 128 most closely matches full Blur transfer but has twice rank 64's parameters and a larger clean loss.
- **Status/next action:** Complete development transfer result. Rank 64 is the practical compressed candidate, but no further ImageNetV2 comparison should be used for selection because that final set has already been consumed.

## Template for Future Entries

### YYYY-MM-DD — Comment/Experiment name

- **Question/objection:**
- **Decision:**
- **Protocol:**
- **Data split manifest:**
- **Command/run name:**
- **Result paths:**
- **Primary result:**
- **Paired statistics:**
- **Clean-control result:**
- **Interpretation:**
- **Limitation:**
- **Supervisor-response wording:**
- **Status/next action:**
## 2026-08-27 — Leakage-Free Replication of Sections 6–7 (Experiment 56)

- **Question:** Do the historical 16-feature causal-ablation and Noise-to-Blur transfer claims survive disjoint feature selection, adapter training, and evaluation?
- **Protocol:** The clean SAE used ImageNet validation `[0,11000)`, harmful features were rediscovered on feature-development splits, three zero-initialized Noise-4 adapters used disjoint seed-specific splits `[15000,36000)`, and causal evaluation used untouched reserve `[36000,46000)`. The 1,000 energy-matched random controls used a separate reserve subset `[46000,47000)`. ImageNetV2 was not accessed.
- **Primary 10K result:** The full adapters gained a mean `+0.74 pp` on Noise-4 and `+0.52 pp` on Blur-4, while changing clean accuracy by `-0.14 pp`. Removing the preregistered 16-feature decoder subspace cost `0.33 pp` on Noise-4, `1.18 pp` on Blur-4, and `0.19 pp` on clean images.
- **Noise conclusion:** The removal cost was significant within the 10K paired evaluation for all adapters, but it did not exceed the 1,000-control random null consistently (`p=0.510, 0.128, 0.819`). The old claim that these 16 directions uniquely mediate Noise repair is not confirmed; Noise repair appears more distributed.
- **Blur conclusion:** Removing the same 16 directions reduced Blur-4 accuracy below the uncorrected baseline for every adapter (mean `-0.65 pp` versus baseline). The cost exceeded 99.9–100% of 1,000 energy-matched controls (`p=0.000999, 0.000999, 0.001998`) and was strongly paired-significant for every seed. The Noise-to-Blur transfer mechanism is robustly confirmed without leakage.
- **Mechanistic interpretation:** The Noise-trained adapter uses a broad/redundant correction for its training corruption, but its unseen Blur benefit is concentrated in the SAE-discovered shared decoder subspace. The directions have a small general clean effect, yet their causal importance is strongly amplified under Blur.
- **Artifact:** `results/sae/experiment56_leakage_free_causal_subspace/full_disjoint_section67_replication_v1/summary.json`.
## 2026-08-27 — Rank-8 Block-6 Repair Propagation (Experiment 57)

- **Question:** Does the compressed rank-8 Block-6 adapter place corrupted representations on a better trajectory that downstream blocks convert into improved classification?
- **Protocol:** Three frozen rank-8 adapters and their corresponding frozen full adapters were evaluated on the same 10,000-image reserve split `[36000,46000)` under Noise-4. The ViT was frozen. Corrections used corrupted hidden states only; paired clean states were used exclusively to quantify representation recovery. ImageNetV2 was not accessed.
- **Rank-8 accuracy:** Gains were `+1.83`, `+1.91`, and `+1.98 pp` across seeds (mean `+1.91 pp`). The full adapters gained `+2.10`, `+2.32`, and `+2.18 pp` (mean `+2.20 pp`). Rank-8 retained about 86.7% of the held-out full-adapter gain while using 14,632 versus 741,120 trainable parameters (98.0% fewer).
- **Trajectory:** Rank-8 immediately increased clean/noisy patch cosine by `+0.0261` at Block 6, while CLS cosine and margin were unchanged at the intervention point. Mean margin gain then rose from `+0.0056` at Block 7 to `+0.0381` at Block 10, `+0.1266` at Block 11, and `+0.2227` at Block 12. Final CLS cosine improved by `+0.0223`.
- **Interpretation:** The rank-8 adapter repairs patch-token information at Block 6; downstream blocks progressively transfer that repaired information into the CLS token and decision margin. The compressed correction is weaker than the full adapter, but its final accuracy was not significantly different from the full adapter for any seed in direct paired tests (`p=0.314, 0.115, 0.462`).
- **Artifact:** `results/sae/experiment57_rank8_repair_propagation/full_noise4_reserve_3seed_v1/summary.json`.
## 2026-08-27 — Held-Out Rank-8 Clean Tradeoff and Blur-4 Transfer (Experiment 58)

- **Question:** Does the rank-8 Block-6 adapter preserve clean accuracy and retain its development Blur-4 transfer on the 10,000-image reserve split?
- **Protocol:** Only the three frozen rank-8 adapters were evaluated on paired clean/Blur-4 images `[36000,46000)`. The ViT and adapters were frozen, and each correction used only the current image's hidden state. Clean counterparts were used for analysis metrics only. ImageNetV2 was not accessed.
- **Clean result:** Baseline clean accuracy was `80.52%`. Corrected accuracies were `80.40`, `80.48`, and `80.47%`, corresponding to `-0.12`, `-0.04`, and `-0.05 pp` (mean `-0.07 pp`). Every paired bootstrap CI crossed zero and McNemar p-values were `0.404`, `0.811`, and `0.748`; there is no evidence of meaningful clean damage.
- **Blur-4 result:** Baseline Blur-4 accuracy was `64.36%`. Gains were `+0.03`, `-0.01`, and `-0.03 pp` (mean approximately `0.00 pp`). Every CI crossed zero and McNemar p-values were `0.903`, `1.000`, and `0.901`. The earlier development mean of `+0.35 pp` did not replicate on the held-out reserve.
- **Trajectory:** Rank-8 reduced Blur-to-clean relative patch L2 distance downstream, but did not improve patch/CLS cosine consistently. Mean final margin increased by only `+0.021`, without an accuracy effect. This is insufficient evidence for useful Blur repair.
- **Conclusion:** Rank-8 provides a strong held-out Noise-4 gain with essentially preserved clean accuracy, but should not be claimed to improve Blur-4. Its practical robustness benefit is Noise-specific on the current held-out evaluation.
- **Artifact:** `results/sae/experiment58_rank8_clean_blur_reserve/full_3seed_reserve_v1/summary.json`.

## 2026-08-31 — Parameter-Matched Block-6 Capacity Controls (Experiment 59)

- **Question:** Is the compact Block-6 result explained merely by adding approximately 15K trainable parameters, or does the correction architecture matter?
- **Protocol:** Compared the rank-8 positional residual adapter against a linear bottleneck, nonlinear MLP bottleneck, fixed-random-projection adapter, and Q/V LoRA. All methods used approximately 14.3K–15.4K trainable parameters, the same frozen ViT, Noise-4 objective, five-epoch/early-stopping schedule, three seed-specific disjoint train/validation splits, and paired Clean/Noise-4/Blur-4 evaluation. ImageNetV2 was not accessed.
- **Three-seed mean changes:** Rank-8 positional: Clean `-1.58 pp`, Noise `+0.42 pp`, Blur `-4.42 pp`; linear bottleneck: `-1.58`, `+0.22`, `-5.23 pp`; MLP bottleneck: `-1.68`, `+0.05`, `-5.08 pp`; random projection: `-0.43`, `+1.12`, `-0.22 pp`; Q/V LoRA: `-0.15`, `+2.45`, `-0.13 pp`.
- **Interpretation:** Equal parameter count does not produce equal robustness. Q/V LoRA is the strongest Noise-specific compact control and nearly preserves clean accuracy, while the matched bottleneck methods generalize poorly to Blur. The result rules out a simple capacity-only explanation, but it does not support the rank-8 architecture as the best compact Noise intervention under this training protocol.
- **Important scope:** These are development-validation results. Validation was separate from training but also used for early stopping/model selection, so it is not an untouched final test. A frozen winner requires one independent final evaluation.
- **Artifact:** `results/sae/experiment59_parameter_matched_block6/full_3seed_matched_capacity_v1/summary.json`.

## 2026-09-03 — Frozen Block-6 Adapter on ImageNet-A OOD (Experiment 60)

- **Question:** Does the frozen Noise-4-trained Block-6 adapter generalize to a genuinely out-of-distribution dataset that was absent from SAE training, feature discovery, adapter training, model selection, and all earlier evaluations?
- **Protocol:** Evaluated all three frozen Experiment-50 `identity_0p2` adapters on all 7,500 ImageNet-A images. No weights, seeds, thresholds, or hyperparameters were selected using ImageNet-A. In the primary standard masked-200 protocol, the frozen ViT still produces all 1,000 ImageNet logits, but top-1 prediction is computed after restricting those logits to the 200 classes represented in ImageNet-A. Full 1,000-class top-1 accuracy is reported separately as a secondary check. Masked-200 ImageNet-A accuracy is therefore not directly comparable to the full-1K ImageNetV2 accuracy reported elsewhere. Native ImageNet-A, online Noise-4, and online Blur-4 used identical paired sample order across the baseline and adapters. Inference used only the current image.
- **Standard masked-200 result:** Native baseline `25.20%`, with adapter gains `+1.11`, `+1.56`, and `+1.17 pp` (mean `+1.28 pp`). Noise-4 baseline `14.03%`, with gains `+2.25`, `+2.59`, and `+1.99 pp` (mean `+2.28 pp`). Blur-4 baseline `9.85%`, with gains `+0.79`, `+1.09`, and `+0.83 pp` (mean `+0.90 pp`).
- **Paired evidence:** Every masked-200 gain was positive, every paired bootstrap interval excluded zero, and every exact McNemar test was significant. Native p-values were `0.000286`, `4.80e-7`, and `0.000152`; Noise p-values were `1.71e-12`, `3.25e-15`, and `3.80e-10`; Blur p-values were `0.000621`, `1.43e-6`, and `0.000136`.
- **Full-1K secondary result:** Mean gains were `+0.77 pp` native, `+1.41 pp` Noise-4, and `+0.65 pp` Blur-4; all three adapter seeds improved every condition.
- **Interpretation:** The Block-6 repair is not limited to the ImageNet validation distribution. It significantly improves naturally difficult OOD images and remains beneficial after additional Noise-4 and Blur-4 corruption. The strongest effect remains on the corruption used for adapter training, but positive Blur transfer also survives the dataset shift.
- **Leakage statement:** ImageNet-A is a separate dataset and contains no images from the ImageNet validation development ranges used for SAE training, SAE feature selection, adapter training, or early stopping. It was downloaded only after the three frozen Block-6 adapter checkpoints and the evaluation protocol had already been selected. No ImageNet-A image, label, or result influenced training, feature discovery, architecture selection, seed selection, thresholds, or hyperparameters. These results must now remain evaluation-only and must not be used to modify the method.
- **Artifact:** `results/sae/experiment60_block6_imageneta_ood/full_imageneta_7500_all3_v1/summary.json`.

## 2026-09-03 — Block-6 Attention Routing vs Value Content (Experiment 61)

- **Question:** Is Noise-4 damage inside Block 6 primarily caused by corrupted attention routing or corrupted value/content vectors?
- **Protocol:** On the previously unused ImageNet validation range `[47000,50000)`, patched per-head Block-6 attention probabilities `A` and value vectors `V` before head concatenation and the attention output projection. Compared corrupted baseline, clean-`A`/corrupt-`V`, corrupt-`A`/clean-`V`, clean-`A`/clean-`V`, wrong-clean-pair `A/V`, and complete clean Block-6 output. The frozen model and paired sample order were preserved. This is an oracle causal diagnostic, not a deployable method. ImageNetV2 and ImageNet-A were not accessed.
- **Baseline:** Clean accuracy was `81.13%`; Noise-4 baseline accuracy was `71.37%`; 387 images were clean-correct and Noise-wrong.
- **Routing result:** Clean attention with corrupted values reached `71.73%`, only `+0.37 pp` (95% CI `[-0.47,+1.17]`, McNemar `p=0.425`). Routing restoration alone is not supported as the primary mechanism.
- **Content result:** Corrupted attention with clean values reached `74.73%`, a significant `+3.37 pp` (95% CI `[+2.40,+4.30]`, `p=5.91e-12`), recovering 134/387 clean-correct Noise failures. This captured 34.5% of the full Block-6 oracle accuracy gain.
- **Full attention result:** Clean attention and clean values reached `74.83%`, `+3.47 pp` (95% CI `[+2.50,+4.47]`, `p=3.81e-12`). This was only `+0.10 pp` above clean-value-only patching, showing that restoring routing adds little once content is restored.
- **Controls:** Patching unrelated clean-image attention and values reduced accuracy to `62.73%` (`-8.63 pp`), ruling out a generic benefit from injecting clean activations. Complete clean Block-6 output restored accuracy to `81.13%` (`+9.77 pp`), establishing the oracle upper bound.
- **Interpretation:** Block-6 recoverability is driven much more by the information carried in its value vectors than by where corrupted queries and keys route attention. This provides a mechanistic explanation for strong Q/V LoRA performance, with the new prediction that its useful component should be concentrated in `V` rather than `Q`. However, attention-value restoration explains only about one third of complete Block-6 recoverability; the remaining effect must involve the incoming residual stream, MLP transformation, or their interaction.
- **Next mechanistic test:** Separate Q-only from V-only LoRA/causal effects and patch the Block-6 attention-sublayer output versus MLP output to localize the remaining two-thirds.
- **Artifact:** `results/sae/experiment61_block6_attention_content/full_noise4_unused3000_v1/summary.json`.

### Blur-4 replication and cross-corruption comparison

- **Blur baseline:** Blur-4 accuracy was `65.57%`; 535 images were clean-correct and Blur-wrong.
- **Routing only:** Clean attention with corrupted values reached `66.00%`, only `+0.43 pp` (95% CI `[-0.27,+1.13]`, McNemar `p=0.271`). As for Noise, routing restoration alone was not significant.
- **Content only:** Corrupted attention with clean values reached `68.57%`, a significant `+3.00 pp` (95% CI `[+2.17,+3.87]`, `p=4.96e-12`), recovering 120/535 clean-correct Blur failures.
- **Full attention:** Clean attention and values reached `70.47%`, a significant `+4.90 pp` (95% CI `[+3.97,+5.87]`, `p=2.01e-25`), recovering 167/535 failures. Full attention exceeded value-only patching by `+1.90 pp` (95% CI `[+1.17,+2.63]`, paired `p=4.36e-7`). Thus routing matters for Blur through interaction with restored content, despite being ineffective alone.
- **Controls and upper bound:** Wrong-pair clean attention/values reduced accuracy by `-7.43 pp`. Complete clean Block-6 output restored `+15.57 pp`. Value-only and full-attention restoration captured 19.3% and 31.5% of the complete Block-6 oracle gain, respectively.
- **Cross-corruption conclusion:** Both corruptions primarily damage value/content. For Noise, adding clean routing beyond clean values gave only `+0.10 pp` (95% CI `[-0.60,+0.80]`, `p=0.853`). For Blur, the additional routing contribution was a significant `+1.90 pp`. Blur therefore produces a broader attention mechanism failure: content corruption is primary, but correct routing becomes useful once content is restored. Most complete Block-6 recoverability remains outside the attention `A/V` decomposition, motivating residual-stream-versus-MLP restoration next.
- **Blur artifact:** `results/sae/experiment61_block6_attention_content/full_blur4_unused3000_v1/summary.json`.

## 2026-09-03 — ImageNet-A Replication of Block-6 Mechanism (Experiment 62)

- **Question:** Does the Block-6 routing-versus-value mechanism replicate on a larger, naturally shifted OOD dataset?
- **Protocol:** Repeated the frozen Experiment-61 per-head `A/V` patching protocol on all 7,500 ImageNet-A images under Noise-4 and Blur-4. The primary result uses the standard masked-200 ImageNet-A protocol. The experiment used no adapter, training, feature selection, or parameter tuning; paired clean ImageNet-A activations were used only as oracle causal patches.
- **Noise-4:** Baseline `13.93%`; clean routing/corrupt values `14.03%` (`+0.09 pp`, CI crosses zero, `p=0.723`); corrupt routing/clean values `15.27%` (`+1.33 pp`, `p=1.15e-6`); clean routing/clean values `16.73%` (`+2.80 pp`, `p=1.23e-20`); complete clean Block-6 output `25.20%` (`+11.27 pp`).
- **Blur-4:** Baseline `9.85%`; clean routing/corrupt values `9.88%` (`+0.03 pp`, CI crosses zero, `p=0.939`); corrupt routing/clean values `11.43%` (`+1.57 pp`, `p=7.46e-14`); clean routing/clean values `12.39%` (`+2.53 pp`, `p=2.49e-26`); complete clean Block-6 output `25.20%` (`+15.35 pp`).
- **Direct interaction tests:** Adding clean routing after values were restored contributed `+1.47 pp` for Noise (95% CI `[+1.00,+1.96]`, paired `p=2.22e-9`) and `+0.96 pp` for Blur (95% CI `[+0.59,+1.33]`, `p=5.86e-7`). Routing alone remained ineffective, but correct routing became useful when paired with correct content.
- **Controls:** Wrong-image clean `A/V` reduced Noise accuracy by `-3.96 pp` and Blur by `-2.32 pp`, confirming pair-specific causal information rather than generic clean activation injection.
- **Replication conclusion:** The central finding replicated across dataset shift: restoring value/content is consistently beneficial while restoring attention routing alone is not. On ImageNet-A, routing-value interaction is significant for both corruptions, whereas on the ImageNet validation analysis it was prominent for Blur but negligible for Noise. Thus value corruption is the robust shared mechanism; the magnitude of routing synergy is dataset-dependent.
- **Remaining mechanism:** Full attention restoration captured only 24.9% of full Block-6 Noise recovery and 16.5% of full Blur recovery on ImageNet-A. Most recoverability still lies in the incoming residual stream, MLP, or their interaction.
- **Artifact:** `results/sae/experiment62_block6_imageneta_mechanism/full_7500_noise_blur_v1/summary.json`.

## 2026-09-03 — Block-6 Residual/Attention/MLP Decomposition (Experiment 63)

- **Question:** Does Block 6 itself create the recoverable corruption damage, or is it primarily a useful boundary where damage accumulated in the incoming residual stream can still be corrected?
- **Protocol:** On the established 3,000-image mechanistic split `[47000,50000)`, decomposed Block 6 exactly as `z = x + a + m`, where `x` is the incoming residual stream, `a` is the attention output after `o_proj`, and `m` is the MLP output. Evaluated all eight clean/corrupted component combinations, computed accuracy Shapley contributions, added a sequential clean-attention/recomputed-MLP intervention, and included a wrong-pair complete-block control. Maximum additive reconstruction error was below `3.7e-4`. The frozen ViT used no adapter or training.
- **Noise-4 baseline:** `72.53%`, versus clean `81.13%`; full clean Block-6 restoration recovered `+8.60 pp`.
- **Noise single components:** Clean incoming residual `x` gave `+7.27 pp`; clean attention output `a` gave `+2.47 pp`; clean MLP output `m` gave `+1.17 pp`. All paired tests were significant. Sequentially inserting clean attention and recomputing the MLP gave `+3.27 pp`.
- **Noise Shapley attribution:** Incoming residual `x`: `+6.04 pp` (70.3% of total oracle gain); attention `a`: `+1.36 pp` (15.8%); MLP `m`: `+1.19 pp` (13.9%).
- **Blur-4 baseline:** `65.57%`, versus clean `81.13%`; full clean Block-6 restoration recovered `+15.57 pp`.
- **Blur single components:** Clean incoming residual `x` gave `+13.30 pp`; clean attention output `a` gave `+3.90 pp`; clean MLP output `m` gave `+1.07 pp`. Sequentially inserting clean attention and recomputing the MLP gave `+4.90 pp`.
- **Blur Shapley attribution:** Incoming residual `x`: `+11.71 pp` (75.2% of total oracle gain); attention `a`: `+2.83 pp` (18.2%); MLP `m`: `+1.03 pp` (6.6%).
- **Control:** Replacing Block 6 with an unrelated clean image's full output collapsed accuracy to approximately `0.13%` for both corruptions, confirming image-specific causal information.
- **Interpretation:** Block 6 is highly recoverable mainly because it is a strategically early correction boundary: roughly 70–75% of the recoverable signal is already damaged in the representation entering Block 6. Block-6 attention contributes a smaller but meaningful correction opportunity, while its MLP is not the dominant source of damage. This explains why editing after Block 6 can redirect the later trajectory without requiring Block 6 itself to be the primary corruption-generating module.
- **Methodological caveat:** Factorial additive swaps create hybrid states; Shapley attribution averages across those contexts. The sequential attention intervention supports the attention conclusion under a more on-path recomputation, but clean-input restoration necessarily includes damage accumulated in Blocks 1–5.
- **Artifacts:** `results/sae/experiment63_block6_residual_mlp/full_noise4_3000_v1/summary.json` and `results/sae/experiment63_block6_residual_mlp/full_blur4_3000_v1/summary.json`.

## 2026-09-03 — Early-Block Corruption Origin (Experiment 64)

- **Question:** Which of Blocks 1–5 first creates most of the corruption damage already present at the input to Block 6?
- **Protocol:** Repeated Experiment 63 independently at Blocks 1–5, then aggregated Blocks 1–6. Every block used the same 3,000 paired images `[47000,50000)`, identical Noise-4/Blur-4 implementations and seeds, the frozen ViT, all eight `x/a/m` factorial combinations, paired bootstrap intervals, and exact McNemar tests. No adapter, feature selection, model training, or deployment tuning occurred.
- **Noise-4 result:** At Block 1, the full clean-block oracle recovered `+8.60 pp`. Shapley attribution assigned `+6.42 pp` to the Block-1 MLP (74.6% of the total), `+1.25 pp` to attention, and `+0.93 pp` to the incoming patch-embedding residual. At Block 2, the incoming-residual attribution rose to `+5.66 pp`, showing that the Block-1 damage had entered the residual stream.
- **Blur-4 result:** At Block 1, the full clean-block oracle recovered `+15.57 pp`. Shapley attribution assigned `+11.62 pp` to the Block-1 MLP (74.6%), `+2.43 pp` to attention, and `+1.52 pp` to the incoming residual. At Block 2, incoming-residual attribution rose to `+8.71 pp`.
- **Answer:** For both Noise-4 and Blur-4, **Block 1's MLP is the earliest dominant corruption-damage source**. Approximately three quarters of all recoverable classification damage is causally attributable to that first MLP at the earliest boundary. By Block 2, much of this damage is carried forward in the incoming residual stream; Blocks 2–5 then transform and accumulate it rather than creating the majority from scratch.
- **Why Block 6 remains useful:** Damage originates much earlier, but Block 6 remains a strong deployable intervention point because its representation still retains sufficient recoverable information and later blocks can propagate a compact correction into the CLS prediction.
- **Leakage/status:** This is paired-clean oracle mechanism analysis, not inference-time repair. The `[47000,50000)` split is now explicitly a reused mechanistic-analysis split and is not claimed as untouched final evaluation. Because no architecture, weights, or hyperparameters were selected from these results, this experiment does not contaminate the locked deployment evaluations; future design choices motivated by it must be validated on separate data.
- **Caveat:** Factorial component swaps can produce hybrid off-manifold states. Shapley values reduce ordering bias by averaging interactions but do not prove a single-neuron mechanism. The claim is therefore at the sublayer level: Block-1 MLP computation is the earliest dominant causal source under this intervention definition.
- **Artifacts:** `results/sae/experiment64_early_block_damage_origin/full_noise_blur_blocks1_6_3000_v1/summary.json`, `results/sae/experiment64_early_block_damage_origin/full_noise_blur_blocks1_6_3000_v1/blockwise_damage_origin.csv`, and `results/sae/experiment64_early_block_damage_origin/full_noise_blur_blocks1_6_3000_v1/blockwise_damage_origin.png`.

## 2026-09-04 — Deployable Block-1 Adapter Control (Experiment 46 Extension)

- **Question:** Does intervening at the earliest dominant damage source, Block 1, work better than the previously validated Block-6 repair point?
- **Controlled protocol:** Trained the same `741,120`-parameter patch-token residual adapter used in Experiment 46 after Block 1. Preserved all original settings: frozen ViT, Noise-4, five epochs with patience 2, learning rate `1e-4`, classification and preservation weights `0.05`, training alpha `0.5`, evaluation alpha `1.0`, and three independent seed-specific development train/validation splits from `configs/split_manifest_supervisor_v1.json`.
- **Seed 0:** Noise-4 `68.85% -> 70.65%` (`+1.80 pp`, 95% CI `[+0.15,+3.50]`, McNemar `p=0.040`); clean `80.45% -> 73.35%` (`-7.10 pp`).
- **Seed 1:** Noise-4 `69.65% -> 70.80%` (`+1.15 pp`, 95% CI `[-0.60,+3.00]`, `p=0.224`); clean `80.05% -> 73.10%` (`-6.95 pp`).
- **Seed 2:** Noise-4 `70.75% -> 70.60%` (`-0.15 pp`, 95% CI `[-1.80,+1.55]`, `p=0.909`); clean `80.80% -> 72.70%` (`-8.10 pp`).
- **Aggregate:** Mean Noise-4 change was only `+0.93 pp`, with one negative seed and only one individually significant seed. Mean clean change was `-7.38 pp`.
- **Block-6 comparison:** Under the exact same Experiment-46 protocol, Block 6 achieved mean Noise-4 `+2.20 pp` and mean clean `-1.28 pp`. Block 1 is therefore worse on both robustness gain and clean preservation.
- **Mechanistic interpretation:** The earliest location that generates damage is not necessarily the safest location to intervene. Block-1 features are generic low-level representations used by both clean and corrupted inputs, so a learned correction strongly distorts normal computation. By Block 6, corruption damage is represented in a more repairable form while enough downstream computation remains to propagate the correction.
- **Leakage/status:** Each adapter seed used its locked, disjoint train/validation ranges; ImageNetV2 was not accessed. This is a development layer comparison, not a final held-out claim. Any frozen winner must still be evaluated once on an independent final set.
- **Artifact:** `results/sae/experiment46_layer_specific_adapters/full_3seed_block1_exact_protocol_v1/summary.json`.

## 2026-09-04 — Block-1 Adapter Downstream Propagation (Experiment 65)

- **Question:** When a corrupted image is corrected after Block 1, do later blocks preserve and improve that correction, and how does the same intervention affect clean images?
- **Protocol:** Froze the three Experiment-46 Block-1 adapters and the ViT. For each seed, evaluated its locked 2,000-image validation split and propagated four conditions through Blocks 1–12: clean, adapted clean, Noise-4, and adapted Noise-4. Saved paired per-image representation metrics, true-class logits, margins, and final correctness. No training, selection, or ImageNetV2 access occurred.
- **Final Noise-4:** Seed gains were `+1.80`, `+1.15`, and `-0.15 pp`; mean `+0.93 pp`. Only Seed 0 was individually significant. This exactly reproduces the direct Experiment-46 evaluation.
- **Final clean:** Seed changes were `-7.10`, `-6.95`, and `-8.10 pp`; mean `-7.38 pp`. All three clean degradations were highly significant.
- **Diagnostic logit-lens trajectory:** Mean Noise accuracy changed from `4.08% -> 5.62%` at Block 6, `9.73% -> 12.50%` at Block 8, `24.38% -> 28.12%` at Block 10, and `69.75% -> 70.68%` at Block 12. Thus later blocks initially convert part of the patch correction into a better CLS trajectory, but the advantage is small by the final classifier.
- **Clean trajectory:** Mean clean diagnostic accuracy changed from `8.57% -> 4.92%` at Block 6, `18.18% -> 11.15%` at Block 8, `38.47% -> 26.93%` at Block 10, and `80.43% -> 73.05%` at Block 12. The early adapter's distortion is progressively converted into substantial classification damage.
- **Margin trajectory:** Across seeds, the adapted-Noise mean margin gain peaked around the middle of the network (`+0.17` at Block 6) and fell to about `+0.06` by Block 12. Adapted-clean margin damage grew from about `-0.22` at Block 6 to `-0.95` at Block 12.
- **Conclusion:** Blocks 2–12 do propagate a small useful part of the Block-1 Noise correction, but they also amplify a much larger clean-image distortion. Therefore, directly repairing at the earliest damage-generating layer is not sufficient or safe. Block 6 remains the superior practical intervention boundary because it offers a better robustness/clean trade-off.
- **Caveat:** Intermediate accuracies use the final ViT layernorm/classifier as a diagnostic logit lens; only Block 12 equals native final classification.
- **Artifacts:** `results/sae/experiment65_block1_adapter_propagation/full_noise4_3seed_exact_protocol_v2/summary.json`, paired outcome NPZ files in the same directory, and `logit_lens_accuracy_trajectory.json`.

## 2026-09-04 — Block-1 MLP Neuron Causality (Experiment 66)

- **Protocol:** Ranked all 3,072 Block-1 post-GELU MLP neurons on feature-development images `[0,5000)`, checked ranking reproducibility on disjoint confirmation images `[11000,13000)`, and performed causal evaluation on unused reserve images `[36000,39000)`. The ViT and three selected Block-6 adapters were frozen; ImageNetV2 was not accessed. Tested `k={16,32,64,128,256}` and decoder-output-energy-matched random-neuron controls.
- **Ranking stability:** Only `6/16` top neurons reproduced in the confirmation top 16. Thus very small neuron identities are not stable enough for a strong reproducibility claim.
- **Noise-4:** Selected restoration alone peaked at only `+0.53 pp` (`k=32`) and gave `+0.27 pp` at `k=256`. The energy-matched random `k=256` control gave `+2.03 pp`; selected `k=256` was significantly worse than random by `-1.77 pp` (paired `p=0.00039`). Adding selected `k=256` restoration to Block 6 raised mean gain from `+2.70` to `+3.16 pp`, but the incremental `+0.46 pp` was nonsignificant for every adapter seed.
- **Blur-4:** Selected restoration increased monotonically at larger sets, reaching `+5.70 pp` at `k=256`, versus `+3.80 pp` for the energy-matched random control. The selected set beat random by `+1.90 pp` (95% CI approximately `[+0.93,+2.90]`, paired `p=0.000164`). Combining selected `k=256` restoration with Block 6 produced mean `+4.83 pp`, versus `+0.93 pp` for Block 6 alone. The incremental improvements were `+3.93`, `+3.57`, and `+4.20 pp`, all highly significant.
- **Interpretation:** Block-1 MLP corruption amplification is distributed rather than concentrated in 16 stable neurons. The discovered 256-neuron subspace has Blur-specific causal relevance and complements the Noise-trained Block-6 adapter under oracle restoration. It does not identify a shared Noise-specific neuron mechanism, because random energy-matched restoration performs better for Noise.
- **Deployment limitation:** Selected-neuron restoration uses the exact paired clean activation and is therefore an oracle mechanism experiment. It cannot be included in the deployed method. On a clean image, clean-to-clean neuron restoration is mathematically identity; the clean trade-off of the deployable Block-6 adapter remains governed by its previously measured clean evaluation.
- **Artifacts:** `results/sae/experiment66_block1_mlp_neuron_causality/full_disjoint_noise_blur_k_sweep_v1/summary.json` and `direct_paired_key_comparisons.json`.

## 2026-09-07 — Mixed Noise/Blur Block-6 Adapter (Experiment 67)

- **Question:** Can one deployable Block-6 adapter jointly handle Noise-4 and Blur-4 without sacrificing clean accuracy?
- **Protocol:** Trained one full `741,120`-parameter adapter per seed using balanced Noise-4 and Blur-4 residual/classification losses plus the previously selected clean-identity weight `0.2`. The ViT remained frozen. The original three disjoint seed-specific adapter train/validation partitions were reused, with no hyperparameter sweep and no ImageNetV2 access.
- **Clean:** Changes were `+0.50`, `+0.10`, and `+0.15 pp`; mean `+0.25 pp`. Every confidence interval included zero, so there is no evidence of clean damage or improvement.
- **Noise-4:** Gains were `+3.55`, `+2.90`, and `+3.20 pp`; mean `+3.22 pp`. All three paired comparisons were significant (`p<=2.82e-5`).
- **Blur-4:** Gains were `+6.95`, `+7.60`, and `+6.95 pp`; mean `+7.17 pp`. All three paired comparisons were highly significant (`p<=9.81e-19`).
- **Comparison with Noise-only adapter:** The earlier development adapter achieved approximately `+3.13 pp` Noise, `+1.33 pp` Blur, and `+0.03 pp` clean. Mixed training retained the Noise gain, increased Blur gain by roughly `+5.84 pp`, and preserved clean accuracy.
- **Conclusion:** Within the development protocol, a single Block-6 adapter can learn both corruption-specific corrections without an apparent robustness trade-off. This is the strongest development result so far, but it is not yet evidence of generalization to unseen corruption families.
- **Leakage/status:** Training and validation are disjoint within each seed, and ImageNetV2 was not accessed. Because architecture and training composition were chosen during development, the mixed adapter must be frozen before any independent evaluation. Reusing the same validation splits for comparison is appropriate for development but does not replace final confirmation.
- **Artifact:** `results/sae/experiment67_mixed_noise_blur_block6/full_3seed_noise_blur_identity0p2_v1/summary.json`.
## 2026-09-23 — Audit of the Repeated 74.6% Block-1 MLP Share

- **Supervisor concern:** Noise-4 and Blur-4 both reported a 74.6% Block-1 MLP share, which could indicate accidental caching or result reuse.
- **Audit:** Recomputed all coalition accuracies and Shapley values directly from the two separate 3,000-image `paired_outcomes.npz` files, independently of the Experiment 64 aggregate. Also compared array hashes and elementwise outcomes.
- **Exact result:** Noise-4 is `74.6124%`; Blur-4 is `74.6253%`. They match only after rounding to one decimal place.
- **Independence evidence:** The corruption-baseline arrays differ on `605/3000` images and the MLP-only arrays differ on `342/3000`. Their hashes are distinct. Only the complete clean `x+a+m` restoration arrays are identical, as expected because both reconstruct the same clean Block-1 output.
- **Conclusion:** No caching, denominator, aggregation, or outcome-reuse bug was found. The displayed equality is a rounding coincidence.
- **Corrected scope:** Describe Block-1 MLP as the earliest dominant Transformer-sublayer amplifier among tested components, not necessarily the first source of all damage, because patch embedding was not decomposed.
- **Artifact:** `docs/EXPERIMENT64_746_PERCENT_AUDIT.md`.
