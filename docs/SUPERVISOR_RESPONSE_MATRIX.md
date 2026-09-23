# Supervisor Response Matrix

**Review source:** `Progress_report_review_Iqrar.docx`  
**Purpose:** Track every supervisor objection, the evidence currently available, the claim we may make now, and the exact work still required.

Status labels:

- **Resolved:** sufficient evidence and documentation exist for the stated scope.
- **In progress:** the required experiment or analysis is currently running.
- **Partial:** meaningful evidence exists, but the full requested control is incomplete.
- **Open:** the requested evidence has not yet been produced.
- **Negative result:** tested and unsupported; must be reported rather than hidden.

---

## Executive Priority Table

| Priority | Supervisor concern | Current status | Immediate action |
|---:|---|---|---|
| 1 | Experimental setup was invisible | **Resolved** | Use the new setup section in all reports |
| 2 | Block-6 comparison lacked identity loss | **In progress** | Complete Experiment 130 across Blocks 1–11 and three seeds |
| 3 | No single final method | **Partial** | Freeze the winner after Experiment 130; keep router/SAE variants as ablations |
| 4 | Missing TTA/PEFT/robust-training baselines | **Open/partial** | Implement a fair baseline pipeline after method freeze |
| 5 | Seen-corruption training/test protocol | **Partial** | Report held-out families for every frozen winner; add standardized benchmark |
| 6 | Mechanism did not prospectively predict architecture | **Open** | Pre-register predictions before testing another backbone |
| 7 | Analysis controls and metrics incomplete | **Partial** | Run patch-embedding attribution, oracle alpha sweep, and equal-norm controls |
| 8 | Router/MoE over-framed | **Resolved conceptually; baseline open** | Reframe as repair-or-abstain and compare standard detectors |
| 9 | Circuit random-control result was nonsignificant | **Resolved as negative** | Move to supplement and avoid causal-circuit overclaim |
| 10 | Natural corruption versus adversarial scope | **Resolved conceptually** | Explicitly scope to common corruptions |
| 11 | Only one backbone | **Open and critical** | Replicate prospectively on two additional ViTs |
| 12 | Writing and presentation weaknesses | **In progress** | Rebuild paper after final method and evidence freeze |

---

## 1. Experimental Setup Was Invisible

### Supervisor objection

The report did not clearly identify the backbone, pretraining, datasets, sample counts, corruption definitions, severities, or adapter-training corruptions. It also did not distinguish project-defined online corruptions from official ImageNet-C.

### Response

**Status: Resolved.**

The complete setup is now documented in `docs/EXPERIMENTAL_SETUP_AND_LEAKAGE_PROTOCOL.md`:

- backbone: `google/vit-base-patch16-224`;
- 1,000-class ImageNet classification;
- frozen ViT backbone;
- exact ImageNet index intervals and sample counts;
- three disjoint adapter train/validation partitions;
- online Gaussian Noise-4 and Gaussian Blur-4;
- mixed-corruption adapter training;
- adapter architecture, losses, optimizer, weights, epochs, and seeds;
- deployment assumptions and oracle-analysis boundaries;
- ImageNetV2, reserve, ImageNet-A, and ImageNet-Sketch status.

### Permitted wording

> We evaluate a frozen `google/vit-base-patch16-224` using explicitly partitioned ImageNet development data and independent/frozen-method distribution-shift evaluations. Primary Noise-4 and Blur-4 experiments use project-defined online corruptions and are not described as an official ImageNet-C reproduction.

### Remaining work

- Add the backbone's exact published pretraining description from its model card/reference.
- Build the standardized full ImageNet-C evaluation pipeline if storage and compute permit.

---

## 2. Missing Baselines

### Supervisor objection

The method lacks comparisons with TTA, robust training, and parameter-efficient fine-tuning. Without these, raw accuracy gains cannot establish competitiveness.

### Response

**Status: Partial/Open.**

Completed or partially completed controls include:

- frozen base ViT;
- controlled CFA-style comparison, explicitly not an official reproduction;
- parameter-matched Block-6 methods including Q/V LoRA and small residual adapters;
- full and low-rank residual adapters;
- SAE-space, raw-hidden, hidden-PCA, random-direction, and energy-matched controls;
- corruption-specific and mixed adapters.

Still missing from one standardized table:

- TENT, EATA, SAR, MEMO, DeYO, and normalization-statistics adaptation;
- standard LoRA across blocks, VPT, linear probe, and full fine-tuning on identical data;
- AugMix, DeepAugment, and PixMix checkpoints under matched backbones;
- pixel-space denoising baseline;
- unified latency and batch-dependence measurements.

### Permitted wording

> Current results establish efficacy relative to the frozen backbone and several parameter-matched internal controls, but do not yet establish state-of-the-art robustness against the full TTA/PEFT literature.

### Required next action

After Experiment 130 freezes the final intervention block, define one fairness protocol and evaluate all feasible baselines on the same images, corruption seeds, backbone, and paired outcomes.

---

## 3. Seen-Corruption Train/Test Protocol

### Supervisor objection

Training on Noise/Blur and evaluating on the same families does not demonstrate unseen-corruption generalization. Every model needs held-out-family results.

### Response

**Status: Partial.**

Evidence already available:

- mixed adapter evaluated on held-out online corruption families;
- frozen evaluations on ImageNet-Sketch and ImageNet-A;
- shared rank-32 and rank-128 repair spaces evaluated on held-out families;
- router/selector unseen-family gains were small or inconclusive and are not treated as headline evidence.

Important limitations:

- some held-out families are approximate online implementations rather than official ImageNet-C;
- several datasets/ranges have now influenced development decisions;
- held-out-family results are not yet assembled for every candidate baseline in one table.

### Permitted wording

> Seen-family gains demonstrate targeted repair capacity. General robustness claims rely only on separately reported held-out-family or distribution-shift evaluations.

### Required next action

Freeze one final method, then produce a per-method table containing clean, seen-family, held-out-family, average-corruption, parameters, and latency.

---

## 4. Weak or Post-Hoc Mechanism-to-Design Link

### Supervisor objection

The Q/V LoRA result does not cleanly follow from the Block-7 Query×Key Blur mechanism, performs mainly on Noise, and may resemble default LoRA rather than a mechanistically predicted architecture.

### Response

**Status: Open for the strong predictive claim.**

Current evidence supports **mechanism-informed localization and diagnosis**, but not yet the strongest claim that MI prospectively predicts the optimal parameterization.

The project has learned:

- Blur repair is strongly mediated by attention and includes Query×Key interaction;
- Noise repair is more distributed and becomes strongly integrated through late attention;
- Block 6 provides a useful practical repair boundary in the current backbone;
- Q/V LoRA is a competitive parameter-matched baseline, not yet proven to be uniquely predicted by the mechanism.

### Permitted wording

> Mechanistic analysis guided intervention localization and motivated targeted architecture comparisons. It has not yet prospectively established a universally optimal LoRA parameterization.

### Required next action

Before examining a new backbone, preregister:

1. predicted relative-depth intervention site;
2. predicted parameterization;
3. mechanism-opposed controls such as K-only, MLP-only, and wrong-block insertion;
4. success criteria.

This prospective cross-backbone test is the cleanest way to establish predictive design value.

---

## 5. No Single Final Method

### Supervisor objection

The report presents a full residual adapter, Q/V LoRA, and a router/two-expert system without declaring one primary pipeline.

### Response

**Status: Partial; pending Experiment 130.**

Current provisional hierarchy:

- **Primary candidate:** mixed Noise+Blur residual adapter with identity weight `0.2` and frozen backbone.
- **Layer:** provisionally Block 6, subject to Experiment 130's fair identity-matched sweep.
- **Efficiency variants:** low-rank adapters and compact frozen repair subspaces.
- **Baseline/ablation:** Q/V LoRA and other parameter-matched modules.
- **Interpretability tool:** SAE and causal patching.
- **Optional safety component:** repair-or-abstain detector/router, not the core method.

### Permitted wording

> The primary method will be the identity-preserving mixed-corruption residual adapter at the winning location of the controlled layer sweep. Other architectures are baselines or ablations.

### Required next action

Freeze the final block after Experiment 130. Do not add another primary architecture unless the sweep invalidates the residual-adapter design.

---

## 6. Identity Loss Could Change the Layer Ranking

### Supervisor objection

The earlier deployable layer sweep did not use clean-identity loss. Since clean damage made several layers unattractive, adding identity loss could change the ranking and weaken the Block-6 claim.

### Response

**Status: In progress.**

Experiment 130 trains the same 741,120-parameter mixed Noise/Blur adapter after every Block 1–11 using:

- identity weight `0.2`;
- classification weight `0.05`;
- preservation weight `0.05`;
- identical optimizer and epoch settings;
- three disjoint seed-specific train/validation partitions;
- paired clean, Noise-4, and Blur-4 evaluation.

Only adapter location changes. Block 12 is excluded because a patch-only intervention after the final block cannot alter the already-formed CLS representation.

### Decision rule

- If Block 6 retains the best corruption/clean trade-off, the localization claim is strengthened.
- If another block wins, the final method and narrative must change to that block.

### Artifact

`scripts/experiment130_identity_matched_layer_sweep.py`

---

## 7. Analysis-Methodology Weaknesses

### 7.1 Oracle alpha and missing layer curves

**Status: Partial.** Oracle layer localization exists, but a consolidated full alpha-by-layer curve with clearly stated interpretation is still needed.

Required clarification:

> Oracle restoration measures recoverable information under privileged clean-state interpolation; it is not a deployable accuracy estimate, and full late-layer restoration can become close to clean replay.

### 7.2 Residual attribution and patch embedding

**Status: Partial.** Blocks 1–6 attention/MLP writes were decomposed, but patch embedding was not separately attributed. Therefore, Block-1 MLP is described as the earliest dominant **Transformer-sublayer amplifier tested**, not the absolute origin of corruption.

### 7.3 Repeated 74.6% result

**Status: Resolved.** Independent recomputation from per-image arrays found:

- Noise-4: `74.6124%`;
- Blur-4: `74.6253%`.

The identical one-decimal value is a rounding coincidence. Baseline and MLP-intervention arrays are different. See `docs/EXPERIMENT64_746_PERCENT_AUDIT.md`.

### 7.4 Trajectory metrics and equal-norm controls

**Status: Partial.** Logit-lens propagation, margins, causal subspace controls, hidden-PCA controls, random controls, and energy-matched controls exist. A single consolidated comparison using CKA and cosine after removing high-norm dimensions is still missing.

### 7.5 Magnitude-versus-identity result

**Status: Quantified, figure consolidation pending.** The corrected dominant-feature analysis showed that binary positive-support Jaccard was saturated, while top-K identity overlap and magnitude-based metrics distinguished paired from unrelated images. These numbers need one paper-quality figure and table.

### Required next actions

1. Patch-embedding attribution.
2. Consolidated oracle alpha/layer curves.
3. CKA or outlier-removed similarity plus equal-norm random perturbation.
4. Publication figure for magnitude versus dominant-feature identity.

---

## 8. Router/MoE Was Over-Framed

### Supervisor objection

Two experts are not a compelling MoE, noise-versus-blur classification is easy, unseen-family gain was nonsignificant, and standard shift detectors were missing.

### Response

**Status: Conceptually resolved; detector baselines remain open.**

The router is no longer treated as the primary contribution. The defensible framing is:

> A repair-or-abstain safety mechanism should leave clean and unfamiliar clean-domain images untouched when repair utility is uncertain.

Existing detector/selector experiments found that universal corruption detection from internal statistics is difficult. Several selectors skipped useful repairs or activated unnecessarily. These negative results are retained.

### Permitted wording

> Routing is an optional safety ablation, not the main robustness method. Current evidence does not support a universal corruption detector.

### Required next action

Compare any retained safety gate against Mahalanobis distance, energy score, confidence/entropy, and simple intermediate-feature statistics under clean-domain shift.

---

## 9. Circuit Random Controls Were Not Significant

### Supervisor objection

Beating only 44–45 of 60 random controls corresponds to approximately `p=0.25`, not a significant circuit-specific effect.

### Response

**Status: Resolved as a negative result.**

The candidate head/neuron circuit must not be described as uniquely identified. Later block-level causal decompositions support distributed attention-mediated repair propagation, especially for Noise through Blocks 11–12, but they do not retroactively make the earlier 44–45/60 result significant.

### Permitted wording

> The original fine-grained candidate circuit did not outperform random controls significantly and is reported as a negative result. Stronger evidence exists only at the broader block/pathway level.

### Required action

Move the nonsignificant fine-grained search to the supplement or negative-results section.

---

## 10. Natural Corruptions Versus Adversarial Attacks

### Supervisor objection

The project uses language such as "attack" despite testing common corruptions rather than adaptive adversarial attacks.

### Response

**Status: Resolved conceptually.**

The paper is scoped to common image corruptions and distribution shift. It does not claim adversarial defense.

### Permitted wording

Use:

- corruption;
- distribution shift;
- robustness repair;
- corrupted input.

Avoid:

- adversarial defense;
- attack detection;
- robustness to attacks;
- secure defense.

Adversarial perturbations may be added only as a mechanistic contrast study. Any defense claim would require adaptive attacks against the complete adapter and safety mechanism.

---

## 11. Only One Backbone

### Supervisor objection

Claims about Block 6, Block 7, or specific heads may be idiosyncratic to one model. At least two additional ViTs with different training regimes are required.

### Response

**Status: Open and critical.**

Current findings are explicitly model-specific to `google/vit-base-patch16-224`. Layers are not assumed to transfer by absolute index.

### Required experiment

Replicate the **procedure**, not the layer number, on two additional models:

1. one ViT with a different supervision/pretraining regime, such as DINOv2 or CLIP;
2. one different model scale or architecture variant.

Before running each model, preregister predictions in relative depth and architecture terms. Rediscover the repair site, feature/subspace, and any detector statistics independently.

### Success criterion

The contribution becomes general if the MI procedure predicts useful intervention sites and mechanism-consistent architectures better than opposed controls, even if the exact layer differs.

---

## 12. Writing and Presentation

### Supervisor objection

The report has broken numbering, unclear cross-references, missing related work/formalization, informal prose, causal overclaims, and incomplete multiple-comparison reporting.

### Response

**Status: In progress.**

Completed foundations:

- standalone setup/leakage protocol;
- consolidated research summary;
- supervisor response log;
- explicit positive, partial, rejected, and negative findings;
- corrected 74.6% audit;
- paired statistical-reporting requirements.

### Required final writing structure

1. Introduction and scoped claim.
2. Related work.
3. Formal problem and method.
4. Experimental protocol.
5. Mechanistic findings.
6. Controlled design tests.
7. Final method and baselines.
8. Cross-corruption and cross-backbone generalization.
9. Limitations and negative results.

### Statistical writing rules

- State whether intervals are over images, seeds, or both.
- Use paired image-level inference.
- Apply family-wise or false-discovery correction where many components/layers are tested.
- Do not infer population-level seed variability from only three seeds without qualification.
- Avoid "every step was causally proven."

---

## Recommended Paper Framing

The most defensible framing remains:

> **Localize, then repair: mechanistic analysis identifies where corruption damage becomes practically repairable in frozen Vision Transformers and guides clean-safe parameter-efficient interventions.**

The paper should separate three contributions:

1. **Findings:** dominant feature identity is partly preserved while magnitudes and task evidence change; damage origin and practical repair location are dissociated; repair trajectories differ across corruption families.
2. **Procedure:** causal localization, matched deployable intervention tests, and prospective cross-model validation.
3. **Method:** one identity-preserving residual repair module, with all routers, SAEs, low-rank variants, and LoRA modules treated as diagnostics, efficiency variants, baselines, or ablations.

---

## Immediate Ordered Worklist

1. Complete Experiment 130 and freeze the winning block.
2. Freeze one final method and its hyperparameters.
3. Build the unified baseline/fairness pipeline.
4. Produce held-out-family results for every compared method.
5. Run two preregistered cross-backbone replications.
6. Complete patch-embedding attribution and oracle alpha curves.
7. Add CKA/outlier-robust trajectory controls.
8. Compare repair-or-abstain against standard shift detectors.
9. Build parameter/FLOPs/latency and Pareto tables.
10. Rewrite the final paper around supported claims and move negative results to the supplement.

---

## Authoritative Supporting Artifacts

- Setup and leakage protocol: `docs/EXPERIMENTAL_SETUP_AND_LEAKAGE_PROTOCOL.md`
- 74.6% audit: `docs/EXPERIMENT64_746_PERCENT_AUDIT.md`
- Consolidated evidence: `docs/CONSOLIDATED_RESEARCH_SUMMARY.md`
- Ongoing identity-matched sweep: `scripts/experiment130_identity_matched_layer_sweep.py`
- Split manifest: `configs/split_manifest_supervisor_v1.json`
- Accumulated decisions: `docs/SUPERVISOR_RESPONSE_LOG.md`
