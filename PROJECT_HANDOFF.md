# Project Handoff — SAE-Guided ViT Corruption Robustness

## 1. Purpose of This File

This is the primary handoff for continuing the project in a new Codex session. Read this file before changing code or launching experiments. It records the research question, methods, important paths, completed experiments, definitive results, limitations, and recommended next work.

Project root: `/media/dr-yougart/Iqrar/vit_mi`

Current state: the nine-run multi-seed confirmation of the 16-feature SAE abnormality gate has completed successfully. Do not rerun it unless specifically requested.

Current supervisor feedback, split audit, response drafts, and requested experiment tracker: `docs/SUPERVISOR_RESPONSE_LOG.md`.

Locked candidate split manifest: `configs/split_manifest_supervisor_v1.json`. Automatic audit: `scripts/experiment38_split_protocol_audit.py`. The historical protocol fails isolation; the proposed protocol passes, but clean confirmatory training has not yet been run.

Leakage-free feature rediscovery completed at `results/sae/experiment17_noise_bidirectional_repair/supervisor_disjoint_feature_dev_v1`. It selected 8 over-activated features for direct repair; all 8 recur in the historical harmful-16 set, and the new top-16 ranking overlaps the historical set by 13/16. Direct repair itself remained weak, so this supports feature stability more than standalone correction efficacy.

Leakage-free adapter/gate update: Experiment 39 reproduced positive adapter validation gains across three zero-initialized disjoint seeds (mean +1.02 pp Noise-4), with a small nonsignificant clean trend (mean -0.28 pp). Experiment 41 found that top-8/top-16 SAE gates did not improve over those ungated adapters; top-16 reduced mean adapter Noise accuracy by 0.20 pp and did not improve mean clean accuracy. Treat the historical gate advantage as unconfirmed under the corrected split protocol.

## Base Paper: Gao et al.

The project began as an implementation and extension of:

**Hannah Gao, Isha Agarwal, Dylan Hadfield-Menell, and Rachel Ma, “A Mechanistic Analysis of Adversarial Fine-tuning of Vision Transformers.”**

Repository copy: `references/gao_mechanistic_analysis_adversarial_finetuning_vits.pdf`

Gao et al. compare base and corruption-fine-tuned ViTs using accuracy, attention, logit-lens/knowledge-evolution analysis, and SAE representations. Their Section 4.4 compares corresponding clean/corrupted patch activations using Vanilla and BatchTopK SAEs trained on mixed clean/corrupted data. They report expansion factor 32; BatchTopK additionally uses `alpha = 1/32` and `k = 32`; training uses up to 15 epochs, initial learning rate `1e-3`, cosine annealing, and early stopping. The available text does not provide a reliable numerical Vanilla-SAE sparsity coefficient, so this repository's `lambda = 0.001` must be described as an implementation choice.

Our work first reproduced the paper-style comparison at severity 4, then moved beyond representation similarity toward failure-conditioned feature magnitude, causal restoration, residual intervention, and lightweight frozen-model repair.

## 2. Research Goal

The project asks:

> Can mechanistic interpretability identify where and how common corruptions damage a frozen Vision Transformer, and can that information guide a lightweight inference-time repair without fine-tuning the ViT or requiring a paired clean image at deployment?

The current method is:

1. Use paired clean/corrupted development images for diagnosis only.
2. Causally localize a recoverable ViT layer.
3. Train an SAE and identify features associated with corruption-induced margin deterioration and prediction failure.
4. Train a lightweight residual adapter while keeping the ViT frozen.
5. Use abnormal activity in 16 mechanistically identified SAE features to gate the fixed adapter.
6. At inference, use only the corrupted image. No clean counterpart, labels, or online gradient updates are required.

## 3. Main Model and Data Setup

- Backbone: `google/vit-base-patch16-224`
- Hidden size: 768
- Patch tokens: 196 at 224 × 224 input resolution
- Primary corruptions: online Gaussian Blur and Gaussian Noise
- Primary severity: level 4
- Main independent evaluation: 10,000 ImageNetV2 images
- ViT remains frozen during SAE, adapter, and gate experiments
- Conda environment: `/home/dr-yougart/miniconda3/envs/vit`
- Typical command prefix: `PYTHONPATH=. /home/dr-yougart/miniconda3/envs/vit/bin/python`

The exact local dataset and model paths should be read from each script/configuration and its saved `summary.json`; do not assume an external dataset is available.

## 4. SAE Setup

The paper-style Vanilla SAE used:

- ReLU Vanilla SAE
- Expansion factor: 32
- Input dimension: 768
- Latent features: 24,576
- Sparsity coefficient in this implementation: `lambda = 0.001`
- Initial learning rate: `1e-3`
- Cosine annealing scheduler
- Up to 15 epochs with early stopping
- Mixed clean and severity-4 corrupted activations for corruption-specific SAEs
- Patch representations from the selected late ViT representation; CLS token excluded for the main patch analysis

The paper gave expansion 32 and BatchTopK values `alpha = 1/32`, `k = 32`, but the paper text available to us did not clearly provide a numerical Vanilla-SAE lambda. `0.001` is therefore an implementation choice, not a verified paper constant.

Important SAE checkpoints include:

- `checkpoints/sae/blur4_base_vanilla`
- `checkpoints/sae/blur4_base_vanilla_paper`

Use the exact checkpoint recorded in each experiment's `summary.json`. Different checkpoints/pipelines produced different cosine values and must not be mixed.

## 5. What the SAE Analysis Found

The central observation was not that corruption activates a completely different dictionary. Instead, the identities of active features remain nearly the same while their strengths change.

On a 10,000-image Blur-4 analysis:

| Metric | Correct on clean and blur | Clean correct, blur wrong |
|---|---:|---:|
| Mean patch SAE cosine | 0.9329 | 0.9200 |
| Relative SAE L2 change | 0.3766 | 0.4388 |
| Active-feature Jaccard | 0.9901 | 0.9898 |
| Blur/clean activation-norm ratio | 1.0629 | 1.0978 |
| Images | 6,194 | 1,831 |

Failure cases therefore had about 16.5% larger relative feature change, despite almost unchanged active-feature identity. This changed the research direction from feature appearance/disappearance to harmful feature-strength changes.

Do not confuse this result with an earlier SAE comparison that reported approximately 0.7762 base cosine and 0.8307 Blur-4-fine-tuned cosine. That earlier result used a different SAE checkpoint/analysis pipeline. The approximately 0.93 result is from the later paper-style base SAE feature-change analysis.

## 6. Why Residual Intervention Was Necessary

An SAE reconstruction is lossy. Replacing the original hidden state with `Decode(Encode(h))` mixes the intended feature correction with reconstruction error.

The adopted intervention is:

`h_repaired = h + [Decode(z_corrected) - Decode(z_original)]`

This preserves the original hidden representation and applies only the decoder-space change caused by editing selected SAE features. If no feature is changed, the intervention is effectively the identity. This avoids attributing reconstruction damage to the proposed correction.

## 7. Causal Layer Localization

The layer was selected with an oracle diagnostic, not simply by choosing the layer with the lowest cosine similarity.

For paired clean/Noise-4 development images and each candidate layer:

1. Extract `h_clean` and `h_noise`.
2. Compute `delta = h_clean - h_noise`.
3. Partially restore the noisy hidden state: `h_test = h_noise + alpha * delta`.
4. Run the remaining frozen Transformer blocks.
5. Measure accuracy gain, recovered predictions, damaged predictions, true-class logit change, margin change, and McNemar significance.

Layer 11 was the best partial-recovery location:

- Noise-4 accuracy after intervention: 75.0%
- Accuracy gain: +6.3 percentage points
- Predictions recovered: 135
- Originally correct predictions damaged: 9
- Mean margin change: +0.6914
- Mean true-class logit change: +0.8528
- Exact McNemar p-value: approximately `5.46e-30`

Script: `scripts/experiment19_noise_layer_localization.py`

Result: `results/sae/experiment19_noise_layer_localization/noise_layer_localization_gpu/summary.json`

This is an oracle development analysis because it uses the clean counterpart. It only identifies the intervention location. The deployed adapter does not use clean images.

## 8. Development-Stage Direct SAE Repair

Early direct feature correction on approximately 5,000 development images showed that targeted feature manipulation could improve Blur-4 accuracy by roughly one percentage point.

One remembered result was sometimes described as “1.25%,” but the precise distinction is:

- Selected affine strength: `alpha = 1.25`
- Actual Blur-4 accuracy gain: approximately +1.16 percentage points

These results motivated further work but were not the final independent evaluation. Later methods used residual intervention and stricter held-out evaluation.

## 9. Fixed Residual Adapter

A lightweight patch-wise residual adapter was trained at Block 11 while freezing the ViT. It has 741,120 trainable parameters. At inference it is fixed and performs no online optimization.

Selected seed-2 adapter on 10,000 held-out ImageNetV2 images:

| Condition | Baseline | Fixed adapter | Gain |
|---|---:|---:|---:|
| Clean | 68.37% | 68.74% | +0.37 pp |
| Noise-4 | 57.25% | 58.43% | +1.18 pp |
| Blur-4 | 50.82% | 52.49% | +1.67 pp |

Across three independently trained adapters, Noise-4 gains were approximately +1.17, +0.94, and +1.18 pp.

Relevant scripts:

- `scripts/experiment23_noise_patch_residual_predictor.py`
- `scripts/experiment24_classification_aware_residual_predictor.py`
- `scripts/experiment25_multiseed_independent_confirmation.py`
- `scripts/experiment26_adapter_robustness_ablations.py`
- `scripts/experiment27_corruption_transfer.py`

Three adapter roots used by the latest gate confirmation:

- `results/sae/experiment25_multiseed_confirmation/imagenetv2_multiseed_confirmation/seed_0`
- `results/sae/experiment25_multiseed_confirmation/imagenetv2_multiseed_confirmation/seed_1`
- `results/sae/experiment25_multiseed_confirmation/imagenetv2_multiseed_confirmation/seed_2`

Each contains `classification_weight_0.05.pt`, `train_cache`, and `validation_cache`.

## 10. Mechanistic Validation of the Adapter

The adapter was tested against the concern that it might be only a generic corrector.

For 16 harmful Noise-derived SAE features, the adapter residual was decomposed into:

- its projection into the harmful SAE decoder subspace;
- its orthogonal component;
- energy-matched random SAE subspaces.

The harmful-aligned component alone was not sufficient to reproduce the full gain. However, removing that small component destroyed much of the adapter benefit. The subspace contained only about 1.23% of residual energy, yet its removal was much more damaging than removing energy-matched random subspaces.

Across three adapters and 20 random subspaces per adapter (60 random controls total):

| Condition | Harmful-subspace removal cost | Mean random removal cost | Harmful exceeded random |
|---|---:|---:|---:|
| Clean | -1.42 pp | -0.03 pp | 60/60 |
| Noise-4 | -2.00 pp | -0.13 pp | 60/60 |
| Blur-4 | -4.17 pp | -0.30 pp | 60/60 |

Clean accuracy also fell when this subspace was removed, so it is not corruption-exclusive. A paired difference-in-differences test showed that adapter dependence on it was amplified under corruption:

| Condition | Extra dependence beyond clean | 95% CI | Random-null p-value |
|---|---:|---:|---:|
| Noise-4 | +0.58 pp | +0.08 to +1.10 pp | 0.050 |
| Blur-4 | +2.75 pp | +2.20 to +3.33 pp | 0.00083 |

The correct mechanistic claim is that SAE analysis identified a generally sensitive classification subspace whose role in adapter repair becomes disproportionately important under corruption, especially Blur-4.

Relevant scripts:

- `scripts/experiment31_sae_adapter_causal_mediation.py`
- `scripts/experiment32_multirandom_sae_subspace_test.py`
- `scripts/experiment33_corruption_amplification_test.py`

## 11. SAE Abnormality Gate

Experiment 36 used the 16 harmful SAE features to control how strongly the fixed residual adapter is applied. The ViT, SAE, and residual adapter remained frozen. The gate has only 17 trainable parameters and adds no residual projection.

The first successful run showed:

- Noise-4: 57.25% baseline to 58.63% gated, +1.38 pp
- Blur-4: 50.82% baseline to 53.10% gated, +2.28 pp
- Clean: 68.74% preserved for that adapter/run
- The harmful-feature gate beat 10 random gates, a hidden-state projection gate, and a constant gate

Experiment 37 tested feature counts and scales. A validation grid selected 64 features at scale 1.5, but it did not beat the original 16-feature, scale-2 design on Blur. The locked method therefore remains 16 harmful features with scale 2.

Scripts:

- `scripts/experiment35_sae_guided_residual_adapter.py`
- `scripts/experiment36_sae_abnormality_gate.py`
- `scripts/experiment37_gate_validation_grid.py`

Original successful output:

- `results/sae/experiment36_sae_abnormality_gate/full_gating_10random/summary.json`

## 12. Definitive Nine-Run Gate Confirmation

This experiment is complete. It tested:

- 3 independently trained residual adapters: seeds 0, 1, and 2
- 3 gate-training seeds per adapter: 9 runs total
- 20 random SAE-feature gates for one run per adapter: 60 random gates total
- 10,000 held-out images under Clean, Noise-4, and Blur-4
- Same locked 16 harmful features and scale 2

Gate seeds:

- Adapter 0: 101, 102, 103
- Adapter 1: 201, 202, 203
- Adapter 2: 301, 302, 303

Mean results across all nine runs:

| Method | Clean | Noise-4 | Blur-4 |
|---|---:|---:|---:|
| Baseline | 68.37% | 57.25% | 50.82% |
| Existing adapter | 68.39% | 58.35% | 52.43% |
| 16-feature SAE gate | 68.35% | 58.51% | 52.90% |

Mean gains:

- Gated method over baseline:
  - Noise-4: +1.26 pp
  - Blur-4: +2.08 pp
- Gate over the corresponding ungated adapter:
  - Noise-4: +0.16 pp
  - Blur-4: +0.47 pp
- Clean change versus baseline: approximately -0.02 pp

Reproducibility:

- Harmful gate beat its corresponding adapter in 9/9 runs on Noise-4.
- Harmful gate beat its corresponding adapter in 9/9 runs on Blur-4.
- For each adapter seed, the harmful gate beat all 20 random-feature gates on both corruptions.
- Total random controls beaten: 60/60 per corruption.
- Gate-training seeds produced nearly identical results within each adapter.

Per-adapter mean gated results:

| Adapter seed | Clean | Noise-4 | Blur-4 | Gate increment on Noise | Gate increment on Blur |
|---:|---:|---:|---:|---:|---:|
| 0 | 68.18% | 58.50% | 52.65% | +0.08 pp | +0.27 pp |
| 1 | 68.14% | 58.39% | 52.95% | +0.20 pp | +0.54 pp |
| 2 | 68.74% | 58.63% | 53.10% | +0.20 pp | +0.61 pp |

All nine outputs are under:

- `results/sae/experiment36_sae_abnormality_gate/replication_adapter0_gate101_random20`
- `results/sae/experiment36_sae_abnormality_gate/replication_adapter0_gate102_random0`
- `results/sae/experiment36_sae_abnormality_gate/replication_adapter0_gate103_random0`
- `results/sae/experiment36_sae_abnormality_gate/replication_adapter1_gate201_random20`
- `results/sae/experiment36_sae_abnormality_gate/replication_adapter1_gate202_random0`
- `results/sae/experiment36_sae_abnormality_gate/replication_adapter1_gate203_random0`
- `results/sae/experiment36_sae_abnormality_gate/replication_adapter2_gate301_random20`
- `results/sae/experiment36_sae_abnormality_gate/replication_adapter2_gate302_random0`
- `results/sae/experiment36_sae_abnormality_gate/replication_adapter2_gate303_random0`

Each completed directory contains `summary.json` and Noise/Blur outcome arrays. Do not overwrite these directories because the script uses `exist_ok=False`.

## 13. CFA Comparison

A controlled ImageNetV2 comparison used the same ViT, images, ordering, online corruptions, and severity sweep. CFA used batch size 64, learning rate 0.001, SGD momentum 0.9, one online update per batch, and reset between corruption/severity conditions.

| Condition | Baseline | Fixed adapter | CFA |
|---|---:|---:|---:|
| Blur-1 | 66.05% | 66.64% | 67.16% |
| Blur-2 | 61.96% | 62.67% | 63.89% |
| Blur-3 | 56.18% | 57.97% | 59.42% |
| Blur-4 | 50.82% | 52.49% | 54.84% |
| Blur-5 | 45.01% | 46.98% | 50.23% |
| Noise-1 | 67.19% | 67.55% | 66.97% |
| Noise-2 | 65.63% | 66.32% | 65.54% |
| Noise-3 | 62.39% | 63.30% | 63.10% |
| Noise-4 | 57.25% | 58.43% | 58.11% |
| Noise-5 | 48.37% | 49.82% | 49.10% |

Means across these 10 conditions:

- Baseline: 58.09%
- Fixed adapter: 59.22%, +1.13 pp
- CFA: 59.84%, +1.75 pp

CFA was stronger on Blur and overall. The fixed adapter was stronger at every tested Noise severity and requires no online test-time optimization. This is not an official ImageNet-C CFA reproduction because the required full datasets were unavailable. Do not claim that the project outperforms official CFA generally.

Relevant scripts:

- `scripts/experiment28_cfa_same_setup.py`
- `scripts/experiment29_imagenet_c_benchmark.py`
- `scripts/experiment30_cfa_imagenetv2_severity_sweep.py`
- `third_party/CFA/`

## 14. What Mechanistic Interpretability Contributes

Without MI, the result would only be “a residual adapter improved accuracy.” SAE/causal analysis provides the scientific contribution:

1. It showed that corruption failures involve harmful feature-strength changes despite high cosine similarity and stable feature identity.
2. It identified concrete feature directions associated with true-class margin deterioration.
3. It motivated residual intervention to avoid SAE reconstruction confounds.
4. It causally localized a useful intervention site at Layer 11.
5. It guided a tiny 16-feature, 17-parameter gate for a frozen adapter.
6. It enabled aligned/orthogonal decomposition, energy-matched random controls, clean controls, and corruption-amplification tests.
7. It connected Noise-trained repair to Blur transfer through a shared, SAE-identifiable sensitive subspace.

The defensible contribution is:

> A reusable mechanistic pipeline that localizes a causally recoverable layer, identifies a small harmful SAE subspace, and uses that signal to guide a lightweight frozen-model repair.

## 15. Claims to Make Carefully

Supported:

- The 16-feature gate improves over the corresponding ungated adapter in all nine tested runs.
- It beats 60/60 random gates per corruption in the completed control design.
- The effect is reproducible across three adapter seeds and three gate seeds each.
- Clean accuracy is preserved on average.
- The incremental benefit is stronger for Blur than Noise.
- The SAE subspace is generally classification-sensitive but disproportionately important under corruption.

Do not claim:

- The 16 features alone fully explain the adapter.
- The subspace is corruption-exclusive.
- Layer 11 or the same feature IDs will transfer unchanged to another model.
- The method generally beats official CFA or state of the art.
- The numerical lambda was definitely specified by the source paper.
- Development-set direct SAE repair results are equivalent to the independent ImageNetV2 results.

## 16. Main Limitations

- Only one primary ViT architecture has been studied.
- Layer number, SAE feature IDs, thresholds, adapter weights, and gate parameters are model-specific until shown otherwise.
- The official ImageNet-C/CFA benchmark was not completed.
- Online Gaussian corruptions are not identical to every official benchmark implementation.
- Noise gate improvement beyond the adapter is small (+0.16 pp mean), although consistent.
- Gate-seed replicates on the same adapter use the same evaluation images and are not nine fully independent datasets.
- The harmful subspace is necessary in interaction with broader residual directions, not sufficient alone.

## 17. Recommended Next Work

### Immediate documentation/statistics task

Create a dedicated aggregation script for the nine replication directories that:

- loads every `summary.json` and paired outcome file;
- reports per-run and per-adapter tables;
- computes image-level paired bootstrap confidence intervals for gated versus ungated outcomes;
- accounts for repeated evaluation images by averaging paired effects within adapter/gate groups rather than treating all 90,000 rows as independent;
- reports empirical random-control ranks per adapter;
- saves a machine-readable JSON and publication-quality CSV/plot.

Do not rerun model evaluation for this; all required outcomes are already saved.

### Next scientific experiment: reusable layer locator

Turn Experiment 19 into a model-configurable `Layer Recovery Locator` that a user can run on another compatible ViT.

Inputs should include:

- model/checkpoint identifier;
- paired clean/corrupted development dataset;
- corruption function and severity;
- candidate layers;
- restoration strengths such as 0.1, 0.25, 0.5, and 1.0;
- token scope: full, patch-only, CLS-only;
- sample count and seed.

For every layer/strength, output:

- baseline and repaired accuracy;
- accuracy gain;
- recovered and damaged predictions;
- true-class logit and margin change;
- McNemar p-value and bootstrap confidence interval;
- intervention norm;
- a utility score that rewards recovery and penalizes damage/magnitude.

The tool should rank layers and recommend the safest layer/strength. It must clearly label this as an oracle development diagnostic requiring clean/corrupted pairs. It should not promise that the selected layer transfers across architectures.

After validating the locator on the existing ViT and reproducing Layer 11, test it on a second backbone. For the second model, retrain the SAE and rediscover features; do not reuse the existing feature indices.

## 18. Numbered Experiment Ledger

This ledger records the purpose and decisive outcome of every numbered experiment script. Smoke runs are implementation checks, not scientific results. For exact configurations and all secondary metrics, read the named script and its full-run `summary.json`.

| Exp. | Script | Decisive result |
|---:|---|---|
| 1 | `experiment1_base_blur4_sae_identity_strength.py` | Established clean/Blur-4 SAE identity and strength statistics for the base ViT; high overlap motivated looking beyond feature identity. |
| 2 | `experiment2_sae_strength_vs_classification.py` | Linked larger SAE-strength changes to margin deterioration and clean-correct to blur-wrong failures. |
| 3 | `experiment3_sae_causal_intervention.py` | Tested causal latent edits; selected directions could affect predictions, but full SAE reconstruction remained a confound. |
| 4 | `experiment4_non_oracle_sae_correction.py` | Converted paired discovery into a corrupted-only fixed correction; showed non-oracle feasibility on development data. |
| 5 | `experiment5_sae_correction_strategies.py` | Best early 32-feature Blur strategy used `alpha = 1.25`: 63.86% to 65.02%, +1.16 pp, with a -0.38 pp clean change. |
| 6 | `experiment6_fixed_alpha_calibration.py` | Locked correction strength; an adjacent run improved Blur-4 by +1.04 pp with only -0.14 pp clean change. |
| 7 | `experiment7_noise_finetuning_feature_mechanisms.py` | Tested whether analogous SAE features explain Noise-fine-tuned behavior; showed corruption/model specificity rather than a universal feature list. |
| 8 | `experiment8_blur_features_on_noise.py` | Applying Blur-derived harmful features to Noise did not meaningfully improve Noise, an important negative transfer result. |
| 9 | `experiment9_hybrid_common_specific_correction.py` | Combining a small common feature set with corruption-specific sets did not produce a convincing universal correction. |
| 10 | `experiment10_corruption_agnostic_sae_repair.py` | Used a clean-trained or fixed SAE for corruption-agnostic repair; exposed scalability/resume issues and did not yield strong general repair. |
| 11 | `experiment11_quantile_sae_repair.py` | Fixed, adaptive, and confidence-safe quantile clipping produced approximately zero mean gain; abnormal magnitude alone does not imply causal harm. |
| 12 | `experiment12_failure_targeted_quantile_repair.py` | Failure-associated residual correction gave a cleaner but smaller Blur result: 65.20% to 65.44%, +0.24 pp; random features were near zero. |
| 13 | `experiment13_patch_aware_residual_repair.py` | Correcting 64 features in the top eight abnormal patches gave +0.18 pp Blur, recovered 10 and damaged 1, `p = 0.0117`; random/low-score patches failed. |
| 14 | `experiment14_patch_budget_pareto.py` | Mapped patch-budget trade-offs and confirmed that Blur harm is spatially concentrated rather than requiring every patch to be edited. |
| 15 | `experiment15_independent_frozen_confirmation.py` | Frozen non-oracle Blur correction transferred to unseen ImageNetV2: Blur-4 50.82% to 51.32%, +0.50 pp, with clean accuracy preserved. |
| 16 | `experiment16_frozen_noise_confirmation.py` | The exact frozen Blur rule did not improve Noise-4, confirming that direct SAE correction was corruption-specific. |
| 17 | `experiment17_noise_bidirectional_repair.py` | Discovered Noise-specific over- and under-activated candidates; candidate scarcity and weak direct repair indicated a high-dimensional, image-specific Noise mechanism. This summary supplies the later 16 harmful features. |
| 18 | `experiment18_clean_gated_noise_repair.py` | Clean-range gating protected normal activations but did not create a strong Noise correction. |
| 19 | `experiment19_noise_layer_localization.py` | Oracle partial clean restoration localized the strongest recoverable site to Block 11: +6.3 pp, 135 recovered, 9 damaged, `p ~= 5.46e-30`. |
| 20 | `experiment20_sae_oracle_direction_capacity.py` | Tested whether the SAE decoder dictionary could express the oracle repair direction; SAE-only capacity was insufficient to explain all recoverable information. |
| 21 | `experiment21_noise_affine_latent_prediction.py` | Per-feature affine prediction of clean latents from corrupted latents did not provide strong Noise accuracy recovery. |
| 22 | `experiment22_noise_nearest_neighbor_repair.py` | Nearest-neighbor latent repair was ineffective/insufficient, reinforcing that matching generic clean statistics is not enough. |
| 23 | `experiment23_noise_patch_residual_predictor.py` | Learned a lightweight patch residual predictor at the localized layer, moving from manual SAE edits to learned residual repair with a frozen ViT. |
| 24 | `experiment24_classification_aware_residual_predictor.py` | Added classification-aware training/selection and produced the adapter family used in later confirmation. |
| 25 | `experiment25_multiseed_independent_confirmation.py` | Three independently trained adapters reproduced Noise-4 gains of about +1.17, +0.94, and +1.18 pp. |
| 26 | `experiment26_adapter_robustness_ablations.py` | Ablated adapter design and confirmed the selected fixed adapter improves corruption without requiring ViT fine-tuning or test-time updates. |
| 27 | `experiment27_corruption_transfer.py` | Noise-trained adapter transferred to Blur: selected seed-2 results were Clean +0.37 pp, Noise-4 +1.18 pp, Blur-4 +1.67 pp. |
| 28 | `experiment28_cfa_same_setup.py` | Initial same-setup CFA comparison established a controlled baseline but was not an official CFA/ImageNet-C reproduction. |
| 29 | `experiment29_imagenet_c_benchmark.py` | Prepared official-style ImageNet-C evaluation; full official benchmark remained blocked by unavailable large datasets/statistics. |
| 30 | `experiment30_cfa_imagenetv2_severity_sweep.py` | Controlled severities 1-5: adapter mean 59.22% versus CFA 59.84%; CFA stronger on Blur/overall, adapter stronger at every Noise severity. |
| 31 | `experiment31_sae_adapter_causal_mediation.py` | Harmful-aligned residual alone was insufficient, but removing it destroyed gain; adapter also moved harmful activations closer to paired-clean values. |
| 32 | `experiment32_multirandom_sae_subspace_test.py` | Across 3 adapters and 60 energy-matched random subspaces, harmful-subspace removal was much more damaging: Clean -1.42, Noise -2.00, Blur -4.17 pp. |
| 33 | `experiment33_corruption_amplification_test.py` | Difference-in-differences showed dependence beyond clean: Noise +0.58 pp, 95% CI +0.08 to +1.10; Blur +2.75 pp, CI +2.20 to +3.33. |
| 34 | `experiment34_extreme_corruption_stress_test.py` | Stress-tested higher corruption severity across adapters; confirmed robustness gains persist but do not eliminate severe-corruption degradation. |
| 35 | `experiment35_sae_guided_residual_adapter.py` | Tested SAE-guided adapter variants against random guidance and established the path toward gating an already useful residual adapter. |
| 36 | `experiment36_sae_abnormality_gate.py` | Locked 16-feature gate. Nine-run confirmation: mean Clean 68.35%, Noise-4 58.51%, Blur-4 52.90%; beat ungated adapter 9/9 and all 60 random gates per corruption. |
| 37 | `experiment37_gate_validation_grid.py` | A validation grid selected 64 features/scale 1.5 but failed to beat the original 16-feature/scale-2 gate on Blur; keep the original locked design. |

### Main Failed or Negative Directions

- BatchTopK SAE training did not produce useful results in the initial setup; Vanilla SAE became the primary analysis tool.
- High cosine similarity and active-feature overlap were insufficient explanations because class decisions remained sensitive to magnitude changes.
- Full SAE reconstruction was methodologically unsafe because reconstruction error could contaminate downstream inference.
- Generic clipping, adaptive clipping, and confidence-safe clipping failed because abnormal features are not necessarily harmful features.
- Blur-specific direct feature correction did not transfer to Noise.
- A universal common-plus-specific direct correction did not become a convincing corruption-agnostic method.
- Clean-statistic affine mapping and nearest-neighbor latent repair did not solve Noise.
- SAE harmful directions alone were not sufficient; they function through interaction with broader adapter residual directions.
- A larger 64-feature gate did not outperform the sparse 16-feature gate.
- Official ImageNet-C/CFA reproduction remains incomplete because the required large external data were unavailable.

## 19. Existing Reports to Read

- `reports/sae_mi_robustness_summary_2026-08-20.md` — best concise narrative through Experiment 34
- `reports/sae_robustness_research_report_2026-08-18.md` — detailed early-to-middle experiment history
- `reports/SAE_MI_Robustness_Summary_2026-08-20.pdf`
- `reports/SAE_Blur4_Experiments_2026-08-13.pdf`
- `docs/RESEARCH_LOG.md`
- `docs/PROGRESS_REPORT.md`
- `docs/PAPER_NOTES.md`

This handoff supersedes those reports for the latest Experiment 36 multi-seed gate results, but the older reports contain more detail on Experiments 1–34.

## 20. Working-Tree Safety

The repository currently contains many modified and untracked research files and outputs. Before editing:

- run `git status --short`;
- do not reset, clean, delete, or overwrite existing work;
- use unique `--run-name` values because many scripts reject existing output directories;
- preserve all checkpoints and result directories;
- do not rerun expensive experiments unless a missing result is verified;
- use CUDA for full experiments and smoke-test code changes on tiny samples first.

No process from the completed nine-run confirmation needs to remain active.

## 21. Short Handoff Prompt for a New Codex Session

Use this prompt:

> Read `PROJECT_HANDOFF.md` and `AGENTS.md` completely, then inspect the referenced scripts and saved summaries before making changes. Do not rerun completed experiments or overwrite result directories. First create the statistical aggregator for the completed nine-run Experiment 36 confirmation, including paired confidence intervals and random-control ranks. After reporting those results, propose—but do not automatically launch—the reusable model-configurable Layer Recovery Locator based on Experiment 19.
