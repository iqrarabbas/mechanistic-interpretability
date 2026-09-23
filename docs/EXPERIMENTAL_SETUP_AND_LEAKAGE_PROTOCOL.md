# Experimental Setup and Leakage-Control Protocol

## 1. Purpose

This document makes the experimental setup explicit and defines which results are suitable for development, mechanistic interpretation, confirmation, and final claims. It addresses the supervisor's concern that the earlier report did not clearly state the datasets, split boundaries, model configuration, corruption generation, training protocol, or safeguards against circular feature and method selection.

The central research question is:

> Can mechanistic analysis localize corruption-sensitive computations in a frozen Vision Transformer and guide a lightweight intervention that improves robustness without materially reducing clean accuracy?

The project studies robustness to common image corruptions. It is not presented as a certified adversarial-defense method.

---

## 2. Backbone and Prediction Task

| Item | Configuration |
|---|---|
| Backbone | `google/vit-base-patch16-224` |
| Architecture | ViT-Base, patch size 16, input resolution 224 |
| Classes | 1,000 ImageNet classes |
| Backbone training | Frozen in all adapter, SAE, routing, and causal-analysis experiments |
| Trainable deployment component | Residual adapter only |
| Primary intervention point | After Transformer Block 6; currently being rechecked by Experiment 130 |
| Deployment input | One image only |
| Clean counterpart at deployment | Not used |
| Labels at deployment | Not used |
| Test-time gradients | Not used by the primary adapter |
| SAE at deployment | Not required by the primary adapter |

Clean/corrupted pairs are used only for supervised development targets and oracle mechanistic diagnosis. Any experiment that inserts information from the clean counterpart is explicitly labelled an **oracle analysis** and is not described as deployable.

---

## 3. Datasets

### 3.1 ImageNet validation data

The local ILSVRC-2012 validation set contains 50,000 images under `Dataset/ILSVRC2012_img_val`. Zero-based half-open index intervals are used throughout.

This dataset supports:

- SAE and feature development;
- adapter training and development validation;
- mechanistic paired-clean analyses;
- reserved confirmation experiments.

Because some ranges have now been inspected repeatedly, not every ImageNet validation result is considered untouched final evidence. Every report must state the exact interval and its prior role.

### 3.2 ImageNetV2 matched-frequency

ImageNetV2 matched-frequency contains 10,000 independent images under `external_data/imagenetv2-matched-frequency-format-val`. It was designated as the independent final evaluation set in the locked split manifest.

Historical frozen-method evaluations have already accessed ImageNetV2. Those evaluations remain valid for their already-frozen methods, but ImageNetV2 must not be reused to tune new architectures, layers, gates, thresholds, ranks, or hyperparameters.

### 3.3 Additional distribution-shift datasets

ImageNet-A and ImageNet-Sketch have been used in later frozen-method and mechanistic-replication studies. Their status must be stated per experiment. Repeatedly examined datasets are not relabelled as untouched final tests.

---

## 4. Locked Split Protocol

The authoritative split definition is `configs/split_manifest_supervisor_v1.json`, protocol `proposed_protocol`.

### 4.1 Feature-development group

| Purpose | Dataset | Indices | Size |
|---|---|---:|---:|
| SAE training | ImageNet validation | `[0,10000)` | 10,000 |
| SAE validation | ImageNet validation | `[10000,11000)` | 1,000 |
| Harmful-feature discovery | ImageNet validation | `[0,5000)` | 5,000 |
| Harmful-feature validation | ImageNet validation | `[11000,13000)` | 2,000 |
| Feature-protocol dry run | ImageNet validation | `[13000,15000)` | 2,000 |

Feature-development images may be reused within feature development, but they cannot be used to fit adapters or deployment gates.

### 4.2 Adapter-development group

| Seed | Training interval | Training size | Validation interval | Validation size |
|---:|---:|---:|---:|---:|
| 0 | `[15000,20000)` | 5,000 | `[20000,22000)` | 2,000 |
| 1 | `[22000,27000)` | 5,000 | `[27000,29000)` | 2,000 |
| 2 | `[29000,34000)` | 5,000 | `[34000,36000)` | 2,000 |

The three seeds use distinct image partitions, not merely different random initializations on the same images. Each seed's validation interval is disjoint from its training interval.

### 4.3 Reserve and final groups

| Purpose | Dataset | Indices | Size | Current status |
|---|---|---:|---:|---|
| ImageNet reserve | ImageNet validation | `[36000,50000)` | 14,000 | Used by multiple later studies; confirmation/development, not pristine final data |
| Independent final evaluation | ImageNetV2 | `[0,10000)` | 10,000 | Valid for previously frozen methods; cannot tune subsequent methods |

---

## 5. Corruption Generation

The primary development corruptions are generated online from each clean image:

| Corruption | Severity | Pairing |
|---|---:|---|
| Gaussian Blur | 4 | Same underlying clean image |
| Gaussian Noise | 4 | Same underlying clean image and fixed corruption seed |

The clean and corrupted versions preserve the same class label. Comparisons preserve image order and corruption seeds across methods. Severity sweeps and other corruption families are reported separately and must specify whether they use project-defined online transformations or official ImageNet-C files. Controlled online-corruption evaluations are not described as official ImageNet-C reproduction.

---

## 6. Primary Residual Adapter

For a hidden representation after candidate block `b`, let `h_patch` denote the patch tokens. The adapter predicts a residual correction:

`h'_patch = h_patch + alpha * A(h_patch)`

The CLS token is left unchanged at the intervention point. The corrected token sequence then passes through all remaining frozen ViT blocks and the original classifier.

### 6.1 Architecture

| Property | Value |
|---|---:|
| Full adapter parameters | 741,120 |
| Input/output width | 768 |
| Token scope | Patch tokens only |
| Backbone parameters updated | 0 |
| Training initialization | Zero/neutral initialization |
| Training correction scale | `0.5` |
| Evaluation correction scale | `1.0` |

### 6.2 Training objective

The mixed adapter is trained jointly on Noise-4 and Blur-4. For each corruption, its loss combines:

1. **Residual target loss:** Smooth L1 distance between the predicted patch correction and the paired clean-minus-corrupted hidden-state difference.
2. **Classification loss:** cross-entropy after propagating the corrected representation through the remaining frozen blocks.
3. **Prediction-preservation loss:** discourages margin deterioration for corrupted examples already classified correctly by the base model.
4. **Clean-identity loss:** encourages approximately zero correction on clean hidden states.

The Noise and Blur losses are averaged equally.

### 6.3 Locked hyperparameters

| Hyperparameter | Value |
|---|---:|
| Identity weight | `0.2` |
| Classification weight | `0.05` |
| Preservation weight | `0.05` |
| Smooth L1 beta | `1.0` |
| Learning rate | `1e-4` |
| Weight decay | `1e-4` |
| Optimizer | AdamW |
| Scheduler | Cosine annealing |
| Maximum epochs | 5 |
| Early-stopping patience | 2 |
| Batch size | 4 |
| Adapter seeds | 3 |

Only the adapter is optimized. The ViT remains in evaluation mode with gradients disabled for its parameters.

---

## 7. Current Controlled Layer Sweep

Experiment 130 answers whether Block 6 genuinely provides the best robustness/clean-accuracy trade-off or previously benefited from a more favorable objective.

Adapters after Blocks 1–11 receive exactly the same:

- architecture and parameter count;
- mixed Noise-4/Blur-4 training data;
- identity weight `0.2`;
- classification and preservation losses;
- optimizer, scheduler, epochs, and early stopping;
- three disjoint seed-specific train/validation partitions;
- clean, Noise-4, and Blur-4 paired evaluation.

Block 12 is excluded because a patch-only correction after the final Transformer block cannot modify the already-computed CLS token before classification. The layer sweep is a development-selection experiment. Its winner must be frozen before a new confirmation dataset is accessed.

---

## 8. Mechanistic Interpretability Protocol

Mechanistic interpretability serves two distinct purposes.

### 8.1 Diagnosis and localization

- SAEs quantify corruption-associated changes in dominant features and activation magnitudes.
- Oracle clean-restoration estimates where recoverable information exists.
- Attention/content and residual/MLP patching decompose candidate mechanisms.
- Logit-lens trajectories measure how an intervention changes the downstream classification trajectory.
- Component and subspace ablations test causal necessity and sufficiency against matched controls.

### 8.2 Design guidance

MI motivated the intervention location, patch-token scope, residual rather than reconstructive editing, and compact subspace studies. It does not justify claiming that every SAE feature is uniquely causal. Negative controls are retained and reported.

SAE reconstruction is not inserted directly into the ViT because reconstruction error would confound downstream accuracy. SAE-based interventions use residual decoder differences when applicable.

---

## 9. Leakage and Circularity Safeguards

### 9.1 Required safeguards

1. Feature selection is confined to the feature-development group.
2. Adapter fitting is confined to its seed-specific training interval.
3. Hyperparameter and layer selection use development validation only.
4. Confirmation data cannot be used to revise the selected method and still remain final data.
5. Evaluation uses the corrupted image alone unless explicitly labelled oracle.
6. Paired image order and corruption seeds are identical across compared methods.
7. All three independently trained seeds are preserved in confirmatory reports.
8. Completed expensive evaluations are reused from saved paired outcomes rather than rerun and retuned.

### 9.2 Historical circularity

The historical protocol overlapped harmful-feature discovery with data used to train an initialization adapter. This did not place ImageNetV2 test images into training, but it created mechanism-selection circularity. Results depending on that original feature ranking are labelled historical unless they were repeated under the corrected split protocol.

### 9.3 Dataset-status rule

An image range can be used for passive diagnosis without conventional gradient training, but if its results influence architecture, feature, rank, threshold, or hyperparameter selection, that range becomes development data for subsequent claims. It cannot subsequently be described as untouched final evaluation.

---

## 10. Statistical Reporting

Because methods are evaluated on the same images, inference is paired at the image level.

Required reporting includes:

- baseline and corrected accuracy;
- absolute change in percentage points;
- recovered and damaged prediction counts;
- exact McNemar/binomial test from discordant pairs;
- paired bootstrap confidence interval over images;
- per-seed values and their mean;
- clean accuracy alongside every robustness result.

Repeated seeds evaluated on the same images are not treated as independent image samples. Random-control experiments report the number of controls, empirical rank, percentile, and minimum attainable empirical p-value.

---

## 11. Claim Boundaries

The current defensible claims are:

- a frozen ViT can be improved with a small residual intervention trained on paired development data;
- mechanistic analysis can localize and explain useful intervention points and downstream repair propagation;
- mixed Noise/Blur training can preserve clean accuracy while improving both development corruptions;
- different corruption families may rely on partially different repair pathways or subspaces.

The following claims require additional evidence:

- official state-of-the-art performance on ImageNet-C;
- universal corruption detection;
- transfer of the same layer or mechanism to another backbone;
- certified adversarial robustness;
- a uniquely causal SAE feature dictionary across SAE seeds and architectures.

---

## 12. Reproducibility Artifacts

- Split manifest: `configs/split_manifest_supervisor_v1.json`
- Mixed adapter: `scripts/experiment67_mixed_noise_blur_block6_adapter.py`
- Controlled identity-matched layer sweep: `scripts/experiment130_identity_matched_layer_sweep.py`
- Mixed-adapter results: `results/sae/experiment67_mixed_noise_blur_block6/full_3seed_noise_blur_identity0p2_v1/summary.json`
- Supervisor response log: `docs/SUPERVISOR_RESPONSE_LOG.md`
- Project handoff: `PROJECT_HANDOFF.md`

Every new scientific run must save configuration, split paths, seeds, device, metrics, limitations, and paired outcomes when statistical analysis may be needed.
