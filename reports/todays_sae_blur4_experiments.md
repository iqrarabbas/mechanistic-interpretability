# SAE-Based Blur-4 Robustness Without ViT Retraining

**Daily technical report — 13 August 2026**

## Objective

We tested whether systematic Blur-4-related SAE activation changes can be corrected at inference time without fine-tuning the ViT and without access to the clean counterpart of a new corrupted image.

The central intervention was:

`predicted_clean_z_j = a_j * blurred_z_j + b_j`

`corrected_z_j = blurred_z_j + alpha * (predicted_clean_z_j - blurred_z_j)`

The coefficients, feature set, and correction strength were learned or selected only from development data. Locked test images used only their own observed activations.

## Shared Model and SAE Setup

| Parameter | Value |
|---|---|
| ViT | `google/vit-base-patch16-224` |
| ViT training during intervention experiments | None; weights frozen |
| SAE checkpoint | `checkpoints/sae/blur4_base_vanilla_paper` |
| SAE type | Vanilla ReLU SAE |
| SAE implementation | `paper_vanilla_literal_l2_v3` |
| ViT hidden dimension | 768 |
| SAE expansion factor | 32 |
| SAE latent dimension | 24,576 |
| SAE training sparsity coefficient | lambda = 0.001 |
| SAE training learning rate | 1e-3 |
| SAE training scheduler | Cosine annealing |
| SAE maximum training epochs | 15, with early stopping |
| SAE training data | 10,000 clean/Blur-4 pairs; mixed clean and corrupted activations |
| SAE validation data | 1,000 separate pairs |
| Intervention layer | `hidden_states[-2]`, output after ViT block 10 |
| Tokens modified | 196 patch tokens |
| CLS token | Preserved from the original image representation |
| Downstream computation | Block 11, final LayerNorm, CLS classifier |

## Blur-4 Implementation

Blur-4 was generated dynamically with Pillow:

`PIL.ImageFilter.GaussianBlur(radius=4)`

The implementation is in `corruption/gaussian_blur.py`. It is a Gaussian image corruption, not a gradient-based adversarial attack. Pre-generated ImageNet-C files, OpenCV, and adversarial attack libraries were not used.

## Experiment 4: First Non-Oracle Correction

### Parameters

| Parameter | Value |
|---|---|
| Discovery | 2,000 images, indices 12,000–13,999 |
| Validation | 1,000 images, indices 14,000–14,999 |
| Locked test | 5,000 images, indices 15,000–19,999 |
| Ridge coefficient | 0.05 |
| Minimum directional consistency | 0.60 |
| Feature counts searched | 1, 4, 16, 32 |
| Alpha values searched | 0.25, 0.50, 0.75, 1.00 |
| Random controls | 100 |
| Selected setting | Top 32 affine features, alpha = 0.75 |

Discovery identified 217 candidate features. Repeated leading features included 22,739; 6,966; 17,814; 14,024; and 3,122.

### Locked-Test Results

| Condition | Clean accuracy | Blur-4 accuracy |
|---|---:|---:|
| Original ViT | 80.68% | 64.32% |
| SAE reconstruction baseline | 80.66% | 64.28% |
| Selected correction | **80.68%** | **65.40%** |
| Reverse correction | 80.60% | 62.96% |

The selected correction improved Blur-4 accuracy by **1.12 percentage points**, equivalent to approximately 56 additional correct predictions among 5,000 images. Clean accuracy changed by only **+0.02 percentage points**.

Random-feature and magnitude-matched controls produced approximately zero average gain. The selected correction beat all 100 controls, giving empirical `p = 0.0099`. However, clean-mean/shuffled-pair correction reached 65.34%, suggesting much of the benefit came from shrinking a selected blur-responsive subspace rather than requiring exact image-pair mappings.

## Experiment 5: Larger Feature Sets and Correction Strategies

### Parameters

| Parameter | Value |
|---|---|
| Discovery | 3,000 fresh images, indices 20,000–22,999 |
| Validation | 2,000 fresh images, indices 23,000–24,999 |
| Locked test | 5,000 fresh images, indices 25,000–29,999 |
| Ridge coefficient | 0.05 |
| Minimum directional consistency | 0.55 |
| Feature counts | 32, 64, 128, 256 |
| Alpha values | 0.50, 0.75, 1.00, 1.25 |
| Conditional thresholds | 1.0, 1.5, 2.0 clean standard deviations |
| Strategies | Affine, clean-mean, conditional one-sided correction |
| Random controls | 20 plus magnitude-matched controls |
| Selected setting | Affine, top 32 features, alpha = 1.25 |

Discovery yielded 447 strict FDR-significant candidates and a 713-feature ranked pool. Features 22,739; 6,966; 17,814; and 14,024 appeared again on independent data, supporting a stable Blur-4 response pattern.

### Locked-Test Results

| Condition | Clean accuracy | Blur-4 accuracy |
|---|---:|---:|
| Original ViT | 80.02% | 63.84% |
| SAE reconstruction baseline | 80.08% | 63.86% |
| Selected correction | **79.70%** | **65.02%** |
| Reverse correction | 79.98% | 61.10% |

The correction improved Blur-4 accuracy by **1.16 percentage points**, approximately 58 additional correct predictions, but reduced clean accuracy by **0.38 percentage points**.

Random controls averaged -0.036 percentage points and magnitude-matched controls averaged -0.126 percentage points. The targeted correction beat all 20 controls (`p = 0.0476`). Testing more than 32 features did not improve validation performance; the useful correction appeared concentrated in a small feature subspace.

## Experiment 6: Fine Calibration of Fixed Alpha

Experiment 5's 32 features and affine coefficients were frozen. Only alpha was calibrated.

### Parameters

| Parameter | Value |
|---|---|
| Alpha validation | 5,000 fresh images, indices 30,000–34,999 |
| Locked test | 5,000 fresh images, indices 35,000–39,999 |
| Alpha values | 0.5 to 1.3 in increments of 0.1 |
| Maximum allowed validation clean loss | 0.10 percentage points |
| Selected alpha | 1.3 |

Validation Blur-4 accuracy rose from 65.14% to 66.08%. The alpha curve largely plateaued: alpha 0.8 achieved 66.02%, alpha 0.9 achieved 66.06%, and alpha 1.3 achieved 66.08%.

### Locked-Test Results

| Condition | Clean accuracy | Blur-4 accuracy |
|---|---:|---:|
| Original ViT | 80.90% | 65.20% |
| SAE reconstruction baseline | 80.90% | 65.18% |
| Alpha 1.3 correction | **80.76%** | **66.22%** |
| Reverse correction | 80.80% | 62.26% |

The calibrated correction improved Blur-4 accuracy by **1.04 percentage points**, approximately 52 additional correct predictions, while clean accuracy decreased by **0.14 percentage points**. The clean-loss constraint generalized approximately, but the test loss was slightly larger than the permitted validation loss.

## Combined Findings

| Experiment | Locked-test images | Selected alpha | Blur gain | Clean change |
|---|---:|---:|---:|---:|
| Experiment 4 | 5,000 | 0.75 | **+1.12 pp** | +0.02 pp |
| Experiment 5 | 5,000 | 1.25 | **+1.16 pp** | -0.38 pp |
| Experiment 6 | 5,000 | 1.30 | **+1.04 pp** | -0.14 pp |

The inference-time effect replicated across three disjoint locked test sets: **+1.0 to +1.2 percentage points** of Blur-4 accuracy without changing ViT weights and without clean counterparts at test time.

The repeated reverse-control degradation shows that correction direction matters. Random and magnitude-matched controls show that arbitrary latent modification does not explain the gain. Repeated discovery of the same leading SAE features supports a systematic Blur-4-related feature pattern.

Increasing the number of corrected features and increasing alpha did not break the approximately +1% plateau. Larger alpha values slightly increased the risk to clean accuracy. A conservative fixed choice is approximately alpha 0.8–0.9; alpha 0.75 gave the safest observed clean behavior, while alpha 1.25–1.3 gave similar Blur-4 gains with small clean trade-offs.

## Scientific Contribution and Next Direction

The current contribution is an interpretable, low-cost, inference-time robustness intervention:

- The ViT remains frozen.
- The trained SAE remains frozen.
- No clean counterpart is required at inference.
- Only a small selected SAE feature subspace is modified.
- The gain reproduces on independent locked test sets.

The most promising next experiment is adaptive correction rather than stronger global correction: estimate image risk or corruption strength from the observed SAE activations and model confidence, then choose alpha per image or apply correction only to high-risk images. This preserves the no-fine-tuning contribution and directly addresses the clean-accuracy trade-off and fixed-alpha plateau.

## Result Locations

- Experiment 4: `results/sae/experiment4_non_oracle_correction/full_non_oracle_5000/`
- Experiment 5: `results/sae/experiment5_correction_strategies/full_strategies_5000/`
- Experiment 6: `results/sae/experiment6_fixed_alpha/fine_alpha_5000/`
- Blur implementation: `corruption/gaussian_blur.py`

