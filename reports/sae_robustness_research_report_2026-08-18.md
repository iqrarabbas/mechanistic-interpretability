# SAE-Based Robustness Repair for Vision Transformers

## Detailed Progress Report — 18 August 2026

## 1. Simple Summary

We investigated whether sparse autoencoder (SAE) features can explain why a Vision Transformer (ViT) fails on corrupted images and whether those internal features can be corrected **without fine-tuning or retraining the ViT**.

The clearest successful result is for Gaussian Blur:

- We found SAE features whose Blur-related changes were associated with classification failure.
- We corrected only those features and later restricted correction to the most affected image patches.
- We avoided SAE reconstruction damage by applying the correction as a residual change to the original ViT hidden state.
- On 5,000 ImageNet development images, early feature correction improved Blur-4 accuracy by about **1.1 percentage points**.
- After making the method stricter and fully non-oracle, the final frozen method independently improved Blur-4 accuracy on 10,000 unseen ImageNetV2 images from **50.82% to 51.32%**, a gain of **+0.50 percentage points**.
- The independent gain was statistically significant and clean accuracy was preserved.

The same fixed correction did not improve Gaussian Noise. Later analysis showed that Noise damage is high-dimensional and strongly image-specific. This is an important negative result: SAE repair appears to be corruption-specific rather than universally transferable.

---

## 2. Research Question

The project developed around two connected questions:

1. **Which internal SAE features change when a ViT sees corrupted images?**
2. **Can we correct harmful internal changes at inference time without retraining the ViT and without needing the clean counterpart of a corrupted test image?**

The intended contribution is therefore not adversarial fine-tuning. The ViT remains frozen. Paired clean/corrupted images are used during development to discover systematic mechanisms, but the final correction receives only the new corrupted image.

---

## 3. Models, Corruptions, and SAE Setup

### Vision Transformer

- Base model: `google/vit-base-patch16-224`
- Input resolution: 224 × 224
- Patch grid: 14 × 14 = 196 patches
- ViT parameters remained frozen during SAE analysis and correction experiments.

### Corruptions

- Gaussian Blur, mainly severity 4
- Gaussian Noise, mainly severity 4
- Later independent tests covered severities 1–5.

### Paper-style Vanilla SAE

The main SAE settings were:

- Vanilla ReLU SAE
- Expansion factor: 32
- ViT hidden size: 768
- SAE latent dimension: 24,576
- Sparsity coefficient: λ = 0.001 in our implementation
- Initial learning rate: 1e-3
- Cosine annealing scheduler
- Up to 15 epochs with early stopping
- Mixed clean/corrupted activations for corruption-specific SAEs, following the paper's general setup
- A separate clean-only SAE was later trained for corruption-agnostic analysis.

The BatchTopK settings reported by the paper were expansion 32, α = 1/32, and k = 32. BatchTopK did not produce useful results in our initial attempts, so most later mechanistic experiments used the Vanilla SAE.

---

## 4. What the Original SAE Similarity Analysis Showed

We first followed the paper-style representation analysis:

1. Feed a clean image into a ViT.
2. Feed its corrupted counterpart into the same ViT.
3. Encode corresponding patch representations with the same SAE.
4. Calculate cosine similarity between corresponding clean and corrupted SAE activations.

An early 1,000-pair comparison produced:

- Base ViT mean SAE cosine: 0.7762
- Blur-4 fine-tuned ViT mean SAE cosine: 0.8307
- Difference: +0.0545
- Evaluated patch pairs: 196,000

This indicated that Blur fine-tuning made corresponding clean/Blur patch representations more similar. However, later feature-change analysis with the paper-style base SAE produced patch cosine values around 0.92–0.93. The apparent difference from the earlier 0.77 value came from using a different SAE checkpoint/analysis pipeline and should not be interpreted as a contradiction.

The important lesson was that high cosine similarity does not mean classification is unaffected. Two representations can remain globally aligned while relatively small changes in important directions alter the final class ranking.

---

## 5. Feature Changes and Classification Failure

On 1,000 clean/Blur-4 pairs using the base paper-style SAE:

| Metric | Correct on both | Clean correct → Blur wrong |
|---|---:|---:|
| Mean patch cosine | 0.9318 | 0.9208 |
| Relative L2 change | 0.3809 | 0.4365 |
| Active-feature overlap | 0.9900 | 0.9899 |
| Blur/clean activation norm ratio | 1.0645 | 1.0987 |

Interpretation:

- Failure cases had lower clean/Blur cosine similarity.
- Their relative feature change was approximately 14.6% larger.
- Activation magnitudes changed more.
- The identities of active features were almost unchanged.

This suggested that corruption failure was caused more by **changes in feature strength** than by a completely different set of active SAE features.

---

## 6. Discovering Harmful Blur Features

We then used paired development data to identify features that:

- changed consistently under Blur;
- correlated with deterioration in the true-class margin;
- changed more strongly in clean-correct → Blur-wrong cases;
- had the correct causal direction under intervention.

This produced ranked harmful Blur feature sets, including recurring features such as `6966`, `17814`, `14024`, and others depending on the experiment and SAE checkpoint.

### Early 5,000-image correction result

This is the result remembered as “about 1.25%.” The exact interpretation is:

- **α = 1.25 was the selected affine correction strength.**
- The actual Blur accuracy gain was **+1.16 percentage points**.

Experiment 5, using 32 selected features:

| Metric | Original | Corrected |
|---|---:|---:|
| Clean accuracy | 80.08% | 79.70% |
| Blur-4 accuracy | 63.86% | 65.02% |
| Blur gain | — | **+1.16 pp** |

Random-feature and magnitude-matched controls did not reproduce the gain.

An adjacent non-oracle experiment obtained a similar **+1.12 pp** gain, and a later fixed-alpha experiment obtained:

- Original Blur-4: 65.20%
- Corrected Blur-4: 66.22%
- Gain: **+1.04 pp**
- Clean change: −0.14 pp

These experiments were encouraging, but they were development-stage results. They also reconstructed the complete hidden representation through the SAE, which introduced a possible reconstruction-error confound.

---

## 7. The SAE Reconstruction-Error Problem

An SAE is a lossy compressor:

\[
z=\operatorname{Encode}(h), \qquad \hat h=\operatorname{Decode}(z)
\]

In the original intervention, the downstream ViT received the complete SAE reconstruction:

\[
h'=\operatorname{Decode}(z')
\]

This mixes two effects:

1. the intended SAE feature correction;
2. ordinary SAE reconstruction error.

Our reconstruction baseline did not show a large accuracy collapse—the clean reconstructed accuracy was usually within approximately 0.1 percentage points of the original—but the confound still weakened causal interpretation.

### Residual intervention solution

We replaced full reconstruction with:

\[
h'=h+\left[\operatorname{Decode}(z')-\operatorname{Decode}(z)\right]
\]

The SAE now specifies only the change to apply. If no latent feature changes, then:

\[
z'=z \Rightarrow h'=h
\]

An identity test confirmed that zero correction changed margins only at floating-point scale, around 5 × 10⁻⁷.

This residual formulation became the basis of the final method.

---

## 8. Why Generic Corruption-Agnostic Clipping Failed

We trained a clean-only SAE and asked whether unusually large activations could simply be clipped toward their normal clean range.

Experiment 11 tested:

- fixed clipping;
- image-adaptive clipping;
- confidence-safe adaptive clipping.

Mean corruption accuracy changed by approximately zero. Blur-4 and Noise-4 gains were only a few hundredths of a percentage point.

The conclusion was:

> A feature becoming unusually large means it responds to corruption; it does not prove that the feature is harming classification.

This motivated combining abnormal activation with failure association and causal direction.

---

## 9. Failure-Targeted Residual Repair

Experiment 12 selected features only when they:

- increased under corruption;
- correlated with margin loss;
- increased more during actual classification failures;
- were outside their normal clean activation range.

Using residual intervention, the strongest Blur-4 result was:

- Original: 65.20%
- Corrected: 65.44%
- Gain: +0.24 pp
- Random features: essentially 0 gain

This was smaller than the early affine result, but methodologically cleaner.

---

## 10. Patch-Aware Repair

The same SAE feature may be harmful only in particular image patches. We therefore scored each patch by the weighted abnormal activity of selected harmful features and corrected only the highest-scoring patches.

Experiment 13 selected:

- 64 features
- top 8 patches
- α = 0.5

On 5,000 held-out development images:

- Clean: 81.04% → 81.04%
- Blur-4: 64.38% → 64.56%
- Gain: +0.18 pp
- Recovered: 10
- Damaged: 1
- Paired p-value: 0.0117

Random patches gave only +0.02 pp and low-score patches gave no gain. This showed that patch ranking was meaningful.

---

## 11. Patch-Budget Study

Experiment 14 increased the patch budget while keeping 64 selected features.

| Patch budget | Blur-4 gain |
|---:|---:|
| 4 | +0.12 pp |
| 8 | +0.22 pp |
| 16 | +0.36 pp |
| 32 | **+0.42 pp** |
| All eligible patches | +0.44 pp |

Top-32 patches achieved approximately 95.5% of the all-patch accuracy gain while modifying far fewer patches.

The resulting frozen rule was:

- clean-trained Vanilla SAE;
- 64 failure-associated features;
- clean 99th-percentile activation thresholds;
- top 32 patches;
- α = 1.0;
- residual intervention;
- no confidence filtering.

---

## 12. Independent Unseen-Data Confirmation

We downloaded ImageNetV2 matched-frequency:

- 10,000 unseen images
- 1,000 classes
- 10 images per class
- no parameter tuning on ImageNetV2

Experiment 15 applied the frozen rule to clean images and Blur severities 1–5.

### Main independent result

| Method | Blur-4 accuracy | Gain |
|---|---:|---:|
| Original ViT | 50.82% | — |
| Frozen targeted repair | **51.32%** | **+0.50 pp** |
| Random features | 50.84% | +0.02 pp |
| Random patches | 50.89% | +0.07 pp |
| All patches | 51.27% | +0.45 pp |

Statistics for targeted repair:

- Recovered predictions: 75
- Damaged predictions: 25
- Net recovery: 50 images
- 95% confidence interval: +0.31 to +0.70 pp
- Exact paired p-value: 5.64 × 10⁻⁷

Clean accuracy was preserved:

- Original clean: 68.37%
- Corrected clean: 68.41%
- Change: +0.04 pp, not statistically significant

### Severity trend

| Condition | Original | Corrected | Gain |
|---|---:|---:|---:|
| Clean | 68.37% | 68.41% | +0.04 pp |
| Blur-1 | 66.05% | 66.15% | +0.10 pp |
| Blur-2 | 61.96% | 62.18% | +0.22 pp |
| Blur-3 | 56.18% | 56.46% | +0.28 pp |
| Blur-4 | 50.82% | 51.32% | +0.50 pp |
| Blur-5 | 45.01% | 45.62% | +0.61 pp |

The benefit increased with Blur severity. All preregistered success criteria passed.

This is the strongest result of the project.

---

## 13. Noise Transfer: Important Negative Result

Experiment 16 applied the exact frozen Blur rule to Gaussian Noise on ImageNetV2.

At Noise-4:

- Original: 57.25%
- Corrected: 57.23%
- Change: −0.02 pp
- Confidence interval included zero
- No paired significance

No Noise severity produced a reliable gain. Therefore:

> The successful Blur correction is not a generic beneficial perturbation. It captures a Blur-specific mechanism.

---

## 14. Attempts to Improve Noise

### Noise-specific bidirectional repair

Experiment 17 discovered:

- 51 significant overactivated candidates;
- 26 significant underactivated candidates.

Validation selected an overactivation-only rule with:

- 16 features;
- 32 patches;
- α = 1.0.

On 5,000 development images:

- Noise-4: 71.10% → 71.20%, +0.10 pp
- Clean: 80.90% → 80.80%, −0.10 pp
- Recovered: 13
- Damaged: 8
- Paired p-value: 0.383

Underactivation repair reduced accuracy, while reversing the selected direction reduced Noise accuracy by −0.22 pp. Direction mattered, but the positive result was not reliable.

### Clean-gated Noise repair

Experiment 18 added clean quantile gating to protect clean images.

- Clean: 80.02% → 80.06%
- Noise-4: 69.98% → 69.94%
- Change: −0.04 pp

The gate reduced latent modifications by about 74%, but the earlier +0.10 pp Noise gain did not replicate. This indicated sampling variation or validation overfitting.

The honest conclusion is that current SAE feature correction is not a successful Noise defense.

---

## 15. Noise Layer Localization

Experiment 19 tested whether Noise correction failed because the SAE intervention was applied at the wrong ViT layer.

Using paired clean/Noise images for oracle analysis, we restored 25% of the true clean hidden-state direction.

| Layer | Noise accuracy after restoration | Gain |
|---:|---:|---:|
| 1 | 73.30% | +4.60 pp |
| 3 | 73.60% | +4.90 pp |
| 6 | 74.75% | +6.05 pp |
| 9 | 74.70% | +6.00 pp |
| 11 | **75.00%** | **+6.30 pp** |
| 12 | 75.00% | +6.30 pp |

Original Noise-4 accuracy in this experiment was 68.70%.

At Block 11:

- Patch-only clean restoration: +10.65 pp
- CLS-only restoration: +3.95 pp
- Reversed direction: −42.65 pp
- Another image's shuffled direction: −40.00 pp

Interpretation:

- Block 11 is a suitable intervention site; our existing SAE was already placed at the right late layer.
- Noise has a large theoretically recoverable signal.
- The required correction direction is high-dimensional and strongly image-specific.
- A small fixed set of global SAE features cannot approximate that direction reliably.

---

## 16. Main Scientific Conclusions

### Conclusion 1: Similar representations can still produce different predictions

High clean/corrupted SAE cosine similarity does not imply classification invariance. Relatively small changes in important feature directions can alter class rankings.

### Conclusion 2: Corruption response is not the same as causal harm

Simply clipping unusually large features did not improve accuracy. Harmful features require evidence from failure association, margin deterioration, and causal controls.

### Conclusion 3: Residual intervention is essential

Using the SAE only to define a hidden-state delta removes reconstruction-error confounding and preserves the exact original representation when no correction is applied.

### Conclusion 4: Blur harm is spatially concentrated

Top-ranked patches strongly outperformed random and low-score patches. Thirty-two patches captured almost all of the all-patch benefit.

### Conclusion 5: Blur repair generalizes

The frozen method produced a statistically significant +0.50 pp Blur-4 gain on independent ImageNetV2 while preserving clean accuracy.

### Conclusion 6: Noise requires a different mechanism

The Blur rule did not transfer to Noise. Noise-specific sparse corrections were unstable. Oracle analysis showed that Noise recovery is possible, but its direction is distributed and image-specific.

---

## 17. What We Can Claim

A defensible current claim is:

> A frozen, residual SAE intervention targeting failure-associated features in spatially selected patches can significantly improve ViT robustness to Gaussian Blur without model retraining, without access to a clean counterpart at inference time, and without reducing clean accuracy.

Supporting evidence:

- feature and patch controls;
- residual identity control;
- patch-budget analysis;
- independent ImageNetV2 confirmation;
- severity-dependent gains;
- paired confidence interval and significance test.

We should not claim:

- universal corruption-agnostic repair;
- successful Gaussian Noise defense;
- that average margin improvement always implies accuracy improvement;
- that the early +1.1 pp development gains independently replicated at the same magnitude.

---

## 18. Next Planned Experiment

The next planned experiment is an SAE oracle-direction capacity analysis at Block 11.

For paired clean/Noise images:

\[
\Delta h_{oracle}=h_{clean}-h_{noise}
\]

Compare it with the SAE-representable direction:

\[
\Delta h_{SAE}=Decode(z_{clean})-Decode(z_{noise})
\]

We will compare the clean-trained SAE with the mixed clean/Noise SAE and measure:

- cosine alignment with the oracle direction;
- relative reconstruction error;
- fraction of oracle norm captured;
- patch-level alignment;
- margin and accuracy recovery from SAE oracle restoration.

This will determine whether Noise failure comes from an inadequate SAE dictionary or from the difficulty of predicting the correct image-specific latent change without the clean counterpart.

---

## 19. Final Story in One Paragraph

We began by reproducing the paper's SAE clean/corrupted representation comparison and found that globally similar SAE representations could still accompany major classification failure. We then identified Blur-sensitive features whose strength changes were associated with failure and obtained roughly +1.1 percentage-point Blur gains on 5,000 development images; the remembered value 1.25 was the affine correction strength, while the measured gain was +1.16 points. To remove SAE reconstruction confounding, we introduced residual intervention, then made correction failure-targeted and patch-aware. A frozen rule using 64 features and the top 32 affected patches was tested without retraining and without clean counterparts at inference. On 10,000 unseen ImageNetV2 images it improved Blur-4 accuracy by +0.50 points with strong paired significance, preserved clean accuracy, and produced larger gains at stronger Blur severities. The same rule failed on Noise, and later experiments showed that Noise correction is high-dimensional and image-specific. The project therefore supports a strong, independently validated Blur-specific SAE repair contribution and an informative negative result for Noise transfer.
