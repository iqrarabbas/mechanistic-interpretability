# SAE-Guided Mechanistic Interpretability for ViT Corruption Robustness

## Brief Research Summary — 20 August 2026

## 1. Research Question

We studied why a pretrained Vision Transformer (ViT) loses accuracy under level-4 Gaussian blur and Gaussian noise, and whether mechanistic interpretability (MI) can identify useful internal failure patterns. The practical goal was to improve robustness without fine-tuning the ViT or performing test-time model retraining.

Our final conclusion is:

The SAE identified a small, fragile feature subspace whose activation strengths change abnormally under corruption. A lightweight fixed residual adapter uses this subspace, together with broader hidden-state corrections, to improve corrupted-image accuracy. The subspace is generally important for classification, but the adapter depends on it significantly more under corruption—especially Blur-4.

## 2. SAE Setup

- Backbone: `google/vit-base-patch16-224`.
- SAE type: Vanilla ReLU sparse autoencoder.
- Analysis location: penultimate ViT hidden representation, excluding the CLS token; 196 patch tokens.
- Expansion factor: 32, giving 24,576 SAE features from a 768-dimensional ViT representation.
- Sparsity coefficient used in our implementation: λ = 0.001.
- Training schedule: up to 15 epochs, initial learning rate 1e-3, cosine annealing and early stopping.
- Training data: a mixture of clean and level-4 corrupted activations, following the paper’s general mixed-data SAE protocol.
- Intervention safety: later corrections were added as residual deltas to the original ViT hidden state. We did not replace the hidden state with the lossy SAE reconstruction.

## 3. First Finding: Feature Identities Stayed Similar, but Strengths Changed

At first, clean and Blur-4 SAE activations looked very similar: mean cosine similarity was approximately 0.93 and active-feature overlap was approximately 0.99. This initially suggested that the representation had barely changed.

However, cosine and overlap mostly measure direction and feature identity. They can remain high while feature magnitudes change enough to alter downstream attention and classification. Failure-conditioned analysis revealed exactly this pattern.

| 10,000-image Blur-4 analysis | Correct on clean and blur | Clean correct → blur wrong |
|---|---:|---:|
| Mean patch SAE cosine | 0.9329 | 0.9200 |
| Relative SAE L2 change | 0.3766 | 0.4388 |
| Active-feature Jaccard overlap | 0.9901 | 0.9898 |
| Blur/clean activation-norm ratio | 1.0629 | 1.0978 |
| Images | 6,194 | 1,831 |

When blur caused prediction failure, relative feature change was about 16.5% larger than in images that remained correct. The active feature identities remained almost unchanged, but their strengths moved more strongly. This shifted our research question from “which features appear or disappear?” to “which existing feature strengths move in harmful directions?”

## 4. From Observation to Feature Discovery

Across paired clean/corrupted development images, we identified features that consistently:

- increased or decreased under corruption;
- changed more in clean-correct → corrupt-wrong cases;
- correlated with deterioration in true-class margin;
- generalized across held-out images.

Direct SAE feature correction provided proof of concept. On 10,000 unseen ImageNetV2 images, the frozen targeted Blur rule improved Blur-4 accuracy from 50.82% to 51.32% (+0.50 percentage points) while leaving clean accuracy essentially unchanged. The same direct rule did not meaningfully improve Noise-4, showing that identifying features was useful but hand-designed latent editing was limited.

## 5. Why We Moved to a Residual Adapter

An SAE is a lossy compressor. Replacing a ViT hidden state with the SAE reconstruction can introduce reconstruction error and artificially damage clean accuracy. We therefore used the SAE for discovery and interpretation, while applying corrections as residual changes to the original hidden representation.

Noise analysis and layer-localization experiments showed that patch-token changes were important and that useful corrective information existed before the final ViT block. We trained a lightweight patch-wise residual adapter at Block 11 while freezing the ViT. At inference, this adapter is fixed: it requires no clean counterpart, no labels and no online gradient updates.

The adapter has 741,120 parameters, compared with the much larger frozen ViT backbone.

## 6. Fixed Adapter Results on Unseen ImageNetV2

The table reports the selected seed-2 adapter on 10,000 held-out ImageNetV2 images.

| Condition | Baseline ViT | Fixed adapter | Gain |
|---|---:|---:|---:|
| Clean | 68.37% | 68.74% | +0.37 pp (not significant) |
| Blur-4 | 50.82% | 52.49% | +1.67 pp |
| Noise-4 | 57.25% | 58.43% | +1.18 pp |

Across three independently trained adapter seeds, Noise-4 gains were +1.17, +0.94 and +1.18 pp. This confirmed that the improvement was not specific to one training seed. The Noise-trained adapter also improved Blur, demonstrating cross-corruption transfer.

## 7. Controlled CFA Comparison

We compared the baseline, our fixed adapter and CFA on exactly the same ImageNetV2 images, online Gaussian corruptions and ordering. CFA used its intended target batch size of 64, learning rate 0.001, SGD momentum 0.9, one online update per batch and a reset between each corruption/severity condition.

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

| Mean across Blur/Noise severities 1–5 | Accuracy | Mean gain |
|---|---:|---:|
| Baseline | 58.09% | — |
| Fixed adapter | 59.22% | +1.13 pp |
| CFA | 59.84% | +1.75 pp |

CFA was stronger on Blur and therefore stronger on average. Our adapter was better at every Noise severity. The methods also have different operating costs: CFA performs online test-time optimization and depends on batches, whereas our adapter performs a fixed forward correction without test-time updates.

Important limitation: this was a controlled ImageNetV2 CFA comparison, not an official ImageNet-C reproduction. We used the Hugging Face ViT and ImageNet validation statistics because the full ImageNet training set and ImageNet-C were unavailable.

## 8. Did the SAE Actually Explain the Adapter?

We tested whether the adapter repaired the SAE-identified mechanism or acted as a generic hidden-state corrector.

For 16 harmful Noise-derived SAE features, we projected the adapter residual into:

- the harmful SAE decoder subspace;
- the component orthogonal to that subspace;
- matched random SAE subspaces.

On Noise-4, the full adapter gained +1.18 pp. The harmful-aligned component alone gained only +0.19 pp, but removing the harmful component changed the gain to −1.11 pp below baseline. Thus, the harmful subspace was not sufficient by itself, but it was important through interaction with the rest of the adapter correction.

The adapter also moved harmful activations closer to paired clean values. Mean harmful-feature clean-distance decreased from 3.627 to 3.491. Recovered predictions showed greater normalization than damaged predictions (0.191 versus 0.108).

## 9. Replication and Random-Subspace Controls

The mechanism held across three independently trained adapters. Although the harmful subspace contained only about 1.23% of residual energy, removing it consistently destroyed the gain.

We then tested 20 random SAE subspaces per adapter seed: 60 controls in total. Random components were energy-matched so that each ablation removed the same residual energy as the harmful component.

| Evaluation condition | Mean harmful-subspace removal cost | Mean energy-matched random cost | Harmful exceeded random controls |
|---|---:|---:|---:|
| Clean | −1.42 pp | −0.03 pp | 60/60 |
| Noise-4 | −2.00 pp | −0.13 pp | 60/60 |
| Blur-4 | −4.17 pp | −0.30 pp | 60/60 |

For each adapter seed and condition, the empirical p-value was 1/21 = 0.0476, the minimum possible with 20 controls. Therefore, the SAE-selected subspace was specifically important per unit of removed residual energy, rather than merely important because it captured a large amount of adapter energy.

## 10. Clean Control and Final Mechanistic Interpretation

Removing the harmful subspace also reduced clean accuracy, so it is not a corruption-exclusive pathway. We therefore performed a paired difference-in-differences test: harmful-subspace dependence under corruption minus dependence on the exact clean counterpart.

| Corruption amplification beyond clean | Extra dependence | 95% confidence interval | Random-null p-value |
|---|---:|---:|---:|
| Noise-4 | +0.58 pp | +0.08 to +1.10 pp | 0.050 |
| Blur-4 | +2.75 pp | +2.20 to +3.33 pp | 0.00083 |

The final mechanistic claim is therefore:

SAE analysis identified a generally sensitive classification subspace whose feature strengths are disrupted by corruption. Independently trained adapters consistently use this subspace, together with broader hidden-state directions, to restore predictions. Dependence on the subspace is significantly amplified under corruption and is especially strong for Blur-4, explaining the adapter’s cross-corruption transfer.

## 11. How Mechanistic Interpretability Helped

Without MI, we would only know that a residual adapter improved accuracy. The SAE analysis added the following scientific value:

1. It revealed that failures were associated with feature-strength changes, even though cosine similarity and active-feature identity remained high.
2. It identified concrete features and directions associated with margin deterioration.
3. It motivated residual intervention, avoiding damage from lossy SAE reconstruction.
4. It localized a meaningful internal correction site and guided the adapter design.
5. It enabled causal tests: aligned-only intervention, orthogonal ablation, random subspaces, energy matching, clean controls and paired corruption-amplification analysis.
6. It explained why the Noise-trained adapter transfers to Blur: both gains depend disproportionately on the same SAE-identifiable bottleneck.

Therefore, the contribution is not simply “we trained an adapter.” It is an SAE-guided process that moves from observation, to feature discovery, to intervention design, to causal validation of a shared robustness mechanism.

## 12. Main Limitations

- The official CFA ImageNet-C benchmark was not reproduced because the required datasets were unavailable.
- CFA source statistics used ImageNet validation rather than the official ImageNet training split.
- The harmful subspace is not sufficient by itself and does not explain the complete adapter residual.
- Noise amplification beyond clean is small and borderline significant; Blur amplification is much stronger.
- Results currently use one ViT architecture and online implementations of Gaussian blur/noise.

## 13. Short Takeaway

Corruption did not mainly replace the ViT’s active SAE features; it changed their strengths. SAE-based mechanistic analysis identified a small, fragile subspace related to those failures. A fixed residual adapter improved Blur-4 and Noise-4 accuracy without modifying the ViT at inference. Causal ablations, three adapter seeds, 60 random controls and clean/corrupt paired tests showed that the same subspace is generally important but disproportionately supports corruption repair—especially the transfer from Noise training to Blur robustness.
