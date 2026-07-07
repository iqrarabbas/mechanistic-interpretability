# Paper Notes

Paper:
"A Mechanistic Analysis of Adversarial Fine-tuning of Vision Transformers"

## What the Paper Does
The paper fine-tunes ViT models on common image corruptions and studies the internal changes using mechanistic interpretability.

## Corruptions Used in Paper
- Gaussian Blur level 1
- Gaussian Blur level 2
- Gaussian Blur level 4
- Gaussian Noise level 1
- Gaussian Noise level 2
- Gaussian Noise level 4

The paper trains six separate fine-tuned models.

## Our Simplification
We will only use:
- Gaussian Blur level 4

Reason:
- Smaller and more doable
- Still follows the paper's main idea
- Blur-4 shows clear robustness effect

## Main Analyses to Reproduce
After fine-tuning, compare base ViT and fine-tuned ViT using:

1. Accuracy
   - Top-1
   - Top-5

2. Logit Lens
   - Check when the correct class appears across layers

3. Attention Analysis
   - Attention entropy clean vs blurred images

4. Representation Analysis
   - Hidden representation similarity
   - SAE optional later

## Current Status
Data pipeline and base evaluation are complete.
Next step is fine-tuning one ViT on Gaussian Blur level 4.
