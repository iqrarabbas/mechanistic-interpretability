# Research Log

## Current Experiment

Simplified reproduction of:
"A Mechanistic Analysis of Adversarial Fine-tuning of Vision Transformers"

## Setup
Model:
- `google/vit-base-patch16-224`

Dataset:
- ImageNet ILSVRC2012 validation set

Corruption:
- Gaussian Blur level 4

## Baseline Evaluation

Evaluated base ViT on 1000 ImageNet validation samples.

Results:

| Dataset | Top-1 | Top-5 |
|---|---:|---:|
| Clean | 0.8060 | 0.9640 |
| Gaussian Blur level 4 | 0.6500 | 0.8480 |

## Notes
The clean accuracy is realistic for pretrained ViT.
The blurred accuracy is lower, showing that blur hurts robustness.
This supports the next step: adversarial fine-tuning on blurred images.

## Bugs Fixed
- Wrong Python environment issue: old `.venv` was active.
- Fixed by using Conda environment `vit`.
- Import issue with `data` package fixed by adding `__init__.py`.
- Label mapping issue fixed so ImageNet labels match Hugging Face ViT class IDs.
