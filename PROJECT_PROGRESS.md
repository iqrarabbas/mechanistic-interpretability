# Project Progress: ViT Mechanistic Interpretability

## Goal
Reproduce a simplified version of the paper "A Mechanistic Analysis of Adversarial Fine-tuning of Vision Transformers".

We will use only one corruption instead of six:
- Gaussian Blur level 4

We will fine-tune one ViT model and compare it with the base ViT.

## Paper Simplification
Original paper:
- Blur level 1
- Blur level 2
- Blur level 4
- Noise level 1
- Noise level 2
- Noise level 4
- Six separately fine-tuned models

Our version:
- Gaussian Blur level 4 only
- One fine-tuned model

## Completed
- Created project structure
- Created Conda environment `vit`
- Verified GPU works
- Loaded pretrained Google ViT-B/16
- Predicted one image successfully
- Added Top-5 prediction output
- Loaded ImageNet validation dataset
- Verified ImageNet image folder
- Verified ground-truth file
- Verified meta.mat file
- Built reusable `ImageNetDataset`
- Built DataLoader
- Added Gaussian Blur corruption
- Evaluated base ViT on clean and blurred images

## Current Baseline Results

Using 1000 ImageNet validation samples:

| Dataset | Top-1 | Top-5 |
|---|---:|---:|
| Clean | 0.8060 | 0.9640 |
| Gaussian Blur level 4 | 0.6500 | 0.8480 |

## Interpretation
The base ViT performs well on clean images, but Gaussian Blur reduces performance.

Top-1 drops from 80.6% to 65.0%.
Top-5 drops from 96.4% to 84.8%.

This confirms that blur hurts the base ViT and motivates fine-tuning on blurred images.

## Current Project Structure
- `Dataset/` contains ImageNet data
- `data/imagenet_dataset.py` contains ImageNetDataset
- `corruption/gaussian_blur.py` contains Gaussian Blur
- `evaluation/evaluate.py` contains evaluation function
- `scripts/` contains testing and running scripts
- `results/` stores outputs
- `images/` stores small test images

## Next Step
Create training pipeline:
- Create `training/`
- Create `training/train_blur.py`
- Fine-tune ViT on Gaussian Blur level 4
- Save model checkpoint
- Evaluate fine-tuned model on clean and blurred images
