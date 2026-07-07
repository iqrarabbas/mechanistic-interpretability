# ViT Mechanistic Interpretability

This project is a simplified reproduction of the paper **"A Mechanistic Analysis of Adversarial Fine-tuning of Vision Transformers"**.

The goal is to study how fine-tuning a pretrained Vision Transformer on a specific image corruption changes its robustness. The original paper studies several blur and noise corruptions. This project focuses mainly on **Gaussian Blur level 4**, then checks whether the learned blur robustness transfers to **Gaussian Noise level 4**.

## Project Goal

The project compares:

- A base pretrained ViT model
- A ViT model fine-tuned on Gaussian Blur level 4

The main questions are:

- How much does blur hurt the base ViT?
- Does fine-tuning on blurred images improve blur robustness?
- Does blur fine-tuning preserve clean-image accuracy?
- Does robustness to blur transfer to Gaussian noise?
- Later: how do internal ViT representations change after fine-tuning?

## Model

The model used throughout the project is:

```text
google/vit-base-patch16-224
```

This is a pretrained Vision Transformer Base model with patch size 16, trained for ImageNet 1000-class classification.

## Dataset

The project uses the ImageNet ILSVRC2012 validation set.

Expected dataset layout:

```text
vit_mi/
  Dataset/
    ILSVRC2012_img_val/
      ILSVRC2012_val_00000001.JPEG
      ...
    ILSVRC2012_devkit_t12/
      data/
        ILSVRC2012_validation_ground_truth.txt
        meta.mat
```

The dataset is loaded by `data/imagenet_dataset.py`.

Important details:

- Images are read from the original ImageNet validation folder.
- Corruptions are applied at runtime in memory.
- The original ImageNet images are not modified.
- ImageNet ILSVRC labels are mapped correctly to Hugging Face/PyTorch model class IDs using WordNet IDs.

## Data Splits

The current experiments use the following ImageNet validation ranges:

| Image Range | Purpose |
|---|---|
| 1-10,000 | Blur-4 fine-tuning |
| 10,001-11,000 | Validation during training |
| 11,001-12,000 | Final unseen testing |

This keeps the final evaluation separate from both training and validation.

## Corruptions

### Gaussian Blur Level 4

Implemented in:

```text
corruption/gaussian_blur.py
```

This is the main corruption used for fine-tuning and robustness evaluation.

### Gaussian Noise Level 4

Implemented in:

```text
corruption/gaussian_noise.py
```

This is used to test whether blur fine-tuning transfers to a different corruption type.

## Evaluation Metrics

Evaluation is implemented in:

```text
evaluation/evaluate.py
```

The project reports:

- **Top-1 accuracy**: the highest-scoring prediction must match the true label.
- **Top-5 accuracy**: the true label must appear in the model's five best guesses.
- **Top-10 accuracy**: the true label must appear in the model's ten best guesses.

Top-5 and Top-10 are useful for ImageNet because many classes are visually similar.

## Current Results

### Base ViT Results

Using 1000 ImageNet validation samples:

| Dataset | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Clean | 80.6% | 96.4% | 98.3% |
| Gaussian Blur Level 4 | 65.0% | 84.8% | 89.9% |

The base model performs well on clean images, but blur significantly reduces accuracy.

### Fine-tuned ViT Results

After fine-tuning on Gaussian Blur level 4:

| Dataset | Top-1 | Top-5 | Top-10 |
|---|---:|---:|---:|
| Clean | 79.0% | 93.9% | 96.7% |
| Gaussian Blur Level 4 | 72.4% | 91.2% | 94.9% |

Fine-tuning improves blur robustness while causing only a small clean-accuracy drop.

### Base vs Fine-tuned Comparison

| Metric | Base ViT | Blur-4 Fine-tuned ViT | Change |
|---|---:|---:|---:|
| Clean Top-1 | 80.6% | 79.0% | -1.6% |
| Clean Top-5 | 96.4% | 93.9% | -2.5% |
| Clean Top-10 | 98.3% | 96.7% | -1.6% |
| Blur Top-1 | 65.0% | 72.4% | +7.4% |
| Blur Top-5 | 84.8% | 91.2% | +6.4% |
| Blur Top-10 | 89.9% | 94.9% | +5.0% |

The largest gain is on Blur Top-1 accuracy, which improves by 7.4 percentage points.

### Noise-4 Transfer Evaluation

| Model | Clean Top-1 | Noise-4 Top-1 | Noise-4 Top-5 | Noise-4 Top-10 |
|---|---:|---:|---:|---:|
| Base ViT | 79.8% | 69.3% | 88.1% | 92.6% |
| Blur-4 Fine-tuned ViT | 79.0% | 65.6% | 87.1% | 91.5% |

Blur fine-tuning improves blur robustness, but it does not transfer to Gaussian Noise level 4. Noise accuracy drops slightly after blur-specific fine-tuning.

## Key Finding

The main reproduction result is:

```text
Fine-tuning on Gaussian Blur level 4 improves blur robustness,
while mostly preserving clean accuracy.
```

This matches the main trend from the paper.

## Project Structure

```text
vit_mi/
  Dataset/                  ImageNet validation data and devkit
  checkpoints/              Saved fine-tuned model checkpoints
  corruption/               Runtime corruption functions
    gaussian_blur.py
    gaussian_noise.py
  data/                     Dataset loading and label mapping
    imagenet_dataset.py
  docs/                     Research notes and progress reports
    PAPER_NOTES.md
    PROGRESS_REPORT.md
    RESEARCH_LOG.md
  evaluation/               Evaluation metrics
    evaluate.py
  images/                   Small local test images
  results/                  Result plots
  scripts/                  Evaluation, plotting, and test scripts
  training/                 Fine-tuning scripts
    train_blur.py
  PROJECT_PROGRESS.md
  TODO.md
  requirements.txt
```

## Setup

Create and activate the Conda environment:

```bash
conda create -n vit python=3.11
conda activate vit
```

Install dependencies:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- PyTorch
- TorchVision
- Transformers
- Pillow
- SciPy
- tqdm
- Matplotlib
- scikit-learn
- Accelerate

## Running the Project

Run commands from the project root:

```bash
cd vit_mi
```

Some scripts import local packages such as `data`, `corruption`, and `evaluation`. If imports fail, run scripts with:

```bash
PYTHONPATH=. python scripts/<script_name>.py
```

### Check Setup

```bash
PYTHONPATH=. python scripts/check_setup.py
```

### Test Dataset Loading

```bash
PYTHONPATH=. python scripts/test_dataset.py
PYTHONPATH=. python scripts/test_dataloader.py
PYTHONPATH=. python scripts/test_blur_dataset.py
```

### Predict a Single Image

```bash
PYTHONPATH=. python scripts/predict_image.py
```

### Evaluate Base ViT on Clean and Blur-4 Images

```bash
PYTHONPATH=. python scripts/evaluate_base.py
```

This evaluates the pretrained model on clean images and Gaussian Blur level 4 images.

### Fine-tune ViT on Gaussian Blur Level 4

```bash
PYTHONPATH=. python training/train_blur.py
```

Training setup:

- Training samples: 10,000
- Validation samples: 1,000
- Optimizer: AdamW
- Learning rate: `5e-5`
- Max epochs: 10
- Early stopping patience: 2
- Best checkpoint: `checkpoints/vit_blur4_best`

### Evaluate the Fine-tuned Model

```bash
PYTHONPATH=. python scripts/evaluate_finetuned.py
```

This evaluates the saved checkpoint on the final unseen clean and blurred test split.

### Evaluate Noise Transfer

```bash
PYTHONPATH=. python scripts/evaluate_base_noise.py
PYTHONPATH=. python scripts/evaluate_blur_model_on_noise.py
```

These scripts test Gaussian Noise level 4 performance for the base model and the blur-fine-tuned model.

### Generate Plots

```bash
PYTHONPATH=. python scripts/plot_base_results.py
PYTHONPATH=. python scripts/plot_finetune_results.py
PYTHONPATH=. python scripts/plot_final_results.py
PYTHONPATH=. python scripts/plot_noise_results.py
```

Generated plots are saved in:

```text
results/
```

## Result Plots

The `results/` folder contains plots such as:

- `base_vit_clean_vs_blur_accuracy.png`
- `clean_base_vs_finetuned.png`
- `blur_base_vs_finetuned.png`
- `blur_improvement_after_finetuning.png`
- `base_vit_clean_vs_noise4_accuracy.png`
- `base_vs_finetuned_1epoch_accuracy.png`

## Important Bug Fix: Label Mapping

An important issue was fixed in `data/imagenet_dataset.py`.

ImageNet validation labels use ILSVRC class IDs, but Hugging Face/PyTorch ViT logits use the standard ImageNet model class order. Those are not the same thing.

The dataset loader now maps:

```text
ILSVRC ID -> WordNet ID -> model class ID
```

Without this mapping, evaluation accuracy becomes almost zero even when the model is predicting correctly.

## Documentation Files

Additional project notes are stored in:

- `PROJECT_PROGRESS.md`
- `TODO.md`
- `docs/RESEARCH_LOG.md`
- `docs/PAPER_NOTES.md`
- `docs/PROGRESS_REPORT.md`

`docs/PROGRESS_REPORT.md` contains the most complete written summary of the current experimental results.

## Next Steps

The next phase is mechanistic interpretability analysis:

- Add Expected Calibration Error (ECE)
- Run Logit Lens analysis
- Measure attention entropy on clean vs blurred images
- Compare hidden representations between base and fine-tuned models
- Optionally train a Noise-4 fine-tuned model
- Optionally test adversarial attacks such as FGSM and PGD

## Current Status

Completed:

- Dataset loading
- Correct ImageNet label mapping
- Gaussian Blur level 4 corruption
- Gaussian Noise level 4 corruption
- Base ViT evaluation
- Blur-4 fine-tuning
- Fine-tuned model evaluation
- Noise-transfer evaluation
- Result plots

Next:

- Move from accuracy reproduction to mechanistic analysis.
