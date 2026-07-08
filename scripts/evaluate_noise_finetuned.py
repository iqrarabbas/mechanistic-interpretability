from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from evaluation.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).parent.parent

DATASET_DIR = PROJECT_ROOT / "Dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "vit_noise4_best"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)

model = ViTForImageClassification.from_pretrained(
    CHECKPOINT_DIR
).to(device)

clean_dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=1000,
    start_index=11000,
    corruption=None,
)

noise_dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=1000,
    start_index=11000,
    corruption="noise",
    noise_severity=4,
)

clean_loader = DataLoader(
    clean_dataset,
    batch_size=16,
    shuffle=False,
)

noise_loader = DataLoader(
    noise_dataset,
    batch_size=16,
    shuffle=False,
)

print("Evaluating on clean test images...")
clean_results = evaluate_model(model, clean_loader, device)

print("Evaluating on Noise-4 test images...")
noise_results = evaluate_model(model, noise_loader, device)

print("\nNoise Fine-tuned Model Results")
print("-" * 40)

print(f"Clean Top-1:  {clean_results['top1']:.4f}")
print(f"Clean Top-5:  {clean_results['top5']:.4f}")
print(f"Clean Top-10: {clean_results['top10']:.4f}")

print()

print(f"Noise Top-1:  {noise_results['top1']:.4f}")
print(f"Noise Top-5:  {noise_results['top5']:.4f}")
print(f"Noise Top-10: {noise_results['top10']:.4f}")