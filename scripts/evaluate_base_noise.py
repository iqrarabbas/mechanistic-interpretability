from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from evaluation.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
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

clean_loader = DataLoader(clean_dataset, batch_size=16, shuffle=False)
noise_loader = DataLoader(noise_dataset, batch_size=16, shuffle=False)

print("Evaluating base model on clean images...")
clean_results = evaluate_model(model, clean_loader, device)

print("Evaluating base model on Gaussian noise level 4 images...")
noise_results = evaluate_model(model, noise_loader, device)

print("\nBase Model Results")
print("-" * 40)
print(f"Clean Top-1:  {clean_results['top1']:.4f}")
print(f"Clean Top-5:  {clean_results['top5']:.4f}")
print(f"Clean Top-10: {clean_results['top10']:.4f}")
print(f"Noise Top-1:  {noise_results['top1']:.4f}")
print(f"Noise Top-5:  {noise_results['top5']:.4f}")
print(f"Noise Top-10: {noise_results['top10']:.4f}")
