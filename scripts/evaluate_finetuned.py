from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from evaluation.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "vit_blur4_best"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = ViTForImageClassification.from_pretrained(CHECKPOINT_DIR).to(device)

clean_dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=1000,
    start_index=11000,
    corruption=None,
)

blur_dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=1000,
    start_index=11000,
    corruption="blur",
    blur_severity=4,
)

clean_loader = DataLoader(clean_dataset, batch_size=16, shuffle=False)
blur_loader = DataLoader(blur_dataset, batch_size=16, shuffle=False)

print("Evaluating fine-tuned model on clean images...")
clean_results = evaluate_model(model, clean_loader, device)

print("Evaluating fine-tuned model on blurred images...")
blur_results = evaluate_model(model, blur_loader, device)

print("\nFine-tuned Results")
print("-" * 40)
print(f"Clean Top-1:  {clean_results['top1']:.4f}")
print(f"Clean Top-5:  {clean_results['top5']:.4f}")
print(f"Clean Top-10: {clean_results['top10']:.4f}")
print(f"Blur Top-1:   {blur_results['top1']:.4f}")
print(f"Blur Top-5:   {blur_results['top5']:.4f}")
print(f"Blur Top-10:  {blur_results['top10']:.4f}")