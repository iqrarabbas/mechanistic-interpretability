from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from evaluation.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
).to(device)

clean_dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=1000,
    corruption=None
)

blur_dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=1000,
    corruption="blur"
)

clean_loader = DataLoader(clean_dataset, batch_size=16, shuffle=False)
blur_loader = DataLoader(blur_dataset, batch_size=16, shuffle=False)

# print("Evaluating base model on clean images...")
# clean_top1, clean_top5 = evaluate_model(model, clean_loader, device)

# print("Evaluating base model on blurred images...")
# blur_top1, blur_top5 = evaluate_model(model, blur_loader, device)

# print("\nResults")
# print("-" * 30)
# print(f"Clean Top-1: {clean_top1:.4f}")
# print(f"Clean Top-5: {clean_top5:.4f}")
# print(f"Blur Top-1:  {blur_top1:.4f}")
# print(f"Blur Top-5:  {blur_top5:.4f}")

print("Evaluating base model on clean images...")
clean_results = evaluate_model(model, clean_loader, device)

print("Evaluating base model on blurred images...")
blur_results = evaluate_model(model, blur_loader, device)

print("\nResults")
print("-" * 40)
print(f"Clean Top-1:  {clean_results['top1']:.4f}")
print(f"Clean Top-5:  {clean_results['top5']:.4f}")
print(f"Clean Top-10: {clean_results['top10']:.4f}")
print(f"Blur Top-1:   {blur_results['top1']:.4f}")
print(f"Blur Top-5:   {blur_results['top5']:.4f}")
print(f"Blur Top-10:  {blur_results['top10']:.4f}")