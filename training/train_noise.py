from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification, get_linear_schedule_with_warmup

from data.imagenet_dataset import ImageNetDataset
from evaluation.evaluate import evaluate_model

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "vit_noise4_best"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/vit-base-patch16-224"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

train_dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=10000,
    start_index=0,
    corruption="noise",
    noise_severity=4,
)

val_dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=1000,
    start_index=10000,
    corruption="noise",
    noise_severity=4,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=2,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=2,
)

model = ViTForImageClassification.from_pretrained(MODEL_NAME)
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

epochs = 10
patience = 2
best_val_top1 = 0.0
epochs_without_improvement = 0

total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps,
)

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")

    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(pixel_values=images, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(train_loader)
    print(f"\nEpoch {epoch + 1} average loss: {avg_loss:.4f}")

    print("Evaluating on validation noise set...")
    val_results = evaluate_model(model, val_loader, device)
    val_top1 = val_results["top1"]

    print(
        f"Validation Noise Top-1: {val_results['top1']:.4f} | "
        f"Top-5: {val_results['top5']:.4f} | "
        f"Top-10: {val_results['top10']:.4f}"
    )

    if val_top1 > best_val_top1:
        best_val_top1 = val_top1
        epochs_without_improvement = 0

        print("New best noise model found. Saving checkpoint...")
        model.save_pretrained(CHECKPOINT_DIR)

    else:
        epochs_without_improvement += 1
        print(f"No improvement for {epochs_without_improvement} epoch(s).")

        if epochs_without_improvement >= patience:
            print("Early stopping triggered.")
            break

print("\nNoise training finished.")
print(f"Best validation noise Top-1: {best_val_top1:.4f}")
print(f"Best noise model saved to: {CHECKPOINT_DIR}")