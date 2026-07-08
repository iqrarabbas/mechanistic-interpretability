from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from interpretability.logit_lens import get_correct_class_probs_by_layer

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224"
).to(device)

dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=1,
    start_index=11000,
    corruption="blur",
    blur_severity=4,
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

images, labels = next(iter(loader))
images = images.to(device)
labels = labels.to(device)

probs_by_layer = get_correct_class_probs_by_layer(
    model,
    images,
    labels,
)

print("\nCorrect class probability by layer")
print("-" * 40)

for layer_idx, prob in enumerate(probs_by_layer):
    print(f"Layer {layer_idx:02d}: {prob.item():.6f}")