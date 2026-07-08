from pathlib import Path

import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification
from tqdm import tqdm

from data.imagenet_dataset import ImageNetDataset
from interpretability.logit_lens import get_correct_class_probs_by_layer

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BASE_MODEL_NAME = "google/vit-base-patch16-224"
BLUR_MODEL_DIR = PROJECT_ROOT / "checkpoints" / "vit_blur4_best"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=200,
    start_index=11000,
    corruption="blur",
    blur_severity=4,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=False,
)

base_model = ViTForImageClassification.from_pretrained(
    BASE_MODEL_NAME
).to(device)

blur_model = ViTForImageClassification.from_pretrained(
    BLUR_MODEL_DIR
).to(device)


def average_logit_lens(model, loader, device):
    all_probs = []

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device)

        probs_by_layer = get_correct_class_probs_by_layer(
            model,
            images,
            labels,
        )

        all_probs.append(probs_by_layer)

    all_probs = torch.cat(all_probs, dim=1)

    mean_probs = all_probs.mean(dim=1)

    return mean_probs.numpy()


print("Running Logit Lens on Base model...")
base_probs = average_logit_lens(base_model, loader, device)

print("Running Logit Lens on Blur-4 fine-tuned model...")
blur_probs = average_logit_lens(blur_model, loader, device)

layers = list(range(len(base_probs)))

plt.figure(figsize=(8, 5))
plt.plot(layers, base_probs * 100, marker="o", label="Base ViT")
plt.plot(layers, blur_probs * 100, marker="o", label="Blur-4 Fine-tuned ViT")

plt.xlabel("Layer")
plt.ylabel("Average Correct Class Probability (%)")
plt.title("Logit Lens on Gaussian Blur Level 4 Images")
plt.xticks(layers)
plt.ylim(0, 100)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

output_path = RESULTS_DIR / "logit_lens_base_vs_blur_ft_on_blur4.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved plot to: {output_path}")

print("\nAverage correct-class probability by layer")
print("-" * 50)

for layer, base_p, blur_p in zip(layers, base_probs, blur_probs):
    print(
        f"Layer {layer:02d} | "
        f"Base: {base_p * 100:.2f}% | "
        f"Blur-FT: {blur_p * 100:.2f}%"
    )