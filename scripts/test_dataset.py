from pathlib import Path

from data.imagenet_dataset import ImageNetDataset

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

dataset = ImageNetDataset(DATASET_DIR, max_samples=10)

print("Dataset size:", len(dataset))

image, label = dataset[0]

print("Image tensor shape:", image.shape)
print("Label:", label)