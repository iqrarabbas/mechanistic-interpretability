from pathlib import Path

from torch.utils.data import DataLoader

from data.imagenet_dataset import ImageNetDataset

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

dataset = ImageNetDataset(DATASET_DIR, max_samples=10)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)

print("Number of batches:", len(dataloader))

for images, labels in dataloader:

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    print(labels)

    break