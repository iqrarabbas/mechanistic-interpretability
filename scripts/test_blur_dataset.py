from pathlib import Path
from torch.utils.data import DataLoader
from data.imagenet_dataset import ImageNetDataset

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

dataset = ImageNetDataset(
    DATASET_DIR,
    max_samples=16,
    corruption="blur"
)

dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

for images, labels in dataloader:
    print("Blurred batch images:", images.shape)
    print("Labels:", labels)
    break