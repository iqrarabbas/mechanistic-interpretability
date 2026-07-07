from pathlib import Path
from scipy.io import loadmat

PROJECT_ROOT = Path(__file__).parent.parent
DATASET_DIR = PROJECT_ROOT / "Dataset"

IMAGE_DIR = DATASET_DIR / "ILSVRC2012_img_val"
DEVKIT_DIR = DATASET_DIR / "ILSVRC2012_devkit_t12"
GROUND_TRUTH_FILE = DEVKIT_DIR / "data" / "ILSVRC2012_validation_ground_truth.txt"
META_FILE = DEVKIT_DIR / "data" / "meta.mat"

with open(GROUND_TRUTH_FILE, "r") as f:
    labels = [int(line.strip()) for line in f.readlines()]

meta = loadmat(META_FILE)
synsets = meta["synsets"]

first_image = IMAGE_DIR / "ILSVRC2012_val_00000001.JPEG"
first_label_id = labels[0]

class_name = synsets[first_label_id - 1][0][2][0]

print("Image:", first_image.name)
print("Label ID:", first_label_id)
print("Class name:", class_name)