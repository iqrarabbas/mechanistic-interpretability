from pathlib import Path

from PIL import Image
from scipy.io import loadmat
from torch.utils.data import Dataset
from transformers import AutoImageProcessor

from corruption.gaussian_blur import apply_gaussian_blur
from corruption.gaussian_noise import apply_gaussian_noise


class ImageNetDataset(Dataset):
    def __init__(
        self,
        dataset_dir,
        max_samples=None,
        start_index=0,
        corruption=None,
        blur_severity=4,
        noise_severity=4,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.max_samples = max_samples
        self.start_index = start_index
        self.corruption = corruption
        self.blur_severity = blur_severity
        self.noise_severity = noise_severity

        self.image_dir = self.dataset_dir / "ILSVRC2012_img_val"
        self.devkit_dir = self.dataset_dir / "ILSVRC2012_devkit_t12"

        self.ground_truth_file = (
            self.devkit_dir / "data" / "ILSVRC2012_validation_ground_truth.txt"
        )
        self.meta_file = self.devkit_dir / "data" / "meta.mat"

        self.processor = AutoImageProcessor.from_pretrained(
            "google/vit-base-patch16-224"
        )

        meta = loadmat(self.meta_file)
        synsets = meta["synsets"]

        with open(self.ground_truth_file, "r") as f:
            raw_labels = [int(line.strip()) for line in f.readlines()]

        id_to_wnid = {}

        for i in range(len(synsets)):
            ilsvrc_id = int(synsets[i][0][0][0][0])

            if 1 <= ilsvrc_id <= 1000:
                wnid = synsets[i][0][1][0]
                id_to_wnid[ilsvrc_id] = wnid

        model_wnids = sorted(id_to_wnid.values())

        wnid_to_model_id = {
            wnid: model_id
            for model_id, wnid in enumerate(model_wnids)
        }

        self.label_id_to_model_id = {
            ilsvrc_id: wnid_to_model_id[wnid]
            for ilsvrc_id, wnid in id_to_wnid.items()
        }

        all_labels = [
            self.label_id_to_model_id[label_id]
            for label_id in raw_labels
        ]

        all_image_paths = sorted(self.image_dir.glob("*.JPEG"))

        end_index = None if max_samples is None else start_index + max_samples

        self.image_paths = all_image_paths[start_index:end_index]
        self.labels = all_labels[start_index:end_index]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        label = self.labels[index]

        image = Image.open(image_path).convert("RGB")

        if self.corruption == "blur":
            image = apply_gaussian_blur(
                image,
                severity=self.blur_severity,
            )

        elif self.corruption == "noise":
            image = apply_gaussian_noise(
                image,
                severity=self.noise_severity,
            )

        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label