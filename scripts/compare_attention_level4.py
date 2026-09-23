import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.imagenet_dataset import ImageNetDataset
from interpretability.attention_analysis import get_attention_metrics_by_layer

DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "attention"
BASE_MODEL = "google/vit-base-patch16-224"
MODEL_PATHS = {
    "blur": PROJECT_ROOT / "checkpoints" / "vit_blur4_best",
    "noise": PROJECT_ROOT / "checkpoints" / "vit_noise4_best",
}
METRICS = (
    "clean_entropy",
    "corrupted_entropy",
    "entropy_difference",
    "mean_squared_difference",
    "cosine_similarity",
)


class PairedImageNetDataset(Dataset):
    def __init__(self, corruption, samples, start_index, corruption_seed):
        common = {
            "dataset_dir": DATASET_DIR,
            "max_samples": samples,
            "start_index": start_index,
        }
        self.clean = ImageNetDataset(**common, corruption=None)
        self.corrupted = ImageNetDataset(
            **common,
            corruption=corruption,
            blur_severity=4,
            noise_severity=4,
            corruption_seed=corruption_seed,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean_image, clean_label = self.clean[index]
        corrupted_image, corrupted_label = self.corrupted[index]
        if clean_label != corrupted_label:
            raise RuntimeError("Paired clean and corrupted labels do not match.")
        return clean_image, corrupted_image, clean_label


def analyze_model(model, loader, device):
    weighted_sums = None
    total_samples = 0

    for clean_images, corrupted_images, _ in tqdm(loader, leave=False):
        clean_images = clean_images.to(device)
        corrupted_images = corrupted_images.to(device)
        batch_size = clean_images.size(0)
        batch_metrics = get_attention_metrics_by_layer(
            model, clean_images, corrupted_images
        )

        if weighted_sums is None:
            weighted_sums = [
                {metric: 0.0 for metric in METRICS} for _ in batch_metrics
            ]

        for layer_index, layer_metrics in enumerate(batch_metrics):
            for metric in METRICS:
                weighted_sums[layer_index][metric] += (
                    layer_metrics[metric].item() * batch_size
                )
        total_samples += batch_size

    return {
        metric: [layer[metric] / total_samples for layer in weighted_sums]
        for metric in METRICS
    }


def save_metric_plot(corruption, metric, base_results, tuned_results):
    layers = range(1, len(base_results[metric]) + 1)
    labels = {
        "entropy_difference": "Entropy difference (corrupted - clean)",
        "mean_squared_difference": "Mean squared attention difference",
        "cosine_similarity": "Attention cosine similarity",
    }
    plt.figure(figsize=(8, 5))
    plt.plot(layers, base_results[metric], marker="o", label="Base ViT")
    plt.plot(
        layers,
        tuned_results[metric],
        marker="o",
        label=f"{corruption.title()}-4 fine-tuned ViT",
    )
    plt.xlabel("Transformer layer")
    plt.ylabel(labels[metric])
    plt.title(f"Attention analysis on Gaussian {corruption.title()} severity 4")
    plt.xticks(list(layers))
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(
        RESULTS_DIR / f"{corruption}4_{metric}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def load_model(model_path, device):
    return ViTForImageClassification.from_pretrained(
        model_path,
        attn_implementation="eager",
    ).to(device)


def run_corruption(corruption, args, device):
    dataset = PairedImageNetDataset(
        corruption=corruption,
        samples=args.samples,
        start_index=args.start_index,
        corruption_seed=args.corruption_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Analyzing base ViT attention on clean/{corruption}-4 pairs...")
    base_model = load_model(BASE_MODEL, device)
    base_results = analyze_model(base_model, loader, device)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Analyzing {corruption}-4 fine-tuned ViT attention...")
    tuned_model = load_model(MODEL_PATHS[corruption], device)
    tuned_results = analyze_model(tuned_model, loader, device)
    del tuned_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    output = {
        "corruption": corruption,
        "severity": 4,
        "start_index": args.start_index,
        "samples": args.samples,
        "corruption_seed": args.corruption_seed,
        "entropy_difference_definition": "corrupted_minus_clean",
        "base": base_results,
        "fine_tuned": tuned_results,
    }
    with open(RESULTS_DIR / f"{corruption}4_metrics.json", "w") as output_file:
        json.dump(output, output_file, indent=2)

    for metric in (
        "entropy_difference",
        "mean_squared_difference",
        "cosine_similarity",
    ):
        save_metric_plot(corruption, metric, base_results, tuned_results)


def parse_args():
    parser = argparse.ArgumentParser(description="Level-4 ViT attention analysis")
    parser.add_argument(
        "--corruption", choices=["blur", "noise", "both"], default="both"
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    corruptions = ["blur", "noise"] if args.corruption == "both" else [args.corruption]
    for corruption in corruptions:
        run_corruption(corruption, args, device)
    print(f"Saved attention results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
