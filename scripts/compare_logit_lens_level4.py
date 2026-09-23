import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.imagenet_dataset import ImageNetDataset
from interpretability.logit_lens import (
    get_first_correct_prediction_layers,
    get_layerwise_class_outputs,
)

DATASET_DIR = PROJECT_ROOT / "Dataset"
RESULTS_DIR = PROJECT_ROOT / "results" / "logit_lens"
BASE_MODEL = "google/vit-base-patch16-224"
MODEL_PATHS = {
    "blur": PROJECT_ROOT / "checkpoints" / "vit_blur4_best",
    "noise": PROJECT_ROOT / "checkpoints" / "vit_noise4_best",
}


def analyze_model(model, loader, device):
    probability_batches = []
    first_layer_batches = []

    for images, labels in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        probabilities, predictions = get_layerwise_class_outputs(
            model, images, labels
        )
        probability_batches.append(probabilities)
        first_layer_batches.append(
            get_first_correct_prediction_layers(predictions, labels)
        )

    probabilities = torch.cat(probability_batches, dim=1)
    first_layers = torch.cat(first_layer_batches)
    successful = first_layers >= 1

    return {
        "mean_correct_class_probability": probabilities.mean(dim=1).tolist(),
        "mean_first_correct_layer": (
            first_layers[successful].float().mean().item()
            if successful.any()
            else None
        ),
        "samples_correct_at_any_layer": successful.sum().item(),
        "total_samples": first_layers.numel(),
    }


def save_probability_plot(corruption, base_results, tuned_results):
    layers = range(1, len(base_results["mean_correct_class_probability"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(
        layers,
        base_results["mean_correct_class_probability"],
        marker="o",
        label="Base ViT",
    )
    plt.plot(
        layers,
        tuned_results["mean_correct_class_probability"],
        marker="o",
        label=f"{corruption.title()}-4 fine-tuned ViT",
    )
    plt.xlabel("Transformer layer")
    plt.ylabel("Mean probability of correct class")
    plt.title(f"Logit lens on Gaussian {corruption.title()} severity 4")
    plt.xticks(list(layers))
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.savefig(
        RESULTS_DIR / f"{corruption}4_correct_class_probability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def save_first_layer_plot(corruption, base_results, tuned_results):
    values = [
        base_results["mean_first_correct_layer"],
        tuned_results["mean_first_correct_layer"],
    ]
    labels = ["Base ViT", f"{corruption.title()}-4 fine-tuned ViT"]
    plt.figure(figsize=(7, 5))
    plt.bar(labels, values)
    plt.ylabel("Mean first correct-prediction layer")
    plt.title(f"First correct class emergence on {corruption.title()}-4")
    plt.ylim(0, 12)
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.savefig(
        RESULTS_DIR / f"{corruption}4_first_correct_layer.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def run_corruption(corruption, args, device):
    dataset = ImageNetDataset(
        DATASET_DIR,
        max_samples=args.samples,
        start_index=args.start_index,
        corruption=corruption,
        blur_severity=4,
        noise_severity=4,
        corruption_seed=args.corruption_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print(f"Analyzing base ViT on {corruption}-4...")
    base_model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device)
    base_results = analyze_model(base_model, loader, device)
    del base_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print(f"Analyzing {corruption}-4 fine-tuned ViT...")
    tuned_model = ViTForImageClassification.from_pretrained(
        MODEL_PATHS[corruption]
    ).to(device)
    tuned_results = analyze_model(tuned_model, loader, device)
    del tuned_model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    results = {
        "corruption": corruption,
        "severity": 4,
        "start_index": args.start_index,
        "samples": args.samples,
        "corruption_seed": args.corruption_seed,
        "base": base_results,
        "fine_tuned": tuned_results,
    }
    with open(RESULTS_DIR / f"{corruption}4_metrics.json", "w") as output_file:
        json.dump(results, output_file, indent=2)

    save_probability_plot(corruption, base_results, tuned_results)
    save_first_layer_plot(corruption, base_results, tuned_results)

    print(
        f"{corruption.title()}-4 mean first correct layer: "
        f"base={base_results['mean_first_correct_layer']:.3f}, "
        f"fine-tuned={tuned_results['mean_first_correct_layer']:.3f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Level-4 ViT logit-lens analysis")
    parser.add_argument(
        "--corruption", choices=["blur", "noise", "both"], default="both"
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=16)
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
    print(f"Saved logit-lens results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
