import argparse
import csv
import json
from pathlib import Path
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.imagenet_dataset import ImageNetDataset
from interpretability.sae import VanillaReLUSAE


DATASET_DIR = PROJECT_ROOT / "Dataset"
SAE_DIR = PROJECT_ROOT / "checkpoints" / "sae" / "blur4_base_vanilla_paper"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment1_base_blur4"
BASE_MODEL = "google/vit-base-patch16-224"
EPSILON = 1e-8


class PairedDataset(Dataset):
    def __init__(self, samples, start_index, seed):
        common = dict(
            dataset_dir=DATASET_DIR,
            max_samples=samples,
            start_index=start_index,
        )
        self.clean = ImageNetDataset(**common)
        self.blurred = ImageNetDataset(
            **common,
            corruption="blur",
            blur_severity=4,
            corruption_seed=seed,
        )
        if self.clean.image_paths != self.blurred.image_paths:
            raise RuntimeError("Clean and Blur-4 datasets do not contain identical paths.")
        if self.clean.labels != self.blurred.labels:
            raise RuntimeError("Clean and Blur-4 datasets do not contain identical labels.")

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        blurred, blurred_label = self.blurred[index]
        if label != blurred_label:
            raise RuntimeError(f"Label mismatch at paired index {index}.")
        path = self.clean.image_paths[index]
        return clean, blurred, label, path.name, str(path)


def load_sae(device):
    metadata = json.loads((SAE_DIR / "training.json").read_text())
    if metadata["sae_type"] != "vanilla":
        raise ValueError("Experiment 1 expected the existing Vanilla SAE checkpoint.")
    sae = VanillaReLUSAE(expansion_factor=metadata["expansion_factor"])
    sae.load_state_dict(
        torch.load(SAE_DIR / "model.pt", map_location="cpu", weights_only=True)
    )
    return sae.to(device).eval(), metadata


def safe_mean(values):
    return float(np.mean(values)) if values else None


def safe_median(values):
    return float(np.median(values)) if values else None


def summarize_rows(rows):
    fields = [
        "sae_cosine_similarity",
        "feature_retention",
        "jaccard_similarity",
        "relative_sae_change",
        "clean_sae_norm",
        "blur_sae_norm",
        "norm_ratio",
        "n_shared_features",
        "n_lost_features",
        "n_new_features",
        "shared_feature_mean_signed_change",
        "shared_feature_mean_absolute_change",
        "shared_feature_median_absolute_change",
        "shared_features_percent_increased",
        "shared_features_percent_decreased",
    ]
    return {
        field: {
            "mean": safe_mean([row[field] for row in rows]),
            "median": safe_median([row[field] for row in rows]),
        }
        for field in fields
    } | {"images": len(rows)}


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_results(output_dir, rows, shared_clean, shared_blur):
    cosine = np.array([row["sae_cosine_similarity"] for row in rows])
    retention = np.array([row["feature_retention"] for row in rows])
    relative_change = np.array([row["relative_sae_change"] for row in rows])

    plt.figure(figsize=(7, 5))
    plt.hist(cosine, bins=50, density=True)
    plt.xlabel("Mean corresponding-patch SAE cosine similarity")
    plt.ylabel("Density")
    plt.title("Experiment 1: Clean vs Blur-4 SAE cosine")
    plt.tight_layout()
    plt.savefig(output_dir / "plot1_cosine_distribution.png", dpi=250)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(retention, bins=50, density=True)
    plt.xlabel("Mean patch feature retention (shared / clean active)")
    plt.ylabel("Density")
    plt.title("Experiment 1: Active-feature retention")
    plt.tight_layout()
    plt.savefig(output_dir / "plot2_feature_retention_distribution.png", dpi=250)
    plt.close()

    plt.figure(figsize=(6, 6))
    clean_norm = np.array([row["clean_sae_norm"] for row in rows])
    blur_norm = np.array([row["blur_sae_norm"] for row in rows])
    limits = [min(clean_norm.min(), blur_norm.min()), max(clean_norm.max(), blur_norm.max())]
    plt.scatter(clean_norm, blur_norm, s=8, alpha=0.35)
    plt.plot(limits, limits, "k--", label="y=x")
    plt.xlabel("Mean clean patch SAE norm")
    plt.ylabel("Mean Blur-4 patch SAE norm")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot3_clean_vs_blur_norm.png", dpi=250)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(relative_change, bins=50, density=True)
    plt.xlabel("Mean patch relative SAE activation change")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(output_dir / "plot4_relative_change_distribution.png", dpi=250)
    plt.close()

    plt.figure(figsize=(6, 6))
    limits = [
        min(float(shared_clean.min()), float(shared_blur.min())),
        max(float(shared_clean.max()), float(shared_blur.max())),
    ]
    plt.hexbin(shared_clean, shared_blur, gridsize=80, bins="log", mincnt=1)
    plt.plot(limits, limits, "r--", label="y=x")
    plt.xlabel("Clean activation of shared feature")
    plt.ylabel("Blur-4 activation of shared feature")
    plt.colorbar(label="log count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plot5_shared_feature_strength.png", dpi=250)
    plt.close()

    plt.figure(figsize=(7, 5))
    labels = ["Shared", "Lost", "New"]
    values = [
        np.mean([row["n_shared_features"] for row in rows]),
        np.mean([row["n_lost_features"] for row in rows]),
        np.mean([row["n_new_features"] for row in rows]),
    ]
    plt.bar(labels, values)
    plt.ylabel("Mean features per corresponding patch")
    plt.tight_layout()
    plt.savefig(output_dir / "plot6_shared_lost_new.png", dpi=250)
    plt.close()

    group_a = [row for row in rows if row["clean_correct"] and row["blur_correct"]]
    group_b = [row for row in rows if row["clean_correct"] and not row["blur_correct"]]
    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [
            [row["feature_retention"] for row in group_a],
            [row["feature_retention"] for row in group_b],
        ],
        tick_labels=["Correct→Correct", "Correct→Wrong"],
    )
    plt.ylabel("Feature retention")
    plt.tight_layout()
    plt.savefig(output_dir / "plot7a_group_retention.png", dpi=250)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.boxplot(
        [
            [row["relative_sae_change"] for row in group_a],
            [row["relative_sae_change"] for row in group_b],
        ],
        tick_labels=["Correct→Correct", "Correct→Wrong"],
    )
    plt.ylabel("Relative SAE activation change")
    plt.tight_layout()
    plt.savefig(output_dir / "plot7b_group_relative_change.png", dpi=250)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: SAE identity vs strength")
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--scatter-samples", type=int, default=200000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedDataset(args.samples, args.start_index, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae, sae_metadata = load_sae(device)

    print(f"Device: {device}")
    print("ViT: frozen base ViT-B/16")
    print("SAE type: Vanilla/ReLU")
    print(f"SAE layer: penultimate hidden state (-2), patch tokens only")
    print(f"SAE latent dimension: {sae.latent_dim}")
    print("SAE k: not applicable (Vanilla/ReLU)")
    print("Per-image metrics: mean of 196 corresponding-patch metrics")

    latent_dim = sae.latent_dim
    clean_frequency = torch.zeros(latent_dim, dtype=torch.float64, device=device)
    blur_frequency = torch.zeros_like(clean_frequency)
    clean_active_sum = torch.zeros_like(clean_frequency)
    blur_active_sum = torch.zeros_like(clean_frequency)
    shared_signed_sum = torch.zeros_like(clean_frequency)
    shared_absolute_sum = torch.zeros_like(clean_frequency)
    shared_count = torch.zeros_like(clean_frequency)
    total_patches = 0
    rows = []
    sampled_clean = []
    sampled_blur = []
    sampled_count = 0

    with torch.no_grad():
        for clean, blurred, labels, image_ids, paths in tqdm(loader):
            batch_size = clean.shape[0]
            images = torch.cat((clean, blurred), dim=0).to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            clean_logits, blur_logits = outputs.logits.split(batch_size)
            clean_patches, blur_patches = outputs.hidden_states[-2][:, 1:, :].split(
                batch_size
            )
            patch_count = clean_patches.shape[1]
            clean_latent = sae.encode(clean_patches.flatten(0, 1)).reshape(
                batch_size, patch_count, latent_dim
            )
            blur_latent = sae.encode(blur_patches.flatten(0, 1)).reshape(
                batch_size, patch_count, latent_dim
            )
            clean_active = clean_latent > 0
            blur_active = blur_latent > 0
            shared = clean_active & blur_active
            lost = clean_active & ~blur_active
            new = ~clean_active & blur_active
            union = clean_active | blur_active
            difference = blur_latent - clean_latent

            clean_frequency += clean_active.sum(dim=(0, 1))
            blur_frequency += blur_active.sum(dim=(0, 1))
            clean_active_sum += clean_latent.sum(dim=(0, 1)).double()
            blur_active_sum += blur_latent.sum(dim=(0, 1)).double()
            shared_signed_sum += (difference * shared).sum(dim=(0, 1)).double()
            shared_absolute_sum += (difference.abs() * shared).sum(dim=(0, 1)).double()
            shared_count += shared.sum(dim=(0, 1))
            total_patches += batch_size * patch_count

            remaining = args.scatter_samples - sampled_count
            if remaining > 0:
                shared_clean_values = clean_latent[shared]
                shared_blur_values = blur_latent[shared]
                take = min(remaining, shared_clean_values.numel())
                indices = torch.randperm(shared_clean_values.numel(), device=device)[:take]
                sampled_clean.append(shared_clean_values[indices].cpu())
                sampled_blur.append(shared_blur_values[indices].cpu())
                sampled_count += take

            clean_predictions = clean_logits.argmax(dim=-1)
            blur_predictions = blur_logits.argmax(dim=-1)
            for index in range(batch_size):
                clean_vector = clean_latent[index]
                blur_vector = blur_latent[index]
                shared_mask = shared[index]
                shared_changes = difference[index][shared_mask]
                clean_norms = clean_vector.norm(dim=-1)
                blur_norms = blur_vector.norm(dim=-1)
                clean_counts = clean_active[index].sum(dim=-1)
                shared_counts = shared_mask.sum(dim=-1)
                row = {
                    "image_id": image_ids[index],
                    "image_path": paths[index],
                    "ground_truth": labels[index].item(),
                    "clean_prediction": clean_predictions[index].item(),
                    "blur_prediction": blur_predictions[index].item(),
                    "clean_correct": bool(clean_predictions[index] == labels[index]),
                    "blur_correct": bool(blur_predictions[index] == labels[index]),
                    "sae_cosine_similarity": F.cosine_similarity(
                        clean_vector, blur_vector, dim=-1, eps=EPSILON
                    ).mean().item(),
                    "n_clean_active": clean_counts.float().mean().item(),
                    "n_blur_active": blur_active[index].sum(dim=-1).float().mean().item(),
                    "n_shared_features": shared_counts.float().mean().item(),
                    "n_lost_features": lost[index].sum(dim=-1).float().mean().item(),
                    "n_new_features": new[index].sum(dim=-1).float().mean().item(),
                    "jaccard_similarity": (
                        shared_counts / union[index].sum(dim=-1).clamp_min(1)
                    ).float().mean().item(),
                    "feature_retention": (
                        shared_counts / clean_counts.clamp_min(1)
                    ).float().mean().item(),
                    "clean_sae_norm": clean_norms.mean().item(),
                    "blur_sae_norm": blur_norms.mean().item(),
                    "norm_ratio": (blur_norms / clean_norms.clamp_min(EPSILON)).mean().item(),
                    "relative_sae_change": (
                        difference[index].norm(dim=-1)
                        / clean_norms.clamp_min(EPSILON)
                    ).mean().item(),
                    "shared_feature_mean_signed_change": shared_changes.mean().item(),
                    "shared_feature_mean_absolute_change": shared_changes.abs().mean().item(),
                    "shared_feature_median_absolute_change": shared_changes.abs().median().item(),
                    "shared_features_percent_increased": (
                        (shared_changes > 0).float().mean().item() * 100
                    ),
                    "shared_features_percent_decreased": (
                        (shared_changes < 0).float().mean().item() * 100
                    ),
                }
                rows.append(row)

    feature_rows = []
    for feature_index in range(latent_dim):
        clean_count = clean_frequency[feature_index].item()
        blur_count = blur_frequency[feature_index].item()
        common_count = shared_count[feature_index].item()
        feature_rows.append(
            {
                "feature_index": feature_index,
                "clean_activation_frequency": clean_count / total_patches,
                "blur_activation_frequency": blur_count / total_patches,
                "frequency_change": (blur_count - clean_count) / total_patches,
                "mean_clean_activation": clean_active_sum[feature_index].item()
                / max(clean_count, 1),
                "mean_blur_activation": blur_active_sum[feature_index].item()
                / max(blur_count, 1),
                "mean_signed_strength_change": shared_signed_sum[feature_index].item()
                / max(common_count, 1),
                "mean_absolute_strength_change": shared_absolute_sum[feature_index].item()
                / max(common_count, 1),
            }
        )

    image_fields = list(rows[0].keys())
    feature_fields = list(feature_rows[0].keys())
    write_csv(output_dir / "image_level_metrics.csv", rows, image_fields)
    write_csv(output_dir / "per_feature_metrics.csv", feature_rows, feature_fields)
    write_csv(
        output_dir / "features_ranked_by_strength_change.csv",
        sorted(feature_rows, key=lambda row: abs(row["mean_absolute_strength_change"]), reverse=True),
        feature_fields,
    )
    write_csv(
        output_dir / "features_ranked_by_frequency_change.csv",
        sorted(feature_rows, key=lambda row: abs(row["frequency_change"]), reverse=True),
        feature_fields,
    )

    shared_clean = torch.cat(sampled_clean).numpy()
    shared_blur = torch.cat(sampled_blur).numpy()
    plot_results(output_dir, rows, shared_clean, shared_blur)

    group_a = [row for row in rows if row["clean_correct"] and row["blur_correct"]]
    group_b = [row for row in rows if row["clean_correct"] and not row["blur_correct"]]
    summary = {
        "configuration": {
            **vars(args),
            "model": BASE_MODEL,
            "sae_checkpoint": str(SAE_DIR.relative_to(PROJECT_ROOT)),
            "sae_type": "Vanilla/ReLU",
            "sae_expansion_factor": sae_metadata["expansion_factor"],
            "sae_latent_dimension": latent_dim,
            "sae_k": None,
            "activation_layer": "hidden_states[-2]",
            "tokens": "196 patch tokens; CLS excluded",
            "image_summary": "mean of corresponding-patch metrics",
            "active_threshold": "z > 0",
        },
        "classification": {
            "clean_accuracy": sum(row["clean_correct"] for row in rows) / len(rows),
            "blur4_accuracy": sum(row["blur_correct"] for row in rows) / len(rows),
        },
        "all_images": summarize_rows(rows),
        "group_a_clean_correct_blur_correct": summarize_rows(group_a),
        "group_b_clean_correct_blur_wrong": summarize_rows(group_b),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))

    all_metrics = summary["all_images"]
    print(f"\nNumber of paired images: {len(rows)}")
    print(f"Clean accuracy: {summary['classification']['clean_accuracy']:.2%}")
    print(f"Blur-4 accuracy: {summary['classification']['blur4_accuracy']:.2%}")
    print(f"Mean cosine similarity: {all_metrics['sae_cosine_similarity']['mean']:.6f}")
    print(f"Median cosine similarity: {all_metrics['sae_cosine_similarity']['median']:.6f}")
    print(f"Mean feature retention: {all_metrics['feature_retention']['mean']:.6f}")
    print(f"Mean shared/lost/new: {all_metrics['n_shared_features']['mean']:.2f} / "
          f"{all_metrics['n_lost_features']['mean']:.2f} / {all_metrics['n_new_features']['mean']:.2f}")
    print(f"Mean clean/blur norm: {all_metrics['clean_sae_norm']['mean']:.4f} / "
          f"{all_metrics['blur_sae_norm']['mean']:.4f}")
    print(f"Mean norm ratio: {all_metrics['norm_ratio']['mean']:.6f}")
    print(f"Mean relative change: {all_metrics['relative_sae_change']['mean']:.6f}")
    print(f"Mean shared-feature % increased/decreased: "
          f"{all_metrics['shared_features_percent_increased']['mean']:.2f}% / "
          f"{all_metrics['shared_features_percent_decreased']['mean']:.2f}%")
    for label, group in [("Group A correct→correct", group_a), ("Group B correct→wrong", group_b)]:
        metrics = summarize_rows(group)
        print(f"{label} ({len(group)}): cosine={metrics['sae_cosine_similarity']['mean']:.6f}, "
              f"retention={metrics['feature_retention']['mean']:.6f}, "
              f"relative_change={metrics['relative_sae_change']['mean']:.6f}, "
              f"norm_ratio={metrics['norm_ratio']['mean']:.6f}")
    print(f"Saved Experiment 1 to {output_dir}")


if __name__ == "__main__":
    main()
