import argparse
import json
from pathlib import Path
import sys

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
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints" / "sae" / "blur4_base_vanilla_paper"
RESULTS_DIR = PROJECT_ROOT / "results" / "sae"
BASE_MODEL = "google/vit-base-patch16-224"


class PairedBlurDataset(Dataset):
    def __init__(self, samples, start_index, seed):
        common = {
            "dataset_dir": DATASET_DIR,
            "max_samples": samples,
            "start_index": start_index,
        }
        self.clean = ImageNetDataset(**common)
        self.blurred = ImageNetDataset(
            **common,
            corruption="blur",
            blur_severity=4,
            corruption_seed=seed,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        blurred, _ = self.blurred[index]
        return clean, blurred, label


def load_sae(device):
    metadata = json.loads((CHECKPOINT_DIR / "training.json").read_text())
    if metadata.get("implementation") != "paper_vanilla_literal_l2_v3":
        raise ValueError(f"{CHECKPOINT_DIR} is not a literal-paper Vanilla SAE.")
    sae = VanillaReLUSAE(expansion_factor=metadata["expansion_factor"])
    sae.load_state_dict(
        torch.load(CHECKPOINT_DIR / "model.pt", map_location="cpu", weights_only=True)
    )
    return sae.to(device).eval(), metadata


def empty_group():
    return {
        "images": 0,
        "patch_cosine_sum": 0.0,
        "relative_l2_change_sum": 0.0,
        "active_jaccard_sum": 0.0,
        "clean_active_features_sum": 0.0,
        "blurred_active_features_sum": 0.0,
        "activation_norm_ratio_sum": 0.0,
    }


def add_group_metrics(group, clean_latent, blurred_latent):
    patch_cosine = F.cosine_similarity(clean_latent, blurred_latent, dim=-1).mean()
    relative_l2_change = (
        (blurred_latent - clean_latent).norm(dim=-1)
        / clean_latent.norm(dim=-1).clamp_min(1e-8)
    ).mean()
    clean_active = clean_latent > 0
    blurred_active = blurred_latent > 0
    intersection = (clean_active & blurred_active).sum(dim=-1).float()
    union = (clean_active | blurred_active).sum(dim=-1).clamp_min(1).float()
    norm_ratio = (
        blurred_latent.norm(dim=-1)
        / clean_latent.norm(dim=-1).clamp_min(1e-8)
    ).mean()

    group["images"] += 1
    group["patch_cosine_sum"] += patch_cosine.item()
    group["relative_l2_change_sum"] += relative_l2_change.item()
    group["active_jaccard_sum"] += (intersection / union).mean().item()
    group["clean_active_features_sum"] += clean_active.sum(dim=-1).float().mean().item()
    group["blurred_active_features_sum"] += blurred_active.sum(dim=-1).float().mean().item()
    group["activation_norm_ratio_sum"] += norm_ratio.item()


def summarize_group(group):
    images = group["images"]
    if images == 0:
        return {"images": 0}
    return {
        "images": images,
        "mean_patch_cosine": group["patch_cosine_sum"] / images,
        "mean_relative_l2_change": group["relative_l2_change_sum"] / images,
        "mean_active_feature_jaccard": group["active_jaccard_sum"] / images,
        "mean_clean_active_features": group["clean_active_features_sum"] / images,
        "mean_blurred_active_features": group["blurred_active_features_sum"] / images,
        "mean_blurred_to_clean_activation_norm_ratio": (
            group["activation_norm_ratio_sum"] / images
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test how base-ViT SAE features change under Blur-4"
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--top-features", type=int, default=20)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = PairedBlurDataset(args.samples, args.start_index, args.corruption_seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae, metadata = load_sae(device)

    groups = {
        "both_correct": empty_group(),
        "clean_correct_blur_wrong": empty_group(),
        "both_wrong": empty_group(),
        "clean_wrong_blur_correct": empty_group(),
    }
    feature_absolute_change = torch.zeros(sae.latent_dim, device=device)
    patch_count = 0

    with torch.no_grad():
        for clean, blurred, labels in tqdm(loader):
            batch_size = clean.shape[0]
            images = torch.cat((clean, blurred), dim=0).to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            clean_logits, blurred_logits = outputs.logits.split(batch_size)
            patches = outputs.hidden_states[-2][:, 1:, :]
            clean_patches, blurred_patches = patches.split(batch_size)
            clean_latent = sae.encode(clean_patches.flatten(0, 1)).reshape(
                batch_size, 196, -1
            )
            blurred_latent = sae.encode(blurred_patches.flatten(0, 1)).reshape(
                batch_size, 196, -1
            )

            feature_absolute_change += (
                blurred_latent - clean_latent
            ).abs().sum(dim=(0, 1))
            patch_count += batch_size * 196

            clean_correct = clean_logits.argmax(dim=-1) == labels
            blurred_correct = blurred_logits.argmax(dim=-1) == labels
            for index in range(batch_size):
                if clean_correct[index] and blurred_correct[index]:
                    group_name = "both_correct"
                elif clean_correct[index] and not blurred_correct[index]:
                    group_name = "clean_correct_blur_wrong"
                elif not clean_correct[index] and blurred_correct[index]:
                    group_name = "clean_wrong_blur_correct"
                else:
                    group_name = "both_wrong"
                add_group_metrics(
                    groups[group_name], clean_latent[index], blurred_latent[index]
                )

    mean_feature_change = feature_absolute_change / patch_count
    top_count = min(args.top_features, sae.latent_dim)
    top_values, top_indices = torch.topk(mean_feature_change, top_count)
    summarized_groups = {
        name: summarize_group(group) for name, group in groups.items()
    }
    clean_correct = (
        summarized_groups["both_correct"]["images"]
        + summarized_groups["clean_correct_blur_wrong"]["images"]
    )
    blurred_correct = (
        summarized_groups["both_correct"]["images"]
        + summarized_groups["clean_wrong_blur_correct"]["images"]
    )
    result = {
        "model": "base",
        "corruption": "blur",
        "severity": 4,
        "samples": args.samples,
        "sae_checkpoint": str(CHECKPOINT_DIR.relative_to(PROJECT_ROOT)),
        "lambda": metadata["l1_coefficient"],
        "clean_accuracy": clean_correct / args.samples,
        "blurred_accuracy": blurred_correct / args.samples,
        "prediction_groups": summarized_groups,
        "top_mean_absolute_change_features": [
            {"feature": index.item(), "mean_absolute_change": value.item()}
            for value, index in zip(top_values.cpu(), top_indices.cpu())
        ],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "blur4_base_vanilla_paper_feature_changes.json"
    output_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved feature-change analysis to {output_path}")


if __name__ == "__main__":
    main()
