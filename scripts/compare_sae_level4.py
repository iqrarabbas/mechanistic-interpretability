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
from interpretability.sae import (
    BatchTopKSAE,
    VanillaReLUSAE,
    corresponding_patch_cosine_similarity,
    extract_penultimate_patch_activations,
)

DATASET_DIR = PROJECT_ROOT / "Dataset"
SAE_ROOT = PROJECT_ROOT / "checkpoints" / "sae"
RESULTS_DIR = PROJECT_ROOT / "results" / "sae"
BASE_MODEL = "google/vit-base-patch16-224"


class PairedDataset(Dataset):
    def __init__(self, corruption, samples, start_index, seed):
        common = dict(dataset_dir=DATASET_DIR, max_samples=samples, start_index=start_index)
        self.clean = ImageNetDataset(**common)
        self.corrupted = ImageNetDataset(
            **common,
            corruption=corruption,
            blur_severity=4,
            noise_severity=4,
            corruption_seed=seed,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, _ = self.clean[index]
        corrupted, _ = self.corrupted[index]
        return clean, corrupted


def load_sae(corruption, model_name, sae_type, device):
    suffix = sae_type
    if sae_type == "batchtopk":
        suffix = "batchtopk_reference"
    elif sae_type == "vanilla":
        suffix = "vanilla_paper"
    directory = SAE_ROOT / f"{corruption}4_{model_name}_{suffix}"
    metadata_path = directory / "training.json"
    weights_path = directory / "model.pt"
    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Missing trained SAE at {directory}. Run train_sae_level4.py first."
        )
    metadata = json.loads(metadata_path.read_text())
    if sae_type == "vanilla":
        if metadata.get("implementation") != "paper_vanilla_literal_l2_v3":
            raise ValueError(
                f"{directory} does not use the literal Equation 2 loss. Retrain "
                "it with train_sae_level4.py before comparing paper results."
            )
        sae = VanillaReLUSAE(expansion_factor=metadata["expansion_factor"])
    else:
        sae = BatchTopKSAE(
            expansion_factor=metadata["expansion_factor"],
            k=metadata["k"],
            input_unit_norm=metadata.get("input_unit_norm", True),
            n_batches_to_dead=metadata.get("n_batches_to_dead", 5),
        )
    sae.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    return sae.to(device).eval()


def collect_similarities(vit, sae, loader, device):
    similarities = []
    with torch.no_grad():
        for clean, corrupted in tqdm(loader, leave=False):
            clean = clean.to(device)
            corrupted = corrupted.to(device)
            clean_patches = extract_penultimate_patch_activations(vit, clean)
            corrupted_patches = extract_penultimate_patch_activations(vit, corrupted)
            shape = clean_patches.shape
            clean_latent = sae.encode(clean_patches.reshape(-1, shape[-1]))
            corrupted_latent = sae.encode(corrupted_patches.reshape(-1, shape[-1]))
            cosine = corresponding_patch_cosine_similarity(
                clean_latent, corrupted_latent
            )
            similarities.append(cosine.cpu())
    return torch.cat(similarities)


def summarize(values):
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "count": values.numel(),
        "mean": values.mean().item(),
        "standard_deviation": values.std().item(),
        "q05": quantiles[0].item(),
        "q25": quantiles[1].item(),
        "median": quantiles[2].item(),
        "q75": quantiles[3].item(),
        "q95": quantiles[4].item(),
    }


def run(corruption, sae_type, args, device):
    dataset = PairedDataset(
        corruption, args.samples, args.start_index, args.corruption_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    distributions = {}
    model_names = (
        ("base", "fine_tuned")
        if args.model_scope == "both"
        else (args.model_scope,)
    )
    for model_name in model_names:
        path = (
            BASE_MODEL
            if model_name == "base"
            else PROJECT_ROOT / "checkpoints" / f"vit_{corruption}4_best"
        )
        print(f"Evaluating {corruption}-4 {model_name} {sae_type} SAE...")
        vit = ViTForImageClassification.from_pretrained(path).to(device).eval()
        sae = load_sae(corruption, model_name, sae_type, device)
        distributions[model_name] = collect_similarities(vit, sae, loader, device)
        del vit, sae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    output = {
        "corruption": corruption,
        "severity": 4,
        "sae_type": sae_type,
        "samples": args.samples,
        "patches_per_image": 196,
        "models": {
            model_name: summarize(values)
            for model_name, values in distributions.items()
        },
    }
    result_name = "vanilla_paper" if sae_type == "vanilla" else sae_type
    (RESULTS_DIR / f"{corruption}4_{result_name}_metrics.json").write_text(
        json.dumps(output, indent=2)
    )

    plt.figure(figsize=(8, 5))
    if "base" in distributions:
        plt.hist(
            distributions["base"].numpy(),
            bins=60,
            density=True,
            alpha=0.55,
            label="Base ViT",
        )
    if "fine_tuned" in distributions:
        plt.hist(
            distributions["fine_tuned"].numpy(),
            bins=60,
            density=True,
            alpha=0.55,
            label=f"{corruption.title()}-4 fine-tuned ViT",
        )
    plt.xlabel("Cosine similarity of corresponding patch SAE activations")
    plt.ylabel("Density")
    plt.title(f"{sae_type.title()} SAE representations on {corruption.title()}-4")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.savefig(
        RESULTS_DIR / f"{corruption}4_{result_name}_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    plt.figure(figsize=(7.2, 4.8))
    common = {
        "bins": 60,
        "range": (0.0, 1.0),
        "alpha": 0.55,
        "edgecolor": "white",
        "linewidth": 0.25,
    }
    if "fine_tuned" in distributions:
        plt.hist(
            distributions["fine_tuned"].numpy(),
            label=f"{corruption.title()}-4-tuned model",
            color="#6baed6",
            **common,
        )
    if "base" in distributions:
        plt.hist(
            distributions["base"].numpy(),
            label="Base model",
            color="#fdae6b",
            **common,
        )
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.title(
        f"Cosine Similarities of Vanilla SAE Activations for "
        f"{corruption.title()}-4-Tuned and Base Model"
    )
    plt.legend()
    plt.xlim(0.0, 1.0)
    plt.tight_layout()
    plt.savefig(
        RESULTS_DIR / f"{corruption}4_{result_name}_paper_style.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare level-4 SAE representations")
    parser.add_argument("--corruption", choices=["blur", "noise", "both"], default="both")
    parser.add_argument("--sae-type", choices=["vanilla", "batchtopk", "both"], default="both")
    parser.add_argument(
        "--model-scope",
        choices=["base", "fine_tuned", "both"],
        default="both",
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=11000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    args = parser.parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corruptions = ["blur", "noise"] if args.corruption == "both" else [args.corruption]
    sae_types = ["vanilla", "batchtopk"] if args.sae_type == "both" else [args.sae_type]
    for corruption in corruptions:
        for sae_type in sae_types:
            run(corruption, sae_type, args, device)
    print(f"Saved SAE comparison results to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
