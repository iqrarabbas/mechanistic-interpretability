import argparse
import csv
import json
from pathlib import Path
import random
import sys

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
DEFAULT_SAE_DIRS = {
    "blur": PROJECT_ROOT / "checkpoints" / "sae" / "blur4_base_vanilla_paper",
    "noise": PROJECT_ROOT / "checkpoints" / "sae" / "noise4_base_vanilla_paper",
}
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment49_sae_jaccard_controls"
BASE_MODEL = "google/vit-base-patch16-224"
EPSILON = 1e-8


def cyclic_derangement(size, seed):
    if size < 2:
        raise ValueError("At least two images are required for unrelated controls.")
    generator = np.random.default_rng(seed)
    order = generator.permutation(size)
    mapping = np.empty(size, dtype=np.int64)
    mapping[order] = np.roll(order, -1)
    if np.any(mapping == np.arange(size)):
        raise RuntimeError("Failed to construct a derangement.")
    return mapping.tolist()


class JaccardControlDataset(Dataset):
    def __init__(
        self, samples, start_index, corruption, severity, corruption_seed, control_seed
    ):
        common = {
            "dataset_dir": DATASET_DIR,
            "max_samples": samples,
            "start_index": start_index,
        }
        self.clean = ImageNetDataset(**common)
        corruption_arguments = {
            "corruption": corruption,
            "corruption_seed": corruption_seed,
        }
        corruption_arguments[f"{corruption}_severity"] = severity
        self.corrupted = ImageNetDataset(**common, **corruption_arguments)
        if self.clean.image_paths != self.corrupted.image_paths:
            raise RuntimeError("Clean and corrupted sample order differs.")
        self.unrelated_clean = cyclic_derangement(len(self.clean), control_seed)
        self.shuffled_corrupt = cyclic_derangement(len(self.clean), control_seed + 1)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, _ = self.clean[index]
        paired_corrupt, _ = self.corrupted[index]
        unrelated_index = self.unrelated_clean[index]
        shuffled_index = self.shuffled_corrupt[index]
        unrelated_clean, _ = self.clean[unrelated_index]
        unrelated_corrupt, _ = self.corrupted[unrelated_index]
        shuffled_corrupt, _ = self.corrupted[shuffled_index]
        return (
            clean,
            paired_corrupt,
            unrelated_clean,
            unrelated_corrupt,
            shuffled_corrupt,
            index,
            unrelated_index,
            shuffled_index,
        )


def load_sae(device, sae_dir):
    metadata = json.loads((sae_dir / "training.json").read_text())
    if metadata.get("implementation") != "paper_vanilla_literal_l2_v3":
        raise ValueError("Experiment 49 requires the paper-style Vanilla SAE.")
    sae = VanillaReLUSAE(expansion_factor=metadata["expansion_factor"])
    sae.load_state_dict(
        torch.load(sae_dir / "model.pt", map_location="cpu", weights_only=True)
    )
    return sae.to(device).eval(), metadata


def topk_jaccard(reference, comparison, k):
    reference_indices = reference.topk(k, dim=-1).indices
    comparison_indices = comparison.topk(k, dim=-1).indices
    intersection = (
        reference_indices.unsqueeze(-1) == comparison_indices.unsqueeze(-2)
    ).any(dim=-1).sum(dim=-1).float()
    return intersection / (2 * k - intersection).clamp_min(1)


def comparison_metrics(reference, comparison, topk_values):
    reference_active = reference > 0
    comparison_active = comparison > 0
    intersection = (reference_active & comparison_active).sum(dim=-1).float()
    union = (reference_active | comparison_active).sum(dim=-1).clamp_min(1).float()
    minimum = torch.minimum(reference, comparison).sum(dim=-1)
    maximum = torch.maximum(reference, comparison).sum(dim=-1).clamp_min(EPSILON)
    metrics = {
        "positive_jaccard": intersection / union,
        "weighted_jaccard": minimum / maximum,
        "cosine_similarity": F.cosine_similarity(
            reference, comparison, dim=-1, eps=EPSILON
        ),
    }
    for k in topk_values:
        metrics[f"top{k}_jaccard"] = topk_jaccard(reference, comparison, k)
    return {name: values.mean(dim=-1) for name, values in metrics.items()}


def bootstrap_difference(reference, control, seed, replicates):
    reference = np.asarray(reference, dtype=np.float64)
    control = np.asarray(control, dtype=np.float64)
    differences = reference - control
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 1000):
        count = min(1000, replicates - start)
        indices = generator.integers(0, len(differences), size=(count, len(differences)))
        bootstrap_means[start : start + count] = differences[indices].mean(axis=1)
    return {
        "mean_paired_difference": float(differences.mean()),
        "ci95": [
            float(np.quantile(bootstrap_means, 0.025)),
            float(np.quantile(bootstrap_means, 0.975)),
        ],
        "fraction_paired_greater": float((differences > 0).mean()),
    }


def write_csv(path, rows):
    with path.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 49: compare clean/Blur SAE overlap with unrelated controls"
    )
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--start-index", type=int, default=11000)
    parser.add_argument("--corruption", choices=["blur", "noise"], default="blur")
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--sae-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--control-seed", type=int, default=4900)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--topk", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    random.seed(args.control_seed)
    np.random.seed(args.control_seed)
    torch.manual_seed(args.control_seed)
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run: {output_dir}")
    output_dir.mkdir(parents=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sae_dir = args.sae_checkpoint or DEFAULT_SAE_DIRS[args.corruption]
    if not sae_dir.is_absolute():
        sae_dir = PROJECT_ROOT / sae_dir
    dataset = JaccardControlDataset(
        args.samples,
        args.start_index,
        args.corruption,
        args.severity,
        args.corruption_seed,
        args.control_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae, sae_metadata = load_sae(device, sae_dir)
    if max(args.topk) > sae.latent_dim:
        raise ValueError("A requested top-k exceeds the SAE latent dimension.")

    print(f"Device: {device}")
    print(f"Images: {len(dataset)} from index {args.start_index}")
    print(f"Corruption: {args.corruption}-{args.severity}")
    print(f"SAE: {sae_dir}")
    print(f"Top-k controls: {args.topk}")

    condition_names = [
        "paired_clean_corrupt",
        "unrelated_clean_clean",
        "unrelated_clean_corrupt",
        "shuffled_clean_corrupt",
    ]
    metric_names = [
        "positive_jaccard",
        "weighted_jaccard",
        "cosine_similarity",
        *[f"top{k}_jaccard" for k in args.topk],
    ]
    values = {
        condition: {metric: [] for metric in metric_names}
        for condition in condition_names
    }
    rows = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Jaccard controls"):
            image_batches = batch[:5]
            batch_size = image_batches[0].shape[0]
            images = torch.cat(image_batches, dim=0).to(device)
            hidden = model(pixel_values=images, output_hidden_states=True).hidden_states[-2]
            patches = hidden[:, 1:, :]
            latent = sae.encode(patches.flatten(0, 1)).reshape(
                images.shape[0], patches.shape[1], sae.latent_dim
            )
            clean, paired, unrelated_clean, unrelated_corrupt, shuffled_corrupt = latent.split(
                batch_size
            )
            comparisons = {
                "paired_clean_corrupt": paired,
                "unrelated_clean_clean": unrelated_clean,
                "unrelated_clean_corrupt": unrelated_corrupt,
                "shuffled_clean_corrupt": shuffled_corrupt,
            }
            batch_metrics = {
                condition: comparison_metrics(clean, comparison, args.topk)
                for condition, comparison in comparisons.items()
            }
            for index in range(batch_size):
                row = {
                    "dataset_index": int(batch[5][index]),
                    "unrelated_index": int(batch[6][index]),
                    "shuffled_corrupt_index": int(batch[7][index]),
                }
                for condition in condition_names:
                    for metric in metric_names:
                        value = batch_metrics[condition][metric][index].item()
                        values[condition][metric].append(value)
                        row[f"{condition}__{metric}"] = value
                rows.append(row)

    write_csv(output_dir / "image_level_metrics.csv", rows)
    aggregate = {
        condition: {
            metric: {
                "mean": float(np.mean(metric_values)),
                "median": float(np.median(metric_values)),
            }
            for metric, metric_values in condition_values.items()
        }
        for condition, condition_values in values.items()
    }
    paired_tests = {}
    for control in condition_names[1:]:
        paired_tests[control] = {
            metric: bootstrap_difference(
                values["paired_clean_corrupt"][metric],
                values[control][metric],
                args.control_seed + metric_index,
                args.bootstrap_replicates,
            )
            for metric_index, metric in enumerate(metric_names)
        }

    summary = {
        "status": "complete",
        "configuration": {
            **vars(args),
            "device": str(device),
            "model": BASE_MODEL,
            "model_frozen": True,
            "sae_checkpoint": str(sae_dir.relative_to(PROJECT_ROOT)),
            "sae_expansion_factor": sae_metadata["expansion_factor"],
            "sae_latent_dimension": sae.latent_dim,
            "activation_layer": "hidden_states[-2] (Block 11 output)",
            "tokens": "196 patch tokens; corresponding spatial positions",
            "positive_active_definition": "z > 0",
            "unrelated_pairing": "seeded cyclic derangements without self-pairs",
        },
        "aggregate": aggregate,
        "paired_clean_corrupt_minus_controls": paired_tests,
        "interpretation_rule": {
            "supported": "Paired clean/Blur overlap is materially above unrelated controls.",
            "not_supported": "Paired and unrelated overlap are similar; metric reflects broad SAE overlap.",
            "saturated_positive_jaccard": "Use top-k and weighted metrics if z>0 Jaccard saturates.",
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved results to {output_dir}")


if __name__ == "__main__":
    main()
