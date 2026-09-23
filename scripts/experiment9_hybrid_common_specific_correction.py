import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.imagenet_dataset import ImageNetDataset
from scripts.compare_sae_level4 import load_sae
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment3_sae_causal_intervention import downstream_logits


DATASET_DIR = PROJECT_ROOT / "Dataset"
BASE_MODEL = "google/vit-base-patch16-224"
BLUR_RESULTS = PROJECT_ROOT / "results" / "sae" / "experiment5_correction_strategies" / "full_strategies_5000"
NOISE_RESULTS = PROJECT_ROOT / "results" / "sae" / "experiment7_noise_finetuning_features" / "full_noise_ft_5000"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment9_hybrid_correction"
COMMON_BLUR_FEATURES = [6966, 17814, 14024]
COMMON_NOISE_FEATURES = [6966, 17814, 14024]


class TripleDataset(Dataset):
    def __init__(self, samples, start_index, seed):
        common = dict(dataset_dir=DATASET_DIR, max_samples=samples, start_index=start_index)
        self.clean = ImageNetDataset(**common)
        self.blur = ImageNetDataset(**common, corruption="blur", blur_severity=4)
        self.noise = ImageNetDataset(
            **common,
            corruption="noise",
            noise_severity=4,
            corruption_seed=seed,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        blur, blur_label = self.blur[index]
        noise, noise_label = self.noise[index]
        if label != blur_label or label != noise_label:
            raise RuntimeError(f"Label mismatch at {index}")
        return clean, blur, noise, label


def read_statistics(path):
    with path.open() as source:
        rows = list(csv.DictReader(source))
    scale = np.array([float(row["affine_scale"]) for row in rows], dtype=np.float32)
    intercept = np.array([float(row["affine_intercept"]) for row in rows], dtype=np.float32)
    return scale, intercept


def correction(z, components, scale, intercept, reverse_common=False):
    output = z.clone()
    for name, features, alpha in components:
        if not features or alpha == 0:
            continue
        selected = torch.as_tensor(features, device=z.device, dtype=torch.long)
        values = z[..., selected]
        target = values * scale[selected] + intercept[selected]
        direction = -1 if reverse_common and name == "common" else 1
        output[..., selected] += direction * alpha * (target - values)
    return output


def evaluate_branch(model, sae, loader, device, corruption, configurations, scale, intercept):
    totals = {
        name: {"clean_correct": 0, "corrupt_correct": 0, "clean_margin": 0.0, "corrupt_margin": 0.0}
        for name in configurations
    }
    totals["original_vit"] = {
        "clean_correct": 0,
        "corrupt_correct": 0,
        "clean_margin": 0.0,
        "corrupt_margin": 0.0,
    }
    total = 0
    with torch.no_grad():
        for clean, blur, noise, labels in tqdm(loader, desc=f"{corruption.title()} branch"):
            corrupted = blur if corruption == "blur" else noise
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, corrupted]).to(device),
                output_hidden_states=True,
            )
            clean_original, corrupt_original = outputs.logits.split(batch)
            clean_hidden, corrupt_hidden = outputs.hidden_states[-2].split(batch)
            clean_z = sae.encode(clean_hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)
            corrupt_z = sae.encode(corrupt_hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)
            for kind, logits in [("clean", clean_original), ("corrupt", corrupt_original)]:
                margin = classification_margin(logits, labels)[1]
                totals["original_vit"][f"{kind}_correct"] += int((logits.argmax(1) == labels).sum())
                totals["original_vit"][f"{kind}_margin"] += float(margin.sum())
            for name, config in configurations.items():
                clean_modified = correction(clean_z, scale=scale, intercept=intercept, **config)
                corrupt_modified = correction(corrupt_z, scale=scale, intercept=intercept, **config)
                clean_logits = downstream_logits(
                    model, torch.cat([clean_hidden[:, :1], sae.decode(clean_modified)], 1)
                )
                corrupt_logits = downstream_logits(
                    model, torch.cat([corrupt_hidden[:, :1], sae.decode(corrupt_modified)], 1)
                )
                for kind, logits in [("clean", clean_logits), ("corrupt", corrupt_logits)]:
                    margin = classification_margin(logits, labels)[1]
                    totals[name][f"{kind}_correct"] += int((logits.argmax(1) == labels).sum())
                    totals[name][f"{kind}_margin"] += float(margin.sum())
            total += batch
    return {
        name: {
            "clean_accuracy": values["clean_correct"] / total,
            f"{corruption}4_accuracy": values["corrupt_correct"] / total,
            "mean_clean_margin": values["clean_margin"] / total,
            f"mean_{corruption}_margin": values["corrupt_margin"] / total,
            "samples": total,
        }
        for name, values in totals.items()
    }


def branch_configurations(common, specific, alpha_common, alpha_specific):
    return {
        "baseline": {"components": []},
        "common_only": {"components": [("common", common, alpha_common)]},
        "specific_only": {"components": [("specific", specific, alpha_specific)]},
        "hybrid": {
            "components": [
                ("common", common, alpha_common),
                ("specific", specific, alpha_specific),
            ]
        },
        "hybrid_reverse_common": {
            "components": [
                ("common", common, alpha_common),
                ("specific", specific, alpha_specific),
            ],
            "reverse_common": True,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 9: exploratory common plus corruption-specific correction")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--start-index", type=int, default=45000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha-common", type=float, default=0.25)
    parser.add_argument("--alpha-blur", type=float, default=0.75)
    parser.add_argument("--alpha-noise", type=float, default=0.75)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start_index + args.samples > 50000:
        raise ValueError("Requested range exceeds ImageNet validation data")

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blur_summary = json.loads((BLUR_RESULTS / "summary.json").read_text())
    blur_top32 = blur_summary["selected_configuration"]["features"]
    blur_specific = [feature for feature in blur_top32 if feature not in COMMON_BLUR_FEATURES]
    with (NOISE_RESULTS / "base_failure_candidates.csv").open() as source:
        noise_candidates = list(csv.DictReader(source))
    noise_top32 = [int(row["feature_index"]) for row in noise_candidates[:32]]
    noise_specific = [feature for feature in noise_top32 if feature not in COMMON_NOISE_FEATURES]
    blur_scale, blur_intercept = read_statistics(BLUR_RESULTS / "discovery_feature_statistics.csv")
    noise_scale, noise_intercept = read_statistics(NOISE_RESULTS / "base_feature_statistics.csv")
    dataset = TripleDataset(args.samples, args.start_index, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)

    branches = {}
    for corruption, common, specific, alpha_specific, scale, intercept in [
        ("blur", COMMON_BLUR_FEATURES, blur_specific, args.alpha_blur, blur_scale, blur_intercept),
        ("noise", COMMON_NOISE_FEATURES, noise_specific, args.alpha_noise, noise_scale, noise_intercept),
    ]:
        sae = load_sae(corruption, "base", "vanilla", device)
        configurations = branch_configurations(
            common,
            specific,
            args.alpha_common,
            alpha_specific,
        )
        branches[corruption] = evaluate_branch(
            model,
            sae,
            loader,
            device,
            corruption,
            configurations,
            torch.as_tensor(scale, device=device),
            torch.as_tensor(intercept, device=device),
        )
        del sae
        if device.type == "cuda":
            torch.cuda.empty_cache()

    for corruption, results in branches.items():
        baseline = results["baseline"]
        accuracy_key = f"{corruption}4_accuracy"
        for name, values in results.items():
            values["corruption_accuracy_gain_vs_baseline"] = (
                values[accuracy_key] - baseline[accuracy_key]
            )
            values["clean_accuracy_change_vs_baseline"] = (
                values["clean_accuracy"] - baseline["clean_accuracy"]
            )
    summary = {
        "configuration": vars(args) | {
            "status": "exploratory; images were used by Experiment 8 and are not a new locked test",
            "model": BASE_MODEL,
            "common_semantic_features": [6966, 17814, 14024],
            "blur_sae": "checkpoints/sae/blur4_base_vanilla_paper",
            "noise_sae": "checkpoints/sae/noise4_base_vanilla_paper",
        },
        "feature_sets": {
            "blur_common": COMMON_BLUR_FEATURES,
            "blur_specific": blur_specific,
            "noise_common": COMMON_NOISE_FEATURES,
            "noise_specific": noise_specific,
        },
        "branches": branches,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    print(json.dumps(branches, indent=2))
    print(f"Saved exploratory Experiment 9 to {output_dir}")


if __name__ == "__main__":
    main()
