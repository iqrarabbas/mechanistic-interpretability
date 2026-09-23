import argparse
import json
from pathlib import Path
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
from scripts.compare_sae_level4 import load_sae
from scripts.experiment1_base_blur4_sae_identity_strength import EPSILON
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment4_non_oracle_sae_correction import feature_table, fit_affine, write_csv


DATASET_DIR = PROJECT_ROOT / "Dataset"
BASE_MODEL = "google/vit-base-patch16-224"
FINE_TUNED_MODEL = PROJECT_ROOT / "checkpoints" / "vit_noise4_best"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment7_noise_finetuning_features"


class NoisePairedDataset(Dataset):
    def __init__(self, samples, start_index, seed):
        common = dict(dataset_dir=DATASET_DIR, max_samples=samples, start_index=start_index)
        self.clean = ImageNetDataset(**common)
        self.noisy = ImageNetDataset(
            **common,
            corruption="noise",
            noise_severity=4,
            corruption_seed=seed,
        )
        if self.clean.image_paths != self.noisy.image_paths or self.clean.labels != self.noisy.labels:
            raise RuntimeError("Clean and Noise-4 datasets are not aligned")

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        noisy, noisy_label = self.noisy[index]
        if label != noisy_label:
            raise RuntimeError(f"Label mismatch at {index}")
        return clean, noisy, label, self.clean.image_paths[index].name


def collect_statistics(model, sae, loader, device):
    samples = len(loader.dataset)
    latent_dim = sae.latent_dim
    image_delta = np.empty((samples, latent_dim), dtype=np.float32)
    delta_margin = np.empty(samples, dtype=np.float32)
    clean_correct = np.empty(samples, dtype=np.bool_)
    noisy_correct = np.empty(samples, dtype=np.bool_)
    sums = {
        name: torch.zeros(latent_dim, dtype=torch.float64)
        for name in ["x", "y", "xx", "xy", "yy"]
    }
    offset = 0
    with torch.no_grad():
        for clean, noisy, labels, _ in tqdm(loader, desc="Feature statistics"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noisy]).to(device),
                output_hidden_states=True,
            )
            clean_logits, noisy_logits = outputs.logits.split(batch)
            clean_hidden, noisy_hidden = outputs.hidden_states[-2].split(batch)
            clean_patches = clean_hidden[:, 1:]
            noisy_patches = noisy_hidden[:, 1:]
            clean_z = sae.encode(clean_patches.flatten(0, 1)).reshape(batch, 196, -1)
            noisy_z = sae.encode(noisy_patches.flatten(0, 1)).reshape(batch, 196, -1)
            image_delta[offset : offset + batch] = (
                noisy_z.mean(1) - clean_z.mean(1)
            ).cpu().numpy()
            clean_margin = classification_margin(clean_logits, labels)[1]
            noisy_margin = classification_margin(noisy_logits, labels)[1]
            delta_margin[offset : offset + batch] = (noisy_margin - clean_margin).cpu().numpy()
            clean_correct[offset : offset + batch] = (clean_logits.argmax(1) == labels).cpu().numpy()
            noisy_correct[offset : offset + batch] = (noisy_logits.argmax(1) == labels).cpu().numpy()
            sums["x"] += noisy_z.double().sum((0, 1)).cpu()
            sums["y"] += clean_z.double().sum((0, 1)).cpu()
            sums["xx"] += noisy_z.double().square().sum((0, 1)).cpu()
            sums["xy"] += (noisy_z.double() * clean_z.double()).sum((0, 1)).cpu()
            sums["yy"] += clean_z.double().square().sum((0, 1)).cpu()
            offset += batch
    moments = fit_affine(sums, samples * 196, ridge=0.05)
    rows, candidates = feature_table(
        image_delta,
        delta_margin,
        clean_correct,
        noisy_correct,
        moments,
        minimum_consistency=0.60,
    )
    classification = {
        "clean_accuracy": float(clean_correct.mean()),
        "noise4_accuracy": float(noisy_correct.mean()),
        "clean_correct_noise_wrong": int((clean_correct & ~noisy_correct).sum()),
        "mean_delta_margin": float(delta_margin.mean()),
        "fdr_directional_candidates": len(candidates),
    }
    return rows, candidates, classification


def align_features(base_sae, tuned_sae, base_candidates, tuned_rows, top_k):
    selected = base_candidates[:top_k]
    base_indices = torch.tensor([row["feature_index"] for row in selected], dtype=torch.long)
    base_directions = F.normalize(base_sae.decoder.weight[:, base_indices].T.float(), dim=1)
    tuned_directions = F.normalize(tuned_sae.decoder.weight.T.float(), dim=1)
    similarities = base_directions @ tuned_directions.T
    used = set()
    alignments = []
    for position, base_row in enumerate(selected):
        order = torch.argsort(similarities[position], descending=True).tolist()
        tuned_index = next(index for index in order if index not in used)
        used.add(tuned_index)
        tuned_row = tuned_rows[tuned_index]
        alignments.append({
            "base_feature": base_row["feature_index"],
            "matched_tuned_feature": tuned_index,
            "decoder_cosine": float(similarities[position, tuned_index]),
            "base_mean_noise_minus_clean": base_row["mean_blur_minus_clean"],
            "tuned_mean_noise_minus_clean": tuned_row["mean_blur_minus_clean"],
            "base_direction_consistency": base_row["direction_consistency"],
            "tuned_direction_consistency": tuned_row["direction_consistency"],
            "base_margin_correlation": base_row["pearson_delta_margin"],
            "tuned_margin_correlation": tuned_row["pearson_delta_margin"],
            "base_margin_qvalue": base_row["pearson_qvalue"],
            "tuned_margin_qvalue": tuned_row["pearson_qvalue"],
            "same_change_direction": bool(
                np.sign(base_row["mean_blur_minus_clean"])
                == np.sign(tuned_row["mean_blur_minus_clean"])
            ),
            "same_harmful_association_direction": bool(
                np.sign(base_row["pearson_delta_margin"])
                == np.sign(tuned_row["pearson_delta_margin"])
            ),
            "tuned_remains_fdr_significant": bool(tuned_row["pearson_qvalue"] < 0.05),
        })
    return alignments


def summarize_alignments(rows):
    cosines = np.array([row["decoder_cosine"] for row in rows])
    base_changes = np.abs([row["base_mean_noise_minus_clean"] for row in rows])
    tuned_changes = np.abs([row["tuned_mean_noise_minus_clean"] for row in rows])
    base_correlations = np.abs([row["base_margin_correlation"] for row in rows])
    tuned_correlations = np.abs([row["tuned_margin_correlation"] for row in rows])
    return {
        "matched_features": len(rows),
        "mean_decoder_cosine": float(cosines.mean()),
        "matches_cosine_at_least_0_5": int((cosines >= 0.5).sum()),
        "same_change_direction_fraction": float(np.mean([row["same_change_direction"] for row in rows])),
        "same_harmful_association_direction_fraction": float(np.mean([
            row["same_harmful_association_direction"] for row in rows
        ])),
        "tuned_fdr_significant_fraction": float(np.mean([
            row["tuned_remains_fdr_significant"] for row in rows
        ])),
        "mean_absolute_change_base": float(base_changes.mean()),
        "mean_absolute_change_tuned": float(tuned_changes.mean()),
        "mean_change_reduction_fraction": float(1 - tuned_changes.mean() / max(base_changes.mean(), EPSILON)),
        "mean_absolute_margin_correlation_base": float(base_correlations.mean()),
        "mean_absolute_margin_correlation_tuned": float(tuned_correlations.mean()),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 7: Noise-4 feature mechanisms after fine-tuning")
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--start-index", type=int, default=40000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--top-features", type=int, default=32)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start_index < 40000:
        raise ValueError("Use fresh indices >=40000; lower indices were used previously")
    if args.start_index + args.samples > 50000:
        raise ValueError("Requested range exceeds ImageNet validation data")

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = NoisePairedDataset(args.samples, args.start_index, args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    results = {}
    saes = {}
    for model_name, model_path in [("base", BASE_MODEL), ("fine_tuned", FINE_TUNED_MODEL)]:
        print(f"Analyzing {model_name} model")
        model = ViTForImageClassification.from_pretrained(model_path).to(device).eval()
        model.requires_grad_(False)
        sae = load_sae("noise", model_name, "vanilla", device)
        rows, candidates, classification = collect_statistics(model, sae, loader, device)
        write_csv(output_dir / f"{model_name}_feature_statistics.csv", rows)
        write_csv(output_dir / f"{model_name}_failure_candidates.csv", candidates)
        results[model_name] = {
            "classification": classification,
            "top_candidates": candidates[: args.top_features],
        }
        saes[model_name] = sae.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    tuned_rows = list(csv_rows(output_dir / "fine_tuned_feature_statistics.csv"))
    tuned_rows = [{key: parse_value(value) for key, value in row.items()} for row in tuned_rows]
    alignments = align_features(
        saes["base"],
        saes["fine_tuned"],
        results["base"]["top_candidates"],
        tuned_rows,
        args.top_features,
    )
    write_csv(output_dir / "matched_feature_mechanisms.csv", alignments)
    summary = {
        "configuration": vars(args) | {
            "corruption": "Gaussian noise",
            "severity": 4,
            "base_model": BASE_MODEL,
            "fine_tuned_model": str(FINE_TUNED_MODEL.relative_to(PROJECT_ROOT)),
            "base_sae": "checkpoints/sae/noise4_base_vanilla_paper",
            "fine_tuned_sae": "checkpoints/sae/noise4_fine_tuned_vanilla_paper",
            "layer": "hidden_states[-2], 196 patch tokens",
            "alignment": "greedy one-to-one maximum positive decoder cosine",
        },
        "base": results["base"],
        "fine_tuned": results["fine_tuned"],
        "matched_feature_summary": summarize_alignments(alignments),
        "matched_features": alignments,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (output_dir / "config.json").write_text(json.dumps(summary["configuration"], indent=2))
    print(json.dumps({
        "base": results["base"]["classification"],
        "fine_tuned": results["fine_tuned"]["classification"],
        "matched_feature_summary": summary["matched_feature_summary"],
    }, indent=2))
    print(f"Saved Experiment 7 to {output_dir}")


def csv_rows(path):
    import csv

    with path.open() as source:
        yield from csv.DictReader(source)


def parse_value(value):
    if value == "":
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


if __name__ == "__main__":
    main()
