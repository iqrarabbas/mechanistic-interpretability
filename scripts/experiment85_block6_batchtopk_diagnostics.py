import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from interpretability.sae import BatchTopKSAE
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL


ROOT = Path(__file__).parent.parent
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment85_block6_batchtopk_diagnostics"
DEFAULT_SAE = ACTIVE_ROOT / "results/sae/experiment84_block6_clean_sae_calibration/full_seed0_1000train_200val_3epoch_v1/batchtopk_lambda_1em03/model.pt"
BLOCK = 6
INPUT_DIM = 768
EXPANSION_FACTOR = 32
K = 32
EPSILON = 1e-8


def atomic_json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


class PairedDataset(Dataset):
    def __init__(self, start, samples, corruption, severity, corruption_seed):
        common = {"dataset_dir": ROOT / "Dataset", "start_index": start, "max_samples": samples}
        self.clean = ImageNetDataset(**common)
        arguments = common | {"corruption": corruption, "corruption_seed": corruption_seed}
        arguments[f"{corruption}_severity"] = severity
        self.corrupted = ImageNetDataset(**arguments)
        if self.clean.image_paths != self.corrupted.image_paths:
            raise RuntimeError("Clean/corrupted sample orders differ")

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        corrupted, corrupted_label = self.corrupted[index]
        if label != corrupted_label:
            raise RuntimeError("Clean/corrupted labels differ")
        return clean, corrupted, label, self.clean.start_index + index


def load_sae(path, device):
    sae = BatchTopKSAE(
        input_dim=INPUT_DIM,
        expansion_factor=EXPANSION_FACTOR,
        k=K,
        input_unit_norm=True,
        n_batches_to_dead=5,
    )
    sae.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return sae.to(device).eval()


def encode_topk_per_patch(sae, hidden):
    patches = hidden[:, 1:].flatten(0, 1)
    normalized, _, _ = sae.preprocess_inputs(patches)
    preactivations = sae.preactivations(normalized)
    values, indices = preactivations.topk(K, dim=-1, sorted=True)
    latent = torch.zeros_like(preactivations).scatter(1, indices, values)
    return latent.reshape(hidden.shape[0], hidden.shape[1] - 1, -1), indices.reshape(
        hidden.shape[0], hidden.shape[1] - 1, K
    )


def true_class_margin(logits, labels):
    true = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    return true - masked.max(dim=1).values


def collect_clean_statistics(model, sae, loader, device):
    latent_dim = sae.latent_dim
    sums = torch.zeros(latent_dim, dtype=torch.float64)
    squares = torch.zeros(latent_dim, dtype=torch.float64)
    positive = torch.zeros(latent_dim, dtype=torch.int64)
    patches = 0
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Clean reference statistics"):
            hidden = model(pixel_values=images.to(device), output_hidden_states=True).hidden_states[BLOCK]
            latent, _ = encode_topk_per_patch(sae, hidden)
            flat = latent.flatten(0, 1).cpu().double()
            sums += flat.sum(0)
            squares += flat.square().sum(0)
            positive += (flat > 0).sum(0)
            patches += flat.shape[0]
    mean = sums / patches
    variance = (squares / patches - mean.square()).clamp_min(0)
    return {
        "mean": mean.float(),
        "std": variance.sqrt().float(),
        "prevalence": (positive.double() / patches).float(),
        "patches": patches,
    }


def topk_jaccard(first, second):
    intersection = (first.unsqueeze(-1) == second.unsqueeze(-2)).any(-1).sum(-1).float()
    return intersection / (2 * K - intersection).clamp_min(1)


def bootstrap_mean(values, seed, replicates=10000):
    values = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        indices = generator.integers(0, len(values), size=(count, len(values)))
        estimates[start : start + count] = values[indices].mean(1)
    return {
        "mean": float(values.mean()),
        "ci95": [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))],
    }


def ranked_features(values, eligible, count=100, descending=True):
    values = np.asarray(values)
    eligible_indices = np.flatnonzero(eligible)
    order = np.argsort(values[eligible_indices])
    if descending:
        order = order[::-1]
    return [
        {"feature": int(index), "value": float(values[index])}
        for index in eligible_indices[order[:count]]
    ]


def analyze_corruption(model, sae, loader, clean_stats, device, corruption, seed):
    latent_dim = sae.latent_dim
    delta_sum = np.zeros(latent_dim, dtype=np.float64)
    absolute_delta_sum = np.zeros(latent_dim, dtype=np.float64)
    abnormality_sum = np.zeros(latent_dim, dtype=np.float64)
    enter = np.zeros(latent_dim, dtype=np.int64)
    leave = np.zeros(latent_dim, dtype=np.int64)
    failure_delta_sum = np.zeros(latent_dim, dtype=np.float64)
    both_correct_delta_sum = np.zeros(latent_dim, dtype=np.float64)
    failure_images = both_correct_images = images_seen = patches_seen = 0
    paired_jaccard = []
    margin_changes = []
    clean_correct = []
    corrupted_correct = []
    per_image_deltas = []
    all_clean_indices = []
    all_corrupted_indices = []
    image_indices = []
    mean = clean_stats["mean"].to(device)
    std = clean_stats["std"].to(device).clamp_min(1e-4)
    with torch.no_grad():
        for clean, corrupted, labels, indices in tqdm(loader, desc=f"Diagnosing {corruption}"):
            batch_size = clean.shape[0]
            pixels = torch.cat((clean, corrupted)).to(device)
            labels = labels.to(device)
            outputs = model(pixel_values=pixels, output_hidden_states=True)
            clean_hidden, corrupted_hidden = outputs.hidden_states[BLOCK].split(batch_size)
            clean_latent, clean_indices = encode_topk_per_patch(sae, clean_hidden)
            corrupted_latent, corrupted_indices = encode_topk_per_patch(sae, corrupted_hidden)
            delta = corrupted_latent - clean_latent
            delta_flat = delta.flatten(0, 1)
            corrupted_flat = corrupted_latent.flatten(0, 1)
            delta_sum += delta_flat.sum(0).cpu().double().numpy()
            absolute_delta_sum += delta_flat.abs().sum(0).cpu().double().numpy()
            abnormality_sum += ((corrupted_flat - mean) / std).abs().sum(0).cpu().double().numpy()
            clean_active = clean_latent > 0
            corrupted_active = corrupted_latent > 0
            enter += ((~clean_active) & corrupted_active).flatten(0, 1).sum(0).cpu().numpy()
            leave += (clean_active & (~corrupted_active)).flatten(0, 1).sum(0).cpu().numpy()
            clean_logits, corrupted_logits = outputs.logits.split(batch_size)
            clean_ok = clean_logits.argmax(1) == labels
            corrupted_ok = corrupted_logits.argmax(1) == labels
            failure = clean_ok & (~corrupted_ok)
            both = clean_ok & corrupted_ok
            image_delta = delta.mean(1)
            if failure.any():
                failure_delta_sum += image_delta[failure].sum(0).cpu().double().numpy()
                failure_images += int(failure.sum())
            if both.any():
                both_correct_delta_sum += image_delta[both].sum(0).cpu().double().numpy()
                both_correct_images += int(both.sum())
            paired_jaccard.extend(topk_jaccard(clean_indices, corrupted_indices).mean(1).cpu().tolist())
            clean_margin = true_class_margin(clean_logits, labels)
            corrupted_margin = true_class_margin(corrupted_logits, labels)
            margin_changes.extend((corrupted_margin - clean_margin).cpu().tolist())
            clean_correct.extend(clean_ok.cpu().numpy())
            corrupted_correct.extend(corrupted_ok.cpu().numpy())
            per_image_deltas.append(image_delta.cpu().numpy().astype(np.float32))
            all_clean_indices.append(clean_indices.cpu().numpy().astype(np.int32))
            all_corrupted_indices.append(corrupted_indices.cpu().numpy().astype(np.int32))
            image_indices.extend(indices.numpy().tolist())
            images_seen += batch_size
            patches_seen += clean_indices.shape[0] * clean_indices.shape[1]
    clean_top = torch.from_numpy(np.concatenate(all_clean_indices))
    corrupt_top = torch.from_numpy(np.concatenate(all_corrupted_indices))
    unrelated = torch.roll(clean_top, shifts=1, dims=0)
    shuffled_corrupt = torch.roll(corrupt_top, shifts=1, dims=0)
    unrelated_jaccard = topk_jaccard(clean_top, unrelated).mean(1).numpy()
    shuffled_jaccard = topk_jaccard(clean_top, shuffled_corrupt).mean(1).numpy()
    image_deltas = np.concatenate(per_image_deltas)
    margin_changes_array = np.asarray(margin_changes)
    centered_delta = image_deltas - image_deltas.mean(0, keepdims=True)
    centered_margin = margin_changes_array - margin_changes_array.mean()
    covariance = (centered_delta * centered_margin[:, None]).mean(0)
    correlation = covariance / (image_deltas.std(0) * margin_changes_array.std() + EPSILON)
    prevalence = clean_stats["prevalence"].numpy()
    eligible = prevalence >= 0.001
    failure_contrast = failure_delta_sum / max(failure_images, 1) - both_correct_delta_sum / max(both_correct_images, 1)
    summary = {
        "corruption": corruption,
        "images": images_seen,
        "patches": patches_seen,
        "clean_accuracy": float(np.mean(clean_correct)),
        "corrupted_accuracy": float(np.mean(corrupted_correct)),
        "clean_correct_corrupt_wrong_images": failure_images,
        "both_correct_images": both_correct_images,
        "paired_top32_jaccard": bootstrap_mean(paired_jaccard, seed),
        "unrelated_clean_top32_jaccard": bootstrap_mean(unrelated_jaccard, seed + 1),
        "shuffled_clean_corrupt_top32_jaccard": bootstrap_mean(shuffled_jaccard, seed + 2),
        "margin_change": bootstrap_mean(margin_changes, seed + 3),
        "feature_eligibility": "clean-reference prevalence >= 0.001",
        "top_strengthened": ranked_features(delta_sum / patches_seen, eligible),
        "top_weakened": ranked_features(delta_sum / patches_seen, eligible, descending=False),
        "top_absolute_change": ranked_features(absolute_delta_sum / patches_seen, eligible),
        "top_standardized_abnormality": ranked_features(abnormality_sum / patches_seen, eligible),
        "top_entering_dominant_set": ranked_features(enter / patches_seen, eligible),
        "top_leaving_dominant_set": ranked_features(leave / patches_seen, eligible),
        "top_failure_specific_change": ranked_features(np.abs(failure_contrast), eligible),
        "top_margin_harm_correlation": ranked_features(-correlation, eligible),
    }
    arrays = {
        "image_indices": np.asarray(image_indices),
        "clean_correct": np.asarray(clean_correct),
        "corrupted_correct": np.asarray(corrupted_correct),
        "margin_change": margin_changes_array.astype(np.float32),
        "paired_top32_jaccard": np.asarray(paired_jaccard, dtype=np.float32),
        "unrelated_top32_jaccard": unrelated_jaccard.astype(np.float32),
        "shuffled_top32_jaccard": shuffled_jaccard.astype(np.float32),
        "mean_feature_delta_per_image": image_deltas,
    }
    return summary, arrays


def main():
    global BLOCK
    parser = argparse.ArgumentParser(description="Experiment 85: corrected BatchTopK Block-6 corruption diagnostics")
    parser.add_argument("--block", type=int, default=6)
    parser.add_argument("--reference-start", type=int, default=10200)
    parser.add_argument("--reference-samples", type=int, default=800)
    parser.add_argument("--diagnostic-start", type=int, default=11000)
    parser.add_argument("--diagnostic-samples", type=int, default=1000)
    parser.add_argument("--split", choices=["development", "confirmation"], default="development")
    parser.add_argument("--corruptions", nargs="+", choices=["noise", "blur"], default=["noise", "blur"])
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sae-checkpoint", type=Path, default=DEFAULT_SAE)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.block <= 12:
        raise ValueError("--block must be between 1 and 12")
    BLOCK = args.block
    if args.reference_start < 10200 or args.reference_start + args.reference_samples > 11000:
        raise ValueError("Clean reference split must remain inside [10200,11000)")
    allowed_diagnostic_range = {
        "development": (11000, 12000),
        "confirmation": (12000, 13000),
    }[args.split]
    if (
        args.diagnostic_start < allowed_diagnostic_range[0]
        or args.diagnostic_start + args.diagnostic_samples > allowed_diagnostic_range[1]
    ):
        raise ValueError(
            f"{args.split} split must remain inside {allowed_diagnostic_range}"
        )
    output_dir = args.output_root / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_sae(args.sae_checkpoint, device)
    reference_loader = DataLoader(
        ImageNetDataset(ROOT / "Dataset", args.reference_samples, args.reference_start),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    clean_stats = collect_clean_statistics(model, sae, reference_loader, device)
    results = {}
    for offset, corruption in enumerate(args.corruptions):
        summary_path = output_dir / f"{corruption}_summary.json"
        if args.resume and summary_path.exists():
            print(f"Skipping completed corruption: {corruption}")
            results[corruption] = json.loads(summary_path.read_text())
            continue
        dataset = PairedDataset(
            args.diagnostic_start,
            args.diagnostic_samples,
            corruption,
            args.severity,
            args.corruption_seed,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        summary, arrays = analyze_corruption(
            model, sae, loader, clean_stats, device, corruption, args.corruption_seed + offset * 100
        )
        atomic_json_write(summary_path, summary)
        np.savez_compressed(output_dir / f"{corruption}_paired_outcomes.npz", **arrays)
        results[corruption] = summary
    final = {
        "configuration": vars(args) | {
            "sae_checkpoint": str(args.sae_checkpoint.resolve()),
            "model": BASE_MODEL,
            "block": BLOCK,
            "expansion_factor": EXPANSION_FACTOR,
            "per_patch_topk": K,
            "vit_frozen": True,
            "sae_frozen": True,
        },
        "results": results,
        "methodological_guardrails": [
            "SAE is diagnostic only; no SAE reconstruction enters the ViT forward pass.",
            "Dominant identities use exactly top-32 features independently per patch.",
            "Raw positive-feature Jaccard is not used because Experiment 49 demonstrated saturation.",
            "Clean reference [10200,11000) and paired diagnostic [11000,12000) are disjoint from SAE train/calibration data.",
            "Unrelated and shuffled-pair controls establish the overlap baseline.",
            "This seed-0 development study is preliminary; stable claims require three independently trained SAE seeds and held-out confirmation.",
        ],
        "status": "development diagnostic; not final evaluation",
    }
    atomic_json_write(output_dir / "summary.json", final)
    print(json.dumps({name: {key: value for key, value in result.items() if key in ("clean_accuracy", "corrupted_accuracy", "paired_top32_jaccard", "unrelated_clean_top32_jaccard", "margin_change")} for name, result in results.items()}, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
