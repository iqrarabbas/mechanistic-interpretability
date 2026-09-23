import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment85_block6_batchtopk_diagnostics import encode_topk_per_patch
from scripts.experiment88_oracle_sae_residual_repair import (
    BLOCK,
    ROOT,
    atomic_json_write,
    load_sae,
    selected_features,
)
from interpretability.sae import BatchTopKSAE
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment89_sae_affine_clean_prior"


def paired_loaders(start, samples, corruption, batch_size, workers, seed):
    common = {"dataset_dir": ROOT / "Dataset", "start_index": start, "max_samples": samples}
    clean = DataLoader(ImageNetDataset(**common), batch_size=batch_size, shuffle=False, num_workers=workers)
    arguments = common | {"corruption": corruption, "corruption_seed": seed}
    arguments[f"{corruption}_severity"] = 4
    corrupt = DataLoader(ImageNetDataset(**arguments), batch_size=batch_size, shuffle=False, num_workers=workers)
    return clean, corrupt


def fit_affine(model, sae, device, features, start, samples, corruption, batch_size, workers, seed):
    clean_loader, corrupt_loader = paired_loaders(start, samples, corruption, batch_size, workers, seed)
    selected = torch.as_tensor(features, device=device)
    sx = torch.zeros(len(features), dtype=torch.float64, device=device)
    sy = torch.zeros_like(sx)
    sxx = torch.zeros_like(sx)
    sxy = torch.zeros_like(sx)
    count = 0
    with torch.no_grad():
        for (clean, clean_labels), (corrupt, corrupt_labels) in tqdm(zip(clean_loader, corrupt_loader), total=len(clean_loader), desc=f"Fit {corruption} affine"):
            if not torch.equal(clean_labels, corrupt_labels):
                raise RuntimeError("Paired labels differ")
            pixels = torch.cat((clean, corrupt)).to(device)
            hidden = model(pixel_values=pixels, output_hidden_states=True).hidden_states[BLOCK]
            clean_hidden, corrupt_hidden = hidden.split(clean.shape[0])
            clean_latent, _ = encode_topk_per_patch(sae, clean_hidden)
            corrupt_latent, _ = encode_topk_per_patch(sae, corrupt_hidden)
            x = corrupt_latent[..., selected].flatten(0, 1).double()
            y = clean_latent[..., selected].flatten(0, 1).double()
            sx += x.sum(0); sy += y.sum(0); sxx += x.square().sum(0); sxy += (x * y).sum(0)
            count += x.shape[0]
    denominator = (sxx - sx.square() / count).clamp_min(1e-8)
    slope = (sxy - sx * sy / count) / denominator
    intercept = sy / count - slope * sx / count
    return slope.float(), intercept.float(), int(count)


def residual_delta(sae, patches, features, slope, intercept, alpha):
    flat = patches.flatten(0, 1)
    normalized, _, input_std = sae.preprocess_inputs(flat)
    preactivations = sae.preactivations(normalized)
    values, indices = preactivations.topk(32, dim=-1, sorted=False)
    latent = torch.zeros_like(preactivations).scatter(1, indices, values)
    selected = torch.as_tensor(features, device=patches.device)
    current = latent[:, selected]
    predicted = (current * slope + intercept).clamp_min(0)
    coordinate_delta = alpha * (predicted - current)
    decoded = F.linear(coordinate_delta, sae.decoder.weight[:, selected], bias=None)
    return (decoded * input_std).reshape_as(patches)


def evaluate(model, sae, device, features, mapping, start, samples, corruption, batch_size, workers, seed, alphas):
    kwargs = {"corruption": corruption, "corruption_seed": seed}
    if corruption:
        kwargs[f"{corruption}_severity"] = 4
    loader = DataLoader(ImageNetDataset(ROOT / "Dataset", samples, start, **kwargs), batch_size=batch_size, shuffle=False, num_workers=workers)
    correct = {alpha: 0 for alpha in alphas}
    recovered = {alpha: 0 for alpha in alphas}
    damaged = {alpha: 0 for alpha in alphas}
    baseline_correct = total = 0
    slope, intercept = (value.to(device) for value in mapping)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Evaluate {corruption or 'clean'}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            baseline = outputs.logits.argmax(1)
            baseline_correct += int((baseline == labels).sum())
            for alpha in alphas:
                delta = residual_delta(sae, hidden[:, 1:], features, slope, intercept, alpha)
                corrected = torch.cat((hidden[:, :1], hidden[:, 1:] + delta), 1)
                prediction = downstream_from_layer(model, corrected, BLOCK - 1).argmax(1)
                correct[alpha] += int((prediction == labels).sum())
                recovered[alpha] += int(((baseline != labels) & (prediction == labels)).sum())
                damaged[alpha] += int(((baseline == labels) & (prediction != labels)).sum())
            total += labels.shape[0]
    baseline_accuracy = baseline_correct / total
    return {
        "baseline_accuracy": baseline_accuracy,
        "methods": {
            str(alpha): {
                "alpha": alpha,
                "accuracy": correct[alpha] / total,
                "accuracy_change_pp": 100 * (correct[alpha] / total - baseline_accuracy),
                "recovered": recovered[alpha],
                "damaged": damaged[alpha],
            }
            for alpha in alphas
        },
        "images": total,
    }


def main():
    global BLOCK
    parser = argparse.ArgumentParser(description="Experiment 89: label-free SAE affine clean-prior repair")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block", type=int, default=6)
    parser.add_argument("--sae-checkpoint", type=Path)
    parser.add_argument("--diagnostic-summary", type=Path)
    parser.add_argument("--top-features", type=int, default=100)
    parser.add_argument("--fit-start", type=int, default=47000)
    parser.add_argument("--fit-samples", type=int, default=500)
    parser.add_argument("--validation-start", type=int, default=47500)
    parser.add_argument("--validation-samples", type=int, default=500)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    BLOCK = args.block
    if (args.fit_start, args.fit_start + args.fit_samples) != (47000, 47500):
        raise ValueError("Affine fitting is locked to [47000,47500)")
    if (args.validation_start, args.validation_start + args.validation_samples) != (47500, 48000):
        raise ValueError("Validation is locked to [47500,48000)")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval(); model.requires_grad_(False)
    if args.sae_checkpoint:
        sae = BatchTopKSAE(expansion_factor=32, k=32, input_unit_norm=True, n_batches_to_dead=5)
        sae.load_state_dict(torch.load(args.sae_checkpoint, map_location="cpu", weights_only=True))
        sae = sae.to(device).eval()
    else:
        sae = load_sae(args.seed, device)
    generator = np.random.default_rng(8900 + args.seed)
    results = {}
    for corruption in ("noise", "blur"):
        if args.diagnostic_summary:
            diagnostic_results = json.loads(args.diagnostic_summary.read_text())["results"][corruption]
            groups = {
                "strengthened": [row["feature"] for row in diagnostic_results["top_strengthened"][: args.top_features]],
                "weakened": [row["feature"] for row in diagnostic_results["top_weakened"][: args.top_features]],
            }
        else:
            groups = selected_features(args.seed, corruption)
        features = groups["strengthened"] + groups["weakened"]
        slope, intercept, patches = fit_affine(model, sae, device, features, args.fit_start, args.fit_samples, corruption, args.batch_size, args.workers, args.corruption_seed)
        corrupted = evaluate(model, sae, device, features, (slope, intercept), args.validation_start, args.validation_samples, corruption, args.batch_size, args.workers, args.corruption_seed, args.alphas)
        clean = evaluate(model, sae, device, features, (slope, intercept), args.validation_start, args.validation_samples, None, args.batch_size, args.workers, args.corruption_seed, args.alphas)
        winner = max(args.alphas, key=lambda alpha: (corrupted["methods"][str(alpha)]["accuracy"], clean["methods"][str(alpha)]["accuracy"]))
        random_results = []
        for control in range(args.random_controls):
            random_features = generator.choice(sae.latent_dim, size=len(features), replace=False).tolist()
            random_slope, random_intercept, _ = fit_affine(model, sae, device, random_features, args.fit_start, args.fit_samples, corruption, args.batch_size, args.workers, args.corruption_seed)
            random_eval = evaluate(model, sae, device, random_features, (random_slope, random_intercept), args.validation_start, args.validation_samples, corruption, args.batch_size, args.workers, args.corruption_seed, [winner])
            random_results.append(random_eval["methods"][str(winner)]["accuracy_change_pp"])
        observed = corrupted["methods"][str(winner)]["accuracy_change_pp"]
        results[corruption] = {
            "features": groups,
            "fit_patches": patches,
            "mapping": {"slope": slope.cpu().tolist(), "intercept": intercept.cpu().tolist()},
            "corrupted_validation": corrupted,
            "clean_validation": clean,
            "winner_alpha": winner,
            "random_control_gain_pp": random_results,
            "empirical_p_vs_random": (1 + sum(value >= observed for value in random_results)) / (args.random_controls + 1),
        }
        atomic_json_write(output_dir / f"{corruption}.json", results[corruption])
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "block": BLOCK,
            "labels_used_for_mapping": False,
            "sae_checkpoint": str(args.sae_checkpoint) if args.sae_checkpoint else None,
            "diagnostic_summary": str(args.diagnostic_summary) if args.diagnostic_summary else None,
        },
        "results": results,
        "guardrails": [
            "Paired clean/corrupted representations are used only to fit scalar affine maps; labels are not used.",
            "Inference requires only the corrupted image and an oracle corruption-family choice.",
            "The full SAE reconstruction is never inserted; only selected decoder deltas are added to the original hidden state.",
            "Feature directions are frozen from Experiments 86-87 before this experiment.",
            "Fit [47000,47500) and validation [47500,48000) are disjoint; [48000,49000) remains confirmation-only.",
        ],
        "status": "development; confirmation not accessed",
    }
    atomic_json_write(output_dir / "summary.json", summary)
    print(json.dumps({c: {"winner_alpha": x["winner_alpha"], "gain_pp": x["corrupted_validation"]["methods"][str(x["winner_alpha"])]["accuracy_change_pp"], "clean_change_pp": x["clean_validation"]["methods"][str(x["winner_alpha"])]["accuracy_change_pp"], "p_vs_random": x["empirical_p_vs_random"]} for c, x in results.items()}, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
