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
from interpretability.sae import BatchTopKSAE
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment85_block6_batchtopk_diagnostics import encode_topk_per_patch


ROOT = Path(__file__).parent.parent
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment88_oracle_sae_residual_repair"
STABILITY = ACTIVE_ROOT / "results/sae/experiment86_batchtopk_seed_stability/full_top100_1000controls_v1/summary.json"
BLOCK = 6


def atomic_json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def sae_checkpoint(seed):
    return ACTIVE_ROOT / f"results/sae/experiment84_block6_clean_sae_calibration/full_seed{seed}_1000train_200val_3epoch_v1/batchtopk_lambda_1em03/model.pt"


def load_sae(seed, device):
    sae = BatchTopKSAE(expansion_factor=32, k=32, input_unit_norm=True, n_batches_to_dead=5)
    sae.load_state_dict(torch.load(sae_checkpoint(seed), map_location="cpu", weights_only=True))
    return sae.to(device).eval()


def selected_features(seed, corruption):
    stability = json.loads(STABILITY.read_text())["results"][corruption]
    output = {"strengthened": [], "weakened": []}
    for category, label in (("top_strengthened", "strengthened"), ("top_weakened", "weakened")):
        first = stability[category]["pairwise"]["seed0_to_seed1"]["matches"]
        second = stability[category]["pairwise"]["seed0_to_seed2"]["matches"]
        for left, right in zip(first, second):
            if left["target_category_rank"] is None or right["target_category_rank"] is None:
                continue
            triplet = (left["seed0_feature"], left["target_feature"], right["target_feature"])
            output[label].append(int(triplet[seed]))
    return output


def clean_statistics(model, sae, device, start, samples, batch_size, workers):
    loader = DataLoader(ImageNetDataset(ROOT / "Dataset", samples, start), batch_size=batch_size, shuffle=False, num_workers=workers)
    total = torch.zeros(sae.latent_dim, dtype=torch.float64)
    square = torch.zeros_like(total)
    count = 0
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Clean repair statistics"):
            hidden = model(pixel_values=images.to(device), output_hidden_states=True).hidden_states[BLOCK]
            latent, _ = encode_topk_per_patch(sae, hidden)
            values = latent.flatten(0, 1).cpu().double()
            total += values.sum(0)
            square += values.square().sum(0)
            count += values.shape[0]
    mean = total / count
    std = (square / count - mean.square()).clamp_min(0).sqrt().clamp_min(1e-4)
    return mean.float().to(device), std.float().to(device)


def correction_delta(sae, patches, mean, std, features, alpha, tau):
    shape = patches.shape
    flat = patches.flatten(0, 1)
    normalized, _, input_std = sae.preprocess_inputs(flat)
    preactivations = sae.preactivations(normalized)
    values, indices = preactivations.topk(32, dim=-1, sorted=False)
    latent = torch.zeros_like(preactivations).scatter(1, indices, values)
    corrected = latent.clone()
    selected = torch.as_tensor(features["strengthened"] + features["weakened"], device=patches.device)
    selected_values = latent[:, selected]
    selected_mean = mean[selected]
    selected_std = std[selected]
    lower = (selected_mean - tau * selected_std).clamp_min(0)
    upper = selected_mean + tau * selected_std
    target = torch.maximum(torch.minimum(selected_values, upper), lower)
    active = selected_values > 0
    coordinate_delta = alpha * (target - selected_values) * active
    hidden_delta = F.linear(coordinate_delta, sae.decoder.weight[:, selected], bias=None)
    hidden_delta = hidden_delta * input_std
    return hidden_delta.reshape_as(patches), int(active.sum()), int((coordinate_delta != 0).sum())


def evaluate(model, sae, device, mean, std, features, corruption, start, samples, batch_size, workers, configs, seed):
    kwargs = {"corruption": corruption, "corruption_seed": seed}
    if corruption:
        kwargs[f"{corruption}_severity"] = 4
    loader = DataLoader(ImageNetDataset(ROOT / "Dataset", samples, start, **kwargs), batch_size=batch_size, shuffle=False, num_workers=workers)
    totals = {name: {"correct": 0, "recovered": 0, "damaged": 0, "changed": 0, "active": 0} for name in configs}
    baseline_correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=corruption or "clean"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            baseline_prediction = outputs.logits.argmax(1)
            baseline_correct += int((baseline_prediction == labels).sum())
            for name, config in configs.items():
                delta, active, changed = correction_delta(sae, hidden[:, 1:], mean, std, features, config["alpha"], config["tau"])
                corrected_hidden = torch.cat((hidden[:, :1], hidden[:, 1:] + delta), dim=1)
                prediction = downstream_from_layer(model, corrected_hidden, BLOCK - 1).argmax(1)
                totals[name]["correct"] += int((prediction == labels).sum())
                totals[name]["recovered"] += int(((baseline_prediction != labels) & (prediction == labels)).sum())
                totals[name]["damaged"] += int(((baseline_prediction == labels) & (prediction != labels)).sum())
                totals[name]["changed"] += changed
                totals[name]["active"] += active
            total += labels.shape[0]
    baseline_accuracy = baseline_correct / total
    return {
        "baseline_accuracy": baseline_accuracy,
        "methods": {
            name: {
                "accuracy": values["correct"] / total,
                "accuracy_change_pp": 100 * (values["correct"] / total - baseline_accuracy),
                "recovered": values["recovered"],
                "damaged": values["damaged"],
                "changed_fraction_of_selected_active_coordinates": values["changed"] / max(values["active"], 1),
                **configs[name],
            }
            for name, values in totals.items()
        },
        "images": total,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 88: oracle-family SAE residual repair capacity")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", type=int, default=47000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--reference-start", type=int, default=10200)
    parser.add_argument("--reference-samples", type=int, default=800)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--taus", type=float, nargs="+", default=[1.0, 2.0, 3.0])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start < 47000 or args.start + args.samples > 48000:
        raise ValueError("Repair development is locked inside [47000,48000)")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_sae(args.seed, device)
    mean, std = clean_statistics(model, sae, device, args.reference_start, args.reference_samples, args.batch_size, args.workers)
    configs = {f"alpha_{alpha:g}_tau_{tau:g}": {"alpha": alpha, "tau": tau} for alpha in args.alphas for tau in args.taus}
    results = {}
    for corruption in ("noise", "blur"):
        features = selected_features(args.seed, corruption)
        corruption_results = evaluate(model, sae, device, mean, std, features, corruption, args.start, args.samples, args.batch_size, args.workers, configs, args.corruption_seed)
        clean_results = evaluate(model, sae, device, mean, std, features, None, args.start, args.samples, args.batch_size, args.workers, configs, args.corruption_seed)
        best_name = max(corruption_results["methods"], key=lambda name: (corruption_results["methods"][name]["accuracy"], clean_results["methods"][name]["accuracy"]))
        results[corruption] = {
            "selected_features": features,
            "corrupted": corruption_results,
            "clean": clean_results,
            "development_winner": best_name,
        }
        atomic_json_write(output_dir / f"{corruption}.json", results[corruption])
    summary = {
        "configuration": vars(args) | {"model": BASE_MODEL, "block": BLOCK, "sae_checkpoint": str(sae_checkpoint(args.seed)), "feature_source": str(STABILITY)},
        "results": results,
        "guardrails": [
            "The ViT and SAE are frozen; no adapter is loaded.",
            "The full SAE reconstruction never replaces the original hidden state.",
            "Only residual decoder deltas from frozen Experiment-86 directions are added.",
            "The clean counterpart is not used during inference; corruption family is supplied as an oracle capacity test.",
            "Features were selected on [11000,12000), confirmed on [12000,13000), and correction hyperparameters are developed only on [47000,48000).",
            "[48000,49000) remains reserved for frozen confirmation and [49000,50000) remains untouched.",
        ],
        "status": "oracle-family development capacity test",
    }
    atomic_json_write(output_dir / "summary.json", summary)
    print(json.dumps({c: {"winner": x["development_winner"], "corrupt": x["corrupted"]["methods"][x["development_winner"]], "clean": x["clean"]["methods"][x["development_winner"]]} for c, x in results.items()}, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
