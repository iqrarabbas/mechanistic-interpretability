import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.compare_sae_level4 import load_sae
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment22_noise_nearest_neighbor"
BLOCK_INDEX = 10
PATCHES = 196
WIDTH = 768


def make_loader(samples, start, args):
    return DataLoader(
        PairedCorruptionDataset("noise", samples, start, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def encode(sae, patches):
    batch = patches.shape[0]
    return sae.encode(patches.flatten(0, 1)).reshape(batch, PATCHES, -1)


def build_memory(model, sae, loader, device, output_dir, key_features):
    samples = len(loader.dataset)
    latent_means_path = output_dir / "noise_latent_means.f16"
    residuals_path = output_dir / "sae_patch_residuals.f16"
    hidden_keys_path = output_dir / "noise_hidden_keys.npy"
    predictions_path = output_dir / "noise_predictions.npy"
    latent_means = np.memmap(
        latent_means_path, dtype=np.float16, mode="w+", shape=(samples, sae.latent_dim)
    )
    residuals = np.memmap(
        residuals_path, dtype=np.float16, mode="w+", shape=(samples, PATCHES, WIDTH)
    )
    hidden_keys = np.empty((samples, WIDTH), dtype=np.float32)
    predictions = np.empty(samples, dtype=np.int16)
    feature_change = torch.zeros(sae.latent_dim, dtype=torch.float64)
    offset = 0
    with torch.no_grad():
        for clean, noise, _ in tqdm(loader, desc="Building memory bank"):
            batch = clean.shape[0]
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            _, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            clean_patches, noise_patches = clean_hidden[:, 1:], noise_hidden[:, 1:]
            clean_latent = encode(sae, clean_patches)
            noise_latent = encode(sae, noise_patches)
            decoded_delta = sae.decode(clean_latent.flatten(0, 1)) - sae.decode(
                noise_latent.flatten(0, 1)
            )
            end = offset + batch
            latent_means[offset:end] = noise_latent.mean(1).cpu().numpy().astype(np.float16)
            residuals[offset:end] = decoded_delta.reshape(batch, PATCHES, WIDTH).cpu().numpy().astype(np.float16)
            hidden_keys[offset:end] = noise_patches.mean(1).cpu().numpy()
            predictions[offset:end] = noise_logits.argmax(1).cpu().numpy().astype(np.int16)
            feature_change += (clean_latent - noise_latent).abs().sum((0, 1)).double().cpu()
            offset = end
    latent_means.flush()
    residuals.flush()
    selected_features = torch.topk(feature_change, min(key_features, sae.latent_dim)).indices.numpy()
    sae_keys = np.asarray(latent_means[:, selected_features], dtype=np.float32)
    np.save(output_dir / "sae_keys.npy", sae_keys)
    np.save(hidden_keys_path, hidden_keys)
    np.save(predictions_path, predictions)
    np.save(output_dir / "key_features.npy", selected_features)
    metadata = {
        "samples": samples,
        "latent_dim": sae.latent_dim,
        "key_features": int(selected_features.size),
        "residual_shape": [samples, PATCHES, WIDTH],
        "residual_dtype": "float16",
        "latent_means_size_gib": latent_means_path.stat().st_size / 2**30,
        "residual_size_gib": residuals_path.stat().st_size / 2**30,
    }
    (output_dir / "memory_metadata.json").write_text(json.dumps(metadata, indent=2))
    return load_memory(output_dir, samples)


def load_memory(output_dir, samples):
    metadata = json.loads((output_dir / "memory_metadata.json").read_text())
    if metadata["samples"] != samples:
        raise ValueError("Existing memory bank size does not match --memory-samples")
    return {
        "sae_keys": np.load(output_dir / "sae_keys.npy"),
        "hidden_keys": np.load(output_dir / "noise_hidden_keys.npy"),
        "predictions": np.load(output_dir / "noise_predictions.npy"),
        "features": np.load(output_dir / "key_features.npy"),
        "residuals": np.memmap(
            output_dir / "sae_patch_residuals.f16",
            dtype=np.float16,
            mode="r",
            shape=tuple(metadata["residual_shape"]),
        ),
        "metadata": metadata,
    }


def normalize_keys(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def choose_neighbors(indices, query_predictions, bank_predictions, count, conditioned):
    if not conditioned:
        return indices[:, :count]
    selected = np.empty((indices.shape[0], count), dtype=np.int64)
    for row, prediction in enumerate(query_predictions):
        candidates = indices[row]
        matching = candidates[bank_predictions[candidates] == prediction]
        combined = np.concatenate([matching, candidates[bank_predictions[candidates] != prediction]])
        selected[row] = combined[:count]
    return selected


def memory_residual_mean(residuals, chunk_size=128):
    total = np.zeros((PATCHES, WIDTH), dtype=np.float64)
    for start in range(0, residuals.shape[0], chunk_size):
        total += np.asarray(residuals[start:start + chunk_size], dtype=np.float32).sum(0)
    return (total / residuals.shape[0]).astype(np.float32)


def summarize(original_correct, values):
    correct = np.asarray(values["correct"], dtype=bool)
    recovered = int((~original_correct & correct).sum())
    damaged = int((original_correct & ~correct).sum())
    return {
        "noise4_accuracy": float(correct.mean()),
        "noise4_accuracy_gain": float(correct.mean() - original_correct.mean()),
        "mean_margin_change": float(np.mean(values["margin_change"])),
        "mean_true_logit_change": float(np.mean(values["true_logit_change"])),
        "predictions_recovered": recovered,
        "originally_correct_damaged": damaged,
        "mcnemar_exact_pvalue": float(
            binomtest(recovered, recovered + damaged, 0.5).pvalue
        ) if recovered + damaged else 1.0,
    }


def evaluate(model, sae, loader, device, memory, searchers, configurations, search_depth, controls=False):
    stores = {
        name: {key: [] for key in ["correct", "margin_change", "true_logit_change"]}
        for name in configurations
    }
    if controls:
        for name in ["random_neighbors", "reverse_direction", "mean_memory_residual", "paired_clean_oracle", "residual_identity"]:
            stores[name] = {key: [] for key in ["correct", "margin_change", "true_logit_change"]}
    original_correct, original_margin, original_true = [], [], []
    clean_correct, selected_clean_correct = [], []
    mean_residual = memory_residual_mean(memory["residuals"]) if controls else None
    generator = np.random.default_rng(0)
    selected_config = next(iter(configurations.values())) if controls else None
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Nearest-neighbor evaluation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            noise_true, noise_margin = classification_margin(noise_logits, labels)
            original_correct.extend((noise_logits.argmax(1) == labels).cpu().tolist())
            original_margin.extend(noise_margin.cpu().tolist())
            original_true.extend(noise_true.cpu().tolist())
            clean_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            noise_patches = noise_hidden[:, 1:]
            clean_patches = clean_hidden[:, 1:]
            noise_latent = encode(sae, noise_patches)
            selected_features = torch.as_tensor(
                memory["features"], dtype=torch.long, device=device
            )
            query_keys = {
                "sae": noise_latent.mean(1)[:, selected_features].cpu().numpy(),
                "hidden": noise_patches.mean(1).cpu().numpy(),
            }
            query_prediction = noise_logits.argmax(1).cpu().numpy()
            neighbor_cache = {
                space: searchers[space].kneighbors(
                    normalize_keys(keys), n_neighbors=search_depth, return_distance=False
                )
                for space, keys in query_keys.items()
            }
            residual_cache = {}
            for name, config in configurations.items():
                cache_key = (config["space"], config["neighbors"], config["conditioned"])
                if cache_key not in residual_cache:
                    indices = choose_neighbors(
                        neighbor_cache[config["space"]], query_prediction,
                        memory["predictions"], config["neighbors"], config["conditioned"]
                    )
                    residual_cache[cache_key] = np.asarray(
                        memory["residuals"][indices], dtype=np.float32
                    ).mean(1)
                residual = torch.from_numpy(residual_cache[cache_key]).to(device)
                candidate = torch.cat(
                    [noise_hidden[:, :1], noise_patches + config["alpha"] * residual], dim=1
                )
                logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
                true_logit, margin = classification_margin(logits, labels)
                stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                stores[name]["margin_change"].extend((margin - noise_margin).cpu().tolist())
                stores[name]["true_logit_change"].extend((true_logit - noise_true).cpu().tolist())
            if controls:
                chosen_key = (
                    selected_config["space"], selected_config["neighbors"], selected_config["conditioned"]
                )
                selected_residual = torch.from_numpy(residual_cache[chosen_key]).to(device)
                random_indices = generator.integers(
                    0, memory["metadata"]["samples"], size=(batch, selected_config["neighbors"])
                )
                random_residual = torch.from_numpy(
                    np.asarray(memory["residuals"][random_indices], dtype=np.float32).mean(1)
                ).to(device)
                control_residuals = {
                    "random_neighbors": selected_config["alpha"] * random_residual,
                    "reverse_direction": -selected_config["alpha"] * selected_residual,
                    "mean_memory_residual": selected_config["alpha"] * torch.from_numpy(mean_residual).to(device).expand(batch, -1, -1),
                    "paired_clean_oracle": clean_patches - noise_patches,
                    "residual_identity": torch.zeros_like(noise_patches),
                }
                for name, residual in control_residuals.items():
                    logits = downstream_from_layer(
                        model, torch.cat([noise_hidden[:, :1], noise_patches + residual], dim=1), BLOCK_INDEX
                    )
                    true_logit, margin = classification_margin(logits, labels)
                    stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                    stores[name]["margin_change"].extend((margin - noise_margin).cpu().tolist())
                    stores[name]["true_logit_change"].extend((true_logit - noise_true).cpu().tolist())
                clean_latent = encode(sae, clean_patches)
                clean_key = (
                    clean_latent.mean(1)[:, selected_features].cpu().numpy()
                    if selected_config["space"] == "sae"
                    else clean_patches.mean(1).cpu().numpy()
                )
                clean_indices = searchers[selected_config["space"]].kneighbors(
                    normalize_keys(clean_key), n_neighbors=search_depth, return_distance=False
                )
                clean_indices = choose_neighbors(
                    clean_indices, clean_logits.argmax(1).cpu().numpy(), memory["predictions"],
                    selected_config["neighbors"], selected_config["conditioned"]
                )
                clean_residual = torch.from_numpy(
                    np.asarray(memory["residuals"][clean_indices], dtype=np.float32).mean(1)
                ).to(device)
                clean_candidate = torch.cat(
                    [clean_hidden[:, :1], clean_patches + selected_config["alpha"] * clean_residual], dim=1
                )
                clean_candidate_logits = downstream_from_layer(model, clean_candidate, BLOCK_INDEX)
                selected_clean_correct.extend(
                    (clean_candidate_logits.argmax(1) == labels).cpu().tolist()
                )
    original_correct = np.asarray(original_correct, dtype=bool)
    results = {
        "original_vit": {
            "noise4_accuracy": float(original_correct.mean()),
            "clean_accuracy": float(np.mean(clean_correct)),
            "mean_noise4_margin": float(np.mean(original_margin)),
        }
    }
    results.update({name: summarize(original_correct, values) for name, values in stores.items()})
    if controls:
        selected_name = next(iter(configurations))
        results[selected_name]["clean_accuracy_if_applied"] = float(np.mean(selected_clean_correct))
        results[selected_name]["clean_accuracy_gain_if_applied"] = float(
            np.mean(selected_clean_correct) - np.mean(clean_correct)
        )
    return results


def main():
    parser = argparse.ArgumentParser(description="Experiment 22: training-free Noise-4 nearest-neighbor SAE repair")
    parser.add_argument("--memory-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--memory-start", type=int, default=25000)
    parser.add_argument("--validation-start", type=int, default=30000)
    parser.add_argument("--evaluation-start", type=int, default=35000)
    parser.add_argument("--key-features", type=int, default=128)
    parser.add_argument("--neighbors", type=int, nargs="+", default=[1, 4, 8, 16])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_sae("noise", "base", "vanilla", device)
    if args.resume and (output_dir / "memory_metadata.json").exists():
        memory = load_memory(output_dir, args.memory_samples)
    else:
        memory = build_memory(
            model, sae, make_loader(args.memory_samples, args.memory_start, args),
            device, output_dir, args.key_features
        )
    searchers = {
        space: NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1).fit(
            normalize_keys(memory[f"{space}_keys"])
        )
        for space in ["sae", "hidden"]
    }
    search_depth = min(args.memory_samples, max(64, max(args.neighbors)))
    candidates = {}
    for space in ["sae", "hidden"]:
        for conditioned in [False, True]:
            for neighbors in args.neighbors:
                for alpha in args.alphas:
                    name = f"{space}_{'class_' if conditioned else ''}neighbors{neighbors}_alpha{alpha:g}"
                    candidates[name] = {
                        "space": space, "conditioned": conditioned,
                        "neighbors": neighbors, "alpha": alpha,
                    }
    validation = evaluate(
        model, sae, make_loader(args.validation_samples, args.validation_start, args),
        device, memory, searchers, candidates, search_depth
    )
    selected_name = max(
        candidates,
        key=lambda name: (
            validation[name]["noise4_accuracy"], validation[name]["mean_margin_change"]
        ),
    )
    selected = candidates[selected_name]
    evaluation = evaluate(
        model, sae, make_loader(args.evaluation_samples, args.evaluation_start, args),
        device, memory, searchers, {"selected_neighbors": selected}, search_depth,
        controls=True,
    )
    summary = {
        "configuration": vars(args) | {
            "device": str(device), "vit_block": 11,
            "method": "frozen ViT and SAE; exact nearest-neighbor retrieval; no learned parameters",
        },
        "memory": memory["metadata"],
        "selected": {"validation_name": selected_name} | selected,
        "validation": validation,
        "evaluation": evaluation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"memory": summary["memory"], "selected": summary["selected"], "evaluation": evaluation}, indent=2))
    print(f"Saved Experiment 22 to {output_dir}")


if __name__ == "__main__":
    main()
