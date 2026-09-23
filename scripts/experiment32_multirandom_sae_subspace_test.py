import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment15_independent_frozen_confirmation import ExternalImageNetDataset
from scripts.experiment31_sae_adapter_causal_mediation import FEATURE_SOURCE, decoder_basis, project


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment32_multirandom_subspaces"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "results" / "sae" / "experiment25_multiseed_confirmation"
    / "imagenetv2_multiseed_confirmation"
)
BLOCK_INDEX = 10


class PairedExternalCorruptionDataset(Dataset):
    def __init__(self, root, corruption, severity, samples, start_index, noise_seed):
        self.clean = ExternalImageNetDataset(
            root, 0, samples, start_index, corruption=corruption, seed=noise_seed
        )
        self.corrupt = ExternalImageNetDataset(
            root, severity, samples, start_index, corruption=corruption, seed=noise_seed
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, clean_label = self.clean[index]
        corrupt, corrupt_label = self.corrupt[index]
        if clean_label != corrupt_label:
            raise RuntimeError("Paired ImageNetV2 labels differ")
        return clean, corrupt, clean_label


def evaluate_variants(model, hidden, patches, labels, residuals, chunk_size):
    predictions = []
    for start in range(0, len(residuals), chunk_size):
        chunk = residuals[start:start + chunk_size]
        count = len(chunk)
        candidates = torch.stack([
            torch.cat([hidden[:, :1], patches + residual], dim=1) for residual in chunk
        ])
        logits = downstream_from_layer(
            model,
            candidates.flatten(0, 1),
            BLOCK_INDEX,
        ).reshape(count, hidden.shape[0], -1)
        predictions.extend((logits.argmax(2) == labels[None]).cpu().numpy())
    return predictions


def empirical_summary(full_correct, harmful_correct, random_correct):
    full_accuracy = float(full_correct.mean())
    harmful_accuracy = float(harmful_correct.mean())
    random_accuracies = np.asarray([values.mean() for values in random_correct])
    harmful_cost = full_accuracy - harmful_accuracy
    random_costs = full_accuracy - random_accuracies
    return {
        "full_accuracy": full_accuracy,
        "harmful_removed_accuracy": harmful_accuracy,
        "harmful_removal_cost": harmful_cost,
        "random_removed_accuracy_mean": float(random_accuracies.mean()),
        "random_removed_accuracy_std": float(random_accuracies.std(ddof=1)),
        "random_removal_cost_mean": float(random_costs.mean()),
        "random_removal_cost_std": float(random_costs.std(ddof=1)),
        "random_removal_costs": random_costs.tolist(),
        "harmful_cost_percentile_among_random": float((random_costs < harmful_cost).mean()),
        "empirical_one_sided_pvalue": float((1 + np.sum(random_costs >= harmful_cost)) / (len(random_costs) + 1)),
        "harmful_exceeds_all_random_controls": bool(harmful_cost > random_costs.max()),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 32: multi-random SAE subspace null test")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--feature-source", type=Path, default=FEATURE_SOURCE)
    parser.add_argument("--adapter-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--control-seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--feature-count", type=int, default=16)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--corruption", choices=["clean", "blur", "noise"], default="noise")
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variant-chunk-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if len(args.adapter_seeds) != len(args.control_seeds):
        raise ValueError("adapter-seeds and control-seeds must have equal lengths")

    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device).eval()
    feature_record = json.loads(args.feature_source.read_text())
    harmful_features = feature_record["selected"]["over_features"][:args.feature_count]
    harmful_basis = decoder_basis(sae, harmful_features)
    available = np.setdiff1d(np.arange(sae.latent_dim), harmful_features)

    random_bases = {}
    random_features = {}
    predictors = {}
    for adapter_seed, control_seed in zip(args.adapter_seeds, args.control_seeds):
        generator = np.random.default_rng(control_seed)
        feature_sets = [
            generator.choice(available, args.feature_count, replace=False).tolist()
            for _ in range(args.random_controls)
        ]
        random_features[adapter_seed] = feature_sets
        random_bases[adapter_seed] = [decoder_basis(sae, features) for features in feature_sets]
        checkpoint = CHECKPOINT_ROOT / f"seed_{adapter_seed}" / "classification_weight_0.05.pt"
        predictor = HiddenLinear().to(device)
        predictor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        predictors[adapter_seed] = predictor.eval()

    dataset_corruption = "noise" if args.corruption == "clean" else args.corruption
    dataset_severity = 0 if args.corruption == "clean" else args.severity
    dataset = PairedExternalCorruptionDataset(
        args.image_root,
        dataset_corruption,
        dataset_severity,
        args.samples,
        args.start_index,
        args.noise_seed,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    stores = {
        seed: {"baseline": [], "full": [], "harmful": [], "rank": [[] for _ in range(args.random_controls)], "energy": [[] for _ in range(args.random_controls)]}
        for seed in args.adapter_seeds
    }
    energy = {seed: {"harmful": [], "random": [[] for _ in range(args.random_controls)]} for seed in args.adapter_seeds}

    with torch.no_grad():
        for _, noise, labels in tqdm(loader, desc="Multi-random SAE subspace test"):
            labels = labels.to(device)
            outputs = model(pixel_values=noise.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK_INDEX + 1]
            patches = hidden[:, 1:]
            for adapter_seed in args.adapter_seeds:
                stores[adapter_seed]["baseline"].extend(
                    (outputs.logits.argmax(1) == labels).cpu().numpy()
                )
                residual = predictors[adapter_seed](patches)
                harmful_component = project(residual, harmful_basis)
                harmful_energy = harmful_component.square().sum((1, 2)).clamp_min(1e-12)
                random_components = [project(residual, basis) for basis in random_bases[adapter_seed]]
                rank_residuals = [residual - component for component in random_components]
                energy_residuals = []
                for index, component in enumerate(random_components):
                    random_energy = component.square().sum((1, 2)).clamp_min(1e-12)
                    scale = (harmful_energy / random_energy).sqrt()[:, None, None]
                    energy_residuals.append(residual - scale * component)
                    energy[adapter_seed]["random"][index].extend(random_energy.cpu().tolist())
                energy[adapter_seed]["harmful"].extend(harmful_energy.cpu().tolist())
                variants = [residual, residual - harmful_component] + rank_residuals + energy_residuals
                predictions = evaluate_variants(
                    model, hidden, patches, labels, variants, args.variant_chunk_size
                )
                stores[adapter_seed]["full"].extend(predictions[0])
                stores[adapter_seed]["harmful"].extend(predictions[1])
                for index in range(args.random_controls):
                    stores[adapter_seed]["rank"][index].extend(predictions[2 + index])
                    stores[adapter_seed]["energy"][index].extend(predictions[2 + args.random_controls + index])

    seed_results = {}
    arrays = {}
    for adapter_seed in args.adapter_seeds:
        full = np.asarray(stores[adapter_seed]["full"], dtype=bool)
        baseline = np.asarray(stores[adapter_seed]["baseline"], dtype=bool)
        harmful = np.asarray(stores[adapter_seed]["harmful"], dtype=bool)
        rank = [np.asarray(values, dtype=bool) for values in stores[adapter_seed]["rank"]]
        energy_matched = [np.asarray(values, dtype=bool) for values in stores[adapter_seed]["energy"]]
        seed_results[str(adapter_seed)] = {
            "baseline_accuracy": float(baseline.mean()),
            "full_adapter_accuracy": float(full.mean()),
            "full_adapter_gain": float(full.mean() - baseline.mean()),
            "rank_matched": empirical_summary(full, harmful, rank),
            "energy_matched": empirical_summary(full, harmful, energy_matched),
            "mean_harmful_component_energy": float(np.mean(energy[adapter_seed]["harmful"])),
            "mean_random_component_energies": [float(np.mean(values)) for values in energy[adapter_seed]["random"]],
        }
        arrays[f"seed{adapter_seed}_full"] = full
        arrays[f"seed{adapter_seed}_baseline"] = baseline
        arrays[f"seed{adapter_seed}_harmful_removed"] = harmful
        arrays[f"seed{adapter_seed}_rank_random"] = np.stack(rank)
        arrays[f"seed{adapter_seed}_energy_random"] = np.stack(energy_matched)

    all_harmful_costs = []
    all_random_costs = {"rank_matched": [], "energy_matched": []}
    for result in seed_results.values():
        all_harmful_costs.append(result["rank_matched"]["harmful_removal_cost"])
        for control in all_random_costs:
            all_random_costs[control].extend(result[control]["random_removal_costs"])
    aggregate = {}
    for control, costs in all_random_costs.items():
        costs = np.asarray(costs)
        aggregate[control] = {
            "mean_harmful_removal_cost": float(np.mean(all_harmful_costs)),
            "mean_random_removal_cost": float(costs.mean()),
            "random_controls_total": int(costs.size),
            "fraction_random_costs_below_corresponding_harmful_cost": float(np.mean([
                cost < seed_results[str(seed)][control]["harmful_removal_cost"]
                for seed in args.adapter_seeds
                for cost in seed_results[str(seed)][control]["random_removal_costs"]
            ])),
        }
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "feature_source": str(args.feature_source.resolve()),
            "device": str(device),
            "harmful_features": harmful_features,
            "same_corruption_stream_across_all_tests": True,
            "harmful_feature_discovery_corruption": "noise4",
            "evaluation_corruption": "clean" if args.corruption == "clean" else f"{args.corruption}{args.severity}",
        },
        "seed_results": seed_results,
        "aggregate": aggregate,
        "interpretation_rule": "Specificity is supported when harmful removal cost exceeds the random null distribution for every adapter seed, especially after energy matching.",
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **arrays)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"seed_results": seed_results, "aggregate": aggregate}, indent=2))
    print(f"Saved Experiment 32 to {output_dir}")


if __name__ == "__main__":
    main()
