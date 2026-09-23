import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment31_sae_adapter_causal_mediation import decoder_basis, project


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment56_leakage_free_causal_subspace"
DEFAULT_FEATURE_SOURCE = (
    PROJECT_ROOT / "results" / "sae" / "experiment17_noise_bidirectional_repair"
    / "supervisor_disjoint_feature_dev_v1" / "summary.json"
)
DEFAULT_ADAPTER_ROOT = (
    PROJECT_ROOT / "results" / "sae" / "experiment39_disjoint_adapter_training"
    / "full_disjoint_zero_init_3seed_v1"
)
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "split_manifest_supervisor_v1.json"
BLOCK_INDEX = 10
CONDITIONS = ("clean", "noise", "blur")


def feature_ranking(path, count):
    record = json.loads(path.read_text())
    features = [item["feature_index"] for item in record["top_over_features"][:count]]
    if len(features) != count:
        raise ValueError(f"Feature source has {len(features)} features; expected {count}")
    return features


def verify_protocol(args):
    manifest = json.loads(args.split_manifest.read_text())
    splits = {item["name"]: item for item in manifest[args.split_protocol]["splits"]}
    reserve = splits["unused_imagenet_reserve"]
    intervals = {
        "primary": (args.primary_start, args.primary_start + args.primary_samples),
        "random_null": (args.null_start, args.null_start + args.null_samples),
    }
    for name, (start, end) in intervals.items():
        if start < reserve["start"] or end > reserve["end"]:
            raise ValueError(f"{name} interval [{start}, {end}) is outside reserve {reserve}")
    if max(intervals["primary"][0], intervals["random_null"][0]) < min(
        intervals["primary"][1], intervals["random_null"][1]
    ):
        raise ValueError("Primary and random-null intervals overlap")
    return {"reserve": reserve, "intervals": intervals}


def bootstrap_interval(differences, seed, repetitions):
    values = np.asarray(differences, dtype=np.float64)
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    chunk = 250
    for start in range(0, repetitions, chunk):
        count = min(chunk, repetitions - start)
        indices = generator.integers(0, values.size, size=(count, values.size))
        means[start:start + count] = values[indices].mean(axis=1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_summary(reference, candidate, seed, repetitions):
    reference = np.asarray(reference, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    recovered = int(np.sum(~reference & candidate))
    damaged = int(np.sum(reference & ~candidate))
    discordant = recovered + damaged
    exact_p = float(binomtest(min(recovered, damaged), discordant, 0.5).pvalue) if discordant else 1.0
    differences = candidate.astype(np.int8) - reference.astype(np.int8)
    return {
        "reference_accuracy": float(reference.mean()),
        "candidate_accuracy": float(candidate.mean()),
        "difference": float(differences.mean()),
        "difference_pp": float(100 * differences.mean()),
        "difference_95ci": bootstrap_interval(differences, seed, repetitions),
        "recovered": recovered,
        "damaged": damaged,
        "mcnemar_exact_pvalue": exact_p,
    }


def load_adapters(root, seeds, device):
    adapters = {}
    for seed in seeds:
        checkpoint = root / f"seed_{seed}" / "classification_weight_0.05.pt"
        adapter = HiddenLinear().to(device)
        adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        adapters[seed] = adapter.eval()
    return adapters


def condition_loader(condition, samples, start, seed, batch_size, workers):
    corruption = "noise" if condition in {"clean", "noise"} else "blur"
    dataset = PairedCorruptionDataset(corruption, samples, start, seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def select_images(condition, clean, corrupt):
    return clean if condition == "clean" else corrupt


def downstream_predictions(model, hidden, patches, labels, residuals, chunk_size):
    predictions = []
    for start in range(0, len(residuals), chunk_size):
        chunk = residuals[start:start + chunk_size]
        candidates = torch.stack([
            torch.cat([hidden[:, :1], patches + residual], dim=1) for residual in chunk
        ])
        logits = downstream_from_layer(
            model, candidates.flatten(0, 1), BLOCK_INDEX
        ).reshape(len(chunk), hidden.shape[0], -1)
        predictions.extend((logits.argmax(2) == labels[None]).cpu().numpy())
    return predictions


def evaluate_primary(model, adapters, harmful_basis, args, output_dir, device):
    primary_dir = output_dir / "primary"
    primary_dir.mkdir(exist_ok=True)
    for condition_index, condition in enumerate(CONDITIONS):
        loader = condition_loader(
            condition, args.primary_samples, args.primary_start, args.corruption_seed,
            args.batch_size, args.num_workers,
        )
        stores = {
            seed: {name: [] for name in ("baseline", "full", "harmful_removed", "harmful_only")}
            for seed in args.adapter_seeds
        }
        with torch.no_grad():
            for clean, corrupt, labels in tqdm(loader, desc=f"Primary {condition}"):
                labels = labels.to(device)
                images = select_images(condition, clean, corrupt).to(device)
                outputs = model(pixel_values=images, output_hidden_states=True)
                hidden = outputs.hidden_states[BLOCK_INDEX + 1]
                patches = hidden[:, 1:]
                baseline = (outputs.logits.argmax(1) == labels).cpu().numpy()
                for seed, adapter in adapters.items():
                    residual = adapter(patches)
                    harmful = project(residual, harmful_basis)
                    variants = [residual, residual - harmful, harmful]
                    predictions = downstream_predictions(
                        model, hidden, patches, labels, variants, args.variant_chunk_size
                    )
                    stores[seed]["baseline"].extend(baseline)
                    stores[seed]["full"].extend(predictions[0])
                    stores[seed]["harmful_removed"].extend(predictions[1])
                    stores[seed]["harmful_only"].extend(predictions[2])
        for seed, values in stores.items():
            path = primary_dir / f"{condition}_seed{seed}.npz"
            np.savez_compressed(path, **{name: np.asarray(value, dtype=bool) for name, value in values.items()})
            print(f"Saved {path}")


def random_feature_sets(latent_dim, harmful_features, count, controls, seed):
    available = np.setdiff1d(np.arange(latent_dim), np.asarray(harmful_features))
    generator = np.random.default_rng(seed)
    return [generator.choice(available, count, replace=False).tolist() for _ in range(controls)]


def evaluate_null(model, sae, adapters, harmful_features, harmful_basis, args, output_dir, device):
    null_dir = output_dir / "random_null"
    null_dir.mkdir(exist_ok=True)
    for adapter_seed, adapter in adapters.items():
        feature_sets = random_feature_sets(
            sae.latent_dim, harmful_features, args.feature_count, args.random_controls,
            args.control_seed + adapter_seed,
        )
        (null_dir / f"seed{adapter_seed}_random_features.json").write_text(json.dumps(feature_sets))
        random_bases = [decoder_basis(sae, features) for features in feature_sets]
        for condition in CONDITIONS:
            output_path = null_dir / f"{condition}_seed{adapter_seed}.npz"
            if args.resume and output_path.exists():
                print(f"Skipping completed {output_path}")
                continue
            loader = condition_loader(
                condition, args.null_samples, args.null_start, args.corruption_seed,
                args.batch_size, args.num_workers,
            )
            full_store, harmful_store = [], []
            random_store = [[] for _ in range(args.random_controls)]
            with torch.no_grad():
                for clean, corrupt, labels in tqdm(loader, desc=f"Null {condition} seed {adapter_seed}"):
                    labels = labels.to(device)
                    images = select_images(condition, clean, corrupt).to(device)
                    outputs = model(pixel_values=images, output_hidden_states=True)
                    hidden = outputs.hidden_states[BLOCK_INDEX + 1]
                    patches = hidden[:, 1:]
                    residual = adapter(patches)
                    harmful = project(residual, harmful_basis)
                    harmful_energy = harmful.square().sum((1, 2)).clamp_min(1e-12)
                    fixed_predictions = downstream_predictions(
                        model, hidden, patches, labels, [residual, residual - harmful],
                        args.variant_chunk_size,
                    )
                    full_store.extend(fixed_predictions[0])
                    harmful_store.extend(fixed_predictions[1])
                    for start in range(0, args.random_controls, args.control_basis_chunk):
                        bases = random_bases[start:start + args.control_basis_chunk]
                        variants = []
                        for basis in bases:
                            component = project(residual, basis)
                            random_energy = component.square().sum((1, 2)).clamp_min(1e-12)
                            scale = (harmful_energy / random_energy).sqrt()[:, None, None]
                            variants.append(residual - scale * component)
                        predictions = downstream_predictions(
                            model, hidden, patches, labels, variants, args.variant_chunk_size
                        )
                        for offset, prediction in enumerate(predictions):
                            random_store[start + offset].extend(prediction)
            np.savez_compressed(
                output_path,
                full=np.asarray(full_store, dtype=bool),
                harmful_removed=np.asarray(harmful_store, dtype=bool),
                random_energy_removed=np.asarray(random_store, dtype=bool),
            )
            print(f"Saved {output_path}")


def summarize(args, output_dir, protocol, harmful_features):
    seed_results = {}
    for seed in args.adapter_seeds:
        seed_record = {}
        for condition_index, condition in enumerate(CONDITIONS):
            primary = np.load(output_dir / "primary" / f"{condition}_seed{seed}.npz")
            null = np.load(output_dir / "random_null" / f"{condition}_seed{seed}.npz")
            full = null["full"]
            harmful = null["harmful_removed"]
            random = null["random_energy_removed"]
            harmful_cost = float(full.mean() - harmful.mean())
            random_costs = full.mean() - random.mean(axis=1)
            seed_record[condition] = {
                "primary_full_vs_baseline": paired_summary(
                    primary["baseline"], primary["full"], 5600 + seed * 10 + condition_index,
                    args.bootstrap_repetitions,
                ),
                "primary_harmful_removed_vs_full": paired_summary(
                    primary["full"], primary["harmful_removed"],
                    5700 + seed * 10 + condition_index, args.bootstrap_repetitions,
                ),
                "primary_harmful_removed_vs_baseline": paired_summary(
                    primary["baseline"], primary["harmful_removed"],
                    5800 + seed * 10 + condition_index, args.bootstrap_repetitions,
                ),
                "primary_harmful_only_vs_baseline": paired_summary(
                    primary["baseline"], primary["harmful_only"],
                    5900 + seed * 10 + condition_index, args.bootstrap_repetitions,
                ),
                "random_null": {
                    "images": int(full.size),
                    "controls": int(random.shape[0]),
                    "harmful_removal_cost": harmful_cost,
                    "random_removal_cost_mean": float(random_costs.mean()),
                    "random_removal_cost_std": float(random_costs.std(ddof=1)),
                    "harmful_percentile": float(np.mean(random_costs < harmful_cost)),
                    "empirical_one_sided_pvalue": float(
                        (1 + np.sum(random_costs >= harmful_cost)) / (random_costs.size + 1)
                    ),
                    "minimum_attainable_pvalue": float(1 / (random_costs.size + 1)),
                },
            }
        seed_results[str(seed)] = seed_record
    summary = {
        "configuration": vars(args) | {
            "feature_source": str(args.feature_source.resolve()),
            "adapter_root": str(args.adapter_root.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "harmful_features": harmful_features,
            "vit_block": 11,
            "frozen_components": ["ViT", "SAE", "adapters"],
            "imageNetV2_accessed": False,
        },
        "protocol_verification": protocol,
        "seed_results": seed_results,
        "claim_rule": (
            "The historical causal claim is confirmed only if harmful-feature removal has a "
            "consistent corruption-specific cost across all three adapters and exceeds the "
            "1,000-control energy-matched null."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(seed_results, indent=2))
    print(f"Saved Experiment 56 to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 56: leakage-free replication of causal SAE subspace claims"
    )
    parser.add_argument("--feature-source", type=Path, default=DEFAULT_FEATURE_SOURCE)
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--adapter-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--feature-count", type=int, default=16)
    parser.add_argument("--primary-start", type=int, default=36000)
    parser.add_argument("--primary-samples", type=int, default=10000)
    parser.add_argument("--null-start", type=int, default=46000)
    parser.add_argument("--null-samples", type=int, default=1000)
    parser.add_argument("--random-controls", type=int, default=1000)
    parser.add_argument("--corruption-seed", type=int, default=2026)
    parser.add_argument("--control-seed", type=int, default=56000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--variant-chunk-size", type=int, default=8)
    parser.add_argument("--control-basis-chunk", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--phase", choices=["all", "primary", "null", "summary"], default="all")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    protocol = verify_protocol(args)
    harmful_features = feature_ranking(args.feature_source, args.feature_count)
    if args.phase == "summary":
        summarize(args, output_dir, protocol, harmful_features)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device).eval()
    sae.requires_grad_(False)
    harmful_basis = decoder_basis(sae, harmful_features)
    adapters = load_adapters(args.adapter_root, args.adapter_seeds, device)

    if args.phase in {"all", "primary"}:
        evaluate_primary(model, adapters, harmful_basis, args, output_dir, device)
    if args.phase in {"all", "null"}:
        evaluate_null(
            model, sae, adapters, harmful_features, harmful_basis, args, output_dir, device
        )
    if args.phase == "all":
        summarize(args, output_dir, protocol, harmful_features)


if __name__ == "__main__":
    main()
