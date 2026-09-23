import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from interpretability.sae import BatchTopKSAE
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenCache
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment45_sae_discovered_hidden_subspace import (
    PATCHES,
    WIDTH,
    SharedRawInputRepair,
    hidden_statistics,
    orthonormalize,
    residual_coefficient_scale,
    residual_pca_basis,
    train_method,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment94_batchtopk_blur_hidden_subspace"
BLOCK = 11
TOKENS = 197


class PairedImages(Dataset):
    def __init__(self, start, samples, corruption, seed):
        common = dict(dataset_dir=PROJECT_ROOT / "Dataset", max_samples=samples, start_index=start)
        self.clean = ImageNetDataset(**common)
        kwargs = {"corruption": corruption, "corruption_seed": seed}
        kwargs[f"{corruption}_severity"] = 4
        self.corrupt = ImageNetDataset(**common, **kwargs)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        corrupt, corrupt_label = self.corrupt[index]
        if label != corrupt_label:
            raise RuntimeError("Clean/corrupt label mismatch")
        return clean, corrupt, label


def make_loader(split, corruption, seed, args):
    samples = split["end"] - split["start"]
    if args.max_samples is not None:
        samples = min(samples, args.max_samples)
    return DataLoader(
        PairedImages(split["start"], samples, corruption, seed),
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.workers,
    )


def build_cache(model, loader, device, directory, corruption):
    directory.mkdir(parents=True, exist_ok=True)
    samples = len(loader.dataset)
    shape = (samples, TOKENS, WIDTH)
    clean_cache = np.memmap(directory / "clean.f16", dtype=np.float16, mode="w+", shape=shape)
    corrupt_cache = np.memmap(directory / "noise.f16", dtype=np.float16, mode="w+", shape=shape)
    labels_cache = np.empty(samples, dtype=np.int16)
    offset = 0
    with torch.no_grad():
        for clean, corrupt, labels in tqdm(loader, desc=f"Caching {corruption} {directory.name}"):
            batch = len(labels)
            outputs = model(
                pixel_values=torch.cat((clean, corrupt)).to(device), output_hidden_states=True
            )
            clean_hidden, corrupt_hidden = outputs.hidden_states[BLOCK].split(batch)
            end = offset + batch
            clean_cache[offset:end] = clean_hidden.cpu().numpy().astype(np.float16)
            corrupt_cache[offset:end] = corrupt_hidden.cpu().numpy().astype(np.float16)
            labels_cache[offset:end] = labels.numpy().astype(np.int16)
            offset = end
    clean_cache.flush()
    corrupt_cache.flush()
    np.save(directory / "labels.npy", labels_cache)
    (directory / "metadata.json").write_text(json.dumps({
        "samples": samples,
        "shape": list(shape),
        "dtype": "float16",
        "block": BLOCK,
        "corruption": corruption,
        "severity": 4,
    }, indent=2))


def seed_artifacts(seed, block):
    if block == 6:
        checkpoint = (
            ACTIVE_ROOT / "results/sae/experiment84_block6_clean_sae_calibration"
            / f"full_seed{seed}_1000train_200val_3epoch_v1"
            / "batchtopk_lambda_1em03/model.pt"
        )
        diagnostic_name = f"full_seed{seed}_noise4_blur4_corrected_v1"
    elif block in {3, 9, 11}:
        checkpoint = (
            ACTIVE_ROOT / "results/sae/experiment91_batchtopk_layer_screen"
            / f"block{block}_seed{seed}_1000train_200val_3epoch_v1"
            / "batchtopk_lambda_1em03/model.pt"
        )
        diagnostic_name = f"layer_screen_block{block}_seed{seed}_noise4_blur4_v1"
    else:
        raise ValueError("Only audited Block-3, Block-6, Block-9, and Block-11 SAE artifacts are supported")
    diagnostic = (
        ACTIVE_ROOT / "results/sae/experiment85_block6_batchtopk_diagnostics"
        / diagnostic_name / "summary.json"
    )
    return checkpoint, diagnostic


def load_sae_and_features(seed, device, rank, block, corruption):
    checkpoint, diagnostic_path = seed_artifacts(seed, block)
    sae = BatchTopKSAE(expansion_factor=32, k=32, input_unit_norm=True, n_batches_to_dead=5)
    sae.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    sae = sae.to(device).eval()
    sae.requires_grad_(False)
    diagnostic = json.loads(diagnostic_path.read_text())["results"][corruption]
    half = rank // 2
    features = (
        [row["feature"] for row in diagnostic["top_strengthened"][:half]]
        + [row["feature"] for row in diagnostic["top_weakened"][: rank - half]]
    )
    if len(set(features)) != rank:
        raise RuntimeError(f"Selected {corruption} feature list is not unique")
    return sae, features, checkpoint, diagnostic_path


def evaluate_cache(model, methods, dataset, device, args, seed):
    loader = DataLoader(dataset, batch_size=args.predictor_batch_size, shuffle=False)
    stores = {"baseline": [], "clean": []}
    stores.update({name: [] for name in methods})
    with torch.no_grad():
        for clean_hidden, corrupt_hidden, labels in loader:
            clean_hidden = clean_hidden.to(device)
            corrupt_hidden = corrupt_hidden.to(device)
            labels = labels.to(device)
            baseline_logits = downstream_from_layer(model, corrupt_hidden, BLOCK - 1)
            clean_logits = downstream_from_layer(model, clean_hidden, BLOCK - 1)
            stores["baseline"].extend((baseline_logits.argmax(1) == labels).cpu().tolist())
            stores["clean"].extend((clean_logits.argmax(1) == labels).cpu().tolist())
            for name, method in methods.items():
                patches = corrupt_hidden[:, 1:]
                candidate = torch.cat(
                    (corrupt_hidden[:, :1], patches + args.alpha * method(patches)), dim=1
                )
                logits = downstream_from_layer(model, candidate, BLOCK - 1)
                stores[name].extend((logits.argmax(1) == labels).cpu().tolist())
    arrays = {name: np.asarray(values, dtype=bool) for name, values in stores.items()}
    result = {
        "baseline_accuracy": float(arrays["baseline"].mean()),
        "paired_clean_accuracy": float(arrays["clean"].mean()),
        "methods": {
            name: paired_comparison(
                arrays["baseline"], arrays[name], seed + 9400 + index,
                args.bootstrap_repetitions,
            )
            for index, name in enumerate(methods)
        },
    }
    return result, arrays


def evaluate_clean_cache(model, methods, dataset, device, args, seed):
    loader = DataLoader(dataset, batch_size=args.predictor_batch_size, shuffle=False)
    stores = {"baseline": []}
    stores.update({name: [] for name in methods})
    with torch.no_grad():
        for clean_hidden, _, labels in loader:
            clean_hidden = clean_hidden.to(device)
            labels = labels.to(device)
            baseline_logits = downstream_from_layer(model, clean_hidden, BLOCK - 1)
            stores["baseline"].extend((baseline_logits.argmax(1) == labels).cpu().tolist())
            for name, method in methods.items():
                patches = clean_hidden[:, 1:]
                candidate = torch.cat(
                    (clean_hidden[:, :1], patches + args.alpha * method(patches)), dim=1
                )
                logits = downstream_from_layer(model, candidate, BLOCK - 1)
                stores[name].extend((logits.argmax(1) == labels).cpu().tolist())
    arrays = {name: np.asarray(values, dtype=bool) for name, values in stores.items()}
    return {
        "baseline_accuracy": float(arrays["baseline"].mean()),
        "methods": {
            name: paired_comparison(
                arrays["baseline"], arrays[name], seed + 9450 + index,
                args.bootstrap_repetitions,
            )
            for index, name in enumerate(methods)
        },
    }, arrays


def main():
    global BLOCK
    parser = argparse.ArgumentParser(
        description="BatchTopK corruption-discovered hidden-subspace repair"
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--block", type=int, choices=[3, 6, 9, 11], default=11)
    parser.add_argument("--training-corruption", choices=["blur", "noise"], default="blur")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument(
        "--cache-source-run",
        type=Path,
        help="Reuse audited train/validation caches from another run instead of duplicating them.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--predictor-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--sae-only", action="store_true")
    args = parser.parse_args()
    args.batch_size = args.predictor_batch_size
    BLOCK = args.block
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    training_corruption = args.training_corruption
    transfer_corruption = "noise" if training_corruption == "blur" else "blur"
    if args.rank % 2:
        raise ValueError("--rank must be even for equal strengthened/weakened selection")
    output_dir = args.output_root / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    training, validation, controls, outcomes = {}, {}, {}, {}
    trainable_parameters = WIDTH * args.rank + args.rank + PATCHES * args.rank
    sae_method = f"{training_corruption}_batchtopk_decoder"
    pca_method = f"{training_corruption}_residual_pca"
    method_names = (
        (sae_method,)
        if args.sae_only
        else (sae_method, "random_batchtopk_decoder", "high_variance_hidden", pca_method)
    )
    for seed in args.seeds:
        torch.manual_seed(seed + 9400)
        np.random.seed(seed + 9400)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        cache_seed_dir = (
            args.cache_source_run / f"seed_{seed}"
            if args.cache_source_run is not None
            else seed_dir
        )
        for split_name in ("train", "validation"):
            cache = cache_seed_dir / f"{training_corruption}_{split_name}_cache"
            if args.cache_source_run is not None and not (cache / "metadata.json").exists():
                raise FileNotFoundError(f"Missing reusable cache: {cache}")
            if args.cache_source_run is None and not (
                args.resume and (cache / "metadata.json").exists()
            ):
                build_cache(
                    model, make_loader(splits[seed][split_name], training_corruption, seed, args),
                    device, cache, training_corruption,
                )
        transfer_cache = cache_seed_dir / f"{transfer_corruption}_validation_cache"
        if args.cache_source_run is not None and not (transfer_cache / "metadata.json").exists():
            raise FileNotFoundError(f"Missing reusable cache: {transfer_cache}")
        if args.cache_source_run is None and not (
            args.resume and (transfer_cache / "metadata.json").exists()
        ):
            build_cache(
                model, make_loader(splits[seed]["validation"], transfer_corruption, seed, args),
                device, transfer_cache, transfer_corruption,
            )
        train_data = HiddenCache(cache_seed_dir / f"{training_corruption}_train_cache")
        target_validation = HiddenCache(
            cache_seed_dir / f"{training_corruption}_validation_cache"
        )
        transfer_validation = HiddenCache(transfer_cache)
        if args.max_samples is not None:
            train_data = Subset(train_data, range(min(args.max_samples, len(train_data))))
            target_validation = Subset(target_validation, range(min(args.max_samples, len(target_validation))))
            transfer_validation = Subset(transfer_validation, range(min(args.max_samples, len(transfer_validation))))
        sae, features, checkpoint, diagnostic = load_sae_and_features(
            seed, device, args.rank, BLOCK, training_corruption
        )
        input_mean, input_scale = hidden_statistics(train_data, device, args.predictor_batch_size)
        bases = {
            sae_method: orthonormalize(sae.decoder.weight[:, features]),
        }
        random_features = []
        high_variance_dimensions = torch.empty(0, dtype=torch.long, device=device)
        pca_eigenvalues = torch.empty(0, device=device)
        if not args.sae_only:
            generator = np.random.default_rng(seed + 9400)
            available = np.setdiff1d(np.arange(sae.latent_dim), np.asarray(features))
            random_features = generator.choice(available, args.rank, replace=False).tolist()
            high_variance_dimensions = torch.argsort(input_scale, descending=True)[: args.rank]
            pca_basis, pca_eigenvalues = residual_pca_basis(
                train_data, device, args.predictor_batch_size, args.rank
            )
            bases.update({
                "random_batchtopk_decoder": orthonormalize(sae.decoder.weight[:, random_features]),
                "high_variance_hidden": torch.eye(WIDTH, device=device)[:, high_variance_dimensions],
                pca_method: pca_basis,
            })
        scales = {
            name: residual_coefficient_scale(
                train_data, basis, device, args.predictor_batch_size
            )
            for name, basis in bases.items()
        }
        methods = {
            name: SharedRawInputRepair(basis, input_mean, input_scale, scales[name]).to(device)
            for name, basis in bases.items()
        }
        controls[str(seed)] = {
            "sae_checkpoint": str(checkpoint),
            "diagnostic_summary": str(diagnostic),
            f"{training_corruption}_features": features,
            "random_features": random_features,
            "high_variance_dimensions": high_variance_dimensions.cpu().tolist(),
            "pca_eigenvalues": pca_eigenvalues.cpu().tolist(),
        }
        training[str(seed)] = {}
        for name, method in methods.items():
            torch.manual_seed(seed + 9400)
            np.random.seed(seed + 9400)
            checkpoint_path = seed_dir / f"{name}.pt"
            record_path = seed_dir / f"{name}_training.json"
            if args.resume and checkpoint_path.exists() and record_path.exists():
                method.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
                record = json.loads(record_path.read_text())
            else:
                record = train_method(
                    model, method, train_data, target_validation, args, device, checkpoint_path
                )
                record_path.write_text(json.dumps(record, indent=2))
            method.eval()
            training[str(seed)][name] = record
        validation[str(seed)] = {}
        for corruption, dataset in (
            (training_corruption, target_validation),
            (transfer_corruption, transfer_validation),
        ):
            result, arrays = evaluate_cache(model, methods, dataset, device, args, seed)
            validation[str(seed)][corruption] = result
            for name, values in arrays.items():
                outcomes[f"seed{seed}_{corruption}_{name}"] = values
        clean_result, clean_arrays = evaluate_clean_cache(
            model, methods, target_validation, device, args, seed
        )
        validation[str(seed)]["clean"] = clean_result
        for name, values in clean_arrays.items():
            outcomes[f"seed{seed}_clean_{name}"] = values
        del sae, methods
        if device.type == "cuda":
            torch.cuda.empty_cache()
    comparison = {}
    for name in method_names:
        comparison[name] = {}
        for corruption in (training_corruption, transfer_corruption, "clean"):
            gains = [
                validation[str(seed)][corruption]["methods"][name]["accuracy_difference"]
                for seed in args.seeds
            ]
            comparison[name][f"{corruption}_gains_by_seed"] = gains
            comparison[name][f"mean_{corruption}_gain"] = float(np.mean(gains))
    summary = {
        "configuration": vars(args) | {
            "output_root": str(args.output_root.resolve()),
            "cache_source_run": (
                str(args.cache_source_run.resolve()) if args.cache_source_run else None
            ),
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "block": BLOCK,
            "training_corruption": f"{training_corruption.title()}-4",
            "transfer_corruption": f"{transfer_corruption.title()}-4",
            "feature_selection": f"seed-specific top-8 strengthened plus top-8 weakened {training_corruption} BatchTopK features",
            "trainable_parameters_per_method": trainable_parameters,
            "matched_parameter_count": True,
            "frozen_components": ["ViT", "BatchTopK SAE", "all output bases"],
            "imageNetV2_accessed": False,
            "final_reserve_49000_50000_accessed": False,
            "status": "adapter-development comparison only",
        },
        "audit_of_experiment45": {
            "valid_split_protocol": True,
            "valid_parameter_matching": True,
            "limitations_corrected_here": [
                "Experiment 45 used an older fixed clean Vanilla SAE rather than seed-specific BatchTopK SAEs.",
                "Experiment 45 trained and evaluated only Noise repair.",
            ],
            "experiment45_not_repeated": True,
        },
        "development_splits": {str(seed): splits[seed] for seed in args.seeds},
        "controls": controls,
        "training": training,
        "validation": validation,
        "comparison": comparison,
        "guardrails": [
            "SAE feature discovery [11000,12000) is disjoint from every seed-specific adapter train/validation split.",
            "Every seed uses an independently initialized SAE and predictor.",
            "All four methods have exactly the same trainable architecture and parameter count.",
            "Clean counterparts provide training targets only; inference receives only the corrupted hidden state.",
            "No full SAE reconstruction is inserted; corrections are residual hidden-state deltas.",
            "ImageNetV2 and [49000,50000) are not accessed.",
        ],
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(comparison, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
