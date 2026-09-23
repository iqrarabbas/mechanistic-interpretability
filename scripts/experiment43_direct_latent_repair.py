import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenCache, HiddenLinear
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import (
    DEFAULT_ADAPTER_ROOT,
    DEFAULT_FEATURE_SOURCE,
    evaluate,
    ranked_over_features,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment43_direct_latent_repair"
BLOCK_INDEX = 10
PATCHES = 196
WIDTH = 768


class CoordinatePredictor(nn.Module):
    def __init__(self, coordinate_count):
        super().__init__()
        self.linear = nn.Linear(coordinate_count, coordinate_count)
        self.position = nn.Embedding(PATCHES, coordinate_count)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.zeros_(self.position.weight)

    def forward(self, coordinates):
        positions = torch.arange(PATCHES, device=coordinates.device)
        return self.linear(coordinates) + self.position(positions)[None]


class SAECoordinateRepair(nn.Module):
    def __init__(self, sae, features, mean, scale):
        super().__init__()
        self.sae = sae
        self.sae.requires_grad_(False)
        self.register_buffer("features", torch.as_tensor(features, dtype=torch.long))
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale.clamp_min(1e-4))
        self.predictor = CoordinatePredictor(len(features))

    def coordinates(self, patches):
        shape = patches.shape
        latent = self.sae.encode(patches.reshape(-1, shape[-1]))
        return latent[:, self.features].reshape(*shape[:2], -1)

    def decode_delta(self, delta):
        flat = delta.flatten(0, 1)
        decoder_columns = self.sae.decoder.weight[:, self.features]
        decoded = F.linear(flat, decoder_columns, bias=None)
        return decoded.reshape(delta.shape[0], delta.shape[1], WIDTH)

    def forward(self, patches):
        coordinates = self.coordinates(patches)
        standardized = (coordinates - self.mean) / self.scale
        return self.decode_delta(self.predictor(standardized) * self.scale)

    def target(self, clean_patches, noise_patches):
        with torch.no_grad():
            return (
                self.coordinates(clean_patches) - self.coordinates(noise_patches)
            ) / self.scale

    def predicted_coordinates(self, noise_patches):
        coordinates = self.coordinates(noise_patches)
        return self.predictor((coordinates - self.mean) / self.scale)


class RawCoordinateRepair(nn.Module):
    def __init__(self, dimensions, mean, scale):
        super().__init__()
        self.register_buffer("dimensions", torch.as_tensor(dimensions, dtype=torch.long))
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale.clamp_min(1e-4))
        self.predictor = CoordinatePredictor(len(dimensions))

    def coordinates(self, patches):
        return patches[..., self.dimensions]

    def decode_delta(self, delta):
        output = delta.new_zeros(*delta.shape[:-1], WIDTH)
        output[..., self.dimensions] = delta
        return output

    def forward(self, patches):
        coordinates = self.coordinates(patches)
        return self.decode_delta(
            self.predictor((coordinates - self.mean) / self.scale) * self.scale
        )

    def target(self, clean_patches, noise_patches):
        return (
            self.coordinates(clean_patches) - self.coordinates(noise_patches)
        ) / self.scale

    def predicted_coordinates(self, noise_patches):
        coordinates = self.coordinates(noise_patches)
        return self.predictor((coordinates - self.mean) / self.scale)


def coordinate_statistics(dataset, extractor, device, batch_size):
    total = None
    squared = None
    count = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for _, noise_hidden, _ in loader:
            values = extractor(noise_hidden[:, 1:].to(device)).flatten(0, 1).double()
            if total is None:
                total = torch.zeros(values.shape[-1], dtype=torch.float64, device=device)
                squared = torch.zeros_like(total)
            total += values.sum(0)
            squared += values.square().sum(0)
            count += values.shape[0]
    mean = total / count
    scale = (squared / count - mean.square()).clamp_min(1e-8).sqrt()
    return mean.float(), scale.float()


def top_variance_dimensions(dataset, device, batch_size, count):
    mean, scale = coordinate_statistics(dataset, lambda value: value, device, batch_size)
    dimensions = torch.argsort(scale, descending=True)[:count]
    return dimensions.tolist(), mean[dimensions], scale[dimensions]


def objective(model, method, clean_hidden, noise_hidden, labels, args):
    clean_patches = clean_hidden[:, 1:]
    noise_patches = noise_hidden[:, 1:]
    predicted_coordinates = method.predicted_coordinates(noise_patches)
    target_coordinates = method.target(clean_patches, noise_patches)
    coordinate_loss = F.smooth_l1_loss(
        predicted_coordinates, target_coordinates, beta=args.smooth_l1_beta
    )
    residual = method.decode_delta(predicted_coordinates * method.scale)
    candidate = torch.cat(
        [noise_hidden[:, :1], noise_patches + args.train_alpha * residual], dim=1
    )
    logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, noise_hidden, BLOCK_INDEX)
        baseline_correct = baseline_logits.argmax(1) == labels
        baseline_margin = classification_margin(baseline_logits, labels)[1]
    margin = classification_margin(logits, labels)[1]
    classification_loss = F.cross_entropy(logits, labels)
    preservation_loss = (
        F.relu(baseline_margin[baseline_correct] - margin[baseline_correct]).mean()
        if baseline_correct.any() else margin.new_zeros(())
    )
    total = (
        coordinate_loss
        + args.classification_weight * classification_loss
        + args.preservation_weight * preservation_loss
    )
    return total, coordinate_loss, classification_loss, preservation_loss


def run_epoch(model, method, loader, device, args, optimizer=None):
    method.train(optimizer is not None)
    totals = np.zeros(4, dtype=np.float64)
    samples = 0
    for clean_hidden, noise_hidden, labels in loader:
        clean_hidden = clean_hidden.to(device)
        noise_hidden = noise_hidden.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        losses = objective(model, method, clean_hidden, noise_hidden, labels, args)
        if optimizer is not None:
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(method.predictor.parameters(), 1.0)
            optimizer.step()
        batch = labels.shape[0]
        totals += np.asarray([float(value.detach()) for value in losses]) * batch
        samples += batch
    return dict(zip(["total", "coordinate", "classification", "preservation"], (totals / samples).tolist()))


def train_method(model, method, train_data, validation_data, args, device, checkpoint):
    optimizer = AdamW(method.predictor.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.batch_size, shuffle=False)
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, method, train_loader, device, args, optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(model, method, validation_loader, device, args)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(f"{checkpoint.stem} epoch={epoch} validation={validation_metrics['total']:.5f}")
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            stale = 0
            torch.save(method.predictor.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    method.predictor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    method.eval()
    return {"best_validation_total": best, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Experiment 43: matched direct SAE and raw latent repair")
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--feature-source", type=Path, default=DEFAULT_FEATURE_SOURCE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--coordinate-count", type=int, default=16)
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--max-cache-samples", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    harmful = ranked_over_features(args.feature_source)[: args.coordinate_count]
    if len(harmful) != args.coordinate_count:
        raise ValueError("Feature source has insufficient harmful features")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)
    sae.requires_grad_(False)
    trainable_parameters = args.coordinate_count ** 2 + args.coordinate_count + PATCHES * args.coordinate_count
    training = {}
    validation = {}
    outcomes = {}
    controls = {}

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_data = HiddenCache(args.adapter_root / f"seed_{seed}" / "train_cache")
        validation_data = HiddenCache(args.adapter_root / f"seed_{seed}" / "validation_cache")
        if args.max_cache_samples is not None:
            train_data = Subset(train_data, range(min(args.max_cache_samples, len(train_data))))
            validation_data = Subset(validation_data, range(min(args.max_cache_samples, len(validation_data))))

        generator = np.random.default_rng(seed + 4300)
        available = np.setdiff1d(np.arange(sae.latent_dim), np.asarray(harmful))
        random_features = generator.choice(available, args.coordinate_count, replace=False).tolist()
        raw_dimensions, raw_mean, raw_scale = top_variance_dimensions(
            train_data, device, args.batch_size, args.coordinate_count
        )
        methods = {}
        feature_sets = {"harmful_sae": harmful, "random_sae": random_features}
        for name, features in feature_sets.items():
            extractor = lambda patches, selected=features: sae.encode(
                patches.reshape(-1, WIDTH)
            )[:, selected].reshape(*patches.shape[:2], -1)
            mean, scale = coordinate_statistics(train_data, extractor, device, args.batch_size)
            methods[name] = SAECoordinateRepair(sae, features, mean, scale).to(device)
        methods["raw_hidden"] = RawCoordinateRepair(
            raw_dimensions, raw_mean, raw_scale
        ).to(device)
        controls[f"seed_{seed}"] = {
            "random_sae_features": random_features,
            "raw_top_variance_dimensions": raw_dimensions,
        }

        training[f"seed_{seed}"] = {}
        for name, method in methods.items():
            torch.manual_seed(seed + 4300)
            np.random.seed(seed + 4300)
            checkpoint = seed_dir / f"{name}.pt"
            record_path = seed_dir / f"{name}_training.json"
            if args.resume and checkpoint.exists() and record_path.exists():
                method.predictor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
                method.eval()
                record = json.loads(record_path.read_text())
            else:
                record = train_method(
                    model, method, train_data, validation_data, args, device, checkpoint
                )
                record_path.write_text(json.dumps(record, indent=2))
            training[f"seed_{seed}"][name] = record

        adapter = HiddenLinear().to(device)
        adapter.load_state_dict(torch.load(
            args.adapter_root / f"seed_{seed}" / "classification_weight_0.05.pt",
            map_location=device,
            weights_only=True,
        ))
        adapter.eval()
        evaluated = {"adapter": adapter} | methods
        seed_results, arrays = evaluate(
            model, evaluated, validation_data, args.batch_size, device,
            args.alpha, seed + 4300, args.bootstrap_repetitions,
        )
        validation[f"seed_{seed}"] = seed_results
        for name, values in arrays.items():
            outcomes[f"seed_{seed}_{name}"] = values

    comparison = {}
    for name in ["harmful_sae", "random_sae", "raw_hidden"]:
        noise_gains = []
        clean_gains = []
        for seed in args.seeds:
            result = validation[f"seed_{seed}"]["methods"][name]
            noise_gains.append(result["noise_vs_baseline"]["accuracy_difference"])
            clean_gains.append(result["clean_vs_baseline"]["accuracy_difference"])
        comparison[name] = {
            "mean_noise4_gain_vs_baseline": float(np.mean(noise_gains)),
            "mean_clean_gain_vs_baseline": float(np.mean(clean_gains)),
            "noise4_gains_by_seed": noise_gains,
            "clean_gains_by_seed": clean_gains,
        }
    supported = (
        comparison["harmful_sae"]["mean_noise4_gain_vs_baseline"]
        > comparison["raw_hidden"]["mean_noise4_gain_vs_baseline"]
        and comparison["harmful_sae"]["mean_noise4_gain_vs_baseline"]
        > comparison["random_sae"]["mean_noise4_gain_vs_baseline"]
        and all(
            validation[f"seed_{seed}"]["methods"]["harmful_sae"]["noise_vs_baseline"]["accuracy_difference"] > 0
            for seed in args.seeds
        )
    )
    summary = {
        "configuration": vars(args) | {
            "adapter_root": str(args.adapter_root.resolve()),
            "feature_source": str(args.feature_source.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "vit_block": 11,
            "frozen_components": ["ViT", "clean SAE", "coordinate bases"],
            "trainable_parameters_per_method": trainable_parameters,
            "intervention": "residual decoder delta; no SAE reconstruction replacement",
            "imageNetV2_accessed": False,
            "status": "development comparison only",
        },
        "development_splits": {str(seed): splits[seed] for seed in args.seeds},
        "harmful_features": harmful,
        "controls": controls,
        "training": training,
        "validation": validation,
        "comparison": comparison,
        "sae_direct_repair_supported": supported,
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"comparison": comparison, "sae_direct_repair_supported": supported}, indent=2))
    print(f"Saved Experiment 43 to {output_dir}")


if __name__ == "__main__":
    main()
