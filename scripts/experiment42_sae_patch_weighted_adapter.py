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
from scripts.experiment35_sae_guided_residual_adapter import SelectedSAEFeatures, feature_statistics
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import (
    DEFAULT_ADAPTER_ROOT,
    DEFAULT_FEATURE_SOURCE,
    evaluate,
    ranked_over_features,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment42_sae_patch_weighted_adapter"
BLOCK_INDEX = 10
WIDTH = 768


class UniformScorer(nn.Module):
    def forward(self, patches):
        return patches.new_ones(patches.shape[:2])


class SAEFeatureScorer(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.encoder.requires_grad_(False)

    def forward(self, patches):
        return F.relu(self.encoder.normalized(patches)).mean(-1)


class HiddenVarianceScorer(nn.Module):
    def __init__(self, dimensions, mean, scale):
        super().__init__()
        self.register_buffer("dimensions", torch.as_tensor(dimensions, dtype=torch.long))
        self.register_buffer("mean", mean)
        self.register_buffer("scale", scale.clamp_min(1e-4))

    def forward(self, patches):
        values = patches[..., self.dimensions]
        return ((values - self.mean) / self.scale).abs().mean(-1)


def hidden_statistics(dataset, device, batch_size, feature_count):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    total = torch.zeros(WIDTH, dtype=torch.float64, device=device)
    squared = torch.zeros_like(total)
    count = 0
    with torch.no_grad():
        for clean_hidden, _, _ in loader:
            values = clean_hidden[:, 1:].to(device).flatten(0, 1).double()
            total += values.sum(0)
            squared += values.square().sum(0)
            count += values.shape[0]
    mean = total / count
    scale = (squared / count - mean.square()).clamp_min(1e-8).sqrt()
    dimensions = torch.argsort(scale, descending=True)[:feature_count]
    return dimensions.tolist(), mean[dimensions].float(), scale[dimensions].float()


def normalized_patch_weights(raw_scores, strength, maximum_ratio):
    if strength == 0:
        return torch.ones_like(raw_scores)
    relative = raw_scores / raw_scores.mean(dim=1, keepdim=True).clamp_min(1e-6)
    relative = relative.clamp(0, maximum_ratio)
    weights = 1 + strength * relative
    return weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-6)


def objective(model, adapter, scorer, clean_hidden, noise_hidden, labels, args):
    clean_patches = clean_hidden[:, 1:]
    noise_patches = noise_hidden[:, 1:]
    target = clean_patches - noise_patches
    prediction = adapter(noise_patches)
    with torch.no_grad():
        patch_weights = normalized_patch_weights(
            scorer(noise_patches), args.weight_strength, args.maximum_weight_ratio
        )
    elementwise = F.smooth_l1_loss(prediction, target, beta=args.smooth_l1_beta, reduction="none")
    patch_loss = elementwise.mean(-1)
    residual = (patch_loss * patch_weights).mean()
    candidate = torch.cat(
        [noise_hidden[:, :1], noise_patches + args.train_alpha * prediction], dim=1
    )
    logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, noise_hidden, BLOCK_INDEX)
        baseline_correct = baseline_logits.argmax(1) == labels
        baseline_margin = classification_margin(baseline_logits, labels)[1]
    margin = classification_margin(logits, labels)[1]
    classification = F.cross_entropy(logits, labels)
    preservation = (
        F.relu(baseline_margin[baseline_correct] - margin[baseline_correct]).mean()
        if baseline_correct.any() else margin.new_zeros(())
    )
    total = residual + args.classification_weight * classification + args.preservation_weight * preservation
    return total, residual, classification, preservation, patch_weights.std(unbiased=False)


def run_epoch(model, adapter, scorer, loader, device, args, optimizer=None):
    adapter.train(optimizer is not None)
    scorer.eval()
    totals = np.zeros(5, dtype=np.float64)
    samples = 0
    for clean_hidden, noise_hidden, labels in loader:
        clean_hidden = clean_hidden.to(device)
        noise_hidden = noise_hidden.to(device)
        labels = labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        losses = objective(model, adapter, scorer, clean_hidden, noise_hidden, labels, args)
        if optimizer is not None:
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
        batch = labels.shape[0]
        totals += np.asarray([float(value.detach()) for value in losses]) * batch
        samples += batch
    return dict(zip(
        ["total", "residual", "classification", "preservation", "patch_weight_std"],
        (totals / samples).tolist(),
    ))


def train_adapter(model, scorer, train_data, validation_data, args, device, checkpoint):
    adapter = HiddenLinear().to(device)
    nn.init.zeros_(adapter.linear.weight)
    nn.init.zeros_(adapter.linear.bias)
    nn.init.zeros_(adapter.position.weight)
    optimizer = AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = DataLoader(train_data, batch_size=args.predictor_batch_size, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False)
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, adapter, scorer, train_loader, device, args, optimizer)
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, adapter, scorer, validation_loader, device, args
            )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(f"{checkpoint.stem} epoch={epoch} validation={validation_metrics['total']:.5f}")
        if validation_metrics["total"] < best:
            best = validation_metrics["total"]
            stale = 0
            torch.save(adapter.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    adapter.eval()
    return adapter, {"best_validation_total": best, "history": history}


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 42: SAE patch-weighted full-hidden-state adapter training"
    )
    parser.add_argument("--adapter-root", type=Path, default=DEFAULT_ADAPTER_ROOT)
    parser.add_argument("--feature-source", type=Path, default=DEFAULT_FEATURE_SOURCE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--feature-count", type=int, default=16)
    parser.add_argument("--weight-strength", type=float, default=1.0)
    parser.add_argument("--maximum-weight-ratio", type=float, default=5.0)
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
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-cache-samples", type=int)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    ranking = ranked_over_features(args.feature_source)
    harmful = ranking[: args.feature_count]
    if len(harmful) != args.feature_count:
        raise ValueError("Feature source has insufficient harmful features")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)
    sae.requires_grad_(False)
    training = {}
    validation = {}
    outcome_arrays = {}
    control_features = {}

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_data = HiddenCache(args.adapter_root / f"seed_{seed}" / "train_cache")
        validation_data = HiddenCache(args.adapter_root / f"seed_{seed}" / "validation_cache")
        if args.max_cache_samples is not None:
            train_data = Subset(train_data, range(min(args.max_cache_samples, len(train_data))))
            validation_data = Subset(validation_data, range(min(args.max_cache_samples, len(validation_data))))

        generator = np.random.default_rng(seed + 4200)
        available = np.setdiff1d(np.arange(sae.latent_dim), np.asarray(harmful))
        random_features = generator.choice(available, args.feature_count, replace=False).tolist()
        control_features[f"seed_{seed}"] = {"random_sae": random_features}
        scorers = {"adapter": UniformScorer().to(device)}
        for name, features in (("harmful_sae", harmful), ("random_sae", random_features)):
            mean, scale = feature_statistics(
                train_data, sae, features, device, args.predictor_batch_size
            )
            scorers[name] = SAEFeatureScorer(
                SelectedSAEFeatures(sae, features, mean, scale)
            ).to(device)
        dimensions, hidden_mean, hidden_scale = hidden_statistics(
            train_data, device, args.predictor_batch_size, args.feature_count
        )
        control_features[f"seed_{seed}"]["top_variance_hidden"] = dimensions
        scorers["top_variance_hidden"] = HiddenVarianceScorer(
            dimensions, hidden_mean, hidden_scale
        ).to(device)

        methods = {}
        training[f"seed_{seed}"] = {}
        for variant_index, (name, scorer) in enumerate(scorers.items()):
            torch.manual_seed(seed + 4200)
            np.random.seed(seed + 4200)
            checkpoint = seed_dir / f"{name}.pt"
            record_path = seed_dir / f"{name}_training.json"
            if args.resume and checkpoint.exists() and record_path.exists():
                adapter = HiddenLinear().to(device)
                adapter.load_state_dict(
                    torch.load(checkpoint, map_location=device, weights_only=True)
                )
                adapter.eval()
                record = json.loads(record_path.read_text())
            else:
                adapter, record = train_adapter(
                    model, scorer, train_data, validation_data, args, device, checkpoint
                )
                record_path.write_text(json.dumps(record, indent=2))
            methods[name] = adapter
            training[f"seed_{seed}"][name] = record

        seed_results, arrays = evaluate(
            model,
            methods,
            validation_data,
            args.predictor_batch_size,
            device,
            args.alpha,
            seed + 4200,
            args.bootstrap_repetitions,
        )
        validation[f"seed_{seed}"] = seed_results
        for name, values in arrays.items():
            outcome_arrays[f"seed_{seed}_{name}"] = values

    comparison = {}
    for name in ["harmful_sae", "random_sae", "top_variance_hidden"]:
        noise_differences = []
        clean_differences = []
        for seed in args.seeds:
            methods = validation[f"seed_{seed}"]["methods"]
            noise_differences.append(
                methods[name]["noise_vs_adapter"]["accuracy_difference"]
            )
            clean_differences.append(
                methods[name]["clean_vs_adapter"]["accuracy_difference"]
            )
        comparison[name] = {
            "mean_noise4_difference_vs_uniform": float(np.mean(noise_differences)),
            "mean_clean_difference_vs_uniform": float(np.mean(clean_differences)),
            "beats_uniform_noise_all_seeds": bool(np.all(np.asarray(noise_differences) > 0)),
        }
    harmful_supported = (
        comparison["harmful_sae"]["beats_uniform_noise_all_seeds"]
        and comparison["harmful_sae"]["mean_noise4_difference_vs_uniform"]
        > max(
            comparison["random_sae"]["mean_noise4_difference_vs_uniform"],
            comparison["top_variance_hidden"]["mean_noise4_difference_vs_uniform"],
        )
    )
    summary = {
        "configuration": vars(args) | {
            "adapter_root": str(args.adapter_root.resolve()),
            "feature_source": str(args.feature_source.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "vit_block": 11,
            "initialization": "zero",
            "inference_requires_sae": False,
            "frozen_components": ["ViT", "clean SAE used only for training weights"],
            "imageNetV2_accessed": False,
            "status": "development comparison only",
        },
        "development_splits": {str(seed): splits[seed] for seed in args.seeds},
        "harmful_features": harmful,
        "controls": control_features,
        "training": training,
        "validation": validation,
        "comparison": comparison,
        "sae_patch_weighting_supported": harmful_supported,
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcome_arrays)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "comparison": comparison,
        "sae_patch_weighting_supported": harmful_supported,
    }, indent=2))
    print(f"Saved Experiment 42 to {output_dir}")


if __name__ == "__main__":
    main()
