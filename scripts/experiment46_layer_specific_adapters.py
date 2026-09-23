import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment46_layer_specific_adapters"


def make_loader(split, seed, batch_size, workers, shuffle, max_samples=None):
    samples = split["end"] - split["start"]
    if max_samples is not None:
        samples = min(samples, max_samples)
    dataset = PairedCorruptionDataset("noise", samples, split["start"], seed)
    generator = torch.Generator().manual_seed(seed + 4600)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator if shuffle else None,
    )


def layer_loss(model, adapter, clean_hidden, noise_hidden, labels, layer, args):
    clean_patches = clean_hidden[:, 1:]
    noise_patches = noise_hidden[:, 1:]
    prediction = adapter(noise_patches)
    target = clean_patches - noise_patches
    residual_loss = F.smooth_l1_loss(prediction, target, beta=args.smooth_l1_beta)
    candidate = torch.cat(
        [noise_hidden[:, :1], noise_patches + args.train_alpha * prediction], dim=1
    )
    layer_index = layer - 1
    logits = downstream_from_layer(model, candidate, layer_index)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, noise_hidden, layer_index)
        baseline_correct = baseline_logits.argmax(1) == labels
        baseline_margin = classification_margin(baseline_logits, labels)[1]
    margin = classification_margin(logits, labels)[1]
    classification_loss = F.cross_entropy(logits, labels)
    preservation_loss = (
        F.relu(baseline_margin[baseline_correct] - margin[baseline_correct]).mean()
        if baseline_correct.any() else margin.new_zeros(())
    )
    total = (
        residual_loss
        + args.classification_weight * classification_loss
        + args.preservation_weight * preservation_loss
    )
    return total, residual_loss, classification_loss, preservation_loss


def run_epoch(model, adapters, loader, device, args, optimizers=None):
    training = optimizers is not None
    for adapter in adapters.values():
        adapter.train(training)
    totals = {layer: np.zeros(4, dtype=np.float64) for layer in adapters}
    samples = 0
    description = "Layer adapter training" if training else "Layer adapter validation"
    for clean, noise, labels in tqdm(loader, desc=description, leave=False):
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device),
                output_hidden_states=True,
            )
        batch = clean.shape[0]
        if training:
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
        losses = {}
        for layer, adapter in adapters.items():
            clean_hidden, noise_hidden = outputs.hidden_states[layer].split(batch)
            losses[layer] = layer_loss(
                model, adapter, clean_hidden.detach(), noise_hidden.detach(), labels, layer, args
            )
        if training:
            sum(values[0] for values in losses.values()).backward()
            for layer, adapter in adapters.items():
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizers[layer].step()
        for layer, values in losses.items():
            totals[layer] += np.asarray([float(value.detach()) for value in values]) * batch
        samples += batch
    return {
        str(layer): dict(zip(
            ["total", "residual", "classification", "preservation"],
            (values / samples).tolist(),
        ))
        for layer, values in totals.items()
    }


def train_adapters(model, layers, train_loader, validation_loader, device, args, seed_dir):
    adapters = {layer: HiddenLinear().to(device) for layer in layers}
    for adapter in adapters.values():
        torch.nn.init.zeros_(adapter.linear.weight)
        torch.nn.init.zeros_(adapter.linear.bias)
        torch.nn.init.zeros_(adapter.position.weight)
    optimizers = {
        layer: AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for layer, adapter in adapters.items()
    }
    schedulers = {
        layer: CosineAnnealingLR(optimizers[layer], T_max=args.epochs) for layer in layers
    }
    best = {layer: float("inf") for layer in layers}
    stale = {layer: 0 for layer in layers}
    active = set(layers)
    history = []
    for epoch in range(1, args.epochs + 1):
        active_adapters = {layer: adapters[layer] for layer in active}
        active_optimizers = {layer: optimizers[layer] for layer in active}
        train_metrics = run_epoch(
            model, active_adapters, train_loader, device, args, active_optimizers
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, active_adapters, validation_loader, device, args
            )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        for layer in list(active):
            schedulers[layer].step()
            value = validation_metrics[str(layer)]["total"]
            print(f"seed_dir={seed_dir.name} block={layer} epoch={epoch} validation={value:.5f}")
            if value < best[layer]:
                best[layer] = value
                stale[layer] = 0
                torch.save(adapters[layer].state_dict(), seed_dir / f"block_{layer}.pt")
            else:
                stale[layer] += 1
                if stale[layer] >= args.patience:
                    active.remove(layer)
        if not active:
            break
    for layer, adapter in adapters.items():
        adapter.load_state_dict(torch.load(
            seed_dir / f"block_{layer}.pt", map_location=device, weights_only=True
        ))
        adapter.eval()
    return adapters, {"best_validation_total": best, "history": history}


def evaluate(model, adapters, loader, device, args, seed):
    arrays = {"baseline_clean": [], "baseline_noise": []}
    for layer in adapters:
        arrays[f"block_{layer}_clean"] = []
        arrays[f"block_{layer}_noise"] = []
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Layer adapter paired evaluation"):
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            batch = clean.shape[0]
            clean_logits, noise_logits = outputs.logits.split(batch)
            arrays["baseline_clean"].extend((clean_logits.argmax(1) == labels).cpu().tolist())
            arrays["baseline_noise"].extend((noise_logits.argmax(1) == labels).cpu().tolist())
            for layer, adapter in adapters.items():
                clean_hidden, noise_hidden = outputs.hidden_states[layer].split(batch)
                for condition, hidden in (("clean", clean_hidden), ("noise", noise_hidden)):
                    patches = hidden[:, 1:]
                    candidate = torch.cat(
                        [hidden[:, :1], patches + args.alpha * adapter(patches)], dim=1
                    )
                    logits = downstream_from_layer(model, candidate, layer - 1)
                    arrays[f"block_{layer}_{condition}"].extend(
                        (logits.argmax(1) == labels).cpu().tolist()
                    )
    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    results = {
        "baseline_clean_accuracy": float(arrays["baseline_clean"].mean()),
        "baseline_noise4_accuracy": float(arrays["baseline_noise"].mean()),
        "blocks": {},
    }
    for offset, layer in enumerate(adapters):
        results["blocks"][str(layer)] = {
            "clean_vs_baseline": paired_comparison(
                arrays["baseline_clean"], arrays[f"block_{layer}_clean"],
                seed + offset, args.bootstrap_repetitions,
            ),
            "noise_vs_baseline": paired_comparison(
                arrays["baseline_noise"], arrays[f"block_{layer}_noise"],
                seed + 100 + offset, args.bootstrap_repetitions,
            ),
        }
    return results, arrays


def main():
    parser = argparse.ArgumentParser(description="Experiment 46: deployable adapters across ViT blocks")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--layers", type=int, nargs="+", default=[6, 8, 9, 10, 11, 12])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if any(layer < 1 or layer > 12 for layer in args.layers):
        raise ValueError("Layers must be numbered from 1 through 12")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    training = {}
    validation = {}
    outcomes = {}

    for seed in args.seeds:
        torch.manual_seed(seed + 4600)
        np.random.seed(seed + 4600)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_loader = make_loader(
            splits[seed]["train"], seed, args.image_batch_size,
            args.num_workers, True, args.max_samples,
        )
        validation_loader = make_loader(
            splits[seed]["validation"], seed, args.image_batch_size,
            args.num_workers, False, args.max_samples,
        )
        record_path = seed_dir / "training_record.json"
        checkpoints_exist = all((seed_dir / f"block_{layer}.pt").exists() for layer in args.layers)
        if args.resume and checkpoints_exist and record_path.exists():
            adapters = {layer: HiddenLinear().to(device) for layer in args.layers}
            for layer, adapter in adapters.items():
                adapter.load_state_dict(torch.load(
                    seed_dir / f"block_{layer}.pt", map_location=device, weights_only=True
                ))
                adapter.eval()
            record = json.loads(record_path.read_text())
        else:
            adapters, record = train_adapters(
                model, args.layers, train_loader, validation_loader, device, args, seed_dir
            )
            record_path.write_text(json.dumps(record, indent=2))
        training[f"seed_{seed}"] = record
        result, arrays = evaluate(model, adapters, validation_loader, device, args, seed + 4600)
        validation[f"seed_{seed}"] = result
        for name, values in arrays.items():
            outcomes[f"seed_{seed}_{name}"] = values

    aggregate = {}
    for layer in args.layers:
        noise = [
            validation[f"seed_{seed}"]["blocks"][str(layer)]["noise_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        clean = [
            validation[f"seed_{seed}"]["blocks"][str(layer)]["clean_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        aggregate[str(layer)] = {
            "noise4_gains_by_seed": noise,
            "mean_noise4_gain": float(np.mean(noise)),
            "clean_gains_by_seed": clean,
            "mean_clean_gain": float(np.mean(clean)),
        }
    best_layer = max(
        args.layers,
        key=lambda layer: (aggregate[str(layer)]["mean_noise4_gain"], aggregate[str(layer)]["mean_clean_gain"]),
    )
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "frozen_components": ["ViT"],
            "adapter_trainable_parameters": sum(p.numel() for p in HiddenLinear().parameters()),
            "adapter_scope": "patch tokens only",
            "block_definition": "adapter applied after the numbered transformer block",
            "block12_limitation": "patch-only correction after the final block cannot alter CLS logits; included as a negative control",
            "persistent_hidden_caches_created": False,
            "imageNetV2_accessed": False,
            "status": "development comparison only",
        },
        "development_splits": {str(seed): splits[seed] for seed in args.seeds},
        "training": training,
        "validation": validation,
        "aggregate": aggregate,
        "best_layer": best_layer,
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"aggregate": aggregate, "best_layer": best_layer}, indent=2))
    print(f"Saved Experiment 46 to {output_dir}")


if __name__ == "__main__":
    main()
