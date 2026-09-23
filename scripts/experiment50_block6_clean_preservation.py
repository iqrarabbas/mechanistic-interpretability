import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment46_layer_specific_adapters import make_loader


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment50_block6_clean_preservation"
BLOCK = 6


def variant_name(weight):
    return f"identity_{weight:g}".replace(".", "p")


def adapter_loss(model, adapter, clean_hidden, noise_hidden, labels, identity_weight, args):
    clean_patches = clean_hidden[:, 1:]
    noise_patches = noise_hidden[:, 1:]
    noise_prediction = adapter(noise_patches)
    target = clean_patches - noise_patches
    residual_loss = F.smooth_l1_loss(
        noise_prediction, target, beta=args.smooth_l1_beta
    )
    clean_prediction = adapter(clean_patches)
    identity_loss = F.smooth_l1_loss(
        clean_prediction, torch.zeros_like(clean_prediction), beta=args.smooth_l1_beta
    )
    candidate = torch.cat(
        [noise_hidden[:, :1], noise_patches + args.train_alpha * noise_prediction], dim=1
    )
    logits = downstream_from_layer(model, candidate, BLOCK - 1)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, noise_hidden, BLOCK - 1)
        baseline_correct = baseline_logits.argmax(1) == labels
        baseline_margin = classification_margin(baseline_logits, labels)[1]
    margin = classification_margin(logits, labels)[1]
    classification_loss = F.cross_entropy(logits, labels)
    preservation_loss = (
        F.relu(baseline_margin[baseline_correct] - margin[baseline_correct]).mean()
        if baseline_correct.any()
        else margin.new_zeros(())
    )
    total = (
        residual_loss
        + args.classification_weight * classification_loss
        + args.preservation_weight * preservation_loss
        + identity_weight * identity_loss
    )
    return total, residual_loss, classification_loss, preservation_loss, identity_loss


def run_epoch(model, adapters, loader, device, args, optimizers=None):
    training = optimizers is not None
    for _, adapter in adapters.values():
        adapter.train(training)
    totals = {name: np.zeros(5, dtype=np.float64) for name in adapters}
    samples = 0
    description = "Block-6 preservation training" if training else "Block-6 preservation validation"
    for clean, noise, labels in tqdm(loader, desc=description, leave=False):
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device),
                output_hidden_states=True,
            )
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK].split(clean.shape[0])
        if training:
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
        losses = {
            name: adapter_loss(
                model,
                adapter,
                clean_hidden.detach(),
                noise_hidden.detach(),
                labels,
                weight,
                args,
            )
            for name, (weight, adapter) in adapters.items()
        }
        if training:
            sum(values[0] for values in losses.values()).backward()
            for name, (_, adapter) in adapters.items():
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizers[name].step()
        batch = clean.shape[0]
        for name, values in losses.items():
            totals[name] += np.asarray([float(value.detach()) for value in values]) * batch
        samples += batch
    fields = ["total", "residual", "classification", "preservation", "clean_identity"]
    return {
        name: dict(zip(fields, (values / samples).tolist()))
        for name, values in totals.items()
    }


def train_variants(model, weights, train_loader, validation_loader, device, args, seed_dir):
    adapters = {}
    for weight in weights:
        name = variant_name(weight)
        adapter = HiddenLinear().to(device)
        torch.nn.init.zeros_(adapter.linear.weight)
        torch.nn.init.zeros_(adapter.linear.bias)
        torch.nn.init.zeros_(adapter.position.weight)
        adapters[name] = (weight, adapter)
    optimizers = {
        name: AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        for name, (_, adapter) in adapters.items()
    }
    schedulers = {
        name: CosineAnnealingLR(optimizer, T_max=args.epochs)
        for name, optimizer in optimizers.items()
    }
    best = {name: float("inf") for name in adapters}
    stale = {name: 0 for name in adapters}
    active = set(adapters)
    history = []
    for epoch in range(1, args.epochs + 1):
        active_adapters = {name: adapters[name] for name in active}
        active_optimizers = {name: optimizers[name] for name in active}
        train_metrics = run_epoch(
            model, active_adapters, train_loader, device, args, active_optimizers
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, active_adapters, validation_loader, device, args
            )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        for name in list(active):
            schedulers[name].step()
            value = validation_metrics[name]["total"]
            print(f"seed={seed_dir.name} variant={name} epoch={epoch} validation={value:.6f}")
            if value < best[name]:
                best[name] = value
                stale[name] = 0
                torch.save(adapters[name][1].state_dict(), seed_dir / f"{name}.pt")
            else:
                stale[name] += 1
                if stale[name] >= args.patience:
                    active.remove(name)
        if not active:
            break
    for name, (_, adapter) in adapters.items():
        adapter.load_state_dict(
            torch.load(seed_dir / f"{name}.pt", map_location=device, weights_only=True)
        )
        adapter.eval()
    return adapters, {"best_validation_total": best, "history": history}


def evaluate(model, adapters, loader, device, args, seed):
    arrays = {"baseline_clean": [], "baseline_noise": []}
    for name in adapters:
        arrays[f"{name}_clean"] = []
        arrays[f"{name}_noise"] = []
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Block-6 paired evaluation"):
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            batch = clean.shape[0]
            clean_logits, noise_logits = outputs.logits.split(batch)
            arrays["baseline_clean"].extend((clean_logits.argmax(1) == labels).cpu().tolist())
            arrays["baseline_noise"].extend((noise_logits.argmax(1) == labels).cpu().tolist())
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK].split(batch)
            for name, (_, adapter) in adapters.items():
                for condition, hidden in (("clean", clean_hidden), ("noise", noise_hidden)):
                    patches = hidden[:, 1:]
                    candidate = torch.cat(
                        [hidden[:, :1], patches + args.alpha * adapter(patches)], dim=1
                    )
                    logits = downstream_from_layer(model, candidate, BLOCK - 1)
                    arrays[f"{name}_{condition}"].extend(
                        (logits.argmax(1) == labels).cpu().tolist()
                    )
    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    results = {
        "baseline_clean_accuracy": float(arrays["baseline_clean"].mean()),
        "baseline_noise4_accuracy": float(arrays["baseline_noise"].mean()),
        "variants": {},
    }
    for offset, name in enumerate(adapters):
        results["variants"][name] = {
            "identity_weight": adapters[name][0],
            "clean_vs_baseline": paired_comparison(
                arrays["baseline_clean"], arrays[f"{name}_clean"],
                seed + offset, args.bootstrap_repetitions,
            ),
            "noise_vs_baseline": paired_comparison(
                arrays["baseline_noise"], arrays[f"{name}_noise"],
                seed + 100 + offset, args.bootstrap_repetitions,
            ),
        }
    return results, arrays


def main():
    parser = argparse.ArgumentParser(description="Experiment 50: preserve clean accuracy at Block 6")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--identity-weights", type=float, nargs="+", default=[0.0, 0.2, 1.0, 5.0, 20.0]
    )
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
    parser.add_argument("--clean-loss-tolerance", type=float, default=0.005)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if len(set(args.identity_weights)) != len(args.identity_weights):
        raise ValueError("Identity weights must be unique.")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Block: {BLOCK}; identity weights: {args.identity_weights}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    training = {}
    validation = {}
    outcomes = {}

    for seed in args.seeds:
        torch.manual_seed(seed + 5000)
        np.random.seed(seed + 5000)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir()
        train_loader = make_loader(
            splits[seed]["train"], seed, args.image_batch_size,
            args.num_workers, True, args.max_samples,
        )
        validation_loader = make_loader(
            splits[seed]["validation"], seed, args.image_batch_size,
            args.num_workers, False, args.max_samples,
        )
        adapters, record = train_variants(
            model, args.identity_weights, train_loader, validation_loader, device, args, seed_dir
        )
        training[f"seed_{seed}"] = record
        result, arrays = evaluate(model, adapters, validation_loader, device, args, seed + 5000)
        validation[f"seed_{seed}"] = result
        for name, values in arrays.items():
            outcomes[f"seed_{seed}_{name}"] = values

    aggregate = {}
    for weight in args.identity_weights:
        name = variant_name(weight)
        noise = [
            validation[f"seed_{seed}"]["variants"][name]["noise_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        clean = [
            validation[f"seed_{seed}"]["variants"][name]["clean_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        aggregate[name] = {
            "identity_weight": weight,
            "noise4_gains_by_seed": noise,
            "mean_noise4_gain": float(np.mean(noise)),
            "clean_gains_by_seed": clean,
            "mean_clean_gain": float(np.mean(clean)),
            "eligible_clean_tolerance": float(np.mean(clean)) >= -args.clean_loss_tolerance,
        }
    eligible = [name for name, result in aggregate.items() if result["eligible_clean_tolerance"]]
    selected = max(eligible, key=lambda name: aggregate[name]["mean_noise4_gain"]) if eligible else None
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "block": BLOCK,
            "model": BASE_MODEL,
            "frozen_components": ["ViT"],
            "adapter_trainable_parameters": sum(p.numel() for p in HiddenLinear().parameters()),
            "clean_identity_definition": "SmoothL1(adapter(clean_patch), 0)",
            "imageNetV2_accessed": False,
            "status": "development hyperparameter sweep",
        },
        "development_splits": {str(seed): splits[seed] for seed in args.seeds},
        "training": training,
        "validation": validation,
        "aggregate": aggregate,
        "selection_rule": (
            "Highest mean Noise-4 gain among variants with mean clean loss no worse than "
            f"{-100 * args.clean_loss_tolerance:.2f} pp"
        ),
        "selected_variant": selected,
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"aggregate": aggregate, "selected_variant": selected}, indent=2))
    print(f"Saved Experiment 50 to {output_dir}")


if __name__ == "__main__":
    main()
