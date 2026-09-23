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
from scripts.experiment67_mixed_noise_blur_block6_adapter import Triple, make_loader


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results" / "sae" / "experiment130_identity_matched_layer_sweep"


def layer_loss(model, adapter, clean_hidden, corrupted_hidden, labels, layer, args):
    clean_patches = clean_hidden[:, 1:]
    corrupted_patches = corrupted_hidden[:, 1:]
    correction = adapter(corrupted_patches)
    residual_loss = F.smooth_l1_loss(
        correction, clean_patches - corrupted_patches, beta=args.smooth_l1_beta
    )
    clean_correction = adapter(clean_patches)
    identity_loss = F.smooth_l1_loss(
        clean_correction, torch.zeros_like(clean_correction), beta=args.smooth_l1_beta
    )
    candidate = torch.cat(
        [corrupted_hidden[:, :1], corrupted_patches + args.train_alpha * correction], dim=1
    )
    logits = downstream_from_layer(model, candidate, layer - 1)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, corrupted_hidden, layer - 1)
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
        + args.identity_weight * identity_loss
    )
    return total, residual_loss, classification_loss, preservation_loss, identity_loss


def run_epoch(model, adapter, loader, layer, device, args, optimizer=None):
    training = optimizer is not None
    adapter.train(training)
    totals = np.zeros(5, dtype=np.float64)
    count = 0
    description = f"Block {layer} {'train' if training else 'validation'}"
    for clean, blur, noise, labels in tqdm(loader, desc=description, leave=False):
        labels = labels.to(device)
        with torch.no_grad():
            hidden = model(
                pixel_values=torch.cat([clean, blur, noise]).to(device),
                output_hidden_states=True,
            ).hidden_states[layer]
            clean_hidden, blur_hidden, noise_hidden = hidden.split(len(clean))
        if training:
            optimizer.zero_grad(set_to_none=True)
        blur_losses = layer_loss(
            model, adapter, clean_hidden, blur_hidden, labels, layer, args
        )
        noise_losses = layer_loss(
            model, adapter, clean_hidden, noise_hidden, labels, layer, args
        )
        losses = tuple((blur_value + noise_value) / 2 for blur_value, noise_value in zip(blur_losses, noise_losses))
        if training:
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
        totals += np.asarray([float(value.detach()) for value in losses]) * len(clean)
        count += len(clean)
    return dict(zip(
        ["total", "residual", "classification", "preservation", "identity"],
        (totals / count).tolist(),
    ))


def train_layer(model, train_loader, validation_loader, layer, device, args, layer_dir):
    checkpoint = layer_dir / "best.pt"
    record_path = layer_dir / "training.json"
    if args.resume and checkpoint.exists() and record_path.exists():
        adapter = HiddenLinear().to(device)
        adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        adapter.eval()
        return adapter, json.loads(record_path.read_text())

    adapter = HiddenLinear().to(device)
    for parameter in adapter.parameters():
        torch.nn.init.zeros_(parameter)
    optimizer = AdamW(
        adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best = float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model, adapter, train_loader, layer, device, args, optimizer
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, adapter, validation_loader, layer, device, args
            )
        scheduler.step()
        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        })
        value = validation_metrics["total"]
        print(f"block={layer} epoch={epoch} validation={value:.6f}", flush=True)
        if value < best:
            best = value
            stale = 0
            torch.save(adapter.state_dict(), checkpoint)
        else:
            stale += 1
        if stale >= args.patience:
            break
    record = {"best_validation_total": best, "history": history}
    record_path.write_text(json.dumps(record, indent=2))
    adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    adapter.eval()
    return adapter, record


def evaluate(model, adapters, loader, device, args, seed):
    conditions = ("clean", "noise", "blur")
    arrays = {condition: [] for condition in conditions}
    for layer in adapters:
        for condition in conditions:
            arrays[f"block_{layer}_{condition}"] = []
    with torch.no_grad():
        for clean, blur, noise, labels in tqdm(loader, desc="Layer sweep evaluation"):
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, blur, noise]).to(device),
                output_hidden_states=True,
            )
            batch = len(clean)
            clean_logits, blur_logits, noise_logits = outputs.logits.split(batch)
            logits_by_condition = {
                "clean": clean_logits,
                "noise": noise_logits,
                "blur": blur_logits,
            }
            for condition, logits in logits_by_condition.items():
                arrays[condition].extend((logits.argmax(1) == labels).cpu().tolist())
            for layer, adapter in adapters.items():
                clean_hidden, blur_hidden, noise_hidden = outputs.hidden_states[layer].split(batch)
                hidden_by_condition = {
                    "clean": clean_hidden,
                    "noise": noise_hidden,
                    "blur": blur_hidden,
                }
                for condition, hidden in hidden_by_condition.items():
                    patches = hidden[:, 1:]
                    candidate = torch.cat(
                        [hidden[:, :1], patches + args.alpha * adapter(patches)], dim=1
                    )
                    logits = downstream_from_layer(model, candidate, layer - 1)
                    arrays[f"block_{layer}_{condition}"].extend(
                        (logits.argmax(1) == labels).cpu().tolist()
                    )
    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    result = {
        "baseline": {
            condition: float(arrays[condition].mean()) for condition in conditions
        },
        "blocks": {},
    }
    for layer in adapters:
        result["blocks"][str(layer)] = {}
        for offset, condition in enumerate(conditions):
            result["blocks"][str(layer)][condition] = paired_comparison(
                arrays[condition],
                arrays[f"block_{layer}_{condition}"],
                seed + 100 * layer + offset,
                args.bootstrap,
            )
    return result, arrays


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 130: identity-matched mixed-corruption adapter layer sweep"
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(1, 12)))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--identity-weight", type=float, default=0.2)
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if len(set(args.layers)) != len(args.layers) or any(layer < 1 or layer > 11 for layer in args.layers):
        raise ValueError("Layers must be unique integers from 1 through 11")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)

    training = {}
    validation = {}
    outcomes = {}
    for seed in args.seeds:
        torch.manual_seed(seed + 13000)
        np.random.seed(seed + 13000)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=True)
        train_loader = make_loader(splits[seed]["train"], seed, args, True)
        validation_loader = make_loader(splits[seed]["validation"], seed, args, False)
        adapters = {}
        training[f"seed_{seed}"] = {}
        for layer in args.layers:
            layer_dir = seed_dir / f"block_{layer}"
            layer_dir.mkdir(exist_ok=True)
            adapter, record = train_layer(
                model, train_loader, validation_loader, layer, device, args, layer_dir
            )
            adapters[layer] = adapter
            training[f"seed_{seed}"][str(layer)] = record
        result, arrays = evaluate(
            model, adapters, validation_loader, device, args, seed + 13000
        )
        validation[f"seed_{seed}"] = result
        for name, values in arrays.items():
            outcomes[f"seed_{seed}_{name}"] = values

    aggregate = {}
    for layer in args.layers:
        aggregate[str(layer)] = {}
        for condition in ("clean", "noise", "blur"):
            gains = [
                validation[f"seed_{seed}"]["blocks"][str(layer)][condition]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[str(layer)][condition] = {
                "gains_by_seed": gains,
                "mean_gain": float(np.mean(gains)),
            }
    best_layer = max(
        args.layers,
        key=lambda layer: (
            np.mean([
                aggregate[str(layer)]["noise"]["mean_gain"],
                aggregate[str(layer)]["blur"]["mean_gain"],
            ]),
            aggregate[str(layer)]["clean"]["mean_gain"],
        ),
    )
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "device": str(device),
            "vit_frozen": True,
            "adapter_parameters_per_layer": sum(p.numel() for p in HiddenLinear().parameters()),
            "training_corruptions": ["noise4", "blur4"],
            "adapter_scope": "patch tokens only",
            "imageNetV2_accessed": False,
            "status": "development layer selection; not final evaluation",
        },
        "splits": {str(seed): splits[seed] for seed in args.seeds},
        "training": training,
        "validation": validation,
        "aggregate": aggregate,
        "best_layer": best_layer,
        "limitations": [
            "The same locked development validation partitions are used for layer selection.",
            "A selected winner requires confirmation on data not used for layer selection.",
            "Block 12 is excluded because a patch-only intervention after the final block cannot alter the final CLS logits.",
        ],
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"aggregate": aggregate, "best_layer": best_layer}, indent=2))
    print(f"Saved Experiment 130 to {output_dir}")


if __name__ == "__main__":
    main()
