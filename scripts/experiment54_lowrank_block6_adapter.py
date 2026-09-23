import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment46_layer_specific_adapters import make_loader
from scripts.experiment50_block6_clean_preservation import evaluate, run_epoch


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment54_lowrank_block6_adapter"
FULL_REFERENCE = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment50_block6_clean_preservation"
    / "full_3seed_identity_sweep_v1"
    / "summary.json"
)
WIDTH = 768
PATCHES = 196
IDENTITY_WEIGHT = 0.2


class LowRankHiddenAdapter(nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(WIDTH, rank)
        self.position = nn.Embedding(PATCHES, rank)
        self.up = nn.Linear(rank, WIDTH)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, patches):
        positions = torch.arange(PATCHES, device=patches.device)
        return self.up(self.down(patches) + self.position(positions)[None])


def variant_name(rank):
    return f"rank_{rank}"


def parameter_count(rank):
    return sum(parameter.numel() for parameter in LowRankHiddenAdapter(rank).parameters())


def train_variants(model, ranks, train_loader, validation_loader, device, args, seed_dir):
    adapters = {
        variant_name(rank): (IDENTITY_WEIGHT, LowRankHiddenAdapter(rank).to(device))
        for rank in ranks
    }
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


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 54: low-rank identity-preserving Block-6 adapter sweep"
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--full-reference", type=Path, default=FULL_REFERENCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ranks", type=int, nargs="+", default=[8, 16, 32, 64, 128])
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
    parser.add_argument("--retained-gain-fraction", type=float, default=0.9)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if len(set(args.ranks)) != len(args.ranks) or any(rank <= 0 for rank in args.ranks):
        raise ValueError("Ranks must be unique positive integers")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    reference = json.loads(args.full_reference.read_text())
    reference_gain = reference["aggregate"]["identity_0p2"]["mean_noise4_gain"]
    target_gain = args.retained_gain_fraction * reference_gain
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Ranks: {args.ranks}; target Noise-4 gain: {100 * target_gain:.3f} pp")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    training, validation, outcomes = {}, {}, {}

    for seed in args.seeds:
        torch.manual_seed(seed + 5400)
        np.random.seed(seed + 5400)
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
            model, args.ranks, train_loader, validation_loader, device, args, seed_dir
        )
        training[f"seed_{seed}"] = record
        result, arrays = evaluate(model, adapters, validation_loader, device, args, seed + 5400)
        validation[f"seed_{seed}"] = result
        for name, values in arrays.items():
            outcomes[f"seed_{seed}_{name}"] = values

    aggregate = {}
    for rank in args.ranks:
        name = variant_name(rank)
        noise = [
            validation[f"seed_{seed}"]["variants"][name]["noise_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        clean = [
            validation[f"seed_{seed}"]["variants"][name]["clean_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        parameters = parameter_count(rank)
        aggregate[name] = {
            "rank": rank,
            "trainable_parameters": parameters,
            "parameter_reduction_fraction": 1 - parameters / reference["configuration"]["adapter_trainable_parameters"],
            "noise4_gains_by_seed": noise,
            "mean_noise4_gain": float(np.mean(noise)),
            "retained_full_gain_fraction": float(np.mean(noise) / reference_gain),
            "clean_gains_by_seed": clean,
            "mean_clean_gain": float(np.mean(clean)),
            "eligible_clean_tolerance": float(np.mean(clean)) >= -args.clean_loss_tolerance,
            "eligible_retained_gain": float(np.mean(noise)) >= target_gain,
        }
    eligible = [
        name for name, result in aggregate.items()
        if result["eligible_clean_tolerance"] and result["eligible_retained_gain"]
    ]
    selected = min(eligible, key=lambda name: aggregate[name]["trainable_parameters"]) if eligible else None
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "full_reference": str(args.full_reference.resolve()),
            "device": str(device),
            "block": 6,
            "model": BASE_MODEL,
            "frozen_components": ["ViT"],
            "identity_weight": IDENTITY_WEIGHT,
            "full_adapter_parameters": reference["configuration"]["adapter_trainable_parameters"],
            "full_adapter_reference_mean_noise4_gain": reference_gain,
            "imageNetV2_accessed": False,
            "status": "development architecture sweep",
        },
        "development_splits": {str(seed): splits[seed] for seed in args.seeds},
        "training": training,
        "validation": validation,
        "aggregate": aggregate,
        "selection_rule": (
            "Smallest adapter retaining at least "
            f"{100 * args.retained_gain_fraction:.1f}% of the full adapter mean Noise-4 gain "
            f"with mean clean loss no worse than {-100 * args.clean_loss_tolerance:.2f} pp"
        ),
        "selected_variant": selected,
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"aggregate": aggregate, "selected_variant": selected}, indent=2))
    print(f"Saved Experiment 54 to {output_dir}")


if __name__ == "__main__":
    main()
