import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment51_block6_blur_transfer import evaluate
from scripts.experiment54_lowrank_block6_adapter import LowRankHiddenAdapter, variant_name


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment55_lowrank_blur_transfer"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment54_lowrank_block6_adapter"
    / "full_3seed_ranks8_128_v1"
)


def make_loader(split, seed, batch_size, workers, max_samples):
    samples = split["end"] - split["start"]
    if max_samples is not None:
        samples = min(samples, max_samples)
    return DataLoader(
        PairedCorruptionDataset("blur", samples, split["start"], seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )


def load_adapter(path, rank, device):
    adapter = LowRankHiddenAdapter(rank).to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval()


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 55: frozen low-rank Noise-trained adapters on Blur-4"
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--ranks", type=int, nargs="+")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    source_summary = json.loads((args.source_run / "summary.json").read_text())
    source_ranks = source_summary["configuration"]["ranks"]
    ranks = source_ranks if args.ranks is None else args.ranks
    if not set(ranks).issubset(source_ranks):
        raise ValueError("Requested rank was not trained in the source run")
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Frozen ranks: {ranks}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    validation, outcomes = {}, {}

    for seed in args.seeds:
        adapters = {
            variant_name(rank): load_adapter(
                args.source_run / f"seed_{seed}" / f"{variant_name(rank)}.pt",
                rank,
                device,
            )
            for rank in ranks
        }
        loader = make_loader(
            splits[seed]["validation"], seed, args.image_batch_size,
            args.num_workers, args.max_samples,
        )
        result, arrays = evaluate(
            model, adapters, loader, device, args.alpha,
            seed + 5500, args.bootstrap_repetitions,
        )
        validation[f"seed_{seed}"] = result
        for name, values in arrays.items():
            outcomes[f"seed_{seed}_{name}"] = values

    aggregate = {}
    for rank in ranks:
        name = variant_name(rank)
        blur = [
            validation[f"seed_{seed}"]["variants"][name]["blur4_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        clean = [
            validation[f"seed_{seed}"]["variants"][name]["clean_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        source = source_summary["aggregate"][name]
        aggregate[name] = {
            "rank": rank,
            "trainable_parameters": source["trainable_parameters"],
            "parameter_reduction_fraction": source["parameter_reduction_fraction"],
            "development_noise4_mean_gain": source["mean_noise4_gain"],
            "blur4_gains_by_seed": blur,
            "mean_blur4_gain": float(np.mean(blur)),
            "clean_gains_by_seed": clean,
            "mean_clean_gain": float(np.mean(clean)),
        }
    summary = {
        "configuration": vars(args) | {
            "source_run": str(args.source_run.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "ranks": ranks,
            "device": str(device),
            "model": BASE_MODEL,
            "model_frozen": True,
            "adapters_frozen": True,
            "adapter_training_corruption": "Noise-4",
            "evaluation_corruption": "Blur-4",
            "blur_used_for_training_or_retuning": False,
            "imageNetV2_accessed": False,
            "status": "development frozen transfer evaluation",
        },
        "development_splits": {str(seed): splits[seed]["validation"] for seed in args.seeds},
        "validation": validation,
        "aggregate": aggregate,
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"aggregate": aggregate}, indent=2))
    print(f"Saved Experiment 55 to {output_dir}")


if __name__ == "__main__":
    main()
