import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment24_classification_aware_residual_predictor import (
    HiddenCache,
    build_cache,
    evaluate,
    make_image_loader,
    train_variant,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment39_disjoint_adapter_training"
DEFAULT_MANIFEST = PROJECT_ROOT / "configs" / "split_manifest_supervisor_v1.json"


def locked_seed_splits(manifest_path, protocol, seeds):
    manifest = json.loads(manifest_path.read_text())
    records = {
        split["name"]: split
        for split in manifest[protocol]["splits"]
    }
    result = {}
    for seed in seeds:
        train = records[f"adapter{seed}_gate_train"]
        validation = records[f"adapter{seed}_gate_validation"]
        for split in (train, validation):
            if split["dataset"] != "imagenet_val":
                raise ValueError(f"Unexpected adapter dataset: {split}")
            if split["isolation_group"] != "adapter_gate_development":
                raise ValueError(f"Adapter split is outside adapter_gate_development: {split}")
        result[seed] = {"train": train, "validation": validation}
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 39: leakage-free three-seed Block-11 adapter training"
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--initialization", choices=["zero", "random"], default="zero")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--predictor-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

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
    split_records = {}

    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_splits = splits[seed]
        split_records[str(seed)] = seed_splits
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        seed_args.initial_checkpoint = None
        seed_args.preservation_ratio = args.preservation_weight / args.classification_weight

        for name in ("train", "validation"):
            record = seed_splits[name]
            samples = record["end"] - record["start"]
            if args.max_samples is not None:
                samples = min(samples, args.max_samples)
            cache_dir = seed_dir / f"{name}_cache"
            if not (args.resume and (cache_dir / "metadata.json").exists()):
                build_cache(
                    model,
                    make_image_loader(samples, record["start"], seed_args),
                    device,
                    cache_dir,
                )

        train_data = HiddenCache(seed_dir / "train_cache")
        validation_data = HiddenCache(seed_dir / "validation_cache")
        checkpoint = seed_dir / f"classification_weight_{args.classification_weight:g}.pt"
        training_path = seed_dir / "training_record.json"
        if args.resume and checkpoint.exists() and training_path.exists():
            training_record = json.loads(training_path.read_text())
        else:
            predictor, training_record = train_variant(
                model,
                train_data,
                validation_data,
                seed_args,
                device,
                seed_dir,
                args.classification_weight,
            )
            training_path.write_text(json.dumps(training_record, indent=2))
        predictor_state = torch.load(checkpoint, map_location=device, weights_only=True)
        from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
        predictor = HiddenLinear().to(device)
        predictor.load_state_dict(predictor_state)
        predictor.eval()
        name = f"seed_{seed}"
        training[name] = training_record
        validation[name] = evaluate(
            model,
            {name: predictor},
            DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False),
            device,
            {"fixed_configuration": {"predictor": name, "alpha": args.alpha}},
        )

    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "vit_block": 11,
            "frozen_components": ["ViT"],
            "initialization_checkpoint": None,
            "imageNetV2_accessed": False,
            "status": "adapter development only; final evaluation remains untouched",
        },
        "development_splits": split_records,
        "training": training,
        "validation": validation,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"development_splits": split_records, "validation": validation}, indent=2))
    print(f"Saved Experiment 39 to {output_dir}")


if __name__ == "__main__":
    main()
