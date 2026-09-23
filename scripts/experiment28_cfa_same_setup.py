import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from interpretability.cfa import (
    calculate_source_statistics,
    cfa_online_step,
    collect_layernorm_parameters,
    load_source_statistics,
)
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment15_independent_frozen_confirmation import ExternalImageNetDataset
from scripts.experiment25_multiseed_independent_confirmation import bootstrap_interval


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment28_cfa_same_setup"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
DEFAULT_STATS = PROJECT_ROOT / "checkpoints" / "cfa" / "hf_vit_imagenet_val_statistics.pt"
OFFICIAL_CFA_COMMIT = "c89c573bb4e3b0fbd1fefa5c3f4b94254d1bf5b7"


LOCKED_OUTCOMES = {
    "clean": PROJECT_ROOT / "results" / "sae" / "experiment27_corruption_transfer"
    / "fixed_noise_adapter_transfer" / "conditions" / "clean.npz",
    "blur4": PROJECT_ROOT / "results" / "sae" / "experiment27_corruption_transfer"
    / "fixed_noise_adapter_transfer" / "conditions" / "blur4.npz",
    "noise4": PROJECT_ROOT / "results" / "sae" / "experiment26_adapter_robustness"
    / "fixed_adapter_robustness" / "conditions" / "noise4_seed2026.npz",
}


def make_source_loader(args):
    dataset = ImageNetDataset(
        PROJECT_ROOT / "Dataset",
        max_samples=args.source_samples,
        start_index=args.source_start,
    )
    if len(dataset) != args.source_samples:
        raise ValueError(f"Requested {args.source_samples} source images, found {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=args.source_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def make_target_loader(args, corruption, severity):
    dataset = ExternalImageNetDataset(
        args.image_root,
        severity,
        args.samples,
        args.start_index,
        corruption=corruption,
        seed=args.noise_seed,
    )
    if len(dataset) != args.samples:
        raise ValueError(f"Requested {args.samples} target images, found {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def paired_summary(original, corrected, seed, repetitions):
    difference = corrected.astype(np.float64) - original.astype(np.float64)
    recovered = int((~original & corrected).sum())
    damaged = int((original & ~corrected).sum())
    return {
        "accuracy": float(corrected.mean()),
        "accuracy_gain": float(difference.mean()),
        "accuracy_gain_95ci": bootstrap_interval(difference, seed, repetitions),
        "predictions_recovered": recovered,
        "originally_correct_damaged": damaged,
        "mcnemar_exact_pvalue": float(binomtest(recovered, recovered + damaged, 0.5).pvalue)
        if recovered + damaged else 1.0,
    }


def locked_arrays(condition, samples):
    path = LOCKED_OUTCOMES[condition]
    if not path.exists():
        raise FileNotFoundError(f"Missing locked outcome file: {path}")
    arrays = np.load(path)
    return {
        "original": arrays["original_correct"][:samples].astype(bool),
        "adapter": arrays["adapter_correct"][:samples].astype(bool),
    }


def run_cfa_loader(args, loader, statistics, device, description):
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    parameters, parameter_names = collect_layernorm_parameters(model)
    optimizer = torch.optim.SGD(
        parameters,
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    correct = []
    losses = []
    global_losses = []
    class_losses = []
    for images, labels in tqdm(loader, desc=description):
        labels = labels.to(device)
        logits, loss, parts = cfa_online_step(
            model,
            optimizer,
            images.to(device),
            statistics,
            full_max_moment=args.full_max_moment,
            class_max_moment=args.class_max_moment,
            max_grad_norm=args.max_grad_norm,
        )
        correct.extend((logits.argmax(1) == labels).cpu().tolist())
        losses.append(loss)
        global_losses.append(parts["global_loss"])
        class_losses.append(parts["class_loss"])
    return np.asarray(correct, dtype=bool), {
        "mean_cfa_loss": float(np.mean(losses)),
        "mean_global_loss": float(np.mean(global_losses)),
        "mean_class_loss": float(np.mean(class_losses)),
        "adapted_parameter_count": sum(parameter.numel() for parameter in parameters),
        "adapted_parameter_names": parameter_names,
    }


def run_cfa_condition(args, condition, statistics, device):
    corruption = "noise" if condition in {"clean", "noise4"} else "blur"
    severity = 0 if condition == "clean" else 4
    return run_cfa_loader(
        args,
        make_target_loader(args, corruption, severity),
        statistics,
        device,
        f"CFA {condition}",
    )


def load_sae_references():
    blur = json.loads((
        PROJECT_ROOT / "results" / "sae" / "experiment15_independent_confirmation"
        / "imagenetv2_independent_confirmation_gpu" / "summary.json"
    ).read_text())
    noise = json.loads((
        PROJECT_ROOT / "results" / "sae" / "experiment16_frozen_noise_confirmation"
        / "imagenetv2_frozen_noise_gpu" / "summary.json"
    ).read_text())
    return {
        "clean": blur["evaluation"]["clean"]["frozen_targeted"],
        "blur4": blur["evaluation"]["blur4"]["frozen_targeted"],
        "noise4": noise["evaluation"]["noise4"]["frozen_targeted"],
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 28: CFA versus fixed adapter on identical ImageNetV2 streams")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--source-samples", type=int, default=50000)
    parser.add_argument("--source-start", type=int, default=0)
    parser.add_argument("--source-batch-size", type=int, default=16)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--conditions", nargs="+", default=["clean", "blur4", "noise4"])
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--full-max-moment", type=int, default=3)
    parser.add_argument("--class-max-moment", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    args.statistics.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not args.statistics.exists():
        source_model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
        source_model.requires_grad_(False)
        calculate_source_statistics(
            source_model,
            make_source_loader(args),
            device,
            args.statistics,
            max_moment=args.full_max_moment,
        )
        del source_model
    statistics = load_source_statistics(args.statistics, device)
    cfa_results = {}
    comparison = {}
    for index, condition in enumerate(args.conditions):
        condition_path = output_dir / f"{condition}.json"
        outcomes_path = output_dir / f"{condition}_outcomes.npz"
        locked = locked_arrays(condition, args.samples)
        if args.resume and condition_path.exists() and outcomes_path.exists():
            record = json.loads(condition_path.read_text())
            cfa_results[condition] = record["cfa"]
            comparison[condition] = record["comparison"]
            continue
        cfa_correct, diagnostics = run_cfa_condition(args, condition, statistics, device)
        cfa = paired_summary(
            locked["original"], cfa_correct,
            args.noise_seed + index, args.bootstrap_repetitions,
        ) | diagnostics
        adapter = paired_summary(
            locked["original"], locked["adapter"],
            args.noise_seed + 100 + index, args.bootstrap_repetitions,
        )
        record = {
            "cfa": cfa,
            "comparison": {
                "baseline_accuracy": float(locked["original"].mean()),
                "fixed_adapter": adapter,
                "cfa": cfa,
                "winner_by_accuracy": "cfa" if cfa["accuracy"] > adapter["accuracy"] else "fixed_adapter",
            },
        }
        condition_path.write_text(json.dumps(record, indent=2))
        np.savez_compressed(
            outcomes_path,
            original_correct=locked["original"],
            adapter_correct=locked["adapter"],
            cfa_correct=cfa_correct,
        )
        cfa_results[condition] = cfa
        comparison[condition] = record["comparison"]
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "statistics": str(args.statistics.resolve()),
            "device": str(device),
            "official_cfa_repository_commit": OFFICIAL_CFA_COMMIT,
            "condition_reset": True,
            "stream_order_fixed": True,
        },
        "protocol_status": {
            "algorithm": "CFA equations and official defaults ported to Hugging Face ViT",
            "deviations": [
                "source moments use labeled ImageNet validation because ImageNet train is unavailable",
                "target batch size matches the existing adapter evaluation rather than official batch size 64",
                "backbone is google/vit-base-patch16-224 rather than the official timm checkpoint",
            ],
            "claim": "same-setup CFA comparison, not reproduction of the official ImageNet-C table",
        },
        "source_statistics": {
            "samples": statistics["source_samples"],
            "classes_with_samples": int((statistics["class_counts"] > 0).sum()),
        },
        "cfa": cfa_results,
        "same_setup_comparison": comparison,
        "locked_sae_references": load_sae_references(),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "protocol_status": summary["protocol_status"],
        "same_setup_comparison": comparison,
        "locked_sae_references": summary["locked_sae_references"],
    }, indent=2))
    print(f"Saved Experiment 28 to {output_dir}")


if __name__ == "__main__":
    main()
