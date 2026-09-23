import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, ViTForImageClassification

from interpretability.cfa import calculate_source_statistics, load_source_statistics
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment26_adapter_robustness_ablations import (
    DEFAULT_CHECKPOINT,
    evaluate_condition,
)
from scripts.experiment28_cfa_same_setup import (
    OFFICIAL_CFA_COMMIT,
    paired_summary,
    run_cfa_loader,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment29_imagenet_c"
DEFAULT_STATS = PROJECT_ROOT / "checkpoints" / "cfa" / "hf_vit_imagenet_train_statistics.pt"
CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]


class ProcessedImageFolder(Dataset):
    def __init__(self, root, max_samples=None, start_index=0):
        folder = ImageFolder(root)
        if len(folder.classes) != 1000:
            raise ValueError(f"Expected 1000 ImageNet classes at {root}, found {len(folder.classes)}")
        end = None if max_samples is None else start_index + max_samples
        self.samples = folder.samples[start_index:end]
        self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, label


def make_loader(root, samples, start_index, batch_size, workers):
    dataset = ProcessedImageFolder(root, samples, start_index)
    if samples is not None and len(dataset) != samples:
        raise ValueError(f"Requested {samples} images at {root}, found {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )


def condition_loader(args, corruption, severity):
    root = args.imagenet_c_root / corruption / str(severity)
    if not root.exists():
        raise FileNotFoundError(
            f"Missing ImageNet-C condition {root}. Expected <root>/<corruption>/<severity>/<class>/..."
        )
    return make_loader(
        root, args.samples_per_condition, args.start_index,
        args.batch_size, args.num_workers,
    )


def main():
    parser = argparse.ArgumentParser(description="Experiment 29: fixed adapter and CFA on official ImageNet-C")
    parser.add_argument("--imagenet-c-root", type=Path, required=True)
    parser.add_argument("--imagenet-train-root", type=Path)
    parser.add_argument("--statistics", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--corruptions", nargs="+", default=CORRUPTIONS)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--samples-per-condition", type=int, default=50000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--source-samples", type=int, default=1281167)
    parser.add_argument("--source-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--full-max-moment", type=int, default=3)
    parser.add_argument("--class-max-moment", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-cfa", action="store_true")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    condition_dir = output_dir / "conditions"
    condition_dir.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not args.imagenet_c_root.exists():
        raise FileNotFoundError(f"ImageNet-C root does not exist: {args.imagenet_c_root}")

    base_model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    base_model.requires_grad_(False)
    predictor = HiddenLinear().to(device)
    predictor.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    predictor.eval()

    statistics = None
    if not args.skip_cfa:
        args.statistics.parent.mkdir(parents=True, exist_ok=True)
        if not args.statistics.exists():
            if args.imagenet_train_root is None:
                raise FileNotFoundError(
                    "CFA source statistics are missing. Supply --imagenet-train-root or use --skip-cfa."
                )
            source_loader = make_loader(
                args.imagenet_train_root,
                args.source_samples,
                0,
                args.source_batch_size,
                args.num_workers,
            )
            calculate_source_statistics(
                base_model,
                source_loader,
                device,
                args.statistics,
                max_moment=args.full_max_moment,
            )
        statistics = load_source_statistics(args.statistics, device)

    results = {}
    for corruption_index, corruption in enumerate(args.corruptions):
        results[corruption] = {}
        for severity in args.severities:
            name = f"{corruption}_severity{severity}"
            result_path = condition_dir / f"{name}.json"
            arrays_path = condition_dir / f"{name}.npz"
            if args.resume and result_path.exists() and arrays_path.exists():
                results[corruption][str(severity)] = json.loads(result_path.read_text())
                continue
            loader = condition_loader(args, corruption, severity)
            adapter_result, adapter_arrays = evaluate_condition(
                base_model,
                predictor,
                loader,
                device,
                args.alpha,
                corruption_index * 10 + severity,
                args.bootstrap_repetitions,
            )
            record = {
                "baseline_accuracy": adapter_result["original_accuracy"],
                "fixed_adapter": adapter_result["adapter"],
            }
            arrays = {
                "original_correct": adapter_arrays["original_correct"],
                "adapter_correct": adapter_arrays["adapter_correct"],
            }
            if statistics is not None:
                cfa_loader = condition_loader(args, corruption, severity)
                cfa_correct, diagnostics = run_cfa_loader(
                    args,
                    cfa_loader,
                    statistics,
                    device,
                    f"CFA {name}",
                )
                record["cfa"] = paired_summary(
                    arrays["original_correct"],
                    cfa_correct,
                    corruption_index * 100 + severity,
                    args.bootstrap_repetitions,
                ) | diagnostics
                arrays["cfa_correct"] = cfa_correct
            result_path.write_text(json.dumps(record, indent=2))
            np.savez_compressed(arrays_path, **arrays)
            results[corruption][str(severity)] = record

    method_names = ["fixed_adapter"] + ([] if args.skip_cfa else ["cfa"])
    aggregate = {}
    for method in method_names:
        accuracies = []
        gains = []
        per_corruption = {}
        for corruption in args.corruptions:
            corruption_accuracies = [
                results[corruption][str(severity)][method]["accuracy"]
                for severity in args.severities
            ]
            corruption_gains = [
                results[corruption][str(severity)][method]["accuracy_gain"]
                for severity in args.severities
            ]
            accuracies.extend(corruption_accuracies)
            gains.extend(corruption_gains)
            per_corruption[corruption] = {
                "mean_accuracy": float(np.mean(corruption_accuracies)),
                "mean_error": float(1.0 - np.mean(corruption_accuracies)),
                "mean_accuracy_gain": float(np.mean(corruption_gains)),
            }
        aggregate[method] = {
            "mean_corruption_accuracy": float(np.mean(accuracies)),
            "mean_corruption_error": float(1.0 - np.mean(accuracies)),
            "mean_accuracy_gain": float(np.mean(gains)),
            "per_corruption": per_corruption,
        }
    baseline_accuracies = [
        results[corruption][str(severity)]["baseline_accuracy"]
        for corruption in args.corruptions for severity in args.severities
    ]
    aggregate["baseline"] = {
        "mean_corruption_accuracy": float(np.mean(baseline_accuracies)),
        "mean_corruption_error": float(1.0 - np.mean(baseline_accuracies)),
    }
    summary = {
        "configuration": vars(args) | {
            "imagenet_c_root": str(args.imagenet_c_root.resolve()),
            "imagenet_train_root": str(args.imagenet_train_root.resolve())
            if args.imagenet_train_root else None,
            "statistics": str(args.statistics.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "device": str(device),
            "official_cfa_repository_commit": OFFICIAL_CFA_COMMIT,
            "cfa_reset_between_conditions": True,
        },
        "protocol": {
            "backbone": BASE_MODEL,
            "same_samples_order_and_batch_size_for_all_methods": True,
            "adapter_fixed_without_retuning": True,
            "cfa_online_updates": not args.skip_cfa,
        },
        "conditions": results,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"protocol": summary["protocol"], "aggregate": aggregate}, indent=2))
    print(f"Saved Experiment 29 to {output_dir}")


if __name__ == "__main__":
    main()
