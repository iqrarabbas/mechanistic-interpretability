import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from corruption.gaussian_blur import apply_gaussian_blur
from corruption.gaussian_noise import apply_gaussian_noise
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment15_independent_frozen_confirmation import ExternalImageNetDataset
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment26_adapter_robustness_ablations import (
    DEFAULT_CHECKPOINT,
    evaluate_ablations,
    evaluate_condition,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment27_corruption_transfer"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
BLUR_REFERENCE_PATH = (
    PROJECT_ROOT / "results" / "sae" / "experiment15_independent_confirmation"
    / "imagenetv2_independent_confirmation_gpu" / "summary.json"
)


class MixedCorruptionDataset(Dataset):
    def __init__(self, root, severity, samples, start_index, noise_seed):
        clean = ExternalImageNetDataset(root, 0, samples, start_index)
        self.samples = clean.samples
        self.processor = clean.processor
        self.severity = severity
        self.start_index = start_index
        self.noise_seed = noise_seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        image = apply_gaussian_blur(image, severity=self.severity)
        image = apply_gaussian_noise(
            image,
            severity=self.severity,
            seed=self.noise_seed + self.start_index + index,
        )
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, label


def make_loader(args, corruption, severity):
    if corruption == "mixed":
        dataset = MixedCorruptionDataset(
            args.image_root, severity, args.samples, args.start_index, args.noise_seed
        )
    else:
        dataset = ExternalImageNetDataset(
            args.image_root,
            severity,
            args.samples,
            args.start_index,
            corruption=corruption,
            seed=args.noise_seed,
        )
    if len(dataset) != args.samples:
        raise ValueError(f"Requested {args.samples} images, found {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def locked_blur_reference():
    summary = json.loads(BLUR_REFERENCE_PATH.read_text())
    return {
        "source": str(BLUR_REFERENCE_PATH),
        "method": "frozen targeted clean-SAE correction from Experiment 15",
        "clean": summary["evaluation"]["clean"]["frozen_targeted"],
        "blur4": summary["evaluation"]["blur4"]["frozen_targeted"],
        "configuration": {
            key: summary["configuration"][key]
            for key in ["samples", "start_index", "features", "patch_count", "alpha", "quantile"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 27: fixed Noise-adapter corruption transfer")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--control-severity", type=int, default=4)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
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
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    predictor = HiddenLinear().to(device)
    predictor.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    predictor.eval()

    conditions = [("clean", "noise", 0)] + [
        (f"{corruption}{severity}", corruption, severity)
        for corruption in ["blur", "noise", "mixed"]
        for severity in args.severities
    ]
    results = {}
    for index, (name, corruption, severity) in enumerate(conditions):
        result_path = condition_dir / f"{name}.json"
        arrays_path = condition_dir / f"{name}.npz"
        if args.resume and result_path.exists() and arrays_path.exists():
            results[name] = json.loads(result_path.read_text())
            continue
        result, arrays = evaluate_condition(
            model,
            predictor,
            make_loader(args, corruption, severity),
            device,
            args.alpha,
            args.noise_seed + index,
            args.bootstrap_repetitions,
        )
        result |= {
            "corruption": "clean" if severity == 0 else corruption,
            "severity": severity,
            "mixture_order": "blur_then_noise" if corruption == "mixed" else None,
        }
        result_path.write_text(json.dumps(result, indent=2))
        np.savez_compressed(arrays_path, **arrays)
        results[name] = result

    controls = {}
    for offset, corruption in enumerate(["blur", "noise", "mixed"]):
        control_path = output_dir / f"{corruption}{args.control_severity}_controls.json"
        if args.resume and control_path.exists():
            controls[corruption] = json.loads(control_path.read_text())
            continue
        controls[corruption] = evaluate_ablations(
            model,
            predictor,
            make_loader(args, corruption, args.control_severity),
            device,
            args.alpha,
            args.noise_seed + offset,
            args.bootstrap_repetitions,
            [11],
        )
        control_path.write_text(json.dumps(controls[corruption], indent=2))

    transfer_summary = {}
    for corruption in ["blur", "noise", "mixed"]:
        gains = [results[f"{corruption}{severity}"]["adapter"]["accuracy_gain"] for severity in args.severities]
        transfer_summary[corruption] = {
            "gains_by_severity": {
                str(severity): gain for severity, gain in zip(args.severities, gains)
            },
            "all_gains_positive": bool(np.all(np.asarray(gains) > 0)),
            "mean_gain": float(np.mean(gains)),
        }
    blur_reference = locked_blur_reference()
    blur4_comparison = {
        "noise_adapter_gain": results["blur4"]["adapter"]["accuracy_gain"],
        "blur_sae_rule_gain": blur_reference["blur4"]["accuracy_gain_vs_original"],
        "better_specialized_method": (
            "noise_adapter"
            if results["blur4"]["adapter"]["accuracy_gain"]
            > blur_reference["blur4"]["accuracy_gain_vs_original"]
            else "blur_sae_rule"
        ),
    }
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "device": str(device),
            "fixed_without_retuning": True,
            "adapter_training_corruption": "Noise-4",
            "mixture_definition": "Gaussian blur followed by Gaussian noise at the same severity",
        },
        "condition_results": results,
        "transfer_summary": transfer_summary,
        "severity4_controls": controls,
        "locked_blur_sae_reference": blur_reference,
        "blur4_specialization_comparison": blur4_comparison,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "clean": results["clean"],
        "transfer_summary": transfer_summary,
        "blur4_specialization_comparison": blur4_comparison,
        "severity4_controls": controls,
    }, indent=2))
    print(f"Saved Experiment 27 to {output_dir}")


if __name__ == "__main__":
    main()
