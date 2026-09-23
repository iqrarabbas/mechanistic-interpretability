import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from transformers import AutoImageProcessor, ViTForImageClassification

from corruption.gaussian_blur import apply_gaussian_blur
from corruption.gaussian_noise import apply_gaussian_noise
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae, make_dataset
from scripts.experiment11_quantile_sae_repair import calibrate_quantiles
from scripts.experiment14_patch_budget_pareto import evaluate, pareto_front


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment15_independent_confirmation"
EXPERIMENT12_SUMMARY = PROJECT_ROOT / "results" / "sae" / "experiment12_failure_targeted_repair" / "full_failure_targeted_residual_gpu" / "summary.json"
EXPERIMENT13_SUMMARY = PROJECT_ROOT / "results" / "sae" / "experiment13_patch_aware_repair" / "full_patch_aware_residual_gpu" / "summary.json"


class ExternalImageNetDataset(Dataset):
    def __init__(self, root, severity, max_samples=None, start_index=0, corruption="blur", seed=0):
        root = Path(root)
        class_names = [path.name for path in root.iterdir() if path.is_dir()]
        if len(class_names) != 1000:
            raise ValueError(f"Expected 1000 ImageNet class folders, found {len(class_names)}")
        if all(name.isdigit() for name in class_names):
            if {int(name) for name in class_names} != set(range(1000)):
                raise ValueError("Numeric ImageNetV2 folders must cover labels 0 through 999")
            samples = []
            for name in sorted(class_names, key=int):
                for path in sorted((root / name).iterdir()):
                    if path.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                        samples.append((str(path), int(name)))
            self.label_format = "numeric_imagenetv2"
        elif all(name.startswith("n") for name in class_names):
            image_folder = ImageFolder(root)
            samples = image_folder.samples
            self.label_format = "synset_imagefolder"
        else:
            raise ValueError("Class folders must be ImageNet synsets or numeric ImageNetV2 labels 0-999")
        end = None if max_samples is None else start_index + max_samples
        self.samples = samples[start_index:end]
        self.severity = severity
        self.corruption = corruption
        self.seed = seed
        self.start_index = start_index
        self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.severity and self.corruption == "blur":
            image = apply_gaussian_blur(image, severity=self.severity)
        elif self.severity and self.corruption == "noise":
            image = apply_gaussian_noise(
                image, severity=self.severity, seed=self.seed + self.start_index + index
            )
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, label


def main():
    parser = argparse.ArgumentParser(description="Experiment 15: frozen independent confirmation")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)

    experiment12 = json.loads(EXPERIMENT12_SUMMARY.read_text())
    experiment13 = json.loads(EXPERIMENT13_SUMMARY.read_text())
    features = experiment13["selected"]["features"]
    blur_scores = np.asarray(experiment12["discoveries"]["blur"]["score"])
    noise_scores = np.asarray(experiment12["discoveries"]["noise"]["score"])
    combined_scores = blur_scores + noise_scores
    weights = torch.as_tensor(combined_scores[features], device=device).clamp_min(0)
    weights = weights / weights.mean().clamp_min(1e-8)

    calibration_loader = DataLoader(
        make_dataset(("clean", None, 0), 2000, 30000, 0), batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    calibration = calibrate_quantiles(model, sae, calibration_loader, device, 4, [0.99], 0)
    threshold = calibration["quantiles"]["0.99"].to(device)
    generator = np.random.default_rng(args.seed)
    random_features = generator.choice(sae.latent_dim, 64, replace=False).tolist()
    random_weights = torch.ones(64, device=device)
    frozen = {
        "features": features,
        "weights": weights,
        "alpha": 1.0,
        "patch_count": 32,
        "strategy": "top",
    }
    configurations = {
        "frozen_targeted": frozen,
        "residual_identity": frozen | {"alpha": 0.0},
        "random_feature_control": frozen | {"features": random_features, "weights": random_weights},
        "random_patch_control": frozen | {"strategy": "random"},
        "low_score_patch_control": frozen | {"strategy": "low"},
        "all_patch_control": frozen | {"strategy": "all", "patch_count": 196},
    }

    evaluation = {}
    for severity in range(6):
        condition = "clean" if severity == 0 else f"blur{severity}"
        dataset = ExternalImageNetDataset(args.image_root, severity, args.samples, args.start_index)
        if len(dataset) != args.samples:
            raise ValueError(f"Requested {args.samples} images, found {len(dataset)}")
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        evaluation[condition], arrays = evaluate(model, sae, loader, device, threshold, configurations)
        np.savez_compressed(output_dir / f"{condition}_paired_outcomes.npz", **arrays)

    blur4 = evaluation["blur4"]["frozen_targeted"]
    clean = evaluation["clean"]["frozen_targeted"]
    success = {
        "positive_blur4_gain": blur4["accuracy_gain_vs_original"] > 0,
        "blur4_ci_excludes_zero": blur4["accuracy_gain_95ci"][0] > 0,
        "blur4_mcnemar_below_0.05": blur4["mcnemar_exact_pvalue"] < 0.05,
        "clean_loss_within_0.1pp": clean["accuracy_gain_vs_original"] >= -0.001,
        "beats_random_features": blur4["accuracy"] > evaluation["blur4"]["random_feature_control"]["accuracy"],
        "beats_random_patches": blur4["accuracy"] > evaluation["blur4"]["random_patch_control"]["accuracy"],
    }
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "device": str(device),
            "frozen_without_retuning": True,
            "features": features,
            "patch_count": 32,
            "alpha": 1.0,
            "quantile": 0.99,
        },
        "evaluation": evaluation,
        "blur4_control_pareto": pareto_front(evaluation["blur4"]),
        "preregistered_success_criteria": success,
        "all_success_criteria_met": all(success.values()),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"success": success, "clean": evaluation["clean"], "blur4": evaluation["blur4"]}, indent=2))
    print(f"Saved Experiment 15 to {output_dir}")


if __name__ == "__main__":
    main()
