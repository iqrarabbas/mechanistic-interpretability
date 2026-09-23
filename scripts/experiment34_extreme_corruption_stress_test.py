import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment15_independent_frozen_confirmation import ExternalImageNetDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment25_multiseed_independent_confirmation import paired_summary


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment34_extreme_corruption"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
CHECKPOINT_ROOT = (
    PROJECT_ROOT / "results" / "sae" / "experiment25_multiseed_confirmation"
    / "imagenetv2_multiseed_confirmation"
)
BLOCK_INDEX = 10


class ExtremeCorruptionDataset(Dataset):
    def __init__(self, root, corruption, strength, samples, start_index, seed):
        clean = ExternalImageNetDataset(root, 0, samples, start_index, corruption="noise", seed=seed)
        self.samples = clean.samples
        self.processor = clean.processor
        self.corruption = corruption
        self.strength = strength
        self.start_index = start_index
        self.seed = seed

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.corruption == "blur":
            image = image.filter(ImageFilter.GaussianBlur(radius=self.strength))
        else:
            values = np.asarray(image, dtype=np.float32) / 255.0
            generator = np.random.default_rng(self.seed + self.start_index + index)
            values = np.clip(
                values + generator.normal(0.0, self.strength, values.shape),
                0.0,
                1.0,
            )
            image = Image.fromarray((values * 255).astype(np.uint8))
        pixels = self.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, label


def main():
    parser = argparse.ArgumentParser(description="Experiment 34: fixed-adapter extreme corruption stress test")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--blur-radii", type=float, nargs="+", default=[6.0, 8.0, 10.0])
    parser.add_argument("--noise-stds", type=float, nargs="+", default=[0.50, 0.65, 0.80])
    parser.add_argument("--adapter-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
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
    predictors = {}
    for adapter_seed in args.adapter_seeds:
        checkpoint = CHECKPOINT_ROOT / f"seed_{adapter_seed}" / "classification_weight_0.05.pt"
        predictor = HiddenLinear().to(device)
        predictor.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        predictors[adapter_seed] = predictor.eval()

    conditions = [
        (f"blur_radius_{radius:g}", "blur", radius) for radius in args.blur_radii
    ] + [
        (f"noise_std_{std:g}", "noise", std) for std in args.noise_stds
    ]
    results = {}
    for condition_index, (name, corruption, strength) in enumerate(conditions):
        result_path = condition_dir / f"{name}.json"
        arrays_path = condition_dir / f"{name}.npz"
        if args.resume and result_path.exists() and arrays_path.exists():
            results[name] = json.loads(result_path.read_text())
            print(f"Resumed {name}")
            continue
        dataset = ExtremeCorruptionDataset(
            args.image_root,
            corruption,
            strength,
            args.samples,
            args.start_index,
            args.noise_seed,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
        original_correct = []
        stores = {
            seed: {"correct": [], "margin_change": [], "true_change": []}
            for seed in args.adapter_seeds
        }
        with torch.no_grad():
            for images, labels in tqdm(loader, desc=name):
                labels = labels.to(device)
                outputs = model(pixel_values=images.to(device), output_hidden_states=True)
                hidden = outputs.hidden_states[BLOCK_INDEX + 1]
                patches = hidden[:, 1:]
                baseline_logits = outputs.logits
                baseline_true = baseline_logits.gather(1, labels[:, None]).squeeze(1)
                masked = baseline_logits.clone()
                masked.scatter_(1, labels[:, None], float("-inf"))
                baseline_margin = baseline_true - masked.max(1).values
                original_correct.extend((baseline_logits.argmax(1) == labels).cpu().tolist())
                for adapter_seed, predictor in predictors.items():
                    residual = predictor(patches)
                    candidate = torch.cat([hidden[:, :1], patches + residual], dim=1)
                    logits = downstream_from_layer(model, candidate, BLOCK_INDEX)
                    true_logit = logits.gather(1, labels[:, None]).squeeze(1)
                    candidate_masked = logits.clone()
                    candidate_masked.scatter_(1, labels[:, None], float("-inf"))
                    margin = true_logit - candidate_masked.max(1).values
                    stores[adapter_seed]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                    stores[adapter_seed]["margin_change"].extend((margin - baseline_margin).cpu().tolist())
                    stores[adapter_seed]["true_change"].extend((true_logit - baseline_true).cpu().tolist())

        original = np.asarray(original_correct, dtype=bool)
        record = {
            "corruption": corruption,
            "strength": strength,
            "baseline_accuracy": float(original.mean()),
            "adapters": {},
        }
        arrays = {"original_correct": original}
        for adapter_seed, values in stores.items():
            corrected = np.asarray(values["correct"], dtype=bool)
            margin_change = np.asarray(values["margin_change"], dtype=np.float32)
            true_change = np.asarray(values["true_change"], dtype=np.float32)
            record["adapters"][str(adapter_seed)] = paired_summary(
                original,
                corrected,
                margin_change,
                true_change,
                args.noise_seed + condition_index * 10 + adapter_seed,
                args.bootstrap_repetitions,
            )
            arrays[f"seed{adapter_seed}_correct"] = corrected
        result_path.write_text(json.dumps(record, indent=2))
        np.savez_compressed(arrays_path, **arrays)
        results[name] = record

    trends = {}
    for corruption, strengths in [("blur", args.blur_radii), ("noise", args.noise_stds)]:
        names = [
            f"blur_radius_{value:g}" if corruption == "blur" else f"noise_std_{value:g}"
            for value in strengths
        ]
        trends[corruption] = {
            "strengths": strengths,
            "baseline_accuracies": [results[name]["baseline_accuracy"] for name in names],
            "mean_adapter_gains": [
                float(np.mean([
                    results[name]["adapters"][str(seed)]["accuracy_gain"]
                    for seed in args.adapter_seeds
                ]))
                for name in names
            ],
            "all_seed_gains": {
                name: {
                    str(seed): results[name]["adapters"][str(seed)]["accuracy_gain"]
                    for seed in args.adapter_seeds
                }
                for name in names
            },
        }
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "device": str(device),
            "fixed_without_retuning": True,
            "stress_level_status": "explicit parameters beyond the predefined severity-1-to-5 scale",
            "predefined_severity5_reference": {"blur_radius": 5.0, "noise_std": 0.38},
        },
        "conditions": results,
        "trends": trends,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(trends, indent=2))
    print(f"Saved Experiment 34 to {output_dir}")


if __name__ == "__main__":
    main()
