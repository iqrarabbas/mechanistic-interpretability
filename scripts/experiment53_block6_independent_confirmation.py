import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment15_independent_frozen_confirmation import ExternalImageNetDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment41_disjoint_gate_development import paired_comparison


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment53_block6_independent_confirmation"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment50_block6_clean_preservation"
    / "full_3seed_identity_sweep_v1"
)
BLOCK = 6


class CleanNoiseBlurDataset(Dataset):
    def __init__(self, root, samples, start_index, corruption_seed):
        self.clean = ExternalImageNetDataset(root, 0, samples, start_index)
        self.noise = ExternalImageNetDataset(
            root, 4, samples, start_index, corruption="noise", seed=corruption_seed
        )
        self.blur = ExternalImageNetDataset(
            root, 4, samples, start_index, corruption="blur", seed=corruption_seed
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, clean_label = self.clean[index]
        noise, noise_label = self.noise[index]
        blur, blur_label = self.blur[index]
        if clean_label != noise_label or clean_label != blur_label:
            raise RuntimeError("Paired labels differ")
        return clean, noise, blur, clean_label


def load_adapter(path, device):
    adapter = HiddenLinear().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval()


def evaluate(model, adapters, loader, device, alpha, bootstrap_repetitions, seed):
    conditions = ["clean", "noise4", "blur4"]
    arrays = {f"baseline_{condition}": [] for condition in conditions}
    for adapter_name in adapters:
        for condition in conditions:
            arrays[f"{adapter_name}_{condition}"] = []

    with torch.no_grad():
        for clean, noise, blur, labels in tqdm(loader, desc="Independent Block-6 confirmation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise, blur]).to(device),
                output_hidden_states=True,
            )
            logits = outputs.logits.split(batch)
            hidden = outputs.hidden_states[BLOCK].split(batch)
            for condition, condition_logits in zip(conditions, logits):
                arrays[f"baseline_{condition}"].extend(
                    (condition_logits.argmax(1) == labels).cpu().tolist()
                )
            for adapter_name, adapter in adapters.items():
                for condition, condition_hidden in zip(conditions, hidden):
                    patches = condition_hidden[:, 1:]
                    corrected = torch.cat(
                        [condition_hidden[:, :1], patches + alpha * adapter(patches)], dim=1
                    )
                    corrected_logits = downstream_from_layer(model, corrected, BLOCK - 1)
                    arrays[f"{adapter_name}_{condition}"].extend(
                        (corrected_logits.argmax(1) == labels).cpu().tolist()
                    )

    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    baseline = {
        condition: float(arrays[f"baseline_{condition}"].mean())
        for condition in conditions
    }
    results = {}
    for adapter_offset, adapter_name in enumerate(adapters):
        results[adapter_name] = {}
        for condition_offset, condition in enumerate(conditions):
            results[adapter_name][condition] = paired_comparison(
                arrays[f"baseline_{condition}"],
                arrays[f"{adapter_name}_{condition}"],
                seed + adapter_offset * 100 + condition_offset,
                bootstrap_repetitions,
            )
    return baseline, results, arrays


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 53: frozen Block-6 independent clean/Noise-4/Blur-4 confirmation"
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--selected-variant", default="identity_0p2")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--corruption-seed", type=int, default=2026)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    source_summary = json.loads((args.source_run / "summary.json").read_text())
    if source_summary["selected_variant"] != args.selected_variant:
        raise ValueError("Requested variant does not match the frozen Experiment 50 winner")

    dataset = CleanNoiseBlurDataset(
        args.image_root, args.samples, args.start_index, args.corruption_seed
    )
    if len(dataset) != args.samples:
        raise ValueError(f"Requested {args.samples} images, found {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.image_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    adapters = {
        f"seed_{seed}": load_adapter(
            args.source_run / f"seed_{seed}" / f"{args.selected_variant}.pt", device
        )
        for seed in args.seeds
    }
    baseline, evaluation, arrays = evaluate(
        model,
        adapters,
        loader,
        device,
        args.alpha,
        args.bootstrap_repetitions,
        args.corruption_seed + 5300,
    )
    np.savez_compressed(output_dir / "paired_outcomes.npz", **arrays)

    aggregate = {}
    for condition in ["clean", "noise4", "blur4"]:
        gains = [evaluation[f"seed_{seed}"][condition]["accuracy_difference"] for seed in args.seeds]
        aggregate[condition] = {
            "baseline_accuracy": baseline[condition],
            "gains_by_seed": gains,
            "mean_gain": float(np.mean(gains)),
            "all_seed_gains_positive": bool(np.all(np.asarray(gains) > 0)),
        }
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "source_run": str(args.source_run.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "vit_frozen": True,
            "adapters_frozen": True,
            "adapter_training_corruption": "Noise-4",
            "evaluation_dataset": "ImageNetV2 matched-frequency",
            "evaluation_used_for_selection_or_tuning": False,
            "deployment_uses_clean_counterpart": False,
            "status": "independent frozen confirmation",
        },
        "baseline": baseline,
        "per_adapter": evaluation,
        "aggregate": aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"baseline": baseline, "aggregate": aggregate}, indent=2))
    print(f"Saved Experiment 53 to {output_dir}")


if __name__ == "__main__":
    main()
