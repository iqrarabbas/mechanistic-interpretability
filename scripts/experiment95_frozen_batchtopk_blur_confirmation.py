import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment45_sae_discovered_hidden_subspace import SharedRawInputRepair


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
SOURCE = (
    ACTIVE_ROOT / "results/sae/experiment94_batchtopk_blur_hidden_subspace"
    / "full_3seed_rank16_v1"
)
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment95_frozen_batchtopk_blur_confirmation"
BLOCK = 11
WIDTH = 768
PATCHES = 196
RANK = 16
METHODS = (
    "blur_batchtopk_decoder",
    "random_batchtopk_decoder",
    "high_variance_hidden",
    "blur_residual_pca",
)


class Conditions(Dataset):
    def __init__(self, start, samples, seed):
        common = dict(dataset_dir=PROJECT_ROOT / "Dataset", max_samples=samples, start_index=start)
        self.clean = ImageNetDataset(**common)
        self.blur = ImageNetDataset(**common, corruption="blur", blur_severity=4, corruption_seed=seed)
        self.noise = ImageNetDataset(**common, corruption="noise", noise_severity=4, corruption_seed=seed)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        blur, blur_label = self.blur[index]
        noise, noise_label = self.noise[index]
        if label != blur_label or label != noise_label:
            raise RuntimeError("Condition label mismatch")
        return clean, blur, noise, label


def load_methods(seed, device):
    methods = {}
    for name in METHODS:
        checkpoint = SOURCE / f"seed_{seed}/{name}.pt"
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        basis = torch.zeros(WIDTH, RANK)
        method = SharedRawInputRepair(
            basis,
            torch.zeros(WIDTH),
            torch.ones(WIDTH),
            torch.ones(RANK),
        )
        method.load_state_dict(state)
        method = method.to(device).eval()
        method.requires_grad_(False)
        trainable_architecture_parameters = sum(
            parameter.numel() for parameter in method.parameters()
        )
        if trainable_architecture_parameters != 15440:
            raise RuntimeError(
                f"Unexpected parameter count for {name}: {trainable_architecture_parameters}"
            )
        if method.basis.shape != (WIDTH, RANK):
            raise RuntimeError(f"Unexpected basis shape for {name}: {method.basis.shape}")
        methods[name] = method
    return methods


def evaluate(model, methods, loader, device, seed, bootstrap_repetitions):
    stores = {
        condition: {"baseline": [], **{name: [] for name in METHODS}}
        for condition in ("clean", "blur", "noise")
    }
    with torch.no_grad():
        for clean, blur, noise, labels in tqdm(loader, desc=f"Frozen seed {seed}"):
            labels = labels.to(device)
            images = torch.cat((clean, blur, noise)).to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            batch = len(labels)
            logits = outputs.logits.split(batch)
            hidden = outputs.hidden_states[BLOCK].split(batch)
            for condition, baseline_logits, condition_hidden in zip(
                ("clean", "blur", "noise"), logits, hidden
            ):
                stores[condition]["baseline"].extend(
                    (baseline_logits.argmax(1) == labels).cpu().tolist()
                )
                patches = condition_hidden[:, 1:]
                for name, method in methods.items():
                    candidate = torch.cat(
                        (condition_hidden[:, :1], patches + method(patches)), dim=1
                    )
                    candidate_logits = downstream_from_layer(model, candidate, BLOCK - 1)
                    stores[condition][name].extend(
                        (candidate_logits.argmax(1) == labels).cpu().tolist()
                    )
    arrays = {
        condition: {
            name: np.asarray(values, dtype=bool)
            for name, values in condition_stores.items()
        }
        for condition, condition_stores in stores.items()
    }
    result = {}
    for condition_index, condition in enumerate(("clean", "blur", "noise")):
        baseline = arrays[condition]["baseline"]
        result[condition] = {
            "baseline_accuracy": float(baseline.mean()),
            "methods": {
                name: paired_comparison(
                    baseline,
                    arrays[condition][name],
                    seed + 9500 + condition_index * 10 + method_index,
                    bootstrap_repetitions,
                )
                for method_index, name in enumerate(METHODS)
            },
        }
    return result, arrays


def main():
    parser = argparse.ArgumentParser(description="Experiment 95: frozen Experiment-94 confirmation")
    parser.add_argument("--evaluation-start", type=int, default=49000)
    parser.add_argument("--evaluation-samples", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.smoke:
        if args.evaluation_start < 15000 or args.evaluation_start + args.evaluation_samples > 36000:
            raise ValueError("Smoke test must stay inside adapter-development images [15000,36000)")
    elif (args.evaluation_start, args.evaluation_start + args.evaluation_samples) != (49000, 50000):
        raise ValueError("Frozen confirmation is locked to untouched [49000,50000)")
    source_summary = json.loads((SOURCE / "summary.json").read_text())
    frozen = source_summary["configuration"]
    required = {
        "rank": RANK,
        "classification_weight": 0.05,
        "preservation_weight": 0.05,
        "train_alpha": 0.5,
        "alpha": 1.0,
        "learning_rate": 1e-4,
        "trainable_parameters_per_method": 15440,
    }
    for key, expected in required.items():
        if frozen.get(key) != expected:
            raise RuntimeError(f"Frozen configuration mismatch: {key}={frozen.get(key)}")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    results, outcomes = {}, {}
    for seed in args.seeds:
        methods = load_methods(seed, device)
        loader = DataLoader(
            Conditions(args.evaluation_start, args.evaluation_samples, seed),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
        )
        result, arrays = evaluate(
            model, methods, loader, device, seed, args.bootstrap_repetitions
        )
        results[str(seed)] = result
        for condition, condition_arrays in arrays.items():
            for name, values in condition_arrays.items():
                outcomes[f"seed{seed}_{condition}_{name}"] = values
        del methods
        if device.type == "cuda":
            torch.cuda.empty_cache()
    aggregate = {}
    for name in METHODS:
        aggregate[name] = {}
        for condition in ("clean", "blur", "noise"):
            gains = [
                results[str(seed)][condition]["methods"][name]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[name][f"{condition}_gains_by_seed"] = gains
            aggregate[name][f"mean_{condition}_gain"] = float(np.mean(gains))
    summary = {
        "configuration": vars(args) | {
            "source_experiment": str(SOURCE),
            "model": BASE_MODEL,
            "block": BLOCK,
            "severity": 4,
            "frozen_before_evaluation": True,
            "training_or_tuning_performed": False,
            "clean_counterpart_used_at_inference": False,
            "vit_frozen": True,
            "status": "smoke test" if args.smoke else "one-time frozen reserve confirmation",
        },
        "frozen_configuration": required,
        "results": results,
        "aggregate": aggregate,
        "guardrails": [
            "All checkpoints, bases, features, ranks, scales, and optimization choices come unchanged from Experiment 94.",
            "No model selection or threshold tuning occurs in this script.",
            "Clean, Blur-4, and Noise-4 use identical image order within each seed.",
            "Clean images are a separate reported condition, not an inference-time counterpart.",
            "The ViT and repair checkpoints remain frozen.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
