import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment15_independent_frozen_confirmation import ExternalImageNetDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import (
    DEFAULT_INITIAL_CHECKPOINT,
    HiddenCache,
    HiddenLinear,
    build_cache,
    evaluate,
    make_image_loader,
    train_variant,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment25_multiseed_confirmation"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
BLOCK_INDEX = 10


class PairedExternalDataset(Dataset):
    def __init__(self, root, samples, start_index, noise_seed):
        self.clean = ExternalImageNetDataset(
            root, 0, samples, start_index, corruption="noise", seed=noise_seed
        )
        self.noise = ExternalImageNetDataset(
            root, 4, samples, start_index, corruption="noise", seed=noise_seed
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, clean_label = self.clean[index]
        noise, noise_label = self.noise[index]
        if clean_label != noise_label:
            raise RuntimeError("Paired ImageNetV2 labels differ")
        return clean, noise, clean_label


def bootstrap_interval(differences, seed, repetitions):
    generator = np.random.default_rng(seed)
    means = np.empty(repetitions, dtype=np.float64)
    chunk = 250
    for start in range(0, repetitions, chunk):
        count = min(chunk, repetitions - start)
        indices = generator.integers(0, differences.size, size=(count, differences.size))
        means[start:start + count] = differences[indices].mean(1)
    return [float(value) for value in np.quantile(means, [0.025, 0.975])]


def paired_summary(original_correct, correct, margin_change, true_logit_change, seed, repetitions):
    differences = correct.astype(np.float64) - original_correct.astype(np.float64)
    recovered = int((~original_correct & correct).sum())
    damaged = int((original_correct & ~correct).sum())
    return {
        "accuracy": float(correct.mean()),
        "accuracy_gain": float(differences.mean()),
        "accuracy_gain_95ci": bootstrap_interval(differences, seed, repetitions),
        "mean_margin_change": float(np.mean(margin_change)),
        "mean_true_logit_change": float(np.mean(true_logit_change)),
        "predictions_recovered": recovered,
        "originally_correct_damaged": damaged,
        "mcnemar_exact_pvalue": float(binomtest(recovered, recovered + damaged, 0.5).pvalue)
        if recovered + damaged else 1.0,
    }


def independent_evaluation(model, predictors, loader, device, alpha, seed, repetitions):
    names = list(predictors)
    stores = {
        name: {key: [] for key in ["noise_correct", "clean_correct", "margin_change", "true_logit_change"]}
        for name in names + ["ensemble", "paired_clean_oracle"]
    }
    original_noise_correct, original_clean_correct = [], []
    original_margin, original_true = [], []
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Independent ImageNetV2 confirmation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden, noise_hidden = outputs.hidden_states[BLOCK_INDEX + 1].split(batch)
            noise_true, noise_margin = classification_margin(noise_logits, labels)
            original_noise_correct.extend((noise_logits.argmax(1) == labels).cpu().tolist())
            original_clean_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            original_margin.extend(noise_margin.cpu().tolist())
            original_true.extend(noise_true.cpu().tolist())
            noise_patches = noise_hidden[:, 1:]
            clean_patches = clean_hidden[:, 1:]
            predictions = {
                name: predictor(noise_patches) for name, predictor in predictors.items()
            }
            predictions["ensemble"] = torch.stack(list(predictions.values())).mean(0)
            predictions["paired_clean_oracle"] = clean_patches - noise_patches
            clean_predictions = {
                name: predictor(clean_patches) for name, predictor in predictors.items()
            }
            clean_predictions["ensemble"] = torch.stack(list(clean_predictions.values())).mean(0)
            clean_predictions["paired_clean_oracle"] = torch.zeros_like(clean_patches)
            for name, prediction in predictions.items():
                corrected_noise = torch.cat(
                    [noise_hidden[:, :1], noise_patches + alpha * prediction], dim=1
                )
                corrected_clean = torch.cat(
                    [clean_hidden[:, :1], clean_patches + alpha * clean_predictions[name]], dim=1
                )
                corrected_noise_logits = downstream_from_layer(model, corrected_noise, BLOCK_INDEX)
                corrected_clean_logits = downstream_from_layer(model, corrected_clean, BLOCK_INDEX)
                true_logit, margin = classification_margin(corrected_noise_logits, labels)
                stores[name]["noise_correct"].extend(
                    (corrected_noise_logits.argmax(1) == labels).cpu().tolist()
                )
                stores[name]["clean_correct"].extend(
                    (corrected_clean_logits.argmax(1) == labels).cpu().tolist()
                )
                stores[name]["margin_change"].extend((margin - noise_margin).cpu().tolist())
                stores[name]["true_logit_change"].extend((true_logit - noise_true).cpu().tolist())
    original_noise_correct = np.asarray(original_noise_correct, dtype=bool)
    original_clean_correct = np.asarray(original_clean_correct, dtype=bool)
    results = {
        "original_vit": {
            "noise4_accuracy": float(original_noise_correct.mean()),
            "clean_accuracy": float(original_clean_correct.mean()),
            "samples": int(original_noise_correct.size),
        }
    }
    arrays = {
        "original_noise_correct": original_noise_correct,
        "original_clean_correct": original_clean_correct,
    }
    for index, (name, values) in enumerate(stores.items()):
        noise_correct = np.asarray(values["noise_correct"], dtype=bool)
        clean_correct = np.asarray(values["clean_correct"], dtype=bool)
        results[name] = paired_summary(
            original_noise_correct,
            noise_correct,
            np.asarray(values["margin_change"]),
            np.asarray(values["true_logit_change"]),
            seed + index,
            repetitions,
        ) | {
            "clean_accuracy_if_applied": float(clean_correct.mean()),
            "clean_accuracy_gain_if_applied": float(
                clean_correct.mean() - original_clean_correct.mean()
            ),
        }
        arrays[f"{name}_noise_correct"] = noise_correct
        arrays[f"{name}_clean_correct"] = clean_correct
    return results, arrays


def main():
    parser = argparse.ArgumentParser(description="Experiment 25: fixed multi-seed ImageNetV2 confirmation")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--independent-samples", type=int, default=10000)
    parser.add_argument("--independent-start", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--train-samples", type=int, default=5000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--development-start", type=int, default=0)
    parser.add_argument("--split-stride", type=int, default=7000)
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--initial-checkpoint", type=Path, default=DEFAULT_INITIAL_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--predictor-batch-size", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--noise-seed", type=int, default=2026)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    predictors, training, validation = {}, {}, {}
    split_records = {}
    for position, seed in enumerate(args.seeds):
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        train_start = args.development_start + position * args.split_stride
        validation_start = train_start + args.train_samples
        split_records[str(seed)] = {
            "train_start": train_start,
            "train_end": train_start + args.train_samples,
            "validation_start": validation_start,
            "validation_end": validation_start + args.validation_samples,
        }
        torch.manual_seed(seed)
        np.random.seed(seed)
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = seed
        seed_args.preservation_ratio = args.preservation_weight / args.classification_weight
        for split, samples, start in [
            ("train", args.train_samples, train_start),
            ("validation", args.validation_samples, validation_start),
        ]:
            cache_dir = seed_dir / f"{split}_cache"
            if not (args.resume and (cache_dir / "metadata.json").exists()):
                build_cache(
                    model, make_image_loader(samples, start, seed_args), device, cache_dir
                )
        train_data = HiddenCache(seed_dir / "train_cache")
        validation_data = HiddenCache(seed_dir / "validation_cache")
        checkpoint = seed_dir / f"classification_weight_{args.classification_weight:g}.pt"
        training_path = seed_dir / "training_record.json"
        if args.resume and checkpoint.exists() and training_path.exists():
            predictor = HiddenLinear().to(device)
            predictor.load_state_dict(
                torch.load(checkpoint, map_location=device, weights_only=True)
            )
            predictor.eval()
            training_record = json.loads(training_path.read_text())
        else:
            predictor, training_record = train_variant(
                model, train_data, validation_data, seed_args, device, seed_dir,
                args.classification_weight,
            )
            training_path.write_text(json.dumps(training_record, indent=2))
        name = f"seed_{seed}"
        predictors[name] = predictor
        training[name] = training_record
        validation[name] = evaluate(
            model,
            {name: predictor},
            DataLoader(validation_data, batch_size=args.predictor_batch_size, shuffle=False),
            device,
            {"fixed_configuration": {"predictor": name, "alpha": args.alpha}},
        )
    independent_dataset = PairedExternalDataset(
        args.image_root, args.independent_samples, args.independent_start, args.noise_seed
    )
    if len(independent_dataset) != args.independent_samples:
        raise ValueError(
            f"Requested {args.independent_samples} independent images, found {len(independent_dataset)}"
        )
    independent_loader = DataLoader(
        independent_dataset, batch_size=args.image_batch_size, shuffle=False,
        num_workers=args.num_workers,
    )
    independent, arrays = independent_evaluation(
        model, predictors, independent_loader, device, args.alpha,
        args.noise_seed, args.bootstrap_repetitions,
    )
    np.savez_compressed(output_dir / "independent_paired_outcomes.npz", **arrays)
    seed_gains = np.asarray([independent[f"seed_{seed}"]["accuracy_gain"] for seed in args.seeds])
    aggregate = {
        "mean_seed_accuracy_gain": float(seed_gains.mean()),
        "seed_accuracy_gain_standard_deviation": float(seed_gains.std(ddof=1))
        if seed_gains.size > 1 else 0.0,
        "all_seed_gains_positive": bool((seed_gains > 0).all()),
        "positive_seed_count": int((seed_gains > 0).sum()),
        "seed_count": int(seed_gains.size),
        "ensemble_gain": independent["ensemble"]["accuracy_gain"],
        "ensemble_ci_excludes_zero": independent["ensemble"]["accuracy_gain_95ci"][0] > 0,
        "ensemble_mcnemar_below_0.05": independent["ensemble"]["mcnemar_exact_pvalue"] < 0.05,
    }
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "initial_checkpoint": str(args.initial_checkpoint),
            "device": str(device),
            "fixed_without_retuning": True,
            "vit_block": 11,
            "frozen_components": ["ViT"],
            "sae_status": "not modified and not used by the hidden-state adapter",
            "fixed_initialization_development_range": [25000, 32000],
        },
        "development_splits": split_records,
        "training": training,
        "validation_monitoring_only": validation,
        "independent_evaluation": independent,
        "aggregate_confirmation": aggregate,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "development_splits": split_records,
        "independent_evaluation": independent,
        "aggregate_confirmation": aggregate,
    }, indent=2))
    print(f"Saved Experiment 25 to {output_dir}")


if __name__ == "__main__":
    main()
