import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment15_independent_frozen_confirmation import ExternalImageNetDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment25_multiseed_independent_confirmation import bootstrap_interval


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment26_adapter_robustness"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenetv2-matched-frequency-format-val"
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "results" / "sae" / "experiment25_multiseed_confirmation"
    / "imagenetv2_multiseed_confirmation" / "seed_2" / "classification_weight_0.05.pt"
)
TRAINING_LAYER_INDEX = 10


def paired_metrics(original_correct, corrected, margin_change, true_logit_change, seed, repetitions):
    original_correct = np.asarray(original_correct, dtype=bool)
    corrected = np.asarray(corrected, dtype=bool)
    difference = corrected.astype(np.float64) - original_correct.astype(np.float64)
    recovered = int((~original_correct & corrected).sum())
    damaged = int((original_correct & ~corrected).sum())
    return {
        "accuracy": float(corrected.mean()),
        "accuracy_gain": float(difference.mean()),
        "accuracy_gain_95ci": bootstrap_interval(difference, seed, repetitions),
        "mean_margin_change": float(np.mean(margin_change)),
        "mean_true_logit_change": float(np.mean(true_logit_change)),
        "predictions_recovered": recovered,
        "originally_correct_damaged": damaged,
        "mcnemar_exact_pvalue": float(binomtest(recovered, recovered + damaged, 0.5).pvalue)
        if recovered + damaged else 1.0,
    }


def make_loader(args, severity, noise_seed):
    dataset = ExternalImageNetDataset(
        args.image_root,
        severity,
        args.samples,
        args.start_index,
        corruption="noise",
        seed=noise_seed,
    )
    if len(dataset) != args.samples:
        raise ValueError(f"Requested {args.samples} images, found {len(dataset)}")
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def evaluate_condition(model, predictor, loader, device, alpha, seed, repetitions):
    original_correct, corrected, margin_change, true_logit_change = [], [], [], []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Condition evaluation"):
            labels = labels.to(device)
            outputs = model(pixel_values=images.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[TRAINING_LAYER_INDEX + 1]
            original_true, original_margin = classification_margin(outputs.logits, labels)
            prediction = predictor(hidden[:, 1:])
            candidate = torch.cat(
                [hidden[:, :1], hidden[:, 1:] + alpha * prediction], dim=1
            )
            logits = downstream_from_layer(model, candidate, TRAINING_LAYER_INDEX)
            true_logit, margin = classification_margin(logits, labels)
            original_correct.extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            corrected.extend((logits.argmax(1) == labels).cpu().tolist())
            margin_change.extend((margin - original_margin).cpu().tolist())
            true_logit_change.extend((true_logit - original_true).cpu().tolist())
    original_correct = np.asarray(original_correct, dtype=bool)
    corrected = np.asarray(corrected, dtype=bool)
    summary = {
        "original_accuracy": float(original_correct.mean()),
        "adapter": paired_metrics(
            original_correct, corrected, margin_change, true_logit_change, seed, repetitions
        ),
    }
    return summary, {
        "original_correct": original_correct,
        "adapter_correct": corrected,
        "margin_change": np.asarray(margin_change, dtype=np.float32),
        "true_logit_change": np.asarray(true_logit_change, dtype=np.float32),
    }


def evaluate_ablations(model, predictor, loader, device, alpha, seed, repetitions, layers):
    names = [f"block{layer}" for layer in layers] + [
        "reverse", "shuffled_image", "shuffled_patch", "position_only", "residual_identity"
    ]
    stores = {
        name: {key: [] for key in ["correct", "margin_change", "true_logit_change"]}
        for name in names
    }
    original_correct = []
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Mechanistic ablations"):
            labels = labels.to(device)
            outputs = model(pixel_values=images.to(device), output_hidden_states=True)
            original_true, original_margin = classification_margin(outputs.logits, labels)
            original_correct.extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for block in layers:
                layer_index = block - 1
                hidden = outputs.hidden_states[layer_index + 1]
                residual = alpha * predictor(hidden[:, 1:])
                logits = downstream_from_layer(
                    model,
                    torch.cat([hidden[:, :1], hidden[:, 1:] + residual], dim=1),
                    layer_index,
                )
                true_logit, margin = classification_margin(logits, labels)
                name = f"block{block}"
                stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                stores[name]["margin_change"].extend((margin - original_margin).cpu().tolist())
                stores[name]["true_logit_change"].extend((true_logit - original_true).cpu().tolist())
            hidden = outputs.hidden_states[TRAINING_LAYER_INDEX + 1]
            prediction = predictor(hidden[:, 1:])
            position_only = predictor.position.weight[None].expand_as(prediction)
            controls = {
                "reverse": -alpha * prediction,
                "shuffled_image": alpha * prediction.roll(1, dims=0),
                "shuffled_patch": alpha * prediction.roll(1, dims=1),
                "position_only": alpha * position_only,
                "residual_identity": torch.zeros_like(prediction),
            }
            for name, residual in controls.items():
                logits = downstream_from_layer(
                    model,
                    torch.cat([hidden[:, :1], hidden[:, 1:] + residual], dim=1),
                    TRAINING_LAYER_INDEX,
                )
                true_logit, margin = classification_margin(logits, labels)
                stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                stores[name]["margin_change"].extend((margin - original_margin).cpu().tolist())
                stores[name]["true_logit_change"].extend((true_logit - original_true).cpu().tolist())
    original_correct = np.asarray(original_correct, dtype=bool)
    return {
        "original_accuracy": float(original_correct.mean()),
        "conditions": {
            name: paired_metrics(
                original_correct,
                values["correct"],
                values["margin_change"],
                values["true_logit_change"],
                seed + index,
                repetitions,
            )
            for index, (name, values) in enumerate(stores.items())
        },
    }


def benchmark_latency(model, predictor, images, device, alpha, warmup, repetitions):
    images = images.to(device)

    def baseline():
        return model(pixel_values=images).logits

    def adapted():
        outputs = model(pixel_values=images, output_hidden_states=True)
        hidden = outputs.hidden_states[TRAINING_LAYER_INDEX + 1]
        residual = predictor(hidden[:, 1:])
        candidate = torch.cat([hidden[:, :1], hidden[:, 1:] + alpha * residual], dim=1)
        return downstream_from_layer(model, candidate, TRAINING_LAYER_INDEX)

    with torch.no_grad():
        for _ in range(warmup):
            baseline()
            adapted()
        if device.type == "cuda":
            torch.cuda.synchronize()
        timings = {}
        for name, function in [("baseline", baseline), ("adapter", adapted)]:
            start = time.perf_counter()
            for _ in range(repetitions):
                function()
            if device.type == "cuda":
                torch.cuda.synchronize()
            timings[name] = (time.perf_counter() - start) / repetitions
    return {
        "batch_size": int(images.shape[0]),
        "baseline_seconds_per_batch": timings["baseline"],
        "adapter_seconds_per_batch": timings["adapter"],
        "relative_latency_overhead": timings["adapter"] / timings["baseline"] - 1.0,
        "adapter_parameters": sum(parameter.numel() for parameter in predictor.parameters()),
        "adapter_parameter_megabytes_fp32": sum(
            parameter.numel() * parameter.element_size() for parameter in predictor.parameters()
        ) / 2**20,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 26: fixed adapter robustness and ablations")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--noise-seeds", type=int, nargs="+", default=[2026, 2027, 2028, 2029])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--ablation-layers", type=int, nargs="+", default=[9, 10, 11, 12])
    parser.add_argument("--ablation-seed", type=int, default=2026)
    parser.add_argument("--ablation-severity", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--timing-warmup", type=int, default=10)
    parser.add_argument("--timing-repetitions", type=int, default=50)
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

    conditions = [("clean", 0, args.noise_seeds[0])] + [
        (f"noise{severity}_seed{noise_seed}", severity, noise_seed)
        for noise_seed in args.noise_seeds for severity in args.severities
    ]
    condition_results = {}
    for index, (name, severity, noise_seed) in enumerate(conditions):
        result_path = condition_dir / f"{name}.json"
        arrays_path = condition_dir / f"{name}.npz"
        if args.resume and result_path.exists() and arrays_path.exists():
            condition_results[name] = json.loads(result_path.read_text())
            continue
        result, arrays = evaluate_condition(
            model, predictor, make_loader(args, severity, noise_seed), device,
            args.alpha, noise_seed + index, args.bootstrap_repetitions,
        )
        result |= {"severity": severity, "noise_seed": noise_seed}
        result_path.write_text(json.dumps(result, indent=2))
        np.savez_compressed(arrays_path, **arrays)
        condition_results[name] = result

    ablation_path = output_dir / "ablations.json"
    if args.resume and ablation_path.exists():
        ablations = json.loads(ablation_path.read_text())
    else:
        ablations = evaluate_ablations(
            model,
            predictor,
            make_loader(args, args.ablation_severity, args.ablation_seed),
            device,
            args.alpha,
            args.ablation_seed,
            args.bootstrap_repetitions,
            args.ablation_layers,
        )
        ablation_path.write_text(json.dumps(ablations, indent=2))

    timing_loader = make_loader(args, args.ablation_severity, args.ablation_seed)
    timing_images, _ = next(iter(timing_loader))
    timing = benchmark_latency(
        model, predictor, timing_images, device, args.alpha,
        args.timing_warmup, args.timing_repetitions,
    )
    severity_summary = {}
    for severity in args.severities:
        gains = [
            condition_results[f"noise{severity}_seed{noise_seed}"]["adapter"]["accuracy_gain"]
            for noise_seed in args.noise_seeds
        ]
        severity_summary[f"noise{severity}"] = {
            "mean_accuracy_gain": float(np.mean(gains)),
            "standard_deviation_across_noise_seeds": float(np.std(gains, ddof=1))
            if len(gains) > 1 else 0.0,
            "all_noise_seed_gains_positive": bool(np.all(np.asarray(gains) > 0)),
            "gains": gains,
        }
    summary = {
        "configuration": vars(args) | {
            "image_root": str(args.image_root.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "device": str(device),
            "fixed_without_retuning": True,
            "primary_adapter": "Experiment 25 seed 2",
        },
        "condition_results": condition_results,
        "severity_summary": severity_summary,
        "mechanistic_ablations": ablations,
        "inference_cost": timing,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "clean": condition_results["clean"],
        "severity_summary": severity_summary,
        "mechanistic_ablations": ablations,
        "inference_cost": timing,
    }, indent=2))
    print(f"Saved Experiment 26 to {output_dir}")


if __name__ == "__main__":
    main()
