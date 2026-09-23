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
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment89_sae_affine_clean_prior import fit_affine, residual_delta
from scripts.experiment88_oracle_sae_residual_repair import (
    BLOCK,
    ROOT,
    atomic_json_write,
    load_sae,
    selected_features,
)


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment90_sae_affine_heldout_confirmation"
ALPHA = 0.5


def bootstrap_paired(corrected, baseline, seed, replicates=10000):
    differences = corrected.astype(np.float64) - baseline.astype(np.float64)
    generator = np.random.default_rng(seed)
    estimates = np.empty(replicates)
    for start in range(0, replicates, 500):
        count = min(500, replicates - start)
        indices = generator.integers(0, len(differences), size=(count, len(differences)))
        estimates[start : start + count] = differences[indices].mean(1) * 100
    return [float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))]


def evaluate(model, sae, device, features, mapping, start, samples, corruption, batch_size, workers, seed):
    kwargs = {"corruption": corruption, "corruption_seed": seed}
    if corruption:
        kwargs[f"{corruption}_severity"] = 4
    loader = DataLoader(
        ImageNetDataset(ROOT / "Dataset", samples, start, **kwargs),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    baseline_outcomes = []
    corrected_outcomes = []
    baseline_predictions = []
    corrected_predictions = []
    labels_all = []
    slope, intercept = (value.to(device) for value in mapping)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Confirm {corruption or 'clean'}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            baseline_prediction = outputs.logits.argmax(1)
            delta = residual_delta(sae, hidden[:, 1:], features, slope, intercept, ALPHA)
            corrected_hidden = torch.cat((hidden[:, :1], hidden[:, 1:] + delta), 1)
            corrected_prediction = downstream_from_layer(model, corrected_hidden, BLOCK - 1).argmax(1)
            baseline_predictions.extend(baseline_prediction.cpu().tolist())
            corrected_predictions.extend(corrected_prediction.cpu().tolist())
            labels_all.extend(labels.cpu().tolist())
            baseline_outcomes.extend((baseline_prediction == labels).cpu().tolist())
            corrected_outcomes.extend((corrected_prediction == labels).cpu().tolist())
    baseline = np.asarray(baseline_outcomes, dtype=bool)
    corrected = np.asarray(corrected_outcomes, dtype=bool)
    recovered = int(((~baseline) & corrected).sum())
    damaged = int((baseline & (~corrected)).sum())
    discordant = recovered + damaged
    p_value = float(binomtest(recovered, discordant, 0.5).pvalue) if discordant else 1.0
    return {
        "metrics": {
            "images": int(len(baseline)),
            "baseline_accuracy": float(baseline.mean()),
            "corrected_accuracy": float(corrected.mean()),
            "accuracy_change_pp": float((corrected.mean() - baseline.mean()) * 100),
            "recovered": recovered,
            "damaged": damaged,
            "mcnemar_exact_p": p_value,
            "paired_bootstrap_ci95_pp": bootstrap_paired(corrected, baseline, seed),
        },
        "arrays": {
            "baseline_correct": baseline,
            "corrected_correct": corrected,
            "baseline_prediction": np.asarray(baseline_predictions, dtype=np.int16),
            "corrected_prediction": np.asarray(corrected_predictions, dtype=np.int16),
            "label": np.asarray(labels_all, dtype=np.int16),
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 90: frozen three-seed SAE affine confirmation")
    parser.add_argument("--fit-start", type=int, default=47000)
    parser.add_argument("--fit-samples", type=int, default=500)
    parser.add_argument("--evaluation-start", type=int, default=48000)
    parser.add_argument("--evaluation-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if (args.fit_start, args.fit_start + args.fit_samples) != (47000, 47500):
        raise ValueError("Mapping fit is locked to [47000,47500)")
    if (args.evaluation_start, args.evaluation_start + args.evaluation_samples) != (48000, 49000):
        raise ValueError("Confirmation is locked to [48000,49000)")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    results = {}
    for seed in range(3):
        seed_path = output_dir / f"seed_{seed}.json"
        if args.resume and seed_path.exists():
            print(f"Skipping completed seed {seed}")
            results[str(seed)] = json.loads(seed_path.read_text())
            continue
        sae = load_sae(seed, device)
        groups = selected_features(seed, "noise")
        features = groups["strengthened"] + groups["weakened"]
        slope, intercept, fit_patches = fit_affine(
            model, sae, device, features, args.fit_start, args.fit_samples,
            "noise", args.batch_size, args.workers, args.corruption_seed,
        )
        noise = evaluate(
            model, sae, device, features, (slope, intercept), args.evaluation_start,
            args.evaluation_samples, "noise", args.batch_size, args.workers,
            args.corruption_seed,
        )
        clean = evaluate(
            model, sae, device, features, (slope, intercept), args.evaluation_start,
            args.evaluation_samples, None, args.batch_size, args.workers,
            args.corruption_seed,
        )
        np.savez_compressed(
            output_dir / f"seed_{seed}_paired_outcomes.npz",
            **{f"noise_{key}": value for key, value in noise["arrays"].items()},
            **{f"clean_{key}": value for key, value in clean["arrays"].items()},
        )
        record = {
            "seed": seed,
            "alpha": ALPHA,
            "features": groups,
            "fit_patches": fit_patches,
            "mapping": {"slope": slope.cpu().tolist(), "intercept": intercept.cpu().tolist()},
            "noise": noise["metrics"],
            "clean": clean["metrics"],
        }
        atomic_json_write(seed_path, record)
        results[str(seed)] = record
        del sae
        if device.type == "cuda":
            torch.cuda.empty_cache()
    gains = np.asarray([results[str(seed)]["noise"]["accuracy_change_pp"] for seed in range(3)])
    summary = {
        "configuration": vars(args) | {
            "alpha_frozen_from_experiment89": ALPHA,
            "model": BASE_MODEL,
            "block": BLOCK,
            "vit_frozen": True,
            "sae_frozen": True,
        },
        "seeds": results,
        "aggregate": {
            "mean_noise_gain_pp": float(gains.mean()),
            "noise_gain_pp_by_seed": gains.tolist(),
            "positive_seeds": int((gains > 0).sum()),
            "sign_test_all_three_positive_p_one_sided": 0.125 if np.all(gains > 0) else None,
        },
        "guardrails": [
            "Feature directions were frozen by Experiments 86-87 and alpha=0.5 was frozen by Experiment 89.",
            "No choices are made using [48000,49000); outcomes are paired and saved per image.",
            "Mapping uses paired representations from [47000,47500) without labels; clean counterparts are unavailable and unnecessary at inference.",
            "The full SAE reconstruction is never inserted into the ViT.",
            "[49000,50000) remains untouched.",
        ],
        "next_stage": "Run random-direction controls only if the frozen three-seed gain replicates.",
    }
    atomic_json_write(output_dir / "summary.json", summary)
    print(json.dumps(summary["aggregate"], indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
