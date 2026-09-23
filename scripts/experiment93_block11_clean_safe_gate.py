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
from interpretability.sae import BatchTopKSAE
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
import scripts.experiment89_sae_affine_clean_prior as affine
import scripts.experiment90_sae_affine_heldout_confirmation as confirmation


ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment93_block11_clean_safe_gate"
BLOCK = 11
ALPHA = 0.5


def atomic_json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def artifact_paths(seed):
    sae = (
        ACTIVE_ROOT
        / "results/sae/experiment91_batchtopk_layer_screen"
        / f"block11_seed{seed}_1000train_200val_3epoch_v1"
        / "batchtopk_lambda_1em03/model.pt"
    )
    diagnostic = (
        ACTIVE_ROOT
        / "results/sae/experiment85_block6_batchtopk_diagnostics"
        / f"layer_screen_block11_seed{seed}_noise4_blur4_v1/summary.json"
    )
    return sae, diagnostic


def load_seed(seed, device):
    checkpoint, diagnostic_path = artifact_paths(seed)
    sae = BatchTopKSAE(
        expansion_factor=32, k=32, input_unit_norm=True, n_batches_to_dead=5
    )
    sae.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    sae = sae.to(device).eval()
    diagnostic = json.loads(diagnostic_path.read_text())["results"]["blur"]
    groups = {
        "strengthened": [row["feature"] for row in diagnostic["top_strengthened"][:100]],
        "weakened": [row["feature"] for row in diagnostic["top_weakened"][:100]],
    }
    return sae, groups, checkpoint, diagnostic_path


def confidence_margin(logits):
    top_two = logits.topk(2, dim=1).values
    return top_two[:, 0] - top_two[:, 1]


def collect(model, sae, features, mapping, device, start, samples, corruption, args):
    kwargs = {"corruption": corruption, "corruption_seed": args.corruption_seed}
    if corruption:
        kwargs[f"{corruption}_severity"] = 4
    loader = DataLoader(
        ImageNetDataset(ROOT / "Dataset", samples, start, **kwargs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )
    stores = {key: [] for key in (
        "baseline_correct", "corrected_correct", "baseline_prediction",
        "corrected_prediction", "label", "confidence_delta",
    )}
    slope, intercept = (value.to(device) for value in mapping)
    with torch.no_grad():
        for images, labels in tqdm(loader, desc=f"Gate data {corruption or 'clean'}"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(pixel_values=images, output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            baseline_logits = outputs.logits
            delta = affine.residual_delta(
                sae, hidden[:, 1:], features, slope, intercept, ALPHA
            )
            corrected_hidden = torch.cat((hidden[:, :1], hidden[:, 1:] + delta), 1)
            corrected_logits = downstream_from_layer(model, corrected_hidden, BLOCK - 1)
            baseline_prediction = baseline_logits.argmax(1)
            corrected_prediction = corrected_logits.argmax(1)
            stores["baseline_correct"].extend((baseline_prediction == labels).cpu().tolist())
            stores["corrected_correct"].extend((corrected_prediction == labels).cpu().tolist())
            stores["baseline_prediction"].extend(baseline_prediction.cpu().tolist())
            stores["corrected_prediction"].extend(corrected_prediction.cpu().tolist())
            stores["label"].extend(labels.cpu().tolist())
            stores["confidence_delta"].extend(
                (confidence_margin(corrected_logits) - confidence_margin(baseline_logits)).cpu().tolist()
            )
    return {
        key: np.asarray(value, dtype=np.float32 if key == "confidence_delta" else None)
        for key, value in stores.items()
    }


def gated_outcomes(arrays, threshold):
    apply = arrays["confidence_delta"] >= threshold
    correct = np.where(apply, arrays["corrected_correct"], arrays["baseline_correct"]).astype(bool)
    prediction = np.where(apply, arrays["corrected_prediction"], arrays["baseline_prediction"])
    return correct, prediction, apply


def metrics(arrays, threshold, bootstrap_seed):
    baseline = arrays["baseline_correct"].astype(bool)
    gated, _, apply = gated_outcomes(arrays, threshold)
    recovered = int(((~baseline) & gated).sum())
    damaged = int((baseline & (~gated)).sum())
    discordant = recovered + damaged
    return {
        "images": int(len(baseline)),
        "baseline_accuracy": float(baseline.mean()),
        "gated_accuracy": float(gated.mean()),
        "accuracy_change_pp": float((gated.mean() - baseline.mean()) * 100),
        "recovered": recovered,
        "damaged": damaged,
        "gate_application_rate": float(apply.mean()),
        "mcnemar_exact_p": float(binomtest(recovered, discordant, 0.5).pvalue) if discordant else 1.0,
        "paired_bootstrap_ci95_pp": confirmation.bootstrap_paired(gated, baseline, bootstrap_seed),
    }


def select_threshold(development):
    pooled_deltas = np.concatenate([
        development[str(seed)][condition]["confidence_delta"]
        for seed in range(3)
        for condition in ("blur", "clean")
    ])
    candidates = np.unique(np.concatenate((
        np.quantile(pooled_deltas, np.linspace(0, 1, 101)),
        np.asarray([np.inf]),
    )))
    records = []
    for threshold in candidates:
        blur_base, blur_gate, clean_base, clean_gate, applications = [], [], [], [], []
        for seed in range(3):
            for condition, base_store, gate_store in (
                ("blur", blur_base, blur_gate), ("clean", clean_base, clean_gate)
            ):
                arrays = development[str(seed)][condition]
                gated, _, apply = gated_outcomes(arrays, threshold)
                base_store.append(arrays["baseline_correct"].astype(bool))
                gate_store.append(gated)
                applications.append(apply)
        blur_base = np.concatenate(blur_base)
        blur_gate = np.concatenate(blur_gate)
        clean_base = np.concatenate(clean_base)
        clean_gate = np.concatenate(clean_gate)
        records.append({
            "threshold": float(threshold),
            "blur_gain_pp": float((blur_gate.mean() - blur_base.mean()) * 100),
            "clean_change_pp": float((clean_gate.mean() - clean_base.mean()) * 100),
            "application_rate": float(np.concatenate(applications).mean()),
        })
    eligible = [record for record in records if record["clean_change_pp"] >= 0.0]
    selected = max(
        eligible,
        key=lambda record: (
            record["blur_gain_pp"], record["clean_change_pp"], -record["application_rate"]
        ),
    )
    return selected, records


def main():
    parser = argparse.ArgumentParser(description="Experiment 93: clean-safe Block-11 SAE gate")
    parser.add_argument("--fit-start", type=int, default=47000)
    parser.add_argument("--fit-samples", type=int, default=500)
    parser.add_argument("--gate-start", type=int, default=47500)
    parser.add_argument("--gate-samples", type=int, default=500)
    parser.add_argument("--evaluation-start", type=int, default=48000)
    parser.add_argument("--evaluation-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if (args.fit_start, args.fit_start + args.fit_samples) != (47000, 47500):
        raise ValueError("Mapping fit is locked to [47000,47500)")
    if (args.gate_start, args.gate_start + args.gate_samples) != (47500, 48000):
        raise ValueError("Gate development is locked to [47500,48000)")
    if (args.evaluation_start, args.evaluation_start + args.evaluation_samples) != (48000, 49000):
        raise ValueError("Confirmation is locked to [48000,49000)")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    affine.BLOCK = BLOCK
    confirmation.BLOCK = BLOCK
    development = {}
    mappings = {}
    metadata = {}
    for seed in range(3):
        sae, groups, checkpoint, diagnostic = load_seed(seed, device)
        features = groups["strengthened"] + groups["weakened"]
        slope, intercept, fit_patches = affine.fit_affine(
            model, sae, device, features, args.fit_start, args.fit_samples,
            "blur", args.batch_size, args.workers, args.corruption_seed,
        )
        mappings[str(seed)] = (slope.cpu(), intercept.cpu())
        development[str(seed)] = {
            condition: collect(
                model, sae, features, (slope, intercept), device, args.gate_start,
                args.gate_samples, None if condition == "clean" else "blur", args,
            )
            for condition in ("blur", "clean")
        }
        metadata[str(seed)] = {
            "sae_checkpoint": str(checkpoint),
            "diagnostic_summary": str(diagnostic),
            "features": groups,
            "fit_patches": fit_patches,
        }
        del sae
        torch.cuda.empty_cache()
    selected, grid = select_threshold(development)
    threshold = selected["threshold"]
    print("Selected gate:", selected)
    evaluation = {}
    arrays_to_save = {}
    for seed in range(3):
        sae, groups, _, _ = load_seed(seed, device)
        features = groups["strengthened"] + groups["weakened"]
        evaluation[str(seed)] = {}
        for offset, condition in enumerate(("blur", "clean")):
            arrays = collect(
                model, sae, features, mappings[str(seed)], device,
                args.evaluation_start, args.evaluation_samples,
                None if condition == "clean" else "blur", args,
            )
            evaluation[str(seed)][condition] = metrics(arrays, threshold, seed * 10 + offset)
            for key, value in arrays.items():
                arrays_to_save[f"seed{seed}_{condition}_{key}"] = value
        del sae
        torch.cuda.empty_cache()
    np.savez_compressed(output_dir / "paired_outcomes.npz", **arrays_to_save)
    blur_gains = [evaluation[str(seed)]["blur"]["accuracy_change_pp"] for seed in range(3)]
    clean_changes = [evaluation[str(seed)]["clean"]["accuracy_change_pp"] for seed in range(3)]
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL, "block": BLOCK, "alpha": ALPHA,
            "vit_frozen": True, "saes_frozen": True,
            "gate_signal": "corrected minus baseline top1-minus-top2 logit margin",
        },
        "seed_artifacts": metadata,
        "selected_gate": selected,
        "development_grid": grid,
        "evaluation": evaluation,
        "aggregate": {
            "blur_gain_pp_by_seed": blur_gains,
            "mean_blur_gain_pp": float(np.mean(blur_gains)),
            "clean_change_pp_by_seed": clean_changes,
            "mean_clean_change_pp": float(np.mean(clean_changes)),
            "positive_blur_seeds": int(np.sum(np.asarray(blur_gains) > 0)),
        },
        "guardrails": [
            "The gate threshold is selected jointly across three SAE seeds only on [47500,48000).",
            "Gate selection requires non-negative pooled clean accuracy change on development data.",
            "The frozen gate uses no label or clean counterpart at inference.",
            "[48000,49000) is secondary confirmation previously used for ungated evaluation, not final evaluation.",
            "[49000,50000) remains untouched and must only be used after the method is frozen.",
            "Only residual decoder deltas are added; full SAE reconstructions are never inserted.",
        ],
    }
    atomic_json_write(output_dir / "summary.json", summary)
    print(json.dumps({"selected_gate": selected, "aggregate": summary["aggregate"]}, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
