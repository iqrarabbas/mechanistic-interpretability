import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment45_sae_discovered_hidden_subspace import SharedRawInputRepair
from scripts.experiment49_sae_jaccard_controls import bootstrap_difference
from scripts.experiment52_block6_repair_propagation import (
    STAGES,
    append_values,
    logit_lens,
    representation_metrics,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
SOURCE = (
    ACTIVE_ROOT / "results/sae/experiment96_block6_batchtopk_blur_hidden_subspace"
    / "full_3seed_rank16_v1"
)
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment97_block6_batchtopk_repair_propagation"
INTERVENTION_BLOCK = 6
WIDTH = 768
PATCHES = 196
RANK = 16
METHOD = "blur_batchtopk_decoder"


def load_repair(seed, device):
    state = torch.load(SOURCE / f"seed_{seed}/{METHOD}.pt", map_location="cpu", weights_only=True)
    repair = SharedRawInputRepair(
        torch.zeros(WIDTH, RANK),
        torch.zeros(WIDTH),
        torch.ones(WIDTH),
        torch.ones(RANK),
    )
    repair.load_state_dict(state)
    repair = repair.to(device).eval()
    repair.requires_grad_(False)
    if sum(parameter.numel() for parameter in repair.parameters()) != 15440:
        raise RuntimeError("Frozen repair parameter count mismatch")
    return repair


def make_loader(split, seed, args):
    samples = split["end"] - split["start"]
    if args.max_samples is not None:
        samples = min(samples, args.max_samples)
    return DataLoader(
        PairedCorruptionDataset("blur", samples, split["start"], seed),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )


def evaluate_seed(model, repair, loader, device):
    metric_names = (
        "patch_cosine", "patch_relative_l2", "cls_cosine",
        "cls_relative_l2", "true_logit", "margin",
    )
    stores = {
        stage: {
            condition: {metric: [] for metric in metric_names}
            for condition in ("clean", "blur", "corrected")
        }
        for stage in STAGES
    }
    final_correct = {condition: [] for condition in ("clean", "blur", "corrected")}
    replay_max_error = 0.0
    with torch.no_grad():
        for clean, blur, labels in tqdm(loader, desc="Rank-16 Blur repair propagation"):
            batch = len(labels)
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat((clean, blur)).to(device), output_hidden_states=True
            )
            clean_logits, blur_logits = outputs.logits.split(batch)
            clean_hidden6, blur_hidden6 = outputs.hidden_states[INTERVENTION_BLOCK].split(batch)
            corrected = torch.cat(
                (blur_hidden6[:, :1], blur_hidden6[:, 1:] + repair(blur_hidden6[:, 1:])),
                dim=1,
            )
            blur_replayed = blur_hidden6.clone()
            for stage in STAGES:
                if stage > INTERVENTION_BLOCK:
                    corrected = model.vit.layers[stage - 1](corrected, attention_mask=None)
                    blur_replayed = model.vit.layers[stage - 1](blur_replayed, attention_mask=None)
                clean_hidden, blur_hidden = outputs.hidden_states[stage].split(batch)
                for condition, hidden in (
                    ("clean", clean_hidden),
                    ("blur", blur_hidden),
                    ("corrected", corrected),
                ):
                    representation = representation_metrics(clean_hidden, hidden)
                    _, true_logit, margin = logit_lens(model, hidden, labels)
                    append_values(
                        stores[stage][condition],
                        representation | {"true_logit": true_logit, "margin": margin},
                    )
            corrected_logits, _, _ = logit_lens(model, corrected, labels)
            replayed_logits, _, _ = logit_lens(model, blur_replayed, labels)
            replay_max_error = max(
                replay_max_error, float((replayed_logits - blur_logits).abs().max())
            )
            final_correct["clean"].extend((clean_logits.argmax(1) == labels).cpu().tolist())
            final_correct["blur"].extend((blur_logits.argmax(1) == labels).cpu().tolist())
            final_correct["corrected"].extend((corrected_logits.argmax(1) == labels).cpu().tolist())
    if replay_max_error > 1e-4:
        raise RuntimeError(f"Downstream replay mismatch: {replay_max_error:.8g}")
    stores = {
        stage: {
            condition: {
                metric: np.asarray(values, dtype=np.float64)
                for metric, values in metrics.items()
            }
            for condition, metrics in conditions.items()
        }
        for stage, conditions in stores.items()
    }
    final_correct = {
        condition: np.asarray(values, dtype=bool)
        for condition, values in final_correct.items()
    }
    return stores, final_correct, replay_max_error


def summarize_seed(stores, final_correct, seed, repetitions):
    metric_pairs = {
        "patch_cosine_gain": ("patch_cosine", "corrected", "blur"),
        "patch_relative_l2_reduction": ("patch_relative_l2", "blur", "corrected"),
        "cls_cosine_gain": ("cls_cosine", "corrected", "blur"),
        "cls_relative_l2_reduction": ("cls_relative_l2", "blur", "corrected"),
        "margin_gain": ("margin", "corrected", "blur"),
        "true_logit_gain": ("true_logit", "corrected", "blur"),
    }
    stages = {}
    for stage in STAGES:
        conditions = stores[stage]
        tests = {}
        for offset, (name, (metric, positive, negative)) in enumerate(metric_pairs.items()):
            tests[name] = bootstrap_difference(
                conditions[positive][metric], conditions[negative][metric],
                seed + stage + offset * 20, repetitions,
            )
        stages[str(stage)] = {
            "means": {
                condition: {
                    metric: float(values.mean()) for metric, values in metrics.items()
                }
                for condition, metrics in conditions.items()
            },
            "corrected_minus_blur": tests,
        }
    return {
        "stages": stages,
        "final_classification": paired_comparison(
            final_correct["blur"], final_correct["corrected"], seed + 9700, repetitions
        ),
        "clean_accuracy": float(final_correct["clean"].mean()),
    }


def save_arrays(path, stores, final_correct):
    arrays = {}
    for stage, conditions in stores.items():
        for condition, metrics in conditions.items():
            for metric, values in metrics.items():
                arrays[f"block_{stage}__{condition}__{metric}"] = values
    for condition, values in final_correct.items():
        arrays[f"final__{condition}__correct"] = values
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser(description="Experiment 97: trace rank-16 Block-6 SAE repair")
    parser.add_argument("--source-run", type=Path, default=SOURCE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    source_summary = json.loads((args.source_run / "summary.json").read_text())
    configuration = source_summary["configuration"]
    required = {"block": 6, "rank": 16, "trainable_parameters_per_method": 15440}
    for key, expected in required.items():
        if configuration.get(key) != expected:
            raise RuntimeError(f"Frozen source mismatch: {key}={configuration.get(key)}")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    results = {}
    for seed in args.seeds:
        repair = load_repair(seed, device)
        loader = make_loader(splits[seed]["validation"], seed, args)
        stores, final_correct, replay_error = evaluate_seed(model, repair, loader, device)
        save_arrays(output_dir / f"seed_{seed}_paired_outcomes.npz", stores, final_correct)
        results[str(seed)] = summarize_seed(
            stores, final_correct, seed, args.bootstrap_repetitions
        ) | {"manual_replay_max_absolute_logit_error": replay_error}
        (output_dir / f"seed_{seed}_summary.json").write_text(
            json.dumps(results[str(seed)], indent=2)
        )
    trajectory = {}
    for stage in STAGES:
        trajectory[str(stage)] = {}
        for metric in (
            "patch_cosine_gain", "patch_relative_l2_reduction", "cls_cosine_gain",
            "cls_relative_l2_reduction", "margin_gain", "true_logit_gain",
        ):
            values = [
                results[str(seed)]["stages"][str(stage)]["corrected_minus_blur"][metric]["mean_paired_difference"]
                for seed in args.seeds
            ]
            trajectory[str(stage)][metric] = {"by_seed": values, "mean": float(np.mean(values))}
    final_gains = [
        results[str(seed)]["final_classification"]["accuracy_difference"]
        for seed in args.seeds
    ]
    summary = {
        "configuration": vars(args) | {
            "source_run": str(args.source_run.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "intervention_block": INTERVENTION_BLOCK,
            "tracked_blocks": STAGES,
            "method": METHOD,
            "parameters": 15440,
            "model_frozen": True,
            "repair_frozen": True,
            "corruption": "Blur-4",
            "imageNetV2_accessed": False,
            "final_reserve_accessed": False,
            "status": "development mechanistic propagation analysis",
        },
        "development_splits": {str(seed): splits[seed]["validation"] for seed in args.seeds},
        "per_seed": results,
        "mean_trajectory": trajectory,
        "aggregate": {
            "final_blur_gain_by_seed": final_gains,
            "mean_final_blur_gain": float(np.mean(final_gains)),
        },
        "guardrails": [
            "Uses frozen Experiment 96 checkpoints without training or tuning.",
            "Paired clean images are used only for mechanistic diagnostics, never to construct the repair.",
            "Manual downstream replay must match standard inference within 1e-4 logits.",
            "Only seed-specific adapter validation splits are analyzed.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"aggregate": summary["aggregate"], "trajectory": trajectory}, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
