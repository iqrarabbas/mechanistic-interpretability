import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment49_sae_jaccard_controls import bootstrap_difference
from scripts.experiment52_block6_repair_propagation import (
    append_values,
    load_adapter,
    logit_lens,
    make_loader,
    representation_metrics,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment65_block1_adapter_propagation"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment46_layer_specific_adapters"
    / "full_3seed_block1_exact_protocol_v1"
)
INTERVENTION_BLOCK = 1
STAGES = list(range(INTERVENTION_BLOCK, 13))


def adapt(hidden, adapter, alpha):
    return torch.cat(
        [hidden[:, :1], hidden[:, 1:] + alpha * adapter(hidden[:, 1:])], dim=1
    )


def evaluate_seed(model, adapter, loader, device, alpha):
    metric_names = [
        "patch_cosine", "patch_relative_l2", "cls_cosine", "cls_relative_l2",
        "true_logit", "margin",
    ]
    conditions = ["clean", "adapted_clean", "noise", "adapted_noise"]
    stores = {
        stage: {condition: {metric: [] for metric in metric_names} for condition in conditions}
        for stage in STAGES
    }
    final_correct = {condition: [] for condition in conditions}
    replay_max_absolute_logit_error = 0.0

    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Block-1 adapter propagation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden1, noise_hidden1 = outputs.hidden_states[INTERVENTION_BLOCK].split(batch)
            adapted_clean = adapt(clean_hidden1, adapter, alpha)
            adapted_noise = adapt(noise_hidden1, adapter, alpha)
            clean_replayed = clean_hidden1.clone()
            noise_replayed = noise_hidden1.clone()

            for stage in STAGES:
                if stage > INTERVENTION_BLOCK:
                    layer = model.vit.layers[stage - 1]
                    adapted_clean = layer(adapted_clean, attention_mask=None)
                    adapted_noise = layer(adapted_noise, attention_mask=None)
                    clean_replayed = layer(clean_replayed, attention_mask=None)
                    noise_replayed = layer(noise_replayed, attention_mask=None)
                clean_hidden, noise_hidden = outputs.hidden_states[stage].split(batch)
                for condition, hidden in (
                    ("clean", clean_hidden),
                    ("adapted_clean", adapted_clean),
                    ("noise", noise_hidden),
                    ("adapted_noise", adapted_noise),
                ):
                    representation = representation_metrics(clean_hidden, hidden)
                    _, true_logit, margin = logit_lens(model, hidden, labels)
                    append_values(
                        stores[stage][condition],
                        representation | {"true_logit": true_logit, "margin": margin},
                    )

            adapted_clean_logits, _, _ = logit_lens(model, adapted_clean, labels)
            adapted_noise_logits, _, _ = logit_lens(model, adapted_noise, labels)
            replayed_clean_logits, _, _ = logit_lens(model, clean_replayed, labels)
            replayed_noise_logits, _, _ = logit_lens(model, noise_replayed, labels)
            replay_max_absolute_logit_error = max(
                replay_max_absolute_logit_error,
                float((replayed_clean_logits - clean_logits).abs().max()),
                float((replayed_noise_logits - noise_logits).abs().max()),
            )
            final_correct["clean"].extend((clean_logits.argmax(1) == labels).cpu().tolist())
            final_correct["adapted_clean"].extend(
                (adapted_clean_logits.argmax(1) == labels).cpu().tolist()
            )
            final_correct["noise"].extend((noise_logits.argmax(1) == labels).cpu().tolist())
            final_correct["adapted_noise"].extend(
                (adapted_noise_logits.argmax(1) == labels).cpu().tolist()
            )

    stores = {
        stage: {
            condition: {metric: np.asarray(values) for metric, values in metrics.items()}
            for condition, metrics in stage_conditions.items()
        }
        for stage, stage_conditions in stores.items()
    }
    final_correct = {
        condition: np.asarray(values, dtype=bool) for condition, values in final_correct.items()
    }
    if replay_max_absolute_logit_error > 1e-4:
        raise RuntimeError(f"Manual replay mismatch: {replay_max_absolute_logit_error:.8g}")
    return stores, final_correct, replay_max_absolute_logit_error


def summarize_seed(stores, final_correct, seed, repetitions):
    stages = {}
    for stage in STAGES:
        stage_conditions = stores[stage]
        stages[str(stage)] = {
            "means": {
                condition: {metric: float(values.mean()) for metric, values in metrics.items()}
                for condition, metrics in stage_conditions.items()
            },
            "noise_repair": {
                metric: bootstrap_difference(
                    stage_conditions["adapted_noise"][metric],
                    stage_conditions["noise"][metric], seed + stage * 10, repetitions,
                )
                for metric in ("patch_cosine", "cls_cosine", "true_logit", "margin")
            },
            "clean_change": {
                metric: bootstrap_difference(
                    stage_conditions["adapted_clean"][metric],
                    stage_conditions["clean"][metric], seed + stage * 10 + 1, repetitions,
                )
                for metric in ("patch_cosine", "cls_cosine", "true_logit", "margin")
            },
        }
    return {
        "stages": stages,
        "final_noise": paired_comparison(
            final_correct["noise"], final_correct["adapted_noise"], seed + 6500, repetitions
        ),
        "final_clean": paired_comparison(
            final_correct["clean"], final_correct["adapted_clean"], seed + 6600, repetitions
        ),
    }


def save_outcomes(path, stores, final_correct):
    arrays = {}
    for stage, stage_conditions in stores.items():
        for condition, metrics in stage_conditions.items():
            for metric, values in metrics.items():
                arrays[f"block_{stage}__{condition}__{metric}"] = values
    for condition, values in final_correct.items():
        arrays[f"final__{condition}__correct"] = values
    np.savez_compressed(path, **arrays)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corruption", choices=["noise", "blur"], default="noise")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    source_summary = json.loads((args.source_run / "summary.json").read_text())
    if source_summary["configuration"]["layers"] != [1]:
        raise ValueError("Source run is not the locked Block-1 adapter experiment")
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    results = {}

    for seed in args.seeds:
        adapter = load_adapter(args.source_run / f"seed_{seed}" / "block_1.pt", device)
        loader = make_loader(
            args.corruption, splits[seed]["validation"], seed, args.image_batch_size,
            args.num_workers, args.max_samples,
        )
        stores, final_correct, replay_error = evaluate_seed(
            model, adapter, loader, device, args.alpha
        )
        save_outcomes(output_dir / f"seed_{seed}_paired_outcomes.npz", stores, final_correct)
        results[f"seed_{seed}"] = summarize_seed(
            stores, final_correct, seed, args.bootstrap_repetitions
        )
        results[f"seed_{seed}"]["manual_replay_max_absolute_logit_error"] = replay_error
        (output_dir / f"seed_{seed}_summary.json").write_text(
            json.dumps(results[f"seed_{seed}"], indent=2)
        )

    summary = {
        "configuration": {
            "source_run": str(args.source_run), "split_manifest": str(args.split_manifest),
            "split_protocol": args.split_protocol, "seeds": args.seeds,
            "corruption": args.corruption, "alpha": args.alpha,
            "intervention_block": INTERVENTION_BLOCK, "tracked_blocks": STAGES,
            "model": BASE_MODEL, "model_frozen": True, "adapter_frozen": True,
            "imageNetV2_accessed": False, "status": "development mechanistic analysis",
        },
        "results": results,
        "aggregate": {
            "noise_accuracy_gain_by_seed": [
                results[f"seed_{seed}"]["final_noise"]["accuracy_difference"] for seed in args.seeds
            ],
            "clean_accuracy_change_by_seed": [
                results[f"seed_{seed}"]["final_clean"]["accuracy_difference"] for seed in args.seeds
            ],
        },
        "limitations": [
            "Uses locked development validation splits, not final ImageNetV2 evaluation.",
            "Intermediate logit-lens metrics use the final layernorm and classifier.",
        ],
    }
    summary["aggregate"]["mean_noise_accuracy_gain"] = float(
        np.mean(summary["aggregate"]["noise_accuracy_gain_by_seed"])
    )
    summary["aggregate"]["mean_clean_accuracy_change"] = float(
        np.mean(summary["aggregate"]["clean_accuracy_change_by_seed"])
    )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved Experiment 65 to {output_dir}")


if __name__ == "__main__":
    main()
