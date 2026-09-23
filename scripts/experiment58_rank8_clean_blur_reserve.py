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
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment52_block6_repair_propagation import (
    INTERVENTION_BLOCK,
    STAGES,
    append_values,
    logit_lens,
    representation_metrics,
    save_outcomes,
    summarize_seed,
)
from scripts.experiment54_lowrank_block6_adapter import LowRankHiddenAdapter, parameter_count


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment58_rank8_clean_blur_reserve"
DEFAULT_SOURCE = (
    PROJECT_ROOT / "results" / "sae" / "experiment54_lowrank_block6_adapter"
    / "full_3seed_ranks8_128_v1"
)


def make_loader(args):
    dataset = PairedCorruptionDataset(
        "blur", args.samples, args.start_index, args.corruption_seed
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def load_adapter(path, device):
    adapter = LowRankHiddenAdapter(8).to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval()


def evaluate(model, adapter, loader, device):
    metric_names = [
        "patch_cosine", "patch_relative_l2", "cls_cosine", "cls_relative_l2",
        "true_logit", "margin",
    ]
    stores = {
        stage: {
            condition: {metric: [] for metric in metric_names}
            for condition in ("clean", "noise", "corrected")
        }
        for stage in STAGES
    }
    final_correct = {
        "clean": [], "clean_corrected": [], "noise": [], "corrected": []
    }
    replay_error = 0.0
    with torch.no_grad():
        for clean, blur, labels in tqdm(loader, desc="Rank-8 clean/Blur-4 propagation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, blur]).to(device), output_hidden_states=True
            )
            clean_logits, blur_logits = outputs.logits.split(batch)
            clean_hidden6, blur_hidden6 = outputs.hidden_states[INTERVENTION_BLOCK].split(batch)
            clean_corrected = torch.cat([
                clean_hidden6[:, :1],
                clean_hidden6[:, 1:] + adapter(clean_hidden6[:, 1:]),
            ], dim=1)
            blur_corrected = torch.cat([
                blur_hidden6[:, :1],
                blur_hidden6[:, 1:] + adapter(blur_hidden6[:, 1:]),
            ], dim=1)
            blur_replayed = blur_hidden6.clone()

            for stage in STAGES:
                if stage > INTERVENTION_BLOCK:
                    layer = model.vit.layers[stage - 1]
                    clean_corrected = layer(clean_corrected, attention_mask=None)
                    blur_corrected = layer(blur_corrected, attention_mask=None)
                    blur_replayed = layer(blur_replayed, attention_mask=None)
                clean_hidden, blur_hidden = outputs.hidden_states[stage].split(batch)
                for condition, hidden in (
                    ("clean", clean_hidden),
                    ("noise", blur_hidden),
                    ("corrected", blur_corrected),
                ):
                    representation = representation_metrics(clean_hidden, hidden)
                    _, true_logit, margin = logit_lens(model, hidden, labels)
                    append_values(
                        stores[stage][condition],
                        representation | {"true_logit": true_logit, "margin": margin},
                    )

            clean_corrected_logits, _, _ = logit_lens(model, clean_corrected, labels)
            blur_corrected_logits, _, _ = logit_lens(model, blur_corrected, labels)
            replayed_logits, _, _ = logit_lens(model, blur_replayed, labels)
            replay_error = max(
                replay_error, float((replayed_logits - blur_logits).abs().max().item())
            )
            final_correct["clean"].extend(
                (clean_logits.argmax(1) == labels).cpu().tolist()
            )
            final_correct["clean_corrected"].extend(
                (clean_corrected_logits.argmax(1) == labels).cpu().tolist()
            )
            final_correct["noise"].extend(
                (blur_logits.argmax(1) == labels).cpu().tolist()
            )
            final_correct["corrected"].extend(
                (blur_corrected_logits.argmax(1) == labels).cpu().tolist()
            )
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
        name: np.asarray(values, dtype=bool) for name, values in final_correct.items()
    }
    if replay_error > 1e-4:
        raise RuntimeError(f"Downstream replay error is too large: {replay_error}")
    return stores, final_correct, replay_error


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 58: held-out rank-8 clean tradeoff and Blur-4 propagation"
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=36000)
    parser.add_argument("--corruption-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if args.start_index < 36000 or args.start_index + args.samples > 50000:
        raise ValueError("Evaluation must remain inside reserve [36000, 50000)")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    per_seed = {}

    for seed in args.seeds:
        adapter = load_adapter(args.source_run / f"seed_{seed}" / "rank_8.pt", device)
        stores, final_correct, replay_error = evaluate(
            model, adapter, make_loader(args), device
        )
        save_outcomes(
            output_dir / f"seed_{seed}_paired_outcomes.npz", stores, final_correct
        )
        result = summarize_seed(
            stores, final_correct, 58000 + seed, args.bootstrap_repetitions
        )
        result["clean_corrected_vs_clean"] = paired_comparison(
            final_correct["clean"], final_correct["clean_corrected"],
            58500 + seed, args.bootstrap_repetitions,
        )
        result["manual_replay_max_absolute_logit_error"] = replay_error
        per_seed[str(seed)] = result
        (output_dir / f"seed_{seed}_summary.json").write_text(json.dumps(result, indent=2))

    clean_changes = [
        per_seed[str(seed)]["clean_corrected_vs_clean"]["accuracy_difference"]
        for seed in args.seeds
    ]
    blur_gains = [
        per_seed[str(seed)]["final_classification"]["accuracy_difference"]
        for seed in args.seeds
    ]
    trajectory = {
        str(stage): {
            metric: float(np.mean([
                per_seed[str(seed)]["stages"][str(stage)]["corrected_minus_noise"][metric][
                    "mean_paired_difference"
                ]
                for seed in args.seeds
            ]))
            for metric in (
                "patch_cosine_gain", "patch_relative_l2_reduction", "cls_cosine_gain",
                "cls_relative_l2_reduction", "margin_gain", "true_logit_gain",
            )
        }
        for stage in STAGES
    }
    summary = {
        "configuration": vars(args) | {
            "source_run": str(args.source_run.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "adapter": "rank_8",
            "adapter_parameters": parameter_count(8),
            "intervention_block": 6,
            "frozen_components": ["ViT", "rank-8 adapters"],
            "clean_counterpart_role": "analysis only; correction uses each image's own hidden state",
            "imageNetV2_accessed": False,
            "status": "frozen held-out reserve evaluation",
        },
        "per_seed": per_seed,
        "aggregate": {
            "clean_changes_by_seed": clean_changes,
            "mean_clean_change": float(np.mean(clean_changes)),
            "blur4_gains_by_seed": blur_gains,
            "mean_blur4_gain": float(np.mean(blur_gains)),
            "mean_blur_trajectory": trajectory,
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))
    print(f"Saved Experiment 58 to {output_dir}")


if __name__ == "__main__":
    main()
