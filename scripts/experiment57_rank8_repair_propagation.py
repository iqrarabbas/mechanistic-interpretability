import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment52_block6_repair_propagation import (
    STAGES,
    evaluate_seed,
    save_outcomes,
    summarize_seed,
)
from scripts.experiment54_lowrank_block6_adapter import LowRankHiddenAdapter, parameter_count


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment57_rank8_repair_propagation"
DEFAULT_RANK8_SOURCE = (
    PROJECT_ROOT / "results" / "sae" / "experiment54_lowrank_block6_adapter"
    / "full_3seed_ranks8_128_v1"
)
DEFAULT_FULL_SOURCE = (
    PROJECT_ROOT / "results" / "sae" / "experiment50_block6_clean_preservation"
    / "full_3seed_identity_sweep_v1"
)


def make_loader(args):
    return DataLoader(
        PairedCorruptionDataset(
            args.corruption, args.samples, args.start_index, args.corruption_seed
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )


def load_rank8(path, device):
    adapter = LowRankHiddenAdapter(8).to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval()


def load_full(path, device):
    adapter = HiddenLinear().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval()


def trajectory(summary):
    return {
        str(stage): {
            metric: summary["stages"][str(stage)]["corrected_minus_noise"][metric][
                "mean_paired_difference"
            ]
            for metric in (
                "patch_cosine_gain",
                "patch_relative_l2_reduction",
                "cls_cosine_gain",
                "cls_relative_l2_reduction",
                "margin_gain",
                "true_logit_gain",
            )
        }
        for stage in STAGES
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 57: compare rank-8 and full Block-6 repair propagation"
    )
    parser.add_argument("--rank8-source", type=Path, default=DEFAULT_RANK8_SOURCE)
    parser.add_argument("--full-source", type=Path, default=DEFAULT_FULL_SOURCE)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corruption", choices=["noise", "blur"], default="noise")
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--start-index", type=int, default=36000)
    parser.add_argument("--corruption-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if args.start_index < 36000 or args.start_index + args.samples > 50000:
        raise ValueError("Experiment 57 must remain inside reserve [36000, 50000)")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)

    per_seed = {}
    for seed in args.seeds:
        variants = {
            "rank8": load_rank8(args.rank8_source / f"seed_{seed}" / "rank_8.pt", device),
            "full": load_full(
                args.full_source / f"seed_{seed}" / "identity_0p2.pt", device
            ),
        }
        seed_record = {}
        corrected_arrays = {}
        for variant_index, (name, adapter) in enumerate(variants.items()):
            stores, final_correct, replay_error = evaluate_seed(
                model, adapter, make_loader(args), device, 1.0
            )
            save_outcomes(
                output_dir / f"seed_{seed}_{name}_paired_outcomes.npz",
                stores,
                final_correct,
            )
            result = summarize_seed(
                stores,
                final_correct,
                57000 + seed * 100 + variant_index * 20,
                args.bootstrap_repetitions,
            )
            result["trajectory"] = trajectory(result)
            result["manual_replay_max_absolute_logit_error"] = replay_error
            seed_record[name] = result
            corrected_arrays[name] = final_correct["corrected"]
        seed_record["rank8_vs_full_final"] = paired_comparison(
            corrected_arrays["full"], corrected_arrays["rank8"],
            57500 + seed, args.bootstrap_repetitions,
        )
        per_seed[str(seed)] = seed_record
        (output_dir / f"seed_{seed}_summary.json").write_text(
            json.dumps(seed_record, indent=2)
        )

    aggregate = {}
    for name in ("rank8", "full"):
        aggregate[name] = {
            "final_gains_by_seed": [
                per_seed[str(seed)][name]["final_classification"]["accuracy_difference"]
                for seed in args.seeds
            ],
            "mean_final_gain": float(np.mean([
                per_seed[str(seed)][name]["final_classification"]["accuracy_difference"]
                for seed in args.seeds
            ])),
            "mean_trajectory": {
                str(stage): {
                    metric: float(np.mean([
                        per_seed[str(seed)][name]["trajectory"][str(stage)][metric]
                        for seed in args.seeds
                    ]))
                    for metric in per_seed[str(args.seeds[0])][name]["trajectory"][str(stage)]
                }
                for stage in STAGES
            },
        }
    summary = {
        "configuration": vars(args) | {
            "rank8_source": str(args.rank8_source.resolve()),
            "full_source": str(args.full_source.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "intervention_block": 6,
            "tracked_blocks": STAGES,
            "rank8_parameters": parameter_count(8),
            "full_parameters": 741120,
            "frozen_components": ["ViT", "rank8 adapters", "full adapters"],
            "clean_counterpart_role": "analysis metrics only; never used by either correction",
            "imageNetV2_accessed": False,
            "status": "frozen held-out reserve mechanistic evaluation",
        },
        "per_seed": per_seed,
        "aggregate": aggregate,
        "interpretation_rule": (
            "Propagation is supported when Block-6 patch repair is followed by increasing "
            "downstream CLS/margin benefit and a positive final paired accuracy gain."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"aggregate": aggregate}, indent=2))
    print(f"Saved Experiment 57 to {output_dir}")


if __name__ == "__main__":
    main()
