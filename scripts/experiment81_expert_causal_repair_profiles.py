import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment75_block7_qkv_path_mediation import evaluate
from scripts.experiment76_parameter_matched_oracle_moe import (
    EXPERT_FAMILIES,
    loader,
)
from scripts.experiment54_lowrank_block6_adapter import LowRankHiddenAdapter


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
DEFAULT_EXPERT_ROOT = (
    ACTIVE_ROOT
    / "results/sae/experiment76_parameter_matched_oracle_moe/"
    "full_3seed_rank212_oracle_v1"
)
DEFAULT_OUTPUT_ROOT = (
    ACTIVE_ROOT / "results/sae/experiment81_expert_causal_repair_profiles"
)
CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "gaussian_blur",
    "disk_blur",
    "motion_blur",
)


def load_experts(expert_root, seed, rank, device):
    experts = {}
    paths = {}
    for family in EXPERT_FAMILIES:
        path = expert_root / f"seed_{seed}" / f"{family}_rank{rank}.pt"
        if not path.exists():
            raise FileNotFoundError(path)
        adapter = LowRankHiddenAdapter(rank).to(device)
        adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        experts[family] = adapter.eval()
        paths[family] = str(path.resolve())
    return experts, paths


def compact_result(result):
    conditions = result["conditions"]
    return {
        "baseline_accuracy": result["baseline_to_full"]["reference_accuracy"],
        "adapted_accuracy": result["baseline_to_full"]["candidate_accuracy"],
        "adapter_gain": result["baseline_to_full"]["accuracy_difference"],
        "adapter_gain_95ci": result["baseline_to_full"]["accuracy_difference_95ci"],
        "adapter_gain_mcnemar_exact_pvalue": result["baseline_to_full"][
            "mcnemar_exact_pvalue"
        ],
        "qkv_gain_lost": conditions["revert_qkv"]["adapter_gain_lost"],
        "attention_output_gain_lost": conditions["revert_attention_output"][
            "adapter_gain_lost"
        ],
        "mlp_gain_lost": conditions["revert_mlp_output"]["adapter_gain_lost"],
        "value_gain_lost": conditions["revert_v"]["adapter_gain_lost"],
    }


def aggregate(results, seeds):
    aggregate_results = {}
    for corruption in ("clean", *CORRUPTIONS):
        aggregate_results[corruption] = {}
        for expert in EXPERT_FAMILIES:
            entries = [results[str(seed)][corruption][expert] for seed in seeds]
            compact = [compact_result(entry) for entry in entries]
            aggregate_results[corruption][expert] = {
                key: float(np.mean([entry[key] for entry in compact]))
                for key in (
                    "baseline_accuracy",
                    "adapted_accuracy",
                    "adapter_gain",
                    "qkv_gain_lost",
                    "attention_output_gain_lost",
                    "mlp_gain_lost",
                    "value_gain_lost",
                )
            }
        noise_gain = aggregate_results[corruption]["noise"]["adapter_gain"]
        blur_gain = aggregate_results[corruption]["blur"]["adapter_gain"]
        aggregate_results[corruption]["preferred_expert_by_mean_accuracy_gain"] = (
            "noise" if noise_gain > blur_gain else "blur"
        )
        aggregate_results[corruption]["expert_gain_gap_noise_minus_blur"] = (
            noise_gain - blur_gain
        )
    return aggregate_results


def main():
    parser = argparse.ArgumentParser(
        description="Crossed expert-by-corruption causal Block-7 repair profiles"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=212)
    parser.add_argument(
        "--phase", choices=("discovery", "confirmation"), default="discovery"
    )
    parser.add_argument("--start-index", type=int, default=13000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--expert-root", type=Path, default=DEFAULT_EXPERT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    end_index = args.start_index + args.samples
    locked_interval = {
        "discovery": (13000, 14000),
        "confirmation": (14000, 15000),
    }[args.phase]
    if args.start_index < locked_interval[0] or end_index > locked_interval[1]:
        raise ValueError(f"{args.phase.title()} is locked to {locked_interval}")
    output_dir = args.output_root / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)
    split = {"start": args.start_index, "end": end_index}
    results = {}
    outcomes = {}
    checkpoint_paths = {}
    for seed in args.seeds:
        experts, paths = load_experts(args.expert_root, seed, args.rank, device)
        checkpoint_paths[str(seed)] = paths
        results[str(seed)] = {}
        for corruption_index, corruption in enumerate(("clean", *CORRUPTIONS)):
            condition_loader = loader(
                split,
                (corruption,),
                seed,
                args,
                False,
                fixed=corruption,
            )
            results[str(seed)][corruption] = {}
            for expert_index, (expert_name, expert) in enumerate(experts.items()):
                statistical_seed = (
                    81000 + seed * 1000 + corruption_index * 100 + expert_index * 20
                )
                result, correct, margins = evaluate(
                    model,
                    expert,
                    condition_loader,
                    device,
                    args,
                    statistical_seed,
                )
                results[str(seed)][corruption][expert_name] = result
                prefix = f"seed{seed}_{corruption}_{expert_name}"
                for name, values in correct.items():
                    outcomes[f"{prefix}_correct_{name}"] = values
                for name, values in margins.items():
                    outcomes[f"{prefix}_margin_{name}"] = values
    summary = {
        "configuration": vars(args)
        | {
            "expert_root": str(args.expert_root.resolve()),
            "output_root": str(args.output_root.resolve()),
            "model": BASE_MODEL,
            "vit_frozen": True,
            "experts_frozen": True,
            "adapter_location": "after Block 6",
            "decomposed_layer": 7,
            "analysis_phase": args.phase,
            "analysis_range": [args.start_index, end_index],
            "mechanism_discovery_range": [13000, 14000],
            "mechanism_confirmation_range": [14000, 15000],
            "expert_training_ranges": [[15000, 20000], [22000, 27000], [29000, 34000]],
            "expert_validation_ranges": [[20000, 22000], [27000, 29000], [34000, 36000]],
            "uses_clean_counterpart": False,
            "uses_corruption_label_at_inference": False,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": f"mechanism {args.phase} only; no architecture or router fitted",
        },
        "checkpoint_paths": checkpoint_paths,
        "results": results,
        "aggregate": aggregate(results, args.seeds),
        "interpretation_rules": [
            "Adapter gain measures whether an expert helps each corruption.",
            "Gain lost after component reversion measures causal mediation or necessity, not sufficiency.",
            "Mechanistic labels require agreement between the locked discovery and confirmation analyses.",
        ],
        "leakage_guardrails": [
            "Neither [13000,14000) nor [14000,15000) overlaps any expert train or validation range.",
            "The confirmation protocol was frozen after discovery and uses no discovery-driven tuning.",
            "No final benchmark, ImageNetV2, or ImageNet-Sketch is accessed.",
            "These images were used in earlier feature-protocol development, so this is not an untouched final benchmark.",
            "Any router motivated by this result must be trained and validated on separate partitions.",
        ],
        "required_architecture_control": {
            "status": "not run by this experiment",
            "description": "Train one parameter-matched monolithic Block-6 adapter on the same six corruptions, with matched image exposure and optimization, before attributing an MoE gain to specialization.",
        },
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
