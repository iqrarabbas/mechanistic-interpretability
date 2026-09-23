import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment101_routed_batchtopk_sae_repairs import (
    BLOCK,
    EvaluationDataset,
    THRESHOLD_SUMMARY,
    evaluate,
    load_repair,
    load_router,
)


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment102_safe_routed_blur_severity_sweep"


def main():
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    args.methods = ["safe_router", "always_blur"]

    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    split_map = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    thresholds = json.loads(THRESHOLD_SUMMARY.read_text())["results"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)

    results = {}
    outcomes = {}
    for seed in args.seeds:
        router, mean, std = load_router(seed, device)
        repairs = {family: load_repair(seed, family, device) for family in ("noise", "blur")}
        threshold = thresholds[str(seed)]["selected_clean_probability_threshold"]
        start = split_map[seed]["validation"]["start"] + 166
        samples = split_map[seed]["validation"]["end"] - start
        if args.max_samples is not None:
            samples = min(samples, args.max_samples)
        results[str(seed)] = {}
        for severity in args.severities:
            result, arrays = evaluate(
                model,
                router,
                mean,
                std,
                repairs,
                threshold,
                EvaluationDataset(start, samples, "gaussian_blur", seed, severity),
                "blur",
                device,
                args,
                102000 + seed * 1000 + severity * 20,
            )
            results[str(seed)][str(severity)] = result
            for name, values in arrays.items():
                outcomes[f"seed{seed}_severity{severity}_{name}"] = values
        del router, repairs
        torch.cuda.empty_cache()

    aggregate = {}
    for severity in args.severities:
        key = str(severity)
        aggregate[key] = {}
        for method in args.methods:
            gains = [
                results[str(seed)][key]["methods"][method]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[key][method] = {
                "gains_by_seed": gains,
                "mean_gain": float(np.mean(gains)),
            }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "block": BLOCK,
            "repair_training_severity": 4,
            "router_training_severity": 4,
            "all_components_frozen": True,
            "clean_counterpart_used_at_inference": False,
            "corruption_label_used_by_safe_router": False,
            "imageNetV2_accessed": False,
            "status": "frozen development severity generalization; no tuning",
        },
        "evaluation_ranges": {
            str(seed): [
                split_map[seed]["validation"]["start"] + 166,
                split_map[seed]["validation"]["end"],
            ]
            for seed in args.seeds
        },
        "results": results,
        "aggregate": aggregate,
        "guardrails": [
            "The router, thresholds, SAEs, repair bases, and repair predictors remain frozen.",
            "Severity is varied only during evaluation; no severity-specific selection occurs.",
            "The same image indices are paired across all severities and methods.",
            "This development range was used to select the repair checkpoint, so it is not a pristine final benchmark.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
