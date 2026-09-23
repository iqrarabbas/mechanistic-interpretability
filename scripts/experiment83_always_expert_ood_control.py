import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT, BLOCK
from scripts.experiment77_learned_moe_router import load_experts
from scripts.experiment82_ood_repair_utility_router import (
    EvaluationDataset,
    UNSEEN,
    make_loader,
)


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment83_always_expert_ood_control"
SOURCE_RUN = (
    ACTIVE_ROOT
    / "results/sae/experiment82_ood_repair_utility_router/"
    "full_3seed_ood_utility_router_v1"
)
SOURCE_METHODS = (
    "mixed",
    "uniform",
    "family_router",
    "early_utility",
    "mechanism_utility",
    "oracle_margin",
)


def evaluate_condition(model, experts, data_loader, source_path, device, args, seed):
    correct = {"baseline": [], "always_noise": [], "always_blur": []}
    with torch.no_grad():
        for pixels, labels in tqdm(
            data_loader, desc="always-expert OOD control", leave=False
        ):
            labels = labels.to(device)
            outputs = model(pixel_values=pixels.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            patches = hidden[:, 1:]
            correct["baseline"].extend(
                (outputs.logits.argmax(dim=1) == labels).cpu().tolist()
            )
            for family in ("noise", "blur"):
                corrected = torch.cat(
                    (hidden[:, :1], patches + experts[family](patches)), dim=1
                )
                logits = downstream_from_layer(model, corrected, BLOCK - 1)
                correct[f"always_{family}"].extend(
                    (logits.argmax(dim=1) == labels).cpu().tolist()
                )
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    source = np.load(source_path)
    if not np.array_equal(correct["baseline"], source["correct_baseline"]):
        raise RuntimeError(f"Baseline replay mismatch against {source_path}")
    source_arrays = {
        name: source[f"correct_{name}"].astype(bool) for name in SOURCE_METHODS
    }
    result = {
        "versus_baseline": {
            name: paired_comparison(
                correct["baseline"], values, seed + offset, args.bootstrap
            )
            for offset, (name, values) in enumerate(correct.items())
            if name != "baseline"
        },
        "mechanism_router_comparisons": {
            name: paired_comparison(
                values,
                source_arrays["mechanism_utility"],
                seed + 20 + offset,
                args.bootstrap,
            )
            for offset, (name, values) in enumerate(
                (
                    ("always_noise", correct["always_noise"]),
                    ("always_blur", correct["always_blur"]),
                    ("family_router", source_arrays["family_router"]),
                    ("uniform", source_arrays["uniform"]),
                )
            )
        },
    }
    return result, correct


def aggregate(results, seeds):
    conditions = sorted(results[str(seeds[0])])
    methods = ("always_noise", "always_blur")
    output = {}
    for method in methods:
        output[method] = {
            "mean_accuracy": float(
                np.mean(
                    [
                        results[str(seed)][condition]["versus_baseline"][method][
                            "candidate_accuracy"
                        ]
                        for seed in seeds
                        for condition in conditions
                    ]
                )
            ),
            "mean_gain": float(
                np.mean(
                    [
                        results[str(seed)][condition]["versus_baseline"][method][
                            "accuracy_difference"
                        ]
                        for seed in seeds
                        for condition in conditions
                    ]
                )
            ),
        }
    output["mechanism_router_minus_always_blur"] = float(
        np.mean(
            [
                results[str(seed)][condition]["mechanism_router_comparisons"][
                    "always_blur"
                ]["accuracy_difference"]
                for seed in seeds
                for condition in conditions
            ]
        )
    )
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Always-Noise and always-Blur controls for Experiment 82"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=212)
    parser.add_argument("--ood-start", type=int, default=14000)
    parser.add_argument("--ood-samples", type=int, default=1000)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--source-run", type=Path, default=SOURCE_RUN)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("The control preserves all three frozen expert seeds")
    if args.ood_start != 14000 or args.ood_start + args.ood_samples > 15000:
        raise ValueError("OOD control is locked to [14000,15000)")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    split = {"start": args.ood_start, "end": args.ood_start + args.ood_samples}
    results = {}
    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        experts = load_experts(seed, args.rank, device)
        results[str(seed)] = {}
        for corruption_index, corruption in enumerate(UNSEEN):
            for severity in args.severities:
                key = f"{corruption}_{severity}"
                result_path = seed_dir / f"{key}.json"
                outcome_path = seed_dir / f"{key}.npz"
                if args.resume and result_path.exists() and outcome_path.exists():
                    result = json.loads(result_path.read_text())
                else:
                    dataset = EvaluationDataset(
                        split, corruption, severity, 2082, args.max_samples
                    )
                    source_path = args.source_run / f"seed_{seed}" / f"ood_{key}.npz"
                    if not source_path.exists():
                        raise FileNotFoundError(source_path)
                    result, correct = evaluate_condition(
                        model,
                        experts,
                        make_loader(dataset, args),
                        source_path,
                        device,
                        args,
                        84000 + seed * 1000 + corruption_index * 100 + severity * 10,
                    )
                    result_path.write_text(json.dumps(result, indent=2))
                    np.savez_compressed(outcome_path, **correct)
                results[str(seed)][key] = result
    summary = {
        "configuration": vars(args)
        | {
            "source_run": str(args.source_run.resolve()),
            "model": BASE_MODEL,
            "vit_frozen": True,
            "experts_frozen": True,
            "training": False,
            "selection_or_tuning": False,
            "ood_range": [args.ood_start, args.ood_start + args.ood_samples],
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "post-hoc diagnostic control for Experiment 82",
        },
        "results": results,
        "aggregate": aggregate(results, args.seeds),
        "interpretation_rule": "Mechanistic routing adds value only if it outperforms always-Blur under paired analysis; this control does not tune or retrain the router.",
        "limitations": [
            "The control was motivated after observing Experiment 82 and is therefore a post-hoc diagnostic.",
            "The source images appeared in prior mechanistic work, so this is not a pristine final benchmark.",
            "Online corruptions are controlled approximations rather than official ImageNet-C.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["aggregate"], indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
