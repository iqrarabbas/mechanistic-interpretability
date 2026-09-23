import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment77_learned_moe_router import SketchConditionDataset, load_experts
from scripts.experiment82_ood_repair_utility_router import (
    OUTPUT_ROOT as UTILITY_ROOT,
    UtilityRouter,
    evaluate,
    load_family_router,
)
from scripts.experiment111_leakage_free_head_anomaly_confirmation import freeze_manifest, sha256
from scripts.experiment113_all_corruption_head_detection import PREVIOUS_CONFIRMATIONS
from scripts.experiment114_multisignal_clean_detector import EXPERIMENT113_MANIFEST
from scripts.experiment115_consistency_clean_detector import EXPERIMENT114_MANIFEST
from scripts.experiment116_multiview_clean_detector import EXPERIMENT115_MANIFEST
from scripts.experiment117_paired_detector_fusion import VIEW_MANIFEST
from scripts.experiment107_fresh_sketch_defocus_confirmation import DATA_ROOT


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment118_frozen_utility_router_sketch"
UTILITY_RUN = UTILITY_ROOT / "full_3seed_ood_utility_router_v1"
EXPERIMENT117_MANIFEST = (
    ACTIVE_ROOT / "results/sae/experiment117_paired_detector_fusion/"
    "full_level4_paired_clean_calibrated_fusion_v1/frozen_manifest.json"
)
CONDITIONS = (
    "clean", "shot_noise", "impulse_noise", "defocus", "jpeg",
    "pixelate", "brightness", "contrast",
)
PRIMARY_METHODS = ("mixed", "family_router", "early_utility", "mechanism_utility")


def load_utility_router(seed, name, device):
    path = UTILITY_RUN / f"seed_{seed}" / f"{name}_utility_router.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    router = UtilityRouter(payload["input_dim"]).to(device)
    router.load_state_dict(payload["router"])
    return router.eval(), payload["mean"], payload["std"]


def evaluate_condition(model, experts, mixed, routers, family_router, manifest, condition, seed, args, path):
    if args.resume and path.exists() and path.with_suffix(".npz").exists():
        return json.loads(path.read_text())
    dataset = SketchConditionDataset(
        DATA_ROOT, manifest, None if condition == "clean" else condition,
        args.severity, args.corruption_seed, args.max_samples,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    result, correct, margins, actions = evaluate(
        model, experts, mixed, routers, family_router, loader, model.device,
        args, 118000 + seed * 1000 + CONDITIONS.index(condition) * 20,
    )
    result["condition"] = condition
    result["samples"] = len(dataset)
    result["baseline_accuracy"] = float(correct["baseline"].mean())
    result["deployable_router_selection_is_image_only"] = True
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(
        temporary,
        **{f"correct_{name}": values for name, values in correct.items()},
        **{f"margin_{name}": values for name, values in margins.items()},
        **{f"action_{name}": np.asarray(values, dtype=np.int8) for name, values in actions.items()},
    )
    temporary.replace(path.with_suffix(".npz"))
    path.write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--corruption-seed", type=int, default=118000)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--rank", type=int, default=212)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.rank != 212 or args.severity != 4:
        raise ValueError("The frozen confirmation requires rank 212 and severity 4")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    configuration = vars(args) | {"resume": False}
    configuration_path = output_dir / "configuration.json"
    if configuration_path.exists():
        if json.loads(configuration_path.read_text()) != configuration:
            raise ValueError("Resume arguments differ from the frozen configuration")
    else:
        if args.resume and any(output_dir.iterdir()):
            raise ValueError("Cannot resume a run without its configuration")
        configuration_path.write_text(json.dumps(configuration, indent=2))
    manifest_path = output_dir / "frozen_manifest.json"
    exclusions = (
        *PREVIOUS_CONFIRMATIONS, EXPERIMENT113_MANIFEST, EXPERIMENT114_MANIFEST,
        EXPERIMENT115_MANIFEST, VIEW_MANIFEST, EXPERIMENT117_MANIFEST,
    )
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else freeze_manifest(manifest_path, exclusions)
    if not manifest["items"]:
        raise ValueError("No unseen Sketch images remain")
    print("Manifest", sha256(manifest_path), "images", manifest["samples"], flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    model.requires_grad_(False)
    results = {}
    for seed in (0, 1, 2):
        print("Seed", seed, flush=True)
        experts = load_experts(seed, args.rank, device)
        from scripts.experiment76_parameter_matched_oracle_moe import load_full_mixed
        mixed = load_full_mixed(seed, device)
        family_router = load_family_router(seed, device)[:3]
        routers = {name: load_utility_router(seed, name, device) for name in ("early", "mechanism")}
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        results[str(seed)] = {}
        for condition in CONDITIONS:
            print("Evaluating", condition, flush=True)
            results[str(seed)][condition] = evaluate_condition(
                model, experts, mixed, routers, family_router, manifest, condition,
                seed, args, seed_dir / f"{condition}.json",
            )
        del experts, mixed, routers
        torch.cuda.empty_cache()
    summary = {
        "configuration": configuration | {
            "model": BASE_MODEL,
            "manifest_sha256": sha256(manifest_path),
            "utility_router_checkpoint": str(UTILITY_RUN),
            "backbone_and_experts_frozen": True,
            "clean_counterpart_at_inference": False,
            "true_label_or_corruption_name_used_by_deployable_routers": False,
            "benchmark": "online severity-4 corruptions of new hash-disjoint ImageNet-Sketch images",
        },
        "results": results,
        "limitations": [
            "The utility routers were trained in Experiment 82, not retrained for Sketch.",
            "The source dataset/domain was previously studied, though these image hashes are excluded from prior frozen manifests.",
            "This is not official ImageNet-C; no routing threshold or checkpoint was selected on this new image set.",
            "The oracle-margin method uses true labels and is diagnostic only; exclude it from deployable comparisons.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for condition in CONDITIONS:
        baseline = np.mean([results[str(seed)][condition]["baseline_accuracy"] for seed in (0, 1, 2)])
        row = {name: round(100 * np.mean([
            results[str(seed)][condition]["methods"][name]["candidate_accuracy"]
            for seed in (0, 1, 2)
        ]), 2) for name in PRIMARY_METHODS}
        print(condition, "baseline", round(100 * baseline, 2), row, flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
