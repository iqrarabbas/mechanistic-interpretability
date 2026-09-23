import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT, BLOCK, load_full_mixed
from scripts.experiment77_learned_moe_router import SketchConditionDataset, load_experts
from scripts.experiment82_ood_repair_utility_router import expert_mechanism_features
from scripts.experiment111_leakage_free_head_anomaly_confirmation import OLD_MANIFEST, freeze_manifest, sha256
from scripts.experiment113_all_corruption_head_detection import PREVIOUS_CONFIRMATIONS
from scripts.experiment114_multisignal_clean_detector import EXPERIMENT113_MANIFEST
from scripts.experiment115_consistency_clean_detector import EXPERIMENT114_MANIFEST
from scripts.experiment116_multiview_clean_detector import EXPERIMENT115_MANIFEST
from scripts.experiment117_paired_detector_fusion import VIEW_MANIFEST
from scripts.experiment118_frozen_utility_router_sketch import (
    CONDITIONS, DATA_ROOT, EXPERIMENT117_MANIFEST, OUTPUT_ROOT as EXPERIMENT118_ROOT,
)


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment119_abstaining_repair_selector"
EXPERIMENT118_MANIFEST = (
    EXPERIMENT118_ROOT / "full_frozen_3seed_clean_noise_blur_ood_v1/frozen_manifest.json"
)
ACTIONS = ("none", "noise", "blur")


def prediction_features(logits):
    probabilities = logits.softmax(dim=1)
    top = probabilities.topk(2, dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    return torch.stack((top[:, 0], top[:, 0] - top[:, 1], entropy), dim=1)


def extract_condition(model, experts, mixed, manifest, condition, corruption_seed, args, path):
    if args.resume and path.exists():
        with np.load(path) as saved:
            return {name: saved[name] for name in saved.files}
    dataset = SketchConditionDataset(
        DATA_ROOT, manifest, None if condition == "clean" else condition,
        args.severity, corruption_seed, None,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    features, correctness, mixed_correctness = [], [], []
    with torch.no_grad():
        for batch_index, (pixels, labels) in enumerate(loader):
            outputs = model(pixel_values=pixels.to(model.device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            patches = hidden[:, 1:]
            residuals = {name: expert(patches) for name, expert in experts.items()}
            logits = {"none": outputs.logits}
            for name, residual in residuals.items():
                corrected = torch.cat((hidden[:, :1], patches + residual), dim=1)
                logits[name] = downstream_from_layer(model, corrected, BLOCK - 1)
            mixed_hidden = torch.cat((hidden[:, :1], patches + mixed(patches)), dim=1)
            mixed_logits = downstream_from_layer(model, mixed_hidden, BLOCK - 1)
            mechanism = torch.cat(
                [expert_mechanism_features(model, hidden, residuals[name]) for name in ("noise", "blur")],
                dim=1,
            )
            probability_features = torch.cat([prediction_features(logits[name]) for name in ACTIONS], dim=1)
            base_prediction = logits["none"].argmax(dim=1)
            agreement = torch.stack(
                [(logits[name].argmax(dim=1) == base_prediction).float() for name in ("noise", "blur")],
                dim=1,
            )
            features.append(torch.cat((mechanism, probability_features, agreement), dim=1).cpu().numpy())
            correctness.append(torch.stack(
                [(logits[name].argmax(dim=1) == labels.to(model.device)) for name in ACTIONS], dim=1,
            ).cpu().numpy())
            mixed_correctness.append((mixed_logits.argmax(dim=1) == labels.to(model.device)).cpu().numpy())
            if batch_index % 50 == 0:
                print(f"  {condition}: {batch_index}/{len(loader)} batches", flush=True)
    result = {
        "features": np.concatenate(features).astype(np.float32),
        "correct": np.concatenate(correctness).astype(bool),
        "mixed_correct": np.concatenate(mixed_correctness).astype(bool),
    }
    temporary = path.with_name(path.stem + ".partial.npz")
    np.savez_compressed(temporary, **result)
    temporary.replace(path)
    return result


def collect_split(model, experts, mixed, manifest, seed, args, directory, split):
    data = {}
    corruption_seed = args.corruption_seed + {"train": 0, "validation": 100000, "test": 200000}[split]
    for condition in CONDITIONS:
        print(f"Seed {seed}: {split} {condition}", flush=True)
        data[condition] = extract_condition(
            model, experts, mixed, manifest, condition, corruption_seed, args,
            directory / f"{split}_{condition}.npz",
        )
    return data


def fit_selector(training):
    features = np.concatenate([training[name]["features"] for name in CONDITIONS])
    correct = np.concatenate([training[name]["correct"] for name in CONDITIONS])
    scaler = StandardScaler().fit(features)
    transformed = scaler.transform(features)
    models = {}
    event_rates = {}
    for index, action in enumerate(ACTIONS[1:], start=1):
        recover = (~correct[:, 0] & correct[:, index]).astype(np.int8)
        damage = (correct[:, 0] & ~correct[:, index]).astype(np.int8)
        models[action] = {
            "recover": LogisticRegression(C=0.1, max_iter=1000).fit(transformed, recover),
            "damage": LogisticRegression(C=0.1, max_iter=1000).fit(transformed, damage),
        }
        event_rates[action] = {"recover": float(recover.mean()), "damage": float(damage.mean())}
    return scaler, models, event_rates


def action_scores(features, scaler, models):
    transformed = scaler.transform(features)
    return np.stack([
        models[action]["recover"].predict_proba(transformed)[:, 1]
        - models[action]["damage"].predict_proba(transformed)[:, 1]
        for action in ACTIONS[1:]
    ], axis=1)


def select_actions(scores, threshold):
    strongest = scores.argmax(axis=1) + 1
    return np.where(scores.max(axis=1) > threshold, strongest, 0)


def candidate_accuracy(data, scores, threshold):
    selected = select_actions(scores, threshold)
    return float(data["correct"][np.arange(len(selected)), selected].mean())


def calibrate_threshold(validation, scaler, models, clean_tolerance):
    scores = {name: action_scores(validation[name]["features"], scaler, models) for name in CONDITIONS}
    combined = np.concatenate([values.max(axis=1) for values in scores.values()])
    thresholds = np.unique(np.r_[np.quantile(combined, np.linspace(0, 1, 101)), np.inf])
    clean_baseline = float(validation["clean"]["correct"][:, 0].mean())
    feasible = []
    for threshold in thresholds:
        clean_accuracy = candidate_accuracy(validation["clean"], scores["clean"], threshold)
        if clean_accuracy + 1e-12 < clean_baseline - clean_tolerance:
            continue
        corruption_accuracy = float(np.mean([
            candidate_accuracy(validation[name], scores[name], threshold)
            for name in CONDITIONS[1:]
        ]))
        feasible.append((corruption_accuracy, clean_accuracy, float(threshold)))
    selected = max(feasible)
    return selected[2], {
        "clean_baseline": clean_baseline,
        "clean_selected": selected[1],
        "corruption_macro_selected": selected[0],
        "clean_tolerance": clean_tolerance,
        "thresholds_checked": len(thresholds),
    }


def evaluate_test(test, scaler, models, threshold, seed, args):
    results = {}
    for condition_index, condition in enumerate(CONDITIONS):
        data = test[condition]
        scores = action_scores(data["features"], scaler, models)
        selected = select_actions(scores, threshold)
        baseline = data["correct"][:, 0]
        repaired = data["correct"][np.arange(len(selected)), selected]
        results[condition] = {
            "selected_vs_base": paired_comparison(baseline, repaired, 119000 + seed * 100 + condition_index, args.bootstrap),
            "mixed_vs_base": paired_comparison(baseline, data["mixed_correct"], 120000 + seed * 100 + condition_index, args.bootstrap),
            "route_fractions": {action: float((selected == index).mean()) for index, action in enumerate(ACTIONS)},
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--corruption-seed", type=int, default=119000)
    parser.add_argument("--clean-tolerance", type=float, default=0.002)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.severity != 4 or args.clean_tolerance < 0:
        raise ValueError("Severity 4 and a nonnegative clean tolerance are required")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    config_path = output_dir / "configuration.json"
    configuration = vars(args) | {"resume": False}
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError("Resume arguments differ from the frozen configuration")
    else:
        if args.resume and any(output_dir.iterdir()):
            raise ValueError("Cannot resume without frozen configuration")
        config_path.write_text(json.dumps(configuration, indent=2))
    old_items = json.loads(OLD_MANIFEST.read_text())["items"]
    train_items = old_items[0::3]
    validation_items = old_items[1::3]
    if len(train_items) != 1000 or len(validation_items) != 1000:
        raise ValueError("The old Sketch development manifest is not three images per class")
    train_manifest = {"items": train_items}
    validation_manifest = {"items": validation_items}
    manifest_path = output_dir / "frozen_test_manifest.json"
    exclusions = (
        *PREVIOUS_CONFIRMATIONS, EXPERIMENT113_MANIFEST, EXPERIMENT114_MANIFEST,
        EXPERIMENT115_MANIFEST, VIEW_MANIFEST, EXPERIMENT117_MANIFEST,
        EXPERIMENT118_MANIFEST,
    )
    test_manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else freeze_manifest(manifest_path, exclusions)
    development_hashes = {item["sha256"] for item in train_items + validation_items}
    if development_hashes & {item["sha256"] for item in test_manifest["items"]}:
        raise ValueError("Development/test image hash overlap")
    print("Frozen test manifest", sha256(manifest_path), "images", test_manifest["samples"], flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True,
    ).to(device).eval()
    model.requires_grad_(False)
    results = {}
    for seed in (0, 1, 2):
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        experts = load_experts(seed, 212, device)
        mixed = load_full_mixed(seed, device)
        train = collect_split(model, experts, mixed, train_manifest, seed, args, seed_dir, "train")
        validation = collect_split(model, experts, mixed, validation_manifest, seed, args, seed_dir, "validation")
        scaler, models, rates = fit_selector(train)
        threshold, calibration = calibrate_threshold(validation, scaler, models, args.clean_tolerance)
        (seed_dir / "frozen_selector.json").write_text(json.dumps({
            "threshold": threshold, "calibration": calibration, "event_rates": rates,
        }, indent=2))
        print(f"Seed {seed}: selected threshold {threshold:.6f}; validation {calibration}", flush=True)
        test = collect_split(model, experts, mixed, test_manifest, seed, args, seed_dir, "test")
        results[str(seed)] = evaluate_test(test, scaler, models, threshold, seed, args)
        (seed_dir / "results.json").write_text(json.dumps(results[str(seed)], indent=2))
        del experts, mixed, train, validation, test
        torch.cuda.empty_cache()
    summary = {
        "configuration": configuration | {
            "model": BASE_MODEL,
            "test_manifest_sha256": sha256(manifest_path),
            "development_images": "old Sketch manifest: first/second distinct image per class for train/validation",
            "test_images": "next SHA-disjoint Sketch image per class after excluding all earlier frozen manifests",
            "backbone_and_experts_frozen": True,
            "selector_uses_clean_counterpart_or_label_at_inference": False,
            "benchmark": "online severity-4 corruptions of ImageNet-Sketch, not official ImageNet-C",
        },
        "results": results,
        "limitations": [
            "The development Sketch images were used in earlier research, although they are disjoint from this final image manifest.",
            "The corruption types are shared between selector development and test; this tests new images, not unseen corruption types.",
            "The clean-accuracy tolerance is enforced on validation and may not hold exactly on new images.",
            "Existing frozen experts were trained on ImageNet validation, not Sketch; this is not a from-scratch three-stage training protocol.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    for condition in CONDITIONS:
        gain = np.mean([
            results[str(seed)][condition]["selected_vs_base"]["accuracy_difference"] for seed in (0, 1, 2)
        ])
        no_repair = np.mean([
            results[str(seed)][condition]["route_fractions"]["none"] for seed in (0, 1, 2)
        ])
        print(condition, "mean gain pp", round(100 * gain, 2), "no-repair fraction", round(no_repair, 3), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
