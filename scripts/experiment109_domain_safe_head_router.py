import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment69_imagenet_sketch_frozen_generalization import ManifestDataset
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment99_attention_head_corruption_detector import HeadRouter, attention_head_statistics
from scripts.experiment101_routed_batchtopk_sae_repairs import BLOCK, ROUTER_ROOT, corrected_logits, load_repair
from scripts.experiment107_fresh_sketch_defocus_confirmation import SketchDefocusDataset


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment109_domain_safe_head_router"
PRIOR_SKETCH_MANIFEST = Path("/media/dr-yougart/Iqrar/datasets/imagenet_sketch/frozen_subset_3_per_class_unique_manifest.json")
PRIOR_SKETCH_ROOT = Path("/media/dr-yougart/Iqrar/datasets/imagenet_sketch/extracted/sketch")
FRESH_RUN = ACTIVE_ROOT / "results/sae/experiment107_fresh_sketch_defocus_confirmation/full_1000_disjoint_clean_defocus345_3seed_v1"
FAMILIES = {0: ("clean",), 1: ("gaussian_noise", "shot_noise", "impulse_noise"), 2: ("gaussian_blur", "disk_blur", "motion_blur")}


def extract(model, dataset, device, args, path):
    if path.exists():
        return torch.load(path, weights_only=True)
    values = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for index, pixels_and_labels in enumerate(loader):
            pixels = pixels_and_labels[0].to(device)
            outputs = model(pixel_values=pixels, output_attentions=True, return_dict=True)
            values.append(attention_head_statistics(outputs.attentions).cpu())
            if index % 50 == 0:
                print(f"Sketch feature batches: {index}/{len(loader)}", flush=True)
    values = torch.cat(values)
    torch.save(values, path)
    return values


def load_photo_features(seed_dir, prefix):
    features, labels = [], []
    for label, conditions in FAMILIES.items():
        for condition in conditions:
            values = torch.load(seed_dir / f"{prefix}_{condition}_features.pt", weights_only=True)
            features.append(values)
            labels.append(torch.full((len(values),), label, dtype=torch.long))
    return torch.cat(features), torch.cat(labels)


def fit(seed, sketch_features, device, args):
    seed_dir = ROUTER_ROOT / f"seed_{seed}"
    calibration = torch.load(seed_dir / "clean_calibration_features.pt", weights_only=True)
    mean, std = calibration.mean(0), calibration.std(0).clamp_min(1e-6)
    photo_train, photo_train_labels = load_photo_features(seed_dir, "train")
    photo_validation, photo_validation_labels = load_photo_features(seed_dir, "validation")
    sketch_train, sketch_validation = sketch_features[:2000], sketch_features[2000:]
    train_x = torch.cat((photo_train, sketch_train))
    train_y = torch.cat((photo_train_labels, torch.zeros(len(sketch_train), dtype=torch.long)))
    validation_x = torch.cat((photo_validation, sketch_validation))
    validation_y = torch.cat((photo_validation_labels, torch.zeros(len(sketch_validation), dtype=torch.long)))
    train_x = (train_x - mean) / std
    validation_x = (validation_x - mean) / std
    router = HeadRouter(train_x.shape[1]).to(device)
    optimizer = AdamW(router.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(10900 + seed)
    best, best_state, stale = -1.0, None, 0
    for epoch in range(30):
        router.train()
        loader = DataLoader(TensorDataset(train_x, train_y), batch_size=128, shuffle=True, generator=generator)
        for features, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(router(features.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
        router.eval()
        with torch.no_grad():
            predictions = router(validation_x.to(device)).argmax(1).cpu()
        accuracy = float((predictions == validation_y).float().mean())
        if accuracy > best:
            best, stale = accuracy, 0
            best_state = {name: value.detach().cpu().clone() for name, value in router.state_dict().items()}
        else:
            stale += 1
            if stale >= 5:
                break
    router.load_state_dict(best_state)
    router.eval()
    with torch.no_grad():
        photo_predictions = router(((photo_validation - mean) / std).to(device)).argmax(1).cpu()
        sketch_predictions = router(((sketch_validation - mean) / std).to(device)).argmax(1).cpu()
    return router, mean.to(device), std.to(device), {
        "best_combined_validation_accuracy": best,
        "photo_validation_accuracy": float((photo_predictions == photo_validation_labels).float().mean()),
        "old_sketch_unknown_no_repair_fraction": float((sketch_predictions == 0).float().mean()),
    }


def evaluate(model, router, mean, std, repairs, dataset, device, args):
    baseline, repaired, routes = [], [], []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            outputs = model(pixel_values=pixels, output_hidden_states=True, output_attentions=True, return_dict=True)
            route = router((attention_head_statistics(outputs.attentions) - mean) / std).argmax(1)
            logits = corrected_logits(model, outputs.hidden_states[BLOCK], repairs, route)
            baseline.extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            repaired.extend((logits.argmax(1) == labels).cpu().tolist())
            routes.extend(route.cpu().tolist())
    baseline, repaired, routes = map(np.asarray, (baseline, repaired, routes))
    return {
        "baseline_accuracy": float(baseline.mean()),
        "candidate_accuracy": float(repaired.mean()),
        "accuracy_difference": float(repaired.mean() - baseline.mean()),
        "recovered": int((~baseline & repaired).sum()),
        "damaged": int((baseline & ~repaired).sum()),
        "routing_fractions_unknown_noise_blur": [float((routes == index).mean()) for index in range(3)],
    }


def main():
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[0, 3, 4, 5])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL, attn_implementation="eager", local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    prior_manifest = json.loads(PRIOR_SKETCH_MANIFEST.read_text())
    sketch_features = extract(
        model,
        ManifestDataset(PRIOR_SKETCH_ROOT, prior_manifest, None, 0, 0),
        device, args, output_dir / "old_sketch_head_features.pt",
    )
    fresh_manifest = json.loads((FRESH_RUN / "frozen_manifest.json").read_text())
    progress_path = output_dir / "progress.json"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    calibration = {}
    for seed in args.seeds:
        router, mean, std, calibration[str(seed)] = fit(seed, sketch_features, device, args)
        repairs = {family: load_repair(seed, family, device) for family in ("noise", "blur")}
        results.setdefault(str(seed), {})
        for severity in args.severities:
            key = "clean" if severity == 0 else f"defocus_blur_{severity}"
            if key in results[str(seed)]:
                continue
            print(f"Seed {seed} {key}", flush=True)
            results[str(seed)][key] = evaluate(
                model, router, mean, std, repairs,
                SketchDefocusDataset(fresh_manifest, severity, 107000), device, args,
            )
            progress_path.write_text(json.dumps(results, indent=2))
        del router, repairs
        torch.cuda.empty_cache()
    summary = {
        "configuration": vars(args) | {
            "router": "linear attention-head router with explicit no-repair class calibrated on clean photos plus previously used clean Sketch images",
            "old_sketch_train_validation": [2000, 1000],
            "fresh_sketch_evaluation_samples": 1000,
            "vit_and_repairs_frozen": True,
            "clean_counterpart_used_at_inference": False,
            "post_hoc_status": "router designed after Experiment 107 failure; fresh images are disjoint but reused for this post-hoc diagnostic",
        },
        "calibration": calibration,
        "results": results,
        "limitations": ["A further untouched dataset is required to confirm this revised router."],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
