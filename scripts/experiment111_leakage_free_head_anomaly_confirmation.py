import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment41_disjoint_gate_development import paired_comparison
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment101_routed_batchtopk_sae_repairs import BLOCK, corrected_logits, load_repair
from scripts.experiment107_fresh_sketch_defocus_confirmation import DATA_ROOT
from scripts.experiment110_clean_head_anomaly_detector import (
    EXPERIMENT109,
    anomaly_score,
    fit_clean_detector,
)
from scripts.experiment99_attention_head_corruption_detector import attention_head_statistics
from imagecorruptions import corrupt
from PIL import Image


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment111_leakage_free_head_anomaly_confirmation"
OLD_MANIFEST = Path("/media/dr-yougart/Iqrar/datasets/imagenet_sketch/frozen_subset_3_per_class_unique_manifest.json")
DEV_MANIFEST = ACTIVE_ROOT / "results/sae/experiment107_fresh_sketch_defocus_confirmation/full_1000_disjoint_clean_defocus345_3seed_v1/frozen_manifest.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_manifest(path, extra_excluded_manifests=()):
    excluded_rows = []
    excluded_manifests = (OLD_MANIFEST, DEV_MANIFEST, *extra_excluded_manifests)
    for manifest_path in excluded_manifests:
        excluded_rows.extend(json.loads(manifest_path.read_text())["items"])
    excluded_paths = {row["relative_path"] for row in excluded_rows}
    excluded_hashes = {row["sha256"] for row in excluded_rows}
    selected_hashes = set()
    items = []
    excluded_classes = []
    classes = sorted(folder for folder in DATA_ROOT.iterdir() if folder.is_dir())
    for label, folder in enumerate(classes):
        selected = None
        for image_path in sorted(folder.glob("*.JPEG")):
            relative = str(image_path.relative_to(DATA_ROOT))
            if relative in excluded_paths:
                continue
            image_hash = sha256(image_path)
            if image_hash in excluded_hashes or image_hash in selected_hashes:
                continue
            selected = {"relative_path": relative, "label": label, "sha256": image_hash}
            break
        if selected is None:
            excluded_classes.append(folder.name)
            continue
        items.append(selected)
        selected_hashes.add(selected["sha256"])
    manifest = {
        "selection_rule": "first SHA-unique image per eligible class after excluding all listed prior Sketch manifests; frozen before inference",
        "excluded_manifest_sha256": [sha256(path) for path in excluded_manifests],
        "classes": len(items),
        "samples": len(items),
        "excluded_classes_without_new_unique_image": excluded_classes,
        "items": items,
    }
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


class ConfirmationDataset(Dataset):
    def __init__(self, manifest, severity, corruption_seed):
        self.items = manifest["items"]
        self.severity = severity
        self.corruption_seed = corruption_seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        image = Image.open(DATA_ROOT / row["relative_path"]).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.uint8)
        if self.severity:
            np.random.seed(self.corruption_seed + index)
            values = corrupt(values, corruption_name="defocus_blur", severity=self.severity)
        pixels = torch.from_numpy(np.asarray(values).copy()).permute(2, 0, 1).float()
        pixels = pixels.div_(255.0).sub_(0.5).div_(0.5)
        return pixels, row["label"]


def evaluate(model, mean, std, threshold, repair, dataset, device, args, stat_seed):
    baseline, candidate, detected = [], [], []
    repairs = {"blur": repair}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
            outputs = model(pixel_values=pixels, output_hidden_states=True, output_attentions=True, return_dict=True)
            scores = anomaly_score(attention_head_statistics(outputs.attentions).cpu(), mean, std, args.top_k)
            anomaly = scores > threshold
            logits = corrected_logits(model, outputs.hidden_states[BLOCK], repairs, anomaly.long().to(device) * 2)
            baseline.extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            candidate.extend((logits.argmax(1) == labels).cpu().tolist())
            detected.extend(anomaly.tolist())
    baseline = np.asarray(baseline, dtype=bool)
    candidate = np.asarray(candidate, dtype=bool)
    detected = np.asarray(detected, dtype=bool)
    result = paired_comparison(baseline, candidate, stat_seed, args.bootstrap)
    result["detection_fraction"] = float(detected.mean())
    return result, baseline, candidate, detected


def main():
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[0, 3, 4, 5])
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--clean-quantile", type=float, default=0.99)
    parser.add_argument("--corruption-seed", type=int, default=111000)
    parser.add_argument("--exclude-manifest", type=Path, action="append", default=[])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output_dir / "frozen_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else freeze_manifest(manifest_path, args.exclude_manifest)
    print("Frozen confirmation manifest", sha256(manifest_path), "samples", manifest["samples"], "excluded classes", manifest["excluded_classes_without_new_unique_image"], flush=True)
    sketch_features = torch.load(EXPERIMENT109 / "old_sketch_head_features.pt", weights_only=True)
    progress_path = output_dir / "progress.json"
    outcome_path = output_dir / "paired_outcomes_partial.npz"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    outcomes = {}
    if outcome_path.exists():
        with np.load(outcome_path) as stored:
            outcomes = {name: stored[name] for name in stored.files}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL, attn_implementation="eager", local_files_only=True).to(device).eval()
    model.requires_grad_(False)
    calibration = {}
    for seed in args.seeds:
        mean, std, threshold, calibration[str(seed)] = fit_clean_detector(seed, sketch_features, args.top_k, args.clean_quantile)
        repair = load_repair(seed, "blur", device)
        results.setdefault(str(seed), {})
        for severity in args.severities:
            key = "clean" if severity == 0 else f"defocus_blur_{severity}"
            if key in results[str(seed)]:
                continue
            print(f"Seed {seed} {key}", flush=True)
            result, baseline, candidate, detected = evaluate(
                model, mean, std, threshold, repair,
                ConfirmationDataset(manifest, severity, args.corruption_seed),
                device, args, 1110000 + seed * 100 + severity * 10,
            )
            results[str(seed)][key] = result
            outcomes[f"seed{seed}_{key}_baseline"] = baseline
            outcomes[f"seed{seed}_{key}_candidate"] = candidate
            outcomes[f"seed{seed}_{key}_detected"] = detected
            progress_path.write_text(json.dumps(results, indent=2))
            np.savez_compressed(outcome_path, **outcomes)
        del repair
        torch.cuda.empty_cache()
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    summary = {
        "configuration": vars(args) | {
            "exclude_manifest": [str(path.resolve()) for path in args.exclude_manifest],
            "model": BASE_MODEL,
            "manifest_sha256": sha256(manifest_path),
            "protocol": "frozen Experiment 110 detector and repairs evaluated without tuning on a second SHA-disjoint image set",
            "corruption_examples_used_to_fit_detector": False,
            "clean_counterpart_used_at_inference": False,
            "vit_and_repair_frozen": True,
        },
        "calibration": calibration,
        "results": results,
        "data_independence": [
            "All evaluation hashes are disjoint from the prior 3000-image Sketch calibration manifest.",
            "All evaluation hashes are disjoint from the prior 1000-image Experiments 107-110 development/evaluation manifest.",
            "The manifest was frozen before model inference and no threshold or hyperparameter will be changed from Experiment 110.",
        ],
        "limitation": "ImageNet-Sketch as a dataset/domain was used previously; this is fresh image-level confirmation, not a new-dataset benchmark.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
