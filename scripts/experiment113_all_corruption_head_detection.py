import argparse
import json
from pathlib import Path

import numpy as np
import torch
from imagecorruptions import corrupt, get_corruption_names
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment99_attention_head_corruption_detector import attention_head_statistics
from scripts.experiment107_fresh_sketch_defocus_confirmation import DATA_ROOT
from scripts.experiment110_clean_head_anomaly_detector import EXPERIMENT109, anomaly_score, fit_clean_detector
from scripts.experiment111_leakage_free_head_anomaly_confirmation import freeze_manifest, sha256


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment113_all_corruption_head_detection"
PREVIOUS_CONFIRMATIONS = (
    ACTIVE_ROOT / "results/sae/experiment111_leakage_free_head_anomaly_confirmation/"
    "full_fresh_eligible_classes_top32_q99_3seed_v1/frozen_manifest.json",
    ACTIVE_ROOT / "results/sae/experiment111_leakage_free_head_anomaly_confirmation/"
    "full_fresh_q95_top32_3seed_v1/frozen_manifest.json",
)


class CorruptionDataset(Dataset):
    def __init__(self, manifest, name, severity, seed):
        self.items = manifest["items"]
        self.name = name
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        row = self.items[index]
        image = Image.open(DATA_ROOT / row["relative_path"]).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.uint8)
        if self.name != "clean":
            np.random.seed(self.seed + index)
            values = corrupt(values, corruption_name=self.name, severity=self.severity)
        pixels = torch.from_numpy(np.asarray(values).copy()).permute(2, 0, 1).float()
        pixels = pixels.div_(255.0).sub_(0.5).div_(0.5)
        return pixels


def extract(model, dataset, device, args, path):
    if path.exists():
        return torch.load(path, weights_only=True)
    features = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for batch_index, pixels in enumerate(loader):
            outputs = model(pixel_values=pixels.to(device), output_attentions=True, return_dict=True)
            features.append(attention_head_statistics(outputs.attentions).cpu())
            if batch_index % 50 == 0:
                print(f"  batches {batch_index}/{len(loader)}", flush=True)
    values = torch.cat(features)
    torch.save(values, path)
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--clean-quantile", type=float, default=0.95)
    parser.add_argument("--corruption-seed", type=int, default=113000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest_path = output_dir / "frozen_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text())
        if manifest_path.exists()
        else freeze_manifest(manifest_path, PREVIOUS_CONFIRMATIONS)
    )
    print("Frozen manifest SHA256", sha256(manifest_path), "samples", manifest["samples"], flush=True)

    np.float_ = np.float64
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    conditions = ("clean", *get_corruption_names())
    for name in conditions:
        print("Extracting", name, flush=True)
        extract(
            model,
            CorruptionDataset(manifest, name, args.severity, args.corruption_seed),
            device,
            args,
            output_dir / f"{name}_head_features.pt",
        )

    clean_reference = torch.load(EXPERIMENT109 / "old_sketch_head_features.pt", weights_only=True)
    clean_features = torch.load(output_dir / "clean_head_features.pt", weights_only=True)
    results = {}
    calibration = {}
    for seed in (0, 1, 2):
        mean, std, threshold, calibration[str(seed)] = fit_clean_detector(
            seed, clean_reference, args.top_k, args.clean_quantile
        )
        threshold = float(threshold)
        clean_scores = anomaly_score(clean_features, mean, std, args.top_k).numpy()
        results[str(seed)] = {
            "clean": {
                "samples": len(clean_scores),
                "false_positive_rate": float((clean_scores > threshold).mean()),
                "mean_anomaly_score": float(clean_scores.mean()),
            }
        }
        for name in conditions[1:]:
            features = torch.load(output_dir / f"{name}_head_features.pt", weights_only=True)
            scores = anomaly_score(features, mean, std, args.top_k).numpy()
            results[str(seed)][name] = {
                "samples": len(scores),
                "detection_recall": float((scores > threshold).mean()),
                "auroc_vs_paired_clean_images": float(roc_auc_score(
                    np.concatenate((np.zeros(len(clean_scores)), np.ones(len(scores)))),
                    np.concatenate((clean_scores, scores)),
                )),
                "mean_anomaly_score": float(scores.mean()),
            }

    aggregate = {}
    for name in conditions[1:]:
        recalls = [results[str(seed)][name]["detection_recall"] for seed in (0, 1, 2)]
        aurocs = [results[str(seed)][name]["auroc_vs_paired_clean_images"] for seed in (0, 1, 2)]
        aggregate[name] = {
            "recall_by_seed": recalls,
            "mean_recall": float(np.mean(recalls)),
            "auroc_by_seed": aurocs,
            "mean_auroc": float(np.mean(aurocs)),
        }
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "manifest_sha256": sha256(manifest_path),
            "corruptions": list(conditions[1:]),
            "detector": "frozen clean-only top-32 head-z-score detector with clean-validation q95 threshold",
            "detector_training_or_threshold_tuning_on_this_set": False,
            "benchmark_label": "imagecorruptions algorithms on disjoint ImageNet-Sketch images; not pre-generated official ImageNet-C JPEGs",
            "repair_applied": False,
            "numpy2_fog_compatibility": "np.float_ alias to np.float64",
        },
        "calibration": calibration,
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "All conditions use the same images; seeds are detector replications, not independent test images.",
            "Only severity 4 is tested; no classification repair is applied.",
            "ImageNet-Sketch is a previously used domain, though this frozen image set is SHA-disjoint from the listed prior manifests.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
