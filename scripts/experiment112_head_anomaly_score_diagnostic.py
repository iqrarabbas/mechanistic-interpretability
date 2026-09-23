import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment99_attention_head_corruption_detector import attention_head_statistics
from scripts.experiment101_routed_batchtopk_sae_repairs import ROUTER_ROOT
from scripts.experiment107_fresh_sketch_defocus_confirmation import SketchDefocusDataset
from scripts.experiment110_clean_head_anomaly_detector import EXPERIMENT109, FRESH_RUN, anomaly_score


OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment112_head_anomaly_score_diagnostic"


def extract(model, dataset, device, args, path):
    if path.exists():
        return torch.load(path, weights_only=True)
    features = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for index, (pixels, _) in enumerate(loader):
            outputs = model(pixel_values=pixels.to(device), output_attentions=True, return_dict=True)
            features.append(attention_head_statistics(outputs.attentions).cpu())
            if index % 50 == 0:
                print(f"  batches {index}/{len(loader)}", flush=True)
    values = torch.cat(features)
    torch.save(values, path)
    return values


def score_metrics(clean, corrupted):
    clean = clean.numpy()
    corrupted = corrupted.numpy()
    labels = np.concatenate((np.zeros(len(clean)), np.ones(len(corrupted))))
    scores = np.concatenate((clean, corrupted))
    thresholds = {str(int((1 - rate) * 100)): float(np.quantile(clean, 1 - rate)) for rate in (0.01, 0.05, 0.10)}
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "clean_mean": float(clean.mean()),
        "corrupted_mean": float(corrupted.mean()),
        "recall_at_clean_fpr_1_5_10pct": {
            key: float((corrupted > threshold).mean()) for key, threshold in thresholds.items()
        },
        "clean_thresholds": thresholds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    manifest = json.loads((FRESH_RUN / "frozen_manifest.json").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    conditions = {}
    for severity in (0, 3, 4, 5):
        print("Extracting", severity, flush=True)
        conditions[severity] = extract(
            model, SketchDefocusDataset(manifest, severity, 107000), device, args,
            output_dir / f"development_sketch_severity_{severity}_features.pt",
        )
    old_sketch = torch.load(EXPERIMENT109 / "old_sketch_head_features.pt", weights_only=True)
    results = {}
    for seed in (0, 1, 2):
        seed_dir = ROUTER_ROOT / f"seed_{seed}"
        calibration = torch.load(seed_dir / "clean_calibration_features.pt", weights_only=True)
        photo_train = torch.load(seed_dir / "train_clean_features.pt", weights_only=True)
        photo_clean = torch.cat((calibration, photo_train))
        pooled = torch.cat((photo_clean, old_sketch[:2000]))
        reference_stats = {
            "pooled": (pooled.mean(0), pooled.std(0).clamp_min(1e-6)),
            "sketch_only": (old_sketch[:2000].mean(0), old_sketch[:2000].std(0).clamp_min(1e-6)),
            "photo_only": (photo_clean.mean(0), photo_clean.std(0).clamp_min(1e-6)),
        }
        results[str(seed)] = {}
        for name, (mean, std) in reference_stats.items():
            results[str(seed)][name] = {}
            for top_k in (8, 32, 128, 720):
                clean_score = anomaly_score(conditions[0], mean, std, top_k)
                results[str(seed)][name][str(top_k)] = {
                    str(severity): score_metrics(
                        clean_score,
                        anomaly_score(conditions[severity], mean, std, top_k),
                    )
                    for severity in (3, 4, 5)
                }
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "data_role": "previously inspected Experiment 107 development images only",
            "benchmark_status": "diagnostic, not held-out confirmation",
        },
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
