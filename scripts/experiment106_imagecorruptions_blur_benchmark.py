import argparse
import json
from pathlib import Path

import numpy as np
import torch
import imagecorruptions.corruptions as corruption_operators
from imagecorruptions import corrupt
from PIL import Image
from skimage.filters import gaussian as skimage_gaussian
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST
from scripts.experiment41_disjoint_gate_development import paired_comparison
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT
from scripts.experiment99_attention_head_corruption_detector import attention_head_statistics
from scripts.experiment101_routed_batchtopk_sae_repairs import (
    BLOCK,
    THRESHOLD_SUMMARY,
    corrected_logits,
    load_repair,
    load_router,
)


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment106_imagecorruptions_blur_benchmark"
BLURS = ("defocus_blur", "glass_blur", "motion_blur", "zoom_blur")


def compatible_gaussian(image, sigma=1, multichannel=False, **kwargs):
    if "channel_axis" not in kwargs:
        kwargs["channel_axis"] = -1 if multichannel else None
    return skimage_gaussian(image, sigma=sigma, **kwargs)


corruption_operators.gaussian = compatible_gaussian


def load_reserved_split(manifest_path, protocol):
    manifest = json.loads(manifest_path.read_text())
    records = {record["name"]: record for record in manifest[protocol]["splits"]}
    record = records["unused_imagenet_reserve"]
    if record["dataset"] != "imagenet_val" or record["isolation_group"] != "reserved":
        raise ValueError(f"Invalid reserved split: {record}")
    return record


class StandardizedCorruptionDataset(Dataset):
    def __init__(self, start, samples, corruption, severity, corruption_seed):
        self.base = ImageNetDataset(ROOT / "Dataset", samples, start)
        self.corruption = corruption
        self.severity = severity
        self.corruption_seed = corruption_seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = Image.open(self.base.image_paths[index]).convert("RGB")
        image = image.resize((224, 224), Image.Resampling.BILINEAR)
        values = np.asarray(image, dtype=np.uint8)
        np.random.seed(self.corruption_seed + self.base.start_index + index)
        values = corrupt(
            values,
            corruption_name=self.corruption,
            severity=self.severity,
        )
        pixels = torch.from_numpy(np.asarray(values).copy()).permute(2, 0, 1).float()
        pixels = pixels.div_(255.0).sub_(0.5).div_(0.5)
        return pixels, self.base.labels[index]


def evaluate(model, router, mean, std, repairs, threshold, dataset, device, args, stat_seed):
    outcomes = {"baseline": [], "safe_router": [], "always_blur": []}
    routes = []
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    with torch.no_grad():
        for pixels, labels in loader:
            pixels = pixels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(
                pixel_values=pixels,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            probabilities = router(
                (attention_head_statistics(outputs.attentions) - mean) / std
            ).softmax(1)
            nonclean = probabilities[:, 1:].argmax(1) + 1
            safe_routes = torch.where(
                probabilities[:, 0] >= threshold,
                torch.zeros_like(nonclean),
                nonclean,
            )
            blur_routes = torch.full_like(safe_routes, 2)
            routes.extend(safe_routes.cpu().tolist())
            outcomes["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for name, selected_routes in (
                ("safe_router", safe_routes),
                ("always_blur", blur_routes),
            ):
                logits = corrected_logits(
                    model,
                    outputs.hidden_states[BLOCK],
                    repairs,
                    selected_routes,
                )
                outcomes[name].extend((logits.argmax(1) == labels).cpu().tolist())
    arrays = {name: np.asarray(values, dtype=bool) for name, values in outcomes.items()}
    route_array = np.asarray(routes)
    return {
        "samples": len(dataset),
        "baseline_accuracy": float(arrays["baseline"].mean()),
        "safe_router_fractions_clean_noise_blur": [
            float((route_array == route).mean()) for route in range(3)
        ],
        "methods": {
            name: paired_comparison(
                arrays["baseline"], arrays[name], stat_seed + index, args.bootstrap
            )
            for index, name in enumerate(("safe_router", "always_blur"))
        },
    }, arrays


def main():
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--corruptions", nargs="+", choices=BLURS, default=list(BLURS))
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--corruption-seed", type=int, default=106000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    progress_path = output_dir / "progress.json"
    outcome_path = output_dir / "paired_outcomes_partial.npz"
    reserve = load_reserved_split(args.split_manifest, args.split_protocol)
    start = reserve["start"] + args.start_offset
    if start < reserve["start"] or start + args.samples > reserve["end"]:
        raise ValueError(f"Requested [{start}, {start + args.samples}) outside {reserve}")
    thresholds = json.loads(THRESHOLD_SUMMARY.read_text())["results"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    print(f"Evaluation range: [{start}, {start + args.samples})", flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL,
        attn_implementation="eager",
        local_files_only=True,
    ).to(device).eval()
    model.requires_grad_(False)

    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    outcomes = {}
    if outcome_path.exists():
        with np.load(outcome_path) as stored:
            outcomes = {name: stored[name] for name in stored.files}
    for seed in args.seeds:
        print(f"Repair seed {seed}", flush=True)
        router, mean, std = load_router(seed, device)
        repairs = {family: load_repair(seed, family, device) for family in ("noise", "blur")}
        threshold = thresholds[str(seed)]["selected_clean_probability_threshold"]
        results.setdefault(str(seed), {})
        for corruption_index, corruption in enumerate(args.corruptions):
            results[str(seed)].setdefault(corruption, {})
            for severity in args.severities:
                if str(severity) in results[str(seed)][corruption]:
                    print(f"  {corruption} severity {severity}: already complete", flush=True)
                    continue
                print(f"  {corruption} severity {severity}", flush=True)
                dataset = StandardizedCorruptionDataset(
                    start,
                    args.samples,
                    corruption,
                    severity,
                    args.corruption_seed,
                )
                result, arrays = evaluate(
                    model,
                    router,
                    mean,
                    std,
                    repairs,
                    threshold,
                    dataset,
                    device,
                    args,
                    1060000 + seed * 10000 + corruption_index * 100 + severity * 10,
                )
                results[str(seed)][corruption][str(severity)] = result
                for method, values in arrays.items():
                    outcomes[f"seed{seed}_{corruption}_s{severity}_{method}"] = values
                progress_path.write_text(json.dumps(results, indent=2))
                np.savez_compressed(outcome_path, **outcomes)
        del router, repairs
        torch.cuda.empty_cache()

    aggregate = {}
    for corruption in args.corruptions:
        aggregate[corruption] = {}
        for severity in args.severities:
            severity_results = [
                results[str(seed)][corruption][str(severity)] for seed in args.seeds
            ]
            aggregate[corruption][str(severity)] = {
                "mean_baseline_accuracy": float(
                    np.mean([result["baseline_accuracy"] for result in severity_results])
                ),
                "safe_router_gains_by_seed": [
                    result["methods"]["safe_router"]["accuracy_difference"]
                    for result in severity_results
                ],
                "mean_safe_router_gain": float(
                    np.mean([
                        result["methods"]["safe_router"]["accuracy_difference"]
                        for result in severity_results
                    ])
                ),
                "mean_blur_route_fraction": float(
                    np.mean([
                        result["safe_router_fractions_clean_noise_blur"][2]
                        for result in severity_results
                    ])
                ),
            }

    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "evaluation_start": start,
            "evaluation_end": start + args.samples,
            "corruption_library": "imagecorruptions",
            "corruption_library_protocol": "224x224 uint8 input; official corruption names and severities",
            "benchmark_label": "ImageNet-C-compatible corruptions generated on held-out ImageNet validation images; not the pre-generated official ImageNet-C archive",
            "all_components_frozen": True,
            "clean_counterpart_used_at_inference": False,
            "identical_corrupted_images_across_repair_seeds": True,
            "imageNetV2_accessed": False,
        },
        "reserved_split": reserve,
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "This uses the standardized imagecorruptions implementation, not the downloaded official ImageNet-C JPEG archive.",
            "The reserved ImageNet validation range is disjoint from the locked SAE, adapter, and router development splits.",
            "Later exploratory work may have inspected parts of the nominal reserve, so this is a controlled confirmation benchmark rather than a pristine final test set.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
