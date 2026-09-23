import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from corruption.gaussian_blur import apply_gaussian_blur
from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment76_parameter_matched_oracle_moe import ACTIVE_ROOT, apply_family_corruption
from scripts.experiment99_attention_head_corruption_detector import attention_head_statistics
from scripts.experiment101_routed_batchtopk_sae_repairs import (
    BLOCK,
    THRESHOLD_SUMMARY,
    corrected_logits,
    load_repair,
    load_router,
)


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment105_all_blur_families_batchtopk_repair"
BLURS = ("gaussian_blur", "defocus_blur", "glass_blur", "motion_blur", "zoom_blur")


def center_zoom(image, factor):
    width, height = image.size
    crop_width = max(1, round(width / factor))
    crop_height = max(1, round(height / factor))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    crop = image.crop((left, top, left + crop_width, top + crop_height))
    return crop.resize((width, height), Image.Resampling.BILINEAR)


def zoom_blur(image):
    arrays = [np.asarray(image, dtype=np.float32)]
    for factor in np.linspace(1.05, 1.35, 7):
        arrays.append(np.asarray(center_zoom(image, float(factor)), dtype=np.float32))
    return Image.fromarray(np.clip(np.mean(arrays, axis=0), 0, 255).astype(np.uint8))


def glass_blur(image, seed):
    values = np.asarray(image, dtype=np.float32) / 255.0
    values = gaussian_filter(values, sigma=(1.2, 1.2, 0))
    rng = np.random.default_rng(seed)
    height, width = values.shape[:2]
    y_grid, x_grid = np.meshgrid(
        np.arange(height), np.arange(width), indexing="ij"
    )
    for _ in range(2):
        delta_y = rng.integers(-2, 3, size=(height, width))
        delta_x = rng.integers(-2, 3, size=(height, width))
        source_y = np.clip(y_grid + delta_y, 0, height - 1)
        source_x = np.clip(x_grid + delta_x, 0, width - 1)
        values = values[source_y, source_x]
    values = gaussian_filter(values, sigma=(1.2, 1.2, 0))
    return Image.fromarray(np.clip(values * 255, 0, 255).astype(np.uint8))


def apply_blur(image, name, seed):
    if name == "gaussian_blur":
        return apply_gaussian_blur(image, severity=4)
    if name == "defocus_blur":
        return apply_family_corruption(image, "disk_blur", seed)
    if name == "motion_blur":
        return apply_family_corruption(image, "motion_blur", seed)
    if name == "zoom_blur":
        return zoom_blur(image)
    if name == "glass_blur":
        return glass_blur(image, seed)
    raise ValueError(name)


class BlurDataset(Dataset):
    def __init__(self, start, samples, blur, seed):
        self.base = ImageNetDataset(ROOT / "Dataset", samples, start)
        self.blur = blur
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = Image.open(self.base.image_paths[index]).convert("RGB")
        image = apply_blur(image, self.blur, self.seed + self.base.start_index + index)
        pixels = self.base.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, self.base.labels[index]


def evaluate(model, router, mean, std, repairs, threshold, dataset, device, args, stat_seed):
    outcomes = {"baseline": [], "safe_router": [], "always_blur": []}
    routing = []
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for pixels, labels in loader:
            pixels, labels = pixels.to(device), labels.to(device)
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
                probabilities[:, 0] >= threshold, torch.zeros_like(nonclean), nonclean
            )
            blur_routes = torch.full_like(safe_routes, 2)
            routing.extend(safe_routes.cpu().tolist())
            outcomes["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for name, routes in (("safe_router", safe_routes), ("always_blur", blur_routes)):
                logits = corrected_logits(model, outputs.hidden_states[BLOCK], repairs, routes)
                outcomes[name].extend((logits.argmax(1) == labels).cpu().tolist())
    arrays = {key: np.asarray(value, dtype=bool) for key, value in outcomes.items()}
    routes = np.asarray(routing)
    return {
        "samples": len(dataset),
        "baseline_accuracy": float(arrays["baseline"].mean()),
        "routing_fractions_clean_noise_blur": [float((routes == index).mean()) for index in range(3)],
        "methods": {
            name: paired_comparison(arrays["baseline"], arrays[name], stat_seed + index, args.bootstrap)
            for index, name in enumerate(("safe_router", "always_blur"))
        },
    }, arrays


def main():
    hidden_subspace.BLOCK_INDEX = BLOCK - 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

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
    results, outcomes = {}, {}
    for seed in args.seeds:
        router, mean, std = load_router(seed, device)
        repairs = {family: load_repair(seed, family, device) for family in ("noise", "blur")}
        threshold = thresholds[str(seed)]["selected_clean_probability_threshold"]
        start = split_map[seed]["validation"]["start"] + 166
        samples = split_map[seed]["validation"]["end"] - start
        if args.max_samples is not None:
            samples = min(samples, args.max_samples)
        results[str(seed)] = {}
        for index, blur in enumerate(BLURS):
            result, arrays = evaluate(
                model, router, mean, std, repairs, threshold,
                BlurDataset(start, samples, blur, seed), device, args,
                105000 + seed * 1000 + index * 20,
            )
            results[str(seed)][blur] = result
            for name, values in arrays.items():
                outcomes[f"seed{seed}_{blur}_{name}"] = values
        del router, repairs
        torch.cuda.empty_cache()

    aggregate = {}
    for blur in BLURS:
        aggregate[blur] = {}
        for method in ("safe_router", "always_blur"):
            gains = [results[str(seed)][blur]["methods"][method]["accuracy_difference"] for seed in args.seeds]
            aggregate[blur][method] = {"gains_by_seed": gains, "mean_gain": float(np.mean(gains))}
        aggregate[blur]["mean_safe_blur_routing_fraction"] = float(np.mean([
            results[str(seed)][blur]["routing_fractions_clean_noise_blur"][2] for seed in args.seeds
        ]))
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "blur_families": BLURS,
            "severity": 4,
            "benchmark_label": "controlled deterministic approximations; not official ImageNet-C",
            "all_components_frozen": True,
            "clean_counterpart_used_at_inference": False,
            "imageNetV2_accessed": False,
        },
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "Defocus, glass, motion, and zoom are controlled approximations because imagecorruptions is unavailable.",
            "This is not an official ImageNet-C reproduction.",
            "The development validation range was used for repair checkpoint selection and is not a pristine final benchmark.",
            "Zoom and glass blur were unseen during repair and router fitting.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
