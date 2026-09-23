import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from corruption.gaussian_blur import apply_gaussian_blur
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
import scripts.experiment45_sae_discovered_hidden_subspace as hidden_subspace
from scripts.experiment45_sae_discovered_hidden_subspace import SharedRawInputRepair
from scripts.experiment76_parameter_matched_oracle_moe import (
    ACTIVE_ROOT,
    EXPERT_FAMILIES,
    apply_family_corruption,
)
from scripts.experiment99_attention_head_corruption_detector import (
    HeadRouter,
    attention_head_statistics,
)


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment101_routed_batchtopk_sae_repairs"
ROUTER_ROOT = (
    ACTIVE_ROOT / "results/sae/experiment99_attention_head_corruption_detector/"
    "full_3seed_clean_noise_blur_v1"
)
THRESHOLD_SUMMARY = (
    ACTIVE_ROOT / "results/sae/experiment100_confidence_safe_attention_router/"
    "full_3seed_95pct_retention_v1/summary.json"
)
REPAIR_ROOTS = {
    "blur": ACTIVE_ROOT / "results/sae/experiment96_block6_batchtopk_blur_hidden_subspace/full_3seed_rank16_v1",
    "noise": ACTIVE_ROOT / "results/sae/experiment98_block6_batchtopk_noise_hidden_subspace/full_3seed_rank16_v1",
}
BLOCK = 6


class EvaluationDataset(Dataset):
    def __init__(self, start, samples, condition, seed, severity=4):
        self.base = ImageNetDataset(ROOT / "Dataset", samples, start)
        self.condition = condition
        self.seed = seed
        self.severity = severity

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = Image.open(self.base.image_paths[index]).convert("RGB")
        if self.condition is not None:
            if self.condition == "gaussian_blur":
                image = apply_gaussian_blur(image, severity=self.severity)
            else:
                image = apply_family_corruption(
                    image,
                    self.condition,
                    self.seed + self.base.start_index + index,
                )
        pixels = self.base.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels, self.base.labels[index]


def load_router(seed, device):
    seed_dir = ROUTER_ROOT / f"seed_{seed}"
    calibration = torch.load(
        seed_dir / "clean_calibration_features.pt", map_location="cpu", weights_only=True
    )
    mean = calibration.mean(0).to(device)
    std = calibration.std(0).clamp_min(1e-6).to(device)
    router = HeadRouter(calibration.shape[1]).to(device)
    router.load_state_dict(
        torch.load(seed_dir / "attention_head_router.pt", map_location=device, weights_only=True)["router"]
    )
    return router.eval(), mean, std


def load_repair(seed, family, device):
    run = REPAIR_ROOTS[family]
    summary = json.loads((run / "summary.json").read_text())
    feature_rows = summary["controls"][str(seed)][f"{family}_features"]
    if len(feature_rows) != 16:
        raise RuntimeError(f"Expected 16 {family} features for seed {seed}")
    state = torch.load(
        run / f"seed_{seed}" / f"{family}_batchtopk_decoder.pt",
        map_location=device,
        weights_only=True,
    )
    method = SharedRawInputRepair(
        state["basis"], state["input_mean"], state["input_scale"], state["output_scale"]
    ).to(device)
    method.load_state_dict(state)
    return method.eval()


def corrected_logits(model, hidden, repairs, routes):
    patches = hidden[:, 1:]
    residual = torch.zeros_like(patches)
    for label, family in ((1, "noise"), (2, "blur")):
        mask = routes == label
        if mask.any():
            residual[mask] = repairs[family](patches[mask])
    candidate = torch.cat((hidden[:, :1], patches + residual), dim=1)
    return downstream_from_layer(model, candidate, BLOCK - 1)


def evaluate(model, router, mean, std, repairs, threshold, dataset, family, device, args, stat_seed):
    names = ("baseline", *args.methods)
    outcomes = {name: [] for name in names}
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    with torch.no_grad():
        for pixels, labels in loader:
            pixels = pixels.to(device)
            labels = labels.to(device)
            outputs = model(
                pixel_values=pixels,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
            hidden = outputs.hidden_states[BLOCK]
            probabilities = router((attention_head_statistics(outputs.attentions) - mean) / std).softmax(1)
            original_routes = probabilities.argmax(1)
            nonclean_routes = probabilities[:, 1:].argmax(1) + 1
            safe_routes = torch.where(
                probabilities[:, 0] >= threshold,
                torch.zeros_like(nonclean_routes),
                nonclean_routes,
            )
            all_routes = {
                "original_router": original_routes,
                "safe_router": safe_routes,
                "always_noise": torch.ones_like(original_routes),
                "always_blur": torch.full_like(original_routes, 2),
                "oracle": torch.full_like(
                    original_routes, 0 if family == "clean" else (1 if family == "noise" else 2)
                ),
            }
            routes = {name: all_routes[name] for name in args.methods}
            outcomes["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for name, route in routes.items():
                logits = corrected_logits(model, hidden, repairs, route)
                outcomes[name].extend((logits.argmax(1) == labels).cpu().tolist())
    arrays = {name: np.asarray(values, dtype=bool) for name, values in outcomes.items()}
    return {
        "samples": len(dataset),
        "baseline_accuracy": float(arrays["baseline"].mean()),
        "methods": {
            name: paired_comparison(arrays["baseline"], arrays[name], stat_seed + index, args.bootstrap)
            for index, name in enumerate(names[1:])
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
    parser.add_argument(
        "--evaluation-mode",
        choices=["family_half", "gaussian_blur_remainder"],
        default="family_half",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["original_router", "safe_router", "always_noise", "always_blur", "oracle"],
        default=["original_router", "safe_router", "always_noise", "always_blur", "oracle"],
    )
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
    results = {}
    outcome_store = {}
    for seed in args.seeds:
        router, mean, std = load_router(seed, device)
        repairs = {family: load_repair(seed, family, device) for family in ("noise", "blur")}
        threshold = thresholds[str(seed)]["selected_clean_probability_threshold"]
        validation_start = split_map[seed]["validation"]["start"]
        if args.evaluation_mode == "family_half":
            conditions = [("clean", "clean", None, validation_start + 250, 250)]
            for family, corruptions in EXPERT_FAMILIES.items():
                for condition in corruptions:
                    conditions.append((condition, family, condition, validation_start + 83, 83))
        else:
            conditions = [
                ("clean", "clean", None, validation_start + 500, 1500),
                ("gaussian_blur", "blur", "gaussian_blur", validation_start + 166, 1834),
            ]
        results[str(seed)] = {}
        for index, (key, family, condition, start, samples) in enumerate(conditions):
            if args.max_samples is not None:
                samples = min(samples, args.max_samples)
            result, arrays = evaluate(
                model, router, mean, std, repairs, threshold,
                EvaluationDataset(start, samples, condition, seed),
                family, device, args, 101000 + seed * 1000 + index * 20,
            )
            results[str(seed)][key] = result
            for name, values in arrays.items():
                outcome_store[f"seed{seed}_{key}_{name}"] = values
        del repairs, router
        torch.cuda.empty_cache()
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcome_store)
    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "block": BLOCK,
            "rank": 16,
            "router_source": str(ROUTER_ROOT),
            "threshold_source": str(THRESHOLD_SUMMARY),
            "repair_sources": {key: str(value) for key, value in REPAIR_ROOTS.items()},
            "vit_frozen": True,
            "saes_frozen": True,
            "repairs_frozen": True,
            "router_frozen": True,
            "clean_counterpart_used_at_inference": False,
            "corruption_label_used_by_routed_methods": False,
            "imageNetV2_accessed": False,
            "status": "held-out development-half evaluation; no tuning permitted from these outcomes",
        },
        "evaluation_ranges": {
            str(seed): (
                {
                    "clean": [split_map[seed]["validation"]["start"] + 250, split_map[seed]["validation"]["start"] + 500],
                    "each_corruption": [split_map[seed]["validation"]["start"] + 83, split_map[seed]["validation"]["start"] + 166],
                }
                if args.evaluation_mode == "family_half"
                else {
                    "clean": [split_map[seed]["validation"]["start"] + 500, split_map[seed]["validation"]["end"]],
                    "gaussian_blur": [split_map[seed]["validation"]["start"] + 166, split_map[seed]["validation"]["end"]],
                }
            )
            for seed in args.seeds
        },
        "results": results,
        "guardrails": [
            "The safe threshold was selected on earlier validation-prefix images and is frozen here.",
            "All ViT, SAE, repair, router, and threshold parameters are frozen.",
            "Oracle uses corruption family only as a diagnostic ceiling; routed methods do not.",
            "No clean counterpart is used by any deployment-style method.",
            "ImageNetV2, ImageNet-Sketch, and the final reserve are not accessed.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
