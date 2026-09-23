import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import ViTForImageClassification

from corruption.gaussian_blur import apply_gaussian_blur
from corruption.gaussian_noise import apply_gaussian_noise
from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment76_parameter_matched_oracle_moe import (
    ACTIVE_ROOT,
    EXPERT_FAMILIES,
    apply_family_corruption,
    load_full_mixed,
)
from scripts.experiment77_learned_moe_router import (
    LinearRouter,
    UNSEEN,
    evaluate_and_save,
    load_experts,
)
from scripts.experiment68_unseen_online_corruptions import corrupt


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment79_frozen_router_severity_sweep"
ROUTER_RUN = (
    ACTIVE_ROOT
    / "results/sae/experiment77_learned_moe_router/"
    "full_3seed_linear_router_seen_unseen_v1"
)
MOTION_LENGTHS = (3, 5, 9, 13, 17)
DISK_RADII = (0.75, 1.0, 1.5, 2.0, 3.0)


def shifted_average(image, offsets):
    values = np.asarray(image, dtype=np.float32)
    radius = max(max(abs(y), abs(x)) for y, x in offsets)
    padded = np.pad(values, ((radius, radius), (radius, radius), (0, 0)), mode="reflect")
    height, width = values.shape[:2]
    shifted = [
        padded[radius + y : radius + y + height, radius + x : radius + x + width]
        for y, x in offsets
    ]
    return Image.fromarray(np.clip(np.mean(shifted, axis=0), 0, 255).astype(np.uint8))


def apply_severity_corruption(image, name, severity, seed):
    if severity == 4 and name in {"disk_blur", "motion_blur"}:
        return apply_family_corruption(image, name, seed)
    if name == "gaussian_noise":
        return apply_gaussian_noise(image, severity=severity, seed=seed)
    if name == "gaussian_blur":
        return apply_gaussian_blur(image, severity=severity)
    if name in {"shot_noise", "impulse_noise"} | set(UNSEEN):
        return corrupt(image, name, severity, seed)
    if name == "motion_blur":
        length = MOTION_LENGTHS[severity - 1]
        half = length // 2
        return shifted_average(image, [(0, offset) for offset in range(-half, half + 1)])
    if name == "disk_blur":
        radius = DISK_RADII[severity - 1]
        extent = int(np.ceil(radius))
        offsets = [
            (y, x)
            for y in range(-extent, extent + 1)
            for x in range(-extent, extent + 1)
            if x * x + y * y <= radius * radius
        ]
        return shifted_average(image, offsets)
    raise ValueError(name)


class SeverityDataset(torch.utils.data.Dataset):
    def __init__(self, split, corruption_name, severity, seed, max_samples=None):
        samples = split["end"] - split["start"]
        if max_samples is not None:
            samples = min(samples, max_samples)
        self.base = ImageNetDataset(ROOT / "Dataset", samples, split["start"])
        self.corruption_name = corruption_name
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = Image.open(self.base.image_paths[index]).convert("RGB")
        image = apply_severity_corruption(
            image,
            self.corruption_name,
            self.severity,
            self.seed + self.base.start_index + index,
        )
        pixels = self.base.processor(images=image, return_tensors="pt")["pixel_values"]
        return pixels.squeeze(0), self.base.labels[index]


def load_router(seed, device):
    path = ROUTER_RUN / f"seed_{seed}" / "linear_router.pt"
    payload = torch.load(path, map_location=device, weights_only=True)
    router = LinearRouter().to(device)
    router.load_state_dict(payload["router"])
    return router.eval(), payload["mean"], payload["std"], path


def aggregate(results):
    methods = ("mixed", "uniform", "router_soft", "router_hard")
    output = {}
    for section, section_results in results.items():
        output[section] = {}
        conditions = sorted(section_results["0"])
        severities = sorted({int(condition.rsplit("_", 1)[1]) for condition in conditions})
        for method in methods:
            output[section][method] = {
                "mean_accuracy": float(
                    np.mean(
                        [
                            section_results[str(seed)][condition]["methods"][method][
                                "candidate_accuracy"
                            ]
                            for seed in range(3)
                            for condition in conditions
                        ]
                    )
                ),
                "per_seed_mean_accuracy": [
                    float(
                        np.mean(
                            [
                                section_results[str(seed)][condition]["methods"][method][
                                    "candidate_accuracy"
                                ]
                                for condition in conditions
                            ]
                        )
                    )
                    for seed in range(3)
                ],
            }
        output[section]["router_soft_by_severity"] = {
            str(severity): float(
                np.mean(
                    [
                        section_results[str(seed)][condition]["methods"]["router_soft"][
                            "candidate_accuracy"
                        ]
                        for seed in range(3)
                        for condition in conditions
                        if condition.endswith(f"_{severity}")
                    ]
                )
            )
            for severity in severities
        }
        output[section]["router_mean_noise_probability_by_severity"] = {
            str(severity): float(
                np.mean(
                    [
                        section_results[str(seed)][condition]["routing"][
                            "mean_noise_probability"
                        ]
                        for seed in range(3)
                        for condition in conditions
                        if condition.endswith(f"_{severity}")
                    ]
                )
            )
            for severity in severities
        }
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--rank", type=int, default=212)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--reserve-start", type=int, default=39000)
    parser.add_argument("--reserve-samples", type=int, default=3000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("Confirmatory sweep preserves all three locked adapter seeds")
    if any(severity not in range(1, 6) for severity in args.severities):
        raise ValueError("Severities must be in 1..5")

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    results = {"seen_development": {}, "unseen_reserve": {}}
    checkpoints = {}
    seen_conditions = [
        (corruption, family)
        for family, corruptions in EXPERT_FAMILIES.items()
        for corruption in corruptions
    ]

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        router, mean, std, router_path = load_router(seed, device)
        experts = load_experts(seed, args.rank, device)
        mixed = load_full_mixed(seed, device)
        checkpoints[str(seed)] = {
            "router": str(router_path),
            "experts": {
                family: str(
                    ACTIVE_ROOT
                    / "results/sae/experiment76_parameter_matched_oracle_moe/"
                    "full_3seed_rank212_oracle_v1"
                    / f"seed_{seed}/{family}_rank{args.rank}.pt"
                )
                for family in EXPERT_FAMILIES
            },
        }
        results["seen_development"][str(seed)] = {}
        results["unseen_reserve"][str(seed)] = {}

        for corruption_index, (corruption_name, family) in enumerate(seen_conditions):
            for severity in args.severities:
                key = f"{corruption_name}_{severity}"
                path = seed_dir / f"seen_{key}.json"
                if args.resume and path.exists():
                    result = json.loads(path.read_text())
                else:
                    dataset = SeverityDataset(
                        splits[seed]["validation"],
                        corruption_name,
                        severity,
                        seed,
                        args.max_samples,
                    )
                    result = evaluate_and_save(
                        model,
                        experts,
                        mixed,
                        router,
                        mean,
                        std,
                        dataset,
                        family,
                        device,
                        args,
                        path,
                        79000 + seed * 1000 + corruption_index * 100 + severity * 10,
                    )
                results["seen_development"][str(seed)][key] = result

        reserve_split = {
            "start": args.reserve_start,
            "end": args.reserve_start + args.reserve_samples,
        }
        for corruption_index, corruption_name in enumerate(UNSEEN):
            for severity in args.severities:
                key = f"{corruption_name}_{severity}"
                path = seed_dir / f"unseen_{key}.json"
                if args.resume and path.exists():
                    result = json.loads(path.read_text())
                else:
                    dataset = SeverityDataset(
                        reserve_split,
                        corruption_name,
                        severity,
                        2068,
                        args.max_samples,
                    )
                    result = evaluate_and_save(
                        model,
                        experts,
                        mixed,
                        router,
                        mean,
                        std,
                        dataset,
                        None,
                        device,
                        args,
                        path,
                        80000 + seed * 1000 + corruption_index * 100 + severity * 10,
                    )
                results["unseen_reserve"][str(seed)][key] = result

    summary = {
        "configuration": {
            **vars(args),
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "vit_frozen": True,
            "experts_frozen": True,
            "router_frozen": True,
            "router_training": False,
            "router_selection_or_tuning": False,
            "imagenetv2_accessed": False,
            "imagenet_sketch_accessed": False,
            "benchmark_label": "frozen severity generalization development sweep; online corruptions are not official ImageNet-C",
        },
        "splits": {str(seed): splits[seed] for seed in args.seeds},
        "checkpoints": checkpoints,
        "results": results,
        "aggregate": aggregate(results),
        "limitations": [
            "The seen-family sweep reuses the locked development validation splits and is not an independent final benchmark.",
            "The reserve range appeared in earlier diagnostics and is not pristine independent evaluation.",
            "ImageNetV2 and ImageNet-Sketch are not accessed in this experiment.",
            "Online corruptions approximate common corruption families and are not official ImageNet-C.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["aggregate"], indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
