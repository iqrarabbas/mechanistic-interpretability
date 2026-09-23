import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment54_lowrank_block6_adapter import LowRankHiddenAdapter
from scripts.experiment68_unseen_online_corruptions import corrupt
from scripts.experiment69_imagenet_sketch_frozen_generalization import (
    DEFAULT_DATA_ROOT as SKETCH_ROOT,
    DEFAULT_MANIFEST as SKETCH_MANIFEST,
    file_sha256,
)
from scripts.experiment76_parameter_matched_oracle_moe import (
    ACTIVE_ROOT,
    BLOCK,
    EXPERT_FAMILIES,
    OUTPUT_ROOT as ORACLE_OUTPUT_ROOT,
    apply_family_corruption,
    load_full_mixed,
)


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment77_learned_moe_router"
ORACLE_RUN = ORACLE_OUTPUT_ROOT / "full_3seed_rank212_oracle_v1"
UNSEEN = ("jpeg", "pixelate", "brightness", "contrast")
ROUTER_BLOCKS = (1, 2, 3)


class LinearRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(768 * len(ROUTER_BLOCKS), 1)

    def forward(self, features):
        return self.linear(features).squeeze(-1)


class RouterTrainingDataset(Dataset):
    conditions = tuple(
        (family, corruption)
        for family, corruptions in EXPERT_FAMILIES.items()
        for corruption in corruptions
    )

    def __init__(self, split, seed, max_samples=None):
        samples = split["end"] - split["start"]
        if max_samples is not None:
            samples = min(samples, max_samples)
        self.base = ImageNetDataset(ROOT / "Dataset", samples, split["start"])
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        family, corruption_name = self.conditions[index % len(self.conditions)]
        image = Image.open(self.base.image_paths[index]).convert("RGB")
        image = apply_family_corruption(
            image, corruption_name, self.seed + self.base.start_index + index
        )
        pixels = self.base.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return pixels, int(family == "noise")


class ImageNetConditionDataset(Dataset):
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
        if self.corruption_name is not None:
            if self.corruption_name in {c for values in EXPERT_FAMILIES.values() for c in values}:
                image = apply_family_corruption(
                    image,
                    self.corruption_name,
                    self.seed + self.base.start_index + index,
                )
            else:
                image = corrupt(
                    image,
                    self.corruption_name,
                    self.severity,
                    self.seed + self.base.start_index + index,
                )
        pixels = self.base.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return pixels, self.base.labels[index]


class SketchConditionDataset(Dataset):
    def __init__(self, data_root, manifest, corruption_name, severity, seed, max_samples=None):
        self.data_root = Path(data_root)
        self.items = manifest["items"]
        if max_samples is not None:
            self.items = self.items[:max_samples]
        self.corruption_name = corruption_name
        self.severity = severity
        self.seed = seed
        self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        image = Image.open(self.data_root / item["relative_path"]).convert("RGB")
        if self.corruption_name is not None:
            image = corrupt(
                image, self.corruption_name, self.severity, self.seed + index
            )
        pixels = self.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return pixels, item["label"]


def early_statistics(hidden_states):
    return torch.cat(
        [hidden_states[block][:, 1:].mean(dim=1) for block in ROUTER_BLOCKS], dim=1
    )


def extract_router_features(model, data_loader, device, description):
    features, targets = [], []
    with torch.no_grad():
        for pixels, labels in tqdm(data_loader, desc=description, leave=False):
            outputs = model(pixel_values=pixels.to(device), output_hidden_states=True)
            features.append(early_statistics(outputs.hidden_states).cpu())
            targets.append(torch.as_tensor(labels).long().cpu())
    return torch.cat(features), torch.cat(targets)


def router_loader(dataset, args, shuffle, seed):
    generator = torch.Generator().manual_seed(seed + 7700)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        generator=generator if shuffle else None,
    )


def fit_router(model, splits, seed, device, args, seed_dir):
    checkpoint = seed_dir / "linear_router.pt"
    record_path = seed_dir / "router_training.json"
    if args.resume and checkpoint.exists() and record_path.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        router = LinearRouter().to(device)
        router.load_state_dict(payload["router"])
        return router.eval(), payload["mean"], payload["std"], json.loads(record_path.read_text())

    train_images = router_loader(
        RouterTrainingDataset(splits[seed]["train"], seed, args.max_samples),
        args,
        False,
        seed,
    )
    validation_images = router_loader(
        RouterTrainingDataset(splits[seed]["validation"], seed, args.max_samples),
        args,
        False,
        seed,
    )
    train_x, train_y = extract_router_features(
        model, train_images, device, f"seed {seed} router feature train"
    )
    validation_x, validation_y = extract_router_features(
        model, validation_images, device, f"seed {seed} router feature validation"
    )
    mean = train_x.mean(0)
    std = train_x.std(0).clamp_min(1e-6)
    train_x = (train_x - mean) / std
    validation_x = (validation_x - mean) / std
    train_data = TensorDataset(train_x, train_y.float())
    router = LinearRouter().to(device)
    optimizer = AdamW(router.parameters(), lr=args.router_learning_rate, weight_decay=args.router_weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    best = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.router_epochs + 1):
        router.train()
        total_loss = 0.0
        train_batches = router_loader(train_data, args, True, seed + epoch)
        for batch_x, batch_y in train_batches:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(router(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_x.shape[0]
        router.eval()
        with torch.no_grad():
            logits = router(validation_x.to(device)).cpu()
            accuracy = float(((logits >= 0) == validation_y.bool()).float().mean())
            validation_loss = float(criterion(logits, validation_y.float()))
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_data),
                "validation_loss": validation_loss,
                "validation_family_accuracy": accuracy,
            }
        )
        print(f"seed={seed} router epoch={epoch} validation_family_accuracy={accuracy:.4f}")
        if accuracy > best:
            best = accuracy
            stale = 0
            torch.save(
                {"router": router.state_dict(), "mean": mean, "std": std}, checkpoint
            )
        else:
            stale += 1
            if stale >= args.router_patience:
                break
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    router.load_state_dict(payload["router"])
    record = {"best_validation_family_accuracy": best, "history": history}
    record_path.write_text(json.dumps(record, indent=2))
    return router.eval(), payload["mean"], payload["std"], record


def load_experts(seed, rank, device):
    experts = {}
    for family in EXPERT_FAMILIES:
        adapter = LowRankHiddenAdapter(rank).to(device)
        checkpoint = ORACLE_RUN / f"seed_{seed}" / f"{family}_rank{rank}.pt"
        adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        experts[family] = adapter.eval()
    return experts


def evaluate(model, experts, mixed, router, mean, std, data_loader, oracle_family, device, args, stat_seed):
    names = ["baseline", "mixed", "uniform", "router_soft", "router_hard"]
    if oracle_family is not None:
        names.append("oracle")
    arrays = {name: [] for name in names}
    route_probabilities = []
    with torch.no_grad():
        for pixels, labels in tqdm(data_loader, desc="learned MoE evaluation", leave=False):
            labels = labels.to(device)
            outputs = model(pixel_values=pixels.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            patches = hidden[:, 1:]
            noise_residual = experts["noise"](patches)
            blur_residual = experts["blur"](patches)
            statistics = early_statistics(outputs.hidden_states)
            probabilities = torch.sigmoid(
                router(
                    (statistics - mean.to(device))
                    / std.to(device)
                )
            )
            route_probabilities.extend(probabilities.cpu().tolist())
            soft = probabilities[:, None, None] * noise_residual + (
                1 - probabilities[:, None, None]
            ) * blur_residual
            hard_noise = (probabilities >= 0.5)[:, None, None]
            candidates = {
                "mixed": mixed(patches),
                "uniform": 0.5 * (noise_residual + blur_residual),
                "router_soft": soft,
                "router_hard": torch.where(hard_noise, noise_residual, blur_residual),
            }
            if oracle_family is not None:
                candidates["oracle"] = (
                    noise_residual if oracle_family == "noise" else blur_residual
                )
            arrays["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for name, residual in candidates.items():
                corrected = torch.cat([hidden[:, :1], patches + residual], dim=1)
                logits = downstream_from_layer(model, corrected, BLOCK - 1)
                arrays[name].extend((logits.argmax(1) == labels).cpu().tolist())
    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    probabilities = np.asarray(route_probabilities, dtype=np.float32)
    result = {
        "methods": {
            name: paired_comparison(arrays["baseline"], values, stat_seed + offset, args.bootstrap)
            for offset, (name, values) in enumerate(arrays.items())
            if name != "baseline"
        },
        "routing": {
            "mean_noise_probability": float(probabilities.mean()),
            "hard_noise_fraction": float((probabilities >= 0.5).mean()),
        },
    }
    if oracle_family is not None:
        target = oracle_family == "noise"
        result["routing"]["family_accuracy"] = float(
            ((probabilities >= 0.5) == target).mean()
        )
    return result, arrays, probabilities


def make_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )


def evaluate_and_save(model, experts, mixed, router, mean, std, dataset, oracle_family, device, args, path, stat_seed):
    result, arrays, probabilities = evaluate(
        model,
        experts,
        mixed,
        router,
        mean,
        std,
        make_loader(dataset, args),
        oracle_family,
        device,
        args,
        stat_seed,
    )
    path.write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        path.with_suffix(".npz"),
        **arrays,
        router_noise_probability=probabilities,
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=212)
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--router-patience", type=int, default=5)
    parser.add_argument("--router-learning-rate", type=float, default=1e-3)
    parser.add_argument("--router-weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--reserve-start", type=int, default=39000)
    parser.add_argument("--reserve-samples", type=int, default=3000)
    parser.add_argument("--sketch-root", type=Path, default=SKETCH_ROOT)
    parser.add_argument("--sketch-manifest", type=Path, default=SKETCH_MANIFEST)
    parser.add_argument("--skip-sketch", action="store_true")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    sketch_manifest = None
    if not args.skip_sketch:
        sketch_manifest = json.loads(args.sketch_manifest.read_text())
        if sketch_manifest["samples"] != 3000 or len(
            {item["sha256"] for item in sketch_manifest["items"]}
        ) != 3000:
            raise ValueError("ImageNet-Sketch frozen manifest is invalid")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    router_parameters = sum(parameter.numel() for parameter in LinearRouter().parameters())
    if 2 * 368164 + router_parameters > 741120:
        raise ValueError("MoE exceeds the monolithic adapter parameter budget")

    training = {}
    results = {"development_seen": {}, "reserve_unseen": {}, "sketch_unseen": {}}
    for seed in args.seeds:
        torch.manual_seed(seed + 7700)
        np.random.seed(seed + 7700)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        router, mean, std, training[str(seed)] = fit_router(
            model, splits, seed, device, args, seed_dir
        )
        experts = load_experts(seed, args.rank, device)
        mixed = load_full_mixed(seed, device)

        development_conditions = [("clean", None, None)] + [
            (corruption, corruption, family)
            for family, corruptions in EXPERT_FAMILIES.items()
            for corruption in corruptions
        ]
        results["development_seen"][str(seed)] = {}
        for condition_index, (key, corruption_name, family) in enumerate(development_conditions):
            path = seed_dir / f"development_{key}.json"
            if args.resume and path.exists():
                result = json.loads(path.read_text())
            else:
                dataset = ImageNetConditionDataset(
                    splits[seed]["validation"],
                    corruption_name,
                    args.severity,
                    seed,
                    args.max_samples,
                )
                result = evaluate_and_save(
                    model, experts, mixed, router, mean, std, dataset, family,
                    device, args, path, 77000 + seed * 1000 + condition_index * 20,
                )
            results["development_seen"][str(seed)][key] = result

        results["reserve_unseen"][str(seed)] = {}
        reserve_split = {
            "start": args.reserve_start,
            "end": args.reserve_start + args.reserve_samples,
        }
        for condition_index, corruption_name in enumerate(UNSEEN):
            key = f"{corruption_name}_{args.severity}"
            path = seed_dir / f"reserve_{key}.json"
            if args.resume and path.exists():
                result = json.loads(path.read_text())
            else:
                dataset = ImageNetConditionDataset(
                    reserve_split, corruption_name, args.severity, 2068, args.max_samples
                )
                result = evaluate_and_save(
                    model, experts, mixed, router, mean, std, dataset, None,
                    device, args, path, 78000 + seed * 1000 + condition_index * 20,
                )
            results["reserve_unseen"][str(seed)][key] = result

        if sketch_manifest is not None:
            results["sketch_unseen"][str(seed)] = {}
            for condition_index, corruption_name in enumerate(UNSEEN):
                key = f"{corruption_name}_{args.severity}"
                path = seed_dir / f"sketch_{key}.json"
                if args.resume and path.exists():
                    result = json.loads(path.read_text())
                else:
                    dataset = SketchConditionDataset(
                        args.sketch_root,
                        sketch_manifest,
                        corruption_name,
                        args.severity,
                        2068,
                        args.max_samples,
                    )
                    result = evaluate_and_save(
                        model, experts, mixed, router, mean, std, dataset, None,
                        device, args, path, 79000 + seed * 1000 + condition_index * 20,
                    )
                results["sketch_unseen"][str(seed)][key] = result

    serializable_args = vars(args) | {
        "split_manifest": str(args.split_manifest.resolve()),
        "sketch_root": str(args.sketch_root.resolve()),
        "sketch_manifest": str(args.sketch_manifest.resolve()),
    }
    summary = {
        "configuration": serializable_args
        | {
            "model": BASE_MODEL,
            "vit_frozen": True,
            "experts_frozen": True,
            "block": BLOCK,
            "router_blocks": ROUTER_BLOCKS,
            "router_input": "channel-wise mean of patch tokens from Blocks 1-3",
            "router_parameters": router_parameters,
            "two_expert_parameters": 736328,
            "total_moe_parameters": 736328 + router_parameters,
            "monolithic_adapter_parameters": 741120,
            "router_uses_only_corrupted_image": True,
            "router_uses_clean_counterpart": False,
            "router_uses_corruption_label_at_inference": False,
            "imagenetv2_accessed": False,
            "sketch_manifest_sha256": (
                file_sha256(args.sketch_manifest) if sketch_manifest is not None else None
            ),
        },
        "splits": {str(seed): splits[seed] for seed in args.seeds},
        "training": training,
        "results": results,
        "limitations": [
            "Router family supervision is used during development training, but never at inference.",
            "The ImageNet validation reserve range was used by earlier diagnostics and is not a pristine final benchmark.",
            "ImageNet-Sketch is a frozen cross-domain evaluation and must not be used to tune this router after inspection.",
            "Online corruptions are controlled approximations, not official ImageNet-C.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
