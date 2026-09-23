import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment68_unseen_online_corruptions import corrupt
from scripts.experiment76_parameter_matched_oracle_moe import (
    ACTIVE_ROOT,
    EXPERT_FAMILIES,
    apply_family_corruption,
)


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment99_attention_head_corruption_detector"
SEEN = {
    "clean": (None,),
    "noise": EXPERT_FAMILIES["noise"],
    "blur": EXPERT_FAMILIES["blur"],
}
UNSEEN = ("jpeg", "pixelate", "brightness", "contrast")
LABELS = {"clean": 0, "noise": 1, "blur": 2}
METRICS = ("entropy", "maximum", "cls_entropy", "cls_maximum", "cls_patch_mass")


class ConditionDataset(Dataset):
    def __init__(self, split, condition, severity, seed, samples):
        available = split["end"] - split["start"]
        self.base = ImageNetDataset(ROOT / "Dataset", min(samples, available), split["start"])
        self.condition = condition
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = Image.open(self.base.image_paths[index]).convert("RGB")
        if self.condition is not None:
            if self.condition in {item for values in EXPERT_FAMILIES.values() for item in values}:
                image = apply_family_corruption(
                    image, self.condition, self.seed + self.base.start_index + index
                )
            else:
                image = corrupt(
                    image,
                    self.condition,
                    self.severity,
                    self.seed + self.base.start_index + index,
                )
        pixels = self.base.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        return pixels


class HeadRouter(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.linear = nn.Linear(input_dimension, len(LABELS))

    def forward(self, values):
        return self.linear(values)


def attention_head_statistics(attentions):
    layer_features = []
    epsilon = 1e-12
    for attention in attentions:
        probabilities = attention.clamp_min(epsilon)
        entropy = -(probabilities * probabilities.log()).sum(dim=-1).mean(dim=-1)
        maximum = probabilities.max(dim=-1).values.mean(dim=-1)
        cls = probabilities[:, :, 0]
        cls_entropy = -(cls * cls.log()).sum(dim=-1)
        cls_maximum = cls.max(dim=-1).values
        cls_patch_mass = cls[:, :, 1:].sum(dim=-1)
        layer_features.append(
            torch.stack(
                [entropy, maximum, cls_entropy, cls_maximum, cls_patch_mass], dim=-1
            ).flatten(1)
        )
    return torch.cat(layer_features, dim=1)


def make_loader(dataset, args):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
    )


def extract(model, dataset, device, args, description):
    features = []
    with torch.no_grad():
        for pixels in tqdm(make_loader(dataset, args), desc=description, leave=False):
            outputs = model(
                pixel_values=pixels.to(device),
                output_attentions=True,
                return_dict=True,
            )
            if outputs.attentions is None:
                raise RuntimeError("Attention tensors unavailable; eager attention is required")
            features.append(attention_head_statistics(outputs.attentions).cpu())
    return torch.cat(features)


def cached_extract(path, model, dataset, device, args, description):
    if args.resume and path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    values = extract(model, dataset, device, args, description)
    torch.save(values, path)
    return values


def condition_features(model, split, conditions, samples, seed, device, args, seed_dir, prefix):
    feature_parts = []
    label_parts = []
    names = []
    for family, corruptions in conditions.items():
        samples_per_condition = max(1, samples // len(corruptions))
        for corruption_name in corruptions:
            key = "clean" if corruption_name is None else corruption_name
            path = seed_dir / f"{prefix}_{key}_features.pt"
            dataset = ConditionDataset(
                split, corruption_name, args.severity, seed, samples_per_condition
            )
            values = cached_extract(path, model, dataset, device, args, f"{prefix} {key}")
            feature_parts.append(values)
            label_parts.append(torch.full((len(values),), LABELS[family], dtype=torch.long))
            names.extend([key] * len(values))
    return torch.cat(feature_parts), torch.cat(label_parts), names


def confusion_matrix(targets, predictions):
    matrix = torch.zeros((len(LABELS), len(LABELS)), dtype=torch.long)
    for target, prediction in zip(targets.tolist(), predictions.tolist()):
        matrix[target, prediction] += 1
    return matrix.tolist()


def evaluate(router, features, targets, names, clean_mean, clean_std, device):
    standardized = (features - clean_mean) / clean_std
    with torch.no_grad():
        probabilities = router(standardized.to(device)).softmax(dim=1).cpu()
    predictions = probabilities.argmax(dim=1)
    by_condition = {}
    for name in sorted(set(names)):
        mask = torch.tensor([item == name for item in names])
        by_condition[name] = {
            "samples": int(mask.sum()),
            "accuracy": float((predictions[mask] == targets[mask]).float().mean()),
            "mean_probabilities_clean_noise_blur": probabilities[mask].mean(0).tolist(),
            "mean_clean_abnormality": float(standardized[mask].abs().mean()),
        }
    return {
        "accuracy": float((predictions == targets).float().mean()),
        "confusion_matrix_rows_true_columns_predicted": confusion_matrix(targets, predictions),
        "by_condition": by_condition,
    }


def fit_router(train_features, train_targets, validation, clean_mean, clean_std, seed, args, seed_dir, device):
    checkpoint = seed_dir / "attention_head_router.pt"
    record_path = seed_dir / "router_training.json"
    input_dimension = train_features.shape[1]
    if args.resume and checkpoint.exists() and record_path.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        router = HeadRouter(input_dimension).to(device)
        router.load_state_dict(payload["router"])
        return router.eval(), json.loads(record_path.read_text())

    train_features = (train_features - clean_mean) / clean_std
    data = TensorDataset(train_features, train_targets)
    router = HeadRouter(input_dimension).to(device)
    optimizer = AdamW(router.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed + 9900)
    best_accuracy = -1.0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        router.train()
        loader = DataLoader(data, batch_size=args.router_batch_size, shuffle=True, generator=generator)
        total_loss = 0.0
        for batch_features, batch_targets in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(router(batch_features.to(device)), batch_targets.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(batch_features)
        router.eval()
        result = evaluate(router, *validation, clean_mean, clean_std, device)
        history.append({"epoch": epoch, "training_loss": total_loss / len(data), "validation_accuracy": result["accuracy"]})
        print(f"seed={seed} epoch={epoch} validation_accuracy={result['accuracy']:.4f}")
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            stale = 0
            torch.save({"router": router.state_dict()}, checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    router.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)["router"])
    record = {"best_validation_accuracy": best_accuracy, "history": history}
    record_path.write_text(json.dumps(record, indent=2))
    return router.eval(), record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--calibration-samples", type=int, default=1000)
    parser.add_argument("--train-samples-per-family", type=int, default=1000)
    parser.add_argument("--validation-samples-per-family", type=int, default=500)
    parser.add_argument("--severity", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--router-batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)

    seed_results = {}
    training_records = {}
    split_records = {}
    for seed in args.seeds:
        torch.manual_seed(seed + 9900)
        np.random.seed(seed + 9900)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        source_train = splits[seed]["train"]
        calibration_split = {
            "start": source_train["start"],
            "end": source_train["start"] + args.calibration_samples,
        }
        router_train_split = {
            "start": calibration_split["end"],
            "end": source_train["end"],
        }
        validation_split = splits[seed]["validation"]
        split_records[str(seed)] = {
            "clean_calibration": calibration_split,
            "router_training": router_train_split,
            "validation": validation_split,
        }

        calibration_path = seed_dir / "clean_calibration_features.pt"
        calibration = cached_extract(
            calibration_path,
            model,
            ConditionDataset(calibration_split, None, args.severity, seed, args.calibration_samples),
            device,
            args,
            f"seed {seed} clean calibration",
        )
        clean_mean = calibration.mean(0)
        clean_std = calibration.std(0).clamp_min(1e-6)
        train = condition_features(
            model, router_train_split, SEEN, args.train_samples_per_family,
            seed, device, args, seed_dir, "train"
        )
        validation = condition_features(
            model, validation_split, SEEN, args.validation_samples_per_family,
            seed, device, args, seed_dir, "validation"
        )
        router, training_records[str(seed)] = fit_router(
            train[0], train[1], validation, clean_mean, clean_std,
            seed, args, seed_dir, device
        )
        seed_results[str(seed)] = {
            "seen_validation": evaluate(router, *validation, clean_mean, clean_std, device),
            "unseen_validation": {},
        }
        for condition in UNSEEN:
            features, _, names = condition_features(
                model,
                validation_split,
                {"clean": (condition,)},
                args.validation_samples_per_family,
                seed,
                device,
                args,
                seed_dir,
                "unseen",
            )
            dummy_targets = torch.zeros(len(features), dtype=torch.long)
            seed_results[str(seed)]["unseen_validation"][condition] = evaluate(
                router, features, dummy_targets, names, clean_mean, clean_std, device
            )["by_condition"][condition]

    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "vit_frozen": True,
            "attention_implementation": "eager",
            "classes": LABELS,
            "features": list(METRICS),
            "feature_dimension": 12 * 12 * len(METRICS),
            "router_parameters": (12 * 12 * len(METRICS) + 1) * len(LABELS),
            "router_uses_only_test_image": True,
            "clean_counterpart_used_at_inference": False,
            "corruption_label_used_at_inference": False,
            "imageNetV2_accessed": False,
            "status": "development diagnostic only; no adapter routing or final evaluation",
        },
        "splits": split_records,
        "training": training_records,
        "results": seed_results,
        "guardrails": [
            "Clean calibration, router training, and validation image ranges are disjoint within each seed.",
            "Unseen corruption types are absent from fitting but evaluated on the development validation range.",
            "This phase diagnoses routing information only and does not select or apply an adapter.",
            "ImageNetV2 and the consumed final reserve are not accessed.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(seed_results, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
