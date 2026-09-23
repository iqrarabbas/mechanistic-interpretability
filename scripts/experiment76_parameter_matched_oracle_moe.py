import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from corruption.gaussian_blur import apply_gaussian_blur
from corruption.gaussian_noise import apply_gaussian_noise
from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment50_block6_clean_preservation import adapter_loss
from scripts.experiment54_lowrank_block6_adapter import LowRankHiddenAdapter
from scripts.experiment68_unseen_online_corruptions import corrupt


ROOT = Path(__file__).parent.parent
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment76_parameter_matched_oracle_moe"
MIXED_ROOT = (
    ACTIVE_ROOT
    / "results/sae/experiment67_mixed_noise_blur_block6/"
    "full_3seed_noise_blur_identity0p2_v1"
)
BLOCK = 6
EXPERT_FAMILIES = {
    "noise": ("gaussian_noise", "shot_noise", "impulse_noise"),
    "blur": ("gaussian_blur", "disk_blur", "motion_blur"),
}


def apply_family_corruption(image, name, seed):
    if name == "clean":
        return image
    if name == "gaussian_noise":
        return apply_gaussian_noise(image, severity=4, seed=seed)
    if name == "gaussian_blur":
        return apply_gaussian_blur(image, severity=4)
    if name in {"shot_noise", "impulse_noise"}:
        return corrupt(image, name, 4, seed)
    if name == "disk_blur":
        kernel = [
            0, 0, 1, 0, 0,
            0, 1, 1, 1, 0,
            1, 1, 1, 1, 1,
            0, 1, 1, 1, 0,
            0, 0, 1, 0, 0,
        ]
        return image.filter(ImageFilter.Kernel((5, 5), kernel, scale=sum(kernel)))
    if name == "motion_blur":
        values = np.asarray(image, dtype=np.float32)
        padded = np.pad(values, ((0, 0), (6, 6), (0, 0)), mode="reflect")
        blurred = np.mean([padded[:, offset : offset + values.shape[1]] for offset in range(13)], axis=0)
        return Image.fromarray(np.clip(blurred, 0, 255).astype(np.uint8))
    raise ValueError(name)


class FamilyPairs(Dataset):
    def __init__(self, split, corruptions, seed, fixed=None, max_samples=None):
        samples = split["end"] - split["start"]
        if max_samples is not None:
            samples = min(samples, max_samples)
        self.base = ImageNetDataset(ROOT / "Dataset", samples, split["start"])
        self.corruptions = tuple(corruptions)
        self.seed = seed
        self.fixed = fixed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        path = self.base.image_paths[index]
        label = self.base.labels[index]
        image = Image.open(path).convert("RGB")
        name = self.fixed or self.corruptions[index % len(self.corruptions)]
        corrupted = apply_family_corruption(
            image, name, self.seed + self.base.start_index + index
        )
        clean_pixels = self.base.processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)
        corrupt_pixels = self.base.processor(images=corrupted, return_tensors="pt")["pixel_values"].squeeze(0)
        return clean_pixels, corrupt_pixels, label


def loader(split, corruptions, seed, args, shuffle, fixed=None):
    generator = torch.Generator().manual_seed(seed + 7600)
    return DataLoader(
        FamilyPairs(split, corruptions, seed, fixed, args.max_samples),
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        generator=generator if shuffle else None,
    )


def epoch(model, adapter, data_loader, device, args, optimizer=None):
    training = optimizer is not None
    adapter.train(training)
    totals = np.zeros(5, dtype=np.float64)
    samples = 0
    for clean, corrupted, labels in tqdm(data_loader, desc="expert training" if training else "expert validation", leave=False):
        labels = labels.to(device)
        with torch.no_grad():
            hidden = model(
                pixel_values=torch.cat([clean, corrupted]).to(device),
                output_hidden_states=True,
            ).hidden_states[BLOCK]
            clean_hidden, corrupt_hidden = hidden.split(clean.shape[0])
        if training:
            optimizer.zero_grad(set_to_none=True)
        losses = adapter_loss(
            model, adapter, clean_hidden, corrupt_hidden, labels, args.identity_weight, args
        )
        if training:
            losses[0].backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
        batch = clean.shape[0]
        totals += np.asarray([float(value.detach()) for value in losses]) * batch
        samples += batch
    return dict(zip(["total", "residual", "classification", "preservation", "identity"], (totals / samples).tolist()))


def train_expert(model, family, seed, splits, device, args, seed_dir):
    adapter = LowRankHiddenAdapter(args.rank).to(device)
    checkpoint = seed_dir / f"{family}_rank{args.rank}.pt"
    record_path = seed_dir / f"{family}_training.json"
    if args.resume and checkpoint.exists() and record_path.exists():
        adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        return adapter.eval(), json.loads(record_path.read_text()), checkpoint
    optimizer = AdamW(adapter.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    train_loader = loader(splits[seed]["train"], EXPERT_FAMILIES[family], seed, args, True)
    validation_loader = loader(splits[seed]["validation"], EXPERT_FAMILIES[family], seed, args, False)
    best = float("inf")
    stale = 0
    history = []
    for epoch_index in range(1, args.epochs + 1):
        train_metrics = epoch(model, adapter, train_loader, device, args, optimizer)
        with torch.no_grad():
            validation_metrics = epoch(model, adapter, validation_loader, device, args)
        history.append({"epoch": epoch_index, "train": train_metrics, "validation": validation_metrics})
        scheduler.step()
        value = validation_metrics["total"]
        print(f"seed={seed} expert={family} epoch={epoch_index} validation={value:.6f}")
        if value < best:
            best = value
            stale = 0
            torch.save(adapter.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= args.patience:
                break
    adapter.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    record = {"best_validation_total": best, "history": history}
    record_path.write_text(json.dumps(record, indent=2))
    return adapter.eval(), record, checkpoint


def load_full_mixed(seed, device):
    from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear

    adapter = HiddenLinear().to(device)
    path = MIXED_ROOT / f"seed_{seed}" / "mixed_identity0p2.pt"
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval()


def evaluate_condition(model, experts, mixed, data_loader, oracle_family, device, args, seed):
    arrays = {name: [] for name in ["baseline", "mixed", "oracle", "uniform", "noise_expert", "blur_expert"]}
    with torch.no_grad():
        for _, corrupted, labels in tqdm(data_loader, desc="oracle MoE evaluation", leave=False):
            labels = labels.to(device)
            outputs = model(pixel_values=corrupted.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            patches = hidden[:, 1:]
            residuals = {name: adapter(patches) for name, adapter in experts.items()}
            candidates = {
                "mixed": mixed(patches),
                "oracle": residuals[oracle_family] if oracle_family else 0.5 * (residuals["noise"] + residuals["blur"]),
                "uniform": 0.5 * (residuals["noise"] + residuals["blur"]),
                "noise_expert": residuals["noise"],
                "blur_expert": residuals["blur"],
            }
            arrays["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for name, residual in candidates.items():
                corrected = torch.cat([hidden[:, :1], patches + residual], dim=1)
                logits = downstream_from_layer(model, corrected, BLOCK - 1)
                arrays[name].extend((logits.argmax(1) == labels).cpu().tolist())
    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    return {
        name: paired_comparison(arrays["baseline"], values, seed + offset, args.bootstrap)
        for offset, (name, values) in enumerate(arrays.items()) if name != "baseline"
    }, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=212)
    parser.add_argument("--identity-weight", type=float, default=0.2)
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
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
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    training = {}
    evaluation = {}
    outcomes = {}
    checkpoints = {}
    expert_parameters = sum(p.numel() for p in LowRankHiddenAdapter(args.rank).parameters())
    for seed in args.seeds:
        torch.manual_seed(seed + 7600)
        np.random.seed(seed + 7600)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        experts = {}
        training[str(seed)] = {}
        checkpoints[str(seed)] = {}
        for family in EXPERT_FAMILIES:
            experts[family], record, checkpoint = train_expert(
                model, family, seed, splits, device, args, seed_dir
            )
            training[str(seed)][family] = record
            checkpoints[str(seed)][family] = str(checkpoint)
        mixed = load_full_mixed(seed, device)
        evaluation[str(seed)] = {}
        clean_loader = loader(
            splits[seed]["validation"], ("clean",), seed, args, False, fixed="clean"
        )
        clean_result, clean_arrays = evaluate_condition(
            model, experts, mixed, clean_loader, None, device, args, 2076 + seed * 1000
        )
        evaluation[str(seed)]["clean"] = clean_result
        for name, values in clean_arrays.items():
            outcomes[f"seed{seed}_clean_{name}"] = values
        for family, corruptions in EXPERT_FAMILIES.items():
            for condition_index, corruption_name in enumerate(corruptions):
                condition_loader = loader(
                    splits[seed]["validation"],
                    (corruption_name,),
                    seed,
                    args,
                    False,
                    fixed=corruption_name,
                )
                result, arrays = evaluate_condition(
                    model,
                    experts,
                    mixed,
                    condition_loader,
                    family,
                    device,
                    args,
                    2076 + seed * 1000 + condition_index * 20,
                )
                evaluation[str(seed)][corruption_name] = result
                for name, values in arrays.items():
                    outcomes[f"seed{seed}_{corruption_name}_{name}"] = values
    summary = {
        "configuration": vars(args)
        | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "block": BLOCK,
            "expert_families": EXPERT_FAMILIES,
            "expert_parameters_each": expert_parameters,
            "two_expert_parameters": 2 * expert_parameters,
            "full_mixed_adapter_parameters": 741120,
            "remaining_router_budget": 741120 - 2 * expert_parameters,
            "vit_frozen": True,
            "oracle_uses_known_corruption_family": True,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "parameter-matched MoE oracle feasibility development",
        },
        "splits": {str(seed): splits[seed] for seed in args.seeds},
        "checkpoints": checkpoints,
        "training": training,
        "evaluation": evaluation,
        "decision_rule": "Proceed to a learned router only if oracle routing consistently outperforms both the parameter-matched uniform mixture and the full monolithic mixed adapter without unacceptable clean degradation.",
        "limitations": [
            "Oracle routing uses the known corruption family and is an upper-bound diagnostic, not deployable inference.",
            "Disk and motion blur are controlled online approximations, not official ImageNet-C implementations.",
            "Experts and evaluation use seed-specific disjoint train/validation splits, but this remains development rather than final independent evaluation.",
        ],
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
