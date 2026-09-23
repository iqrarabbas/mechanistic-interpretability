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
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment75_block7_qkv_path_mediation import attention_output, mlp_output, qkv
from scripts.experiment76_parameter_matched_oracle_moe import (
    ACTIVE_ROOT,
    BLOCK,
    EXPERT_FAMILIES,
    apply_family_corruption,
    load_full_mixed,
)
from scripts.experiment77_learned_moe_router import (
    ORACLE_RUN,
    ROUTER_BLOCKS,
    LinearRouter,
    early_statistics,
    load_experts,
)
from scripts.experiment79_frozen_router_severity_sweep import apply_severity_corruption


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment82_ood_repair_utility_router"
FAMILY_ROUTER_ROOT = (
    ACTIVE_ROOT
    / "results/sae/experiment77_learned_moe_router/"
    "full_3seed_linear_router_seen_unseen_v1"
)
SEEN = tuple(
    corruption
    for corruptions in EXPERT_FAMILIES.values()
    for corruption in corruptions
)
UNSEEN = ("jpeg", "pixelate", "brightness", "contrast")
ACTIONS = ("none", "noise", "blur")
BLOCK7_INDEX = 6


class UtilityDataset(Dataset):
    def __init__(self, split, seed, include_clean, max_samples=None):
        samples = split["end"] - split["start"]
        if max_samples is not None:
            samples = min(samples, max_samples)
        self.base = ImageNetDataset(ROOT / "Dataset", samples, split["start"])
        self.seed = seed
        self.conditions = (("clean",) + SEEN) if include_clean else SEEN

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        condition = self.conditions[index % len(self.conditions)]
        image = Image.open(self.base.image_paths[index]).convert("RGB")
        if condition != "clean":
            image = apply_family_corruption(
                image,
                condition,
                self.seed + self.base.start_index + index,
            )
        pixels = self.base.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return pixels, self.base.labels[index], condition


class EvaluationDataset(Dataset):
    def __init__(self, split, corruption, severity, seed, max_samples=None):
        samples = split["end"] - split["start"]
        if max_samples is not None:
            samples = min(samples, max_samples)
        self.base = ImageNetDataset(ROOT / "Dataset", samples, split["start"])
        self.corruption = corruption
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        image = Image.open(self.base.image_paths[index]).convert("RGB")
        if self.corruption != "clean":
            image = apply_severity_corruption(
                image,
                self.corruption,
                self.severity,
                self.seed + self.base.start_index + index,
            )
        pixels = self.base.processor(images=image, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)
        return pixels, self.base.labels[index]


class UtilityRouter(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, len(ACTIONS))

    def forward(self, features):
        return self.linear(features)


def reduced_change(reference, candidate):
    reference_flat = reference.reshape(reference.shape[0], -1).float()
    candidate_flat = candidate.reshape(candidate.shape[0], -1).float()
    difference = candidate_flat - reference_flat
    relative_l2 = difference.norm(dim=1) / reference_flat.norm(dim=1).clamp_min(1e-6)
    cosine = nn.functional.cosine_similarity(reference_flat, candidate_flat, dim=1)
    mean_shift = difference.mean(dim=1) / reference_flat.std(dim=1).clamp_min(1e-6)
    return torch.stack((relative_l2, cosine, mean_shift), dim=1)


def expert_mechanism_features(model, hidden6, residual):
    layer = model.vit.layers[BLOCK7_INDEX]
    corrected6 = torch.cat(
        (hidden6[:, :1], hidden6[:, 1:] + residual), dim=1
    )
    base_norm = layer.layernorm_before(hidden6)
    corrected_norm = layer.layernorm_before(corrected6)
    base_q, base_k, base_v = qkv(layer.attention, base_norm)
    corrected_q, corrected_k, corrected_v = qkv(layer.attention, corrected_norm)
    base_attention = attention_output(
        layer, hidden6, base_q, base_k, base_v
    )
    corrected_attention = attention_output(
        layer, corrected6, corrected_q, corrected_k, corrected_v
    )
    base_after_attention = hidden6 + base_attention
    corrected_after_attention = corrected6 + corrected_attention
    base_mlp = mlp_output(layer, base_after_attention)
    corrected_mlp = mlp_output(layer, corrected_after_attention)
    return torch.cat(
        (
            reduced_change(hidden6[:, 1:], corrected6[:, 1:]),
            reduced_change(base_q, corrected_q),
            reduced_change(base_k, corrected_k),
            reduced_change(base_v, corrected_v),
            reduced_change(base_attention, corrected_attention),
            reduced_change(base_mlp, corrected_mlp),
        ),
        dim=1,
    )


def candidate_outputs(model, experts, outputs, labels):
    hidden = outputs.hidden_states[BLOCK]
    patches = hidden[:, 1:]
    residuals = {name: expert(patches) for name, expert in experts.items()}
    logits = {"none": outputs.logits}
    for name, residual in residuals.items():
        corrected = torch.cat((hidden[:, :1], patches + residual), dim=1)
        logits[name] = downstream_from_layer(model, corrected, BLOCK - 1)
    margins = []
    for name in ACTIONS:
        _, margin = classification_margin(logits[name], labels)
        margins.append(margin)
    return hidden, residuals, logits, torch.stack(margins, dim=1)


def extract_utility_data(model, experts, data_loader, device, description):
    early_features = []
    mechanism_features = []
    targets = []
    margin_utilities = []
    conditions = []
    with torch.no_grad():
        for pixels, labels, batch_conditions in tqdm(
            data_loader, desc=description, leave=False
        ):
            labels = labels.to(device)
            outputs = model(
                pixel_values=pixels.to(device), output_hidden_states=True
            )
            hidden, residuals, _, margins = candidate_outputs(
                model, experts, outputs, labels
            )
            early_features.append(early_statistics(outputs.hidden_states).cpu())
            mechanism_features.append(
                torch.cat(
                    [
                        expert_mechanism_features(model, hidden, residuals[name])
                        for name in ("noise", "blur")
                    ],
                    dim=1,
                ).cpu()
            )
            utility = margins - margins[:, :1]
            targets.append(utility.argmax(dim=1).cpu())
            margin_utilities.append(utility.cpu())
            conditions.extend(batch_conditions)
    return {
        "early": torch.cat(early_features),
        "mechanism": torch.cat(mechanism_features),
        "targets": torch.cat(targets),
        "utilities": torch.cat(margin_utilities),
        "conditions": np.asarray(conditions),
    }


def make_loader(dataset, args, shuffle=False, seed=0):
    generator = torch.Generator().manual_seed(seed + 8200)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        generator=generator if shuffle else None,
    )


def normalize(train, validation):
    mean = train.mean(dim=0)
    std = train.std(dim=0).clamp_min(1e-6)
    return (train - mean) / std, (validation - mean) / std, mean, std


def fit_utility_router(name, train, validation, seed, device, args, seed_dir):
    checkpoint = seed_dir / f"{name}_utility_router.pt"
    record_path = seed_dir / f"{name}_utility_router_training.json"
    if args.resume and checkpoint.exists() and record_path.exists():
        payload = torch.load(checkpoint, map_location=device, weights_only=True)
        router = UtilityRouter(payload["input_dim"]).to(device)
        router.load_state_dict(payload["router"])
        return (
            router.eval(),
            payload["mean"],
            payload["std"],
            json.loads(record_path.read_text()),
            checkpoint,
        )
    train_x, validation_x, mean, std = normalize(
        train[name], validation[name]
    )
    router = UtilityRouter(train_x.shape[1]).to(device)
    optimizer = AdamW(
        router.parameters(),
        lr=args.router_learning_rate,
        weight_decay=args.router_weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    train_data = TensorDataset(train_x, train["targets"], train["utilities"])
    best_utility = -float("inf")
    stale = 0
    history = []
    for epoch in range(1, args.router_epochs + 1):
        router.train()
        total_loss = 0.0
        for batch_x, batch_y, _ in make_loader(
            train_data, args, True, seed + epoch
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(router(batch_x.to(device)), batch_y.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * batch_x.shape[0]
        router.eval()
        with torch.no_grad():
            predictions = router(validation_x.to(device)).argmax(dim=1).cpu()
            rows = torch.arange(len(predictions))
            selected_utility = validation["utilities"][rows, predictions]
            mean_utility = float(selected_utility.mean())
            target_accuracy = float(
                (predictions == validation["targets"]).float().mean()
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / len(train_data),
                "validation_mean_true_class_margin_utility": mean_utility,
                "validation_best_action_accuracy": target_accuracy,
            }
        )
        print(
            f"seed={seed} router={name} epoch={epoch} "
            f"validation_utility={mean_utility:.6f} accuracy={target_accuracy:.4f}"
        )
        if mean_utility > best_utility:
            best_utility = mean_utility
            stale = 0
            torch.save(
                {
                    "router": router.state_dict(),
                    "mean": mean,
                    "std": std,
                    "input_dim": train_x.shape[1],
                },
                checkpoint,
            )
        else:
            stale += 1
            if stale >= args.router_patience:
                break
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    router = UtilityRouter(payload["input_dim"]).to(device)
    router.load_state_dict(payload["router"])
    record = {"best_validation_utility": best_utility, "history": history}
    record_path.write_text(json.dumps(record, indent=2))
    return router.eval(), payload["mean"], payload["std"], record, checkpoint


def load_family_router(seed, device):
    path = FAMILY_ROUTER_ROOT / f"seed_{seed}" / "linear_router.pt"
    payload = torch.load(path, map_location=device, weights_only=True)
    router = LinearRouter().to(device)
    router.load_state_dict(payload["router"])
    return router.eval(), payload["mean"], payload["std"], path


def routed_logits(logits, actions):
    stacked = torch.stack([logits[name] for name in ACTIONS], dim=1)
    rows = torch.arange(len(actions), device=actions.device)
    return stacked[rows, actions]


def evaluate(
    model,
    experts,
    mixed,
    utility_routers,
    family_router,
    data_loader,
    device,
    args,
    statistical_seed,
):
    method_names = (
        "baseline",
        "mixed",
        "uniform",
        "family_router",
        "early_utility",
        "mechanism_utility",
        "oracle_margin",
    )
    correct = {name: [] for name in method_names}
    margins = {name: [] for name in method_names}
    actions = {name: [] for name in ("family_router", "early_utility", "mechanism_utility", "oracle_margin")}
    with torch.no_grad():
        for pixels, labels in tqdm(data_loader, desc="OOD utility evaluation", leave=False):
            labels = labels.to(device)
            outputs = model(pixel_values=pixels.to(device), output_hidden_states=True)
            hidden, residuals, candidate_logits, candidate_margins = candidate_outputs(
                model, experts, outputs, labels
            )
            patches = hidden[:, 1:]
            mixed_hidden = torch.cat((hidden[:, :1], patches + mixed(patches)), dim=1)
            uniform_hidden = torch.cat(
                (
                    hidden[:, :1],
                    patches + 0.5 * (residuals["noise"] + residuals["blur"]),
                ),
                dim=1,
            )
            method_logits = {
                "baseline": candidate_logits["none"],
                "mixed": downstream_from_layer(model, mixed_hidden, BLOCK - 1),
                "uniform": downstream_from_layer(model, uniform_hidden, BLOCK - 1),
            }
            early = early_statistics(outputs.hidden_states)
            family_model, family_mean, family_std = family_router
            family_noise = torch.sigmoid(
                family_model(
                    (early - family_mean.to(device)) / family_std.to(device)
                )
            ) >= 0.5
            family_actions = torch.where(
                family_noise,
                torch.ones_like(family_noise, dtype=torch.long),
                torch.full_like(family_noise, 2, dtype=torch.long),
            )
            mechanism = torch.cat(
                [
                    expert_mechanism_features(model, hidden, residuals[name])
                    for name in ("noise", "blur")
                ],
                dim=1,
            )
            feature_values = {"early": early, "mechanism": mechanism}
            chosen_actions = {"family_router": family_actions}
            for name, (router, mean, std) in utility_routers.items():
                chosen_actions[f"{name}_utility"] = router(
                    (feature_values[name] - mean.to(device)) / std.to(device)
                ).argmax(dim=1)
            chosen_actions["oracle_margin"] = candidate_margins.argmax(dim=1)
            for name, selected in chosen_actions.items():
                method_logits[name] = routed_logits(candidate_logits, selected)
                actions[name].extend(selected.cpu().tolist())
            for name, logits in method_logits.items():
                correct[name].extend((logits.argmax(dim=1) == labels).cpu().tolist())
                _, margin = classification_margin(logits, labels)
                margins[name].extend(margin.cpu().tolist())
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    margins = {name: np.asarray(values, dtype=np.float32) for name, values in margins.items()}
    result = {
        "methods": {
            name: paired_comparison(
                correct["baseline"], values, statistical_seed + offset, args.bootstrap
            )
            for offset, (name, values) in enumerate(correct.items())
            if name != "baseline"
        },
        "routing": {
            name: {
                action: float(np.mean(np.asarray(values) == action_index))
                for action_index, action in enumerate(ACTIONS)
            }
            for name, values in actions.items()
        },
    }
    return result, correct, margins, actions


def evaluate_and_save(
    model,
    experts,
    mixed,
    utility_routers,
    family_router,
    dataset,
    device,
    args,
    statistical_seed,
    path,
):
    if args.resume and path.exists() and path.with_suffix(".npz").exists():
        return json.loads(path.read_text())
    result, correct, margins, actions = evaluate(
        model,
        experts,
        mixed,
        utility_routers,
        family_router,
        make_loader(dataset, args),
        device,
        args,
        statistical_seed,
    )
    path.write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        path.with_suffix(".npz"),
        **{f"correct_{name}": values for name, values in correct.items()},
        **{f"margin_{name}": values for name, values in margins.items()},
        **{
            f"action_{name}": np.asarray(values, dtype=np.int8)
            for name, values in actions.items()
        },
    )
    return result


def extract_or_load_utility_data(
    model, experts, dataset, device, args, description, path
):
    if args.resume and path.exists():
        return torch.load(path, map_location="cpu", weights_only=True)
    data = extract_utility_data(
        model, experts, make_loader(dataset, args), device, description
    )
    cache = {key: value for key, value in data.items() if key != "conditions"}
    torch.save(cache, path)
    return cache


def aggregate(results, seeds):
    methods = (
        "mixed",
        "uniform",
        "family_router",
        "early_utility",
        "mechanism_utility",
        "oracle_margin",
    )
    output = {}
    for section, section_results in results.items():
        conditions = sorted(section_results[str(seeds[0])])
        output[section] = {
            method: {
                "mean_accuracy": float(
                    np.mean(
                        [
                            section_results[str(seed)][condition]["methods"][method][
                                "candidate_accuracy"
                            ]
                            for seed in seeds
                            for condition in conditions
                        ]
                    )
                ),
                "mean_gain": float(
                    np.mean(
                        [
                            section_results[str(seed)][condition]["methods"][method][
                                "accuracy_difference"
                            ]
                            for seed in seeds
                            for condition in conditions
                        ]
                    )
                ),
            }
            for method in methods
        }
    return output


def main():
    parser = argparse.ArgumentParser(
        description="OOD expert selection by predicted repair utility"
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--rank", type=int, default=212)
    parser.add_argument("--router-epochs", type=int, default=30)
    parser.add_argument("--router-patience", type=int, default=5)
    parser.add_argument("--router-learning-rate", type=float, default=1e-3)
    parser.add_argument("--router-weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--ood-start", type=int, default=14000)
    parser.add_argument("--ood-samples", type=int, default=1000)
    parser.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("The full protocol preserves all three expert seeds")
    if args.ood_start != 14000 or args.ood_start + args.ood_samples > 15000:
        raise ValueError("OOD evaluation is locked to [14000,15000)")
    if any(severity not in range(1, 6) for severity in args.severities):
        raise ValueError("Severities must be in 1..5")
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
    results = {"seen_validation": {}, "ood_unseen": {}}
    training = {}
    checkpoints = {}
    for seed in args.seeds:
        torch.manual_seed(seed + 8200)
        np.random.seed(seed + 8200)
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        experts = load_experts(seed, args.rank, device)
        mixed = load_full_mixed(seed, device)
        family_router = load_family_router(seed, device)
        train_data = extract_or_load_utility_data(
            model,
            experts,
            UtilityDataset(splits[seed]["train"], seed, True, args.max_samples),
            device,
            args,
            f"seed {seed} utility train",
            seed_dir / "utility_train_features.pt",
        )
        validation_data = extract_or_load_utility_data(
            model,
            experts,
            UtilityDataset(
                splits[seed]["validation"], seed, True, args.max_samples
            ),
            device,
            args,
            f"seed {seed} utility validation",
            seed_dir / "utility_validation_features.pt",
        )
        utility_routers = {}
        training[str(seed)] = {}
        checkpoints[str(seed)] = {
            "experts": {
                family: str(
                    ORACLE_RUN / f"seed_{seed}" / f"{family}_rank{args.rank}.pt"
                )
                for family in EXPERT_FAMILIES
            },
            "family_router": str(family_router[3]),
        }
        for name in ("early", "mechanism"):
            router, mean, std, record, path = fit_utility_router(
                name,
                train_data,
                validation_data,
                seed,
                device,
                args,
                seed_dir,
            )
            utility_routers[name] = (router, mean, std)
            training[str(seed)][name] = record
            checkpoints[str(seed)][f"{name}_utility_router"] = str(path)
        results["seen_validation"][str(seed)] = {}
        for condition_index, corruption in enumerate(("clean", *SEEN)):
            severity = 4
            dataset = EvaluationDataset(
                splits[seed]["validation"], corruption, severity, seed, args.max_samples
            )
            result = evaluate_and_save(
                model,
                experts,
                mixed,
                utility_routers,
                family_router[:3],
                dataset,
                device,
                args,
                82000 + seed * 1000 + condition_index * 30,
                seed_dir / f"seen_{corruption}.json",
            )
            results["seen_validation"][str(seed)][corruption] = result
        results["ood_unseen"][str(seed)] = {}
        ood_split = {
            "start": args.ood_start,
            "end": args.ood_start + args.ood_samples,
        }
        for condition_index, corruption in enumerate(UNSEEN):
            for severity in args.severities:
                key = f"{corruption}_{severity}"
                dataset = EvaluationDataset(
                    ood_split, corruption, severity, 2082, args.max_samples
                )
                result = evaluate_and_save(
                    model,
                    experts,
                    mixed,
                    utility_routers,
                    family_router[:3],
                    dataset,
                    device,
                    args,
                    83000 + seed * 1000 + condition_index * 100 + severity * 10,
                    seed_dir / f"ood_{key}.json",
                )
                results["ood_unseen"][str(seed)][key] = result
    utility_router_parameters = {
        "early": 768 * len(ROUTER_BLOCKS) * len(ACTIONS) + len(ACTIONS),
        "mechanism": 36 * len(ACTIONS) + len(ACTIONS),
    }
    summary = {
        "configuration": vars(args)
        | {
            "split_manifest": str(args.split_manifest.resolve()),
            "model": BASE_MODEL,
            "vit_frozen": True,
            "experts_frozen": True,
            "mixed_adapter_frozen": True,
            "seen_router_training_corruptions": ["clean", *SEEN],
            "unseen_evaluation_corruptions": list(UNSEEN),
            "utility_target": "argmax true-class margin improvement among none/noise/blur on development data",
            "mechanism_router_input": "36 corrupted-image-only Block-7 response statistics from the two candidate expert corrections",
            "router_parameters": utility_router_parameters,
            "router_uses_clean_counterpart": False,
            "router_uses_corruption_label_at_inference": False,
            "router_uses_ground_truth_at_inference": False,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "OOD development evaluation; not a pristine final benchmark",
        },
        "splits": {str(seed): splits[seed] for seed in args.seeds},
        "checkpoints": checkpoints,
        "training": training,
        "results": results,
        "aggregate": aggregate(results, args.seeds),
        "leakage_guardrails": [
            "Routers train only on clean and six seen corruption conditions from seed-specific adapter/gate development splits.",
            "JPEG, pixelation, brightness, and contrast are completely absent from router fitting and checkpoint selection.",
            "OOD corruption-instance evaluation is locked to [14000,15000), disjoint from all expert and router fitting ranges.",
            "The source images in [14000,15000) were inspected in prior mechanistic work, so this is OOD corruption generalization development rather than a pristine final benchmark.",
            "ImageNetV2 and ImageNet-Sketch are not accessed.",
        ],
        "limitations": [
            "Online corruptions are controlled approximations rather than official ImageNet-C.",
            "The existing family router was selected on the seen validation ranges, so seen-validation comparisons are descriptive.",
            "The six-corruption parameter-matched monolithic adapter remains a required architecture control.",
            "Oracle-margin routing uses labels and is diagnostic only.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary["aggregate"], indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
