import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment54_lowrank_block6_adapter import LowRankHiddenAdapter


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment59_parameter_matched_block6"
BLOCK = 6
WIDTH = 768
PATCHES = 196
IDENTITY_WEIGHT = 0.2


class BottleneckAdapter(nn.Module):
    def __init__(self, bottleneck, activation):
        super().__init__()
        self.down = nn.Linear(WIDTH, bottleneck)
        self.up = nn.Linear(bottleneck, WIDTH)
        self.activation = activation
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, patches):
        hidden = self.down(patches)
        if self.activation == "gelu":
            hidden = F.gelu(hidden)
        return self.up(hidden)


class RandomProjectionAdapter(nn.Module):
    def __init__(self, rank, seed):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        random_matrix = torch.randn(WIDTH, rank, generator=generator)
        basis, _ = torch.linalg.qr(random_matrix, mode="reduced")
        self.register_buffer("basis", basis)
        self.position = nn.Embedding(PATCHES, rank)
        self.up = nn.Linear(rank, WIDTH)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, patches):
        positions = torch.arange(PATCHES, device=patches.device)
        projected = patches @ self.basis
        return self.up(projected + self.position(positions)[None])


class LoRALinear(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.a = nn.Linear(base.in_features, rank, bias=False)
        self.b = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        self.scale = alpha / rank
        self.enabled = False

    def forward(self, inputs):
        output = self.base(inputs)
        if self.enabled:
            output = output + self.scale * self.b(self.a(inputs))
        return output


class Block6AttentionLoRA(nn.Module):
    def __init__(self, model, rank, alpha):
        super().__init__()
        attention = model.vit.layers[BLOCK - 1].attention
        attention.q_proj = LoRALinear(attention.q_proj, rank, alpha)
        attention.v_proj = LoRALinear(attention.v_proj, rank, alpha)
        self.q_proj = attention.q_proj
        self.v_proj = attention.v_proj

    def enable(self, enabled):
        self.q_proj.enabled = enabled
        self.v_proj.enabled = enabled

    def trainable_parameters(self):
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def trainable_count(module):
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


def make_loader(corruption, split, seed, batch_size, workers, shuffle, max_samples):
    samples = split["end"] - split["start"]
    if max_samples is not None:
        samples = min(samples, max_samples)
    generator = torch.Generator().manual_seed(seed + 5900)
    return DataLoader(
        PairedCorruptionDataset(corruption, samples, split["start"], seed),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator if shuffle else None,
    )


def loss_terms(model, clean_base6, corrupt_base6, corrected_clean6, corrected_corrupt6, labels, args):
    clean_patches = clean_base6[:, 1:]
    corrupt_patches = corrupt_base6[:, 1:]
    corrected_clean_patches = corrected_clean6[:, 1:]
    corrected_corrupt_patches = corrected_corrupt6[:, 1:]
    predicted_residual = corrected_corrupt_patches - corrupt_patches
    target = clean_patches - corrupt_patches
    residual_loss = F.smooth_l1_loss(
        predicted_residual, target, beta=args.smooth_l1_beta
    )
    identity_loss = F.smooth_l1_loss(
        corrected_clean_patches, clean_patches, beta=args.smooth_l1_beta
    )
    logits = downstream_from_layer(model, corrected_corrupt6, BLOCK - 1)
    with torch.no_grad():
        baseline_logits = downstream_from_layer(model, corrupt_base6, BLOCK - 1)
        baseline_correct = baseline_logits.argmax(1) == labels
        baseline_margin = classification_margin(baseline_logits, labels)[1]
    margin = classification_margin(logits, labels)[1]
    classification_loss = F.cross_entropy(logits, labels)
    preservation_loss = (
        F.relu(baseline_margin[baseline_correct] - margin[baseline_correct]).mean()
        if baseline_correct.any() else margin.new_zeros(())
    )
    total = (
        residual_loss
        + args.classification_weight * classification_loss
        + args.preservation_weight * preservation_loss
        + IDENTITY_WEIGHT * identity_loss
    )
    return total, residual_loss, classification_loss, preservation_loss, identity_loss


def post_adapter_states(adapter, clean_base6, corrupt_base6, scale):
    clean_patches = clean_base6[:, 1:]
    corrupt_patches = corrupt_base6[:, 1:]
    corrected_clean = torch.cat([
        clean_base6[:, :1], clean_patches + scale * adapter(clean_patches)
    ], dim=1)
    corrected_corrupt = torch.cat([
        corrupt_base6[:, :1], corrupt_patches + scale * adapter(corrupt_patches)
    ], dim=1)
    return corrected_clean, corrected_corrupt


def lora_states(model, lora, clean_input6, corrupt_input6):
    lora.enable(True)
    corrected_clean = model.vit.layers[BLOCK - 1](clean_input6, attention_mask=None)
    corrected_corrupt = model.vit.layers[BLOCK - 1](corrupt_input6, attention_mask=None)
    lora.enable(False)
    return corrected_clean, corrected_corrupt


def base_forward(model, lora, clean, corrupt):
    lora.enable(False)
    with torch.no_grad():
        outputs = model(
            pixel_values=torch.cat([clean, corrupt]), output_hidden_states=True
        )
    batch = clean.shape[0]
    clean_input6, corrupt_input6 = outputs.hidden_states[BLOCK - 1].split(batch)
    clean_base6, corrupt_base6 = outputs.hidden_states[BLOCK].split(batch)
    clean_logits, corrupt_logits = outputs.logits.split(batch)
    return clean_input6, corrupt_input6, clean_base6, corrupt_base6, clean_logits, corrupt_logits


def run_epoch(model, lora, post_adapters, loader, device, args, optimizers=None):
    training = optimizers is not None
    for adapter in post_adapters.values():
        adapter.train(training)
    lora.train(training)
    totals = {name: np.zeros(5, dtype=np.float64) for name in [*post_adapters, "lora_qv"]}
    samples = 0
    description = "Matched baseline training" if training else "Matched baseline validation"
    for clean, noise, labels in tqdm(loader, desc=description, leave=False):
        clean = clean.to(device)
        noise = noise.to(device)
        labels = labels.to(device)
        states = base_forward(model, lora, clean, noise)
        clean_input6, noise_input6, clean_base6, noise_base6 = states[:4]
        if training:
            for optimizer in optimizers.values():
                optimizer.zero_grad(set_to_none=True)
        losses = {}
        for name, adapter in post_adapters.items():
            corrected = post_adapter_states(
                adapter, clean_base6, noise_base6, args.train_alpha
            )
            losses[name] = loss_terms(
                model, clean_base6, noise_base6, *corrected, labels, args
            )
        lora_corrected = lora_states(model, lora, clean_input6, noise_input6)
        losses["lora_qv"] = loss_terms(
            model, clean_base6, noise_base6, *lora_corrected, labels, args
        )
        if training:
            sum(values[0] for values in losses.values()).backward()
            for name, optimizer in optimizers.items():
                module = lora if name == "lora_qv" else post_adapters[name]
                torch.nn.utils.clip_grad_norm_(
                    [p for p in module.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
        batch = labels.shape[0]
        for name, values in losses.items():
            totals[name] += np.asarray([float(value.detach()) for value in values]) * batch
        samples += batch
    fields = ["total", "residual", "classification", "preservation", "clean_identity"]
    return {
        name: dict(zip(fields, (values / samples).tolist()))
        for name, values in totals.items()
    }


def save_modules(seed_dir, post_adapters, lora):
    for name, adapter in post_adapters.items():
        torch.save(adapter.state_dict(), seed_dir / f"{name}.pt")
    torch.save(
        {name: value for name, value in lora.state_dict().items() if "base." not in name},
        seed_dir / "lora_qv.pt",
    )


def load_modules(seed_dir, post_adapters, lora, device):
    for name, adapter in post_adapters.items():
        adapter.load_state_dict(torch.load(
            seed_dir / f"{name}.pt", map_location=device, weights_only=True
        ))
        adapter.eval()
    state = torch.load(seed_dir / "lora_qv.pt", map_location=device, weights_only=True)
    lora.load_state_dict(state, strict=False)
    lora.eval()


def train(model, lora, post_adapters, train_loader, validation_loader, device, args, seed_dir):
    modules = post_adapters | {"lora_qv": lora}
    optimizers = {
        name: AdamW(
            [p for p in module.parameters() if p.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        for name, module in modules.items()
    }
    schedulers = {
        name: CosineAnnealingLR(optimizer, T_max=args.epochs)
        for name, optimizer in optimizers.items()
    }
    best = {name: float("inf") for name in modules}
    stale = {name: 0 for name in modules}
    active = set(modules)
    history = []
    for epoch in range(1, args.epochs + 1):
        active_posts = {name: post_adapters[name] for name in active if name != "lora_qv"}
        active_optimizers = {name: optimizers[name] for name in active}
        train_metrics = run_epoch(
            model, lora, active_posts, train_loader, device, args, active_optimizers
        )
        with torch.no_grad():
            validation_metrics = run_epoch(
                model, lora, active_posts, validation_loader, device, args
            )
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        for name in list(active):
            schedulers[name].step()
            value = validation_metrics[name]["total"]
            print(f"seed={seed_dir.name} method={name} epoch={epoch} validation={value:.6f}")
            if value < best[name]:
                best[name] = value
                stale[name] = 0
                if name == "lora_qv":
                    torch.save(
                        {key: val for key, val in lora.state_dict().items() if "base." not in key},
                        seed_dir / "lora_qv.pt",
                    )
                else:
                    torch.save(post_adapters[name].state_dict(), seed_dir / f"{name}.pt")
            else:
                stale[name] += 1
                if stale[name] >= args.patience:
                    active.remove(name)
        if not active:
            break
    load_modules(seed_dir, post_adapters, lora, device)
    return {"best_validation_total": best, "history": history}


def evaluate(model, lora, post_adapters, loader, condition, device, args, seed):
    methods = [*post_adapters, "lora_qv"]
    baseline, corrected = [], {name: [] for name in methods}
    with torch.no_grad():
        for clean, corrupt, labels in tqdm(loader, desc=f"Evaluate {condition}"):
            clean = clean.to(device)
            corrupt = corrupt.to(device)
            labels = labels.to(device)
            states = base_forward(model, lora, clean, corrupt)
            clean_input6, corrupt_input6, clean_base6, corrupt_base6, clean_logits, corrupt_logits = states
            if condition == "clean":
                input6, base6, logits = clean_input6, clean_base6, clean_logits
            else:
                input6, base6, logits = corrupt_input6, corrupt_base6, corrupt_logits
            baseline.extend((logits.argmax(1) == labels).cpu().tolist())
            for name, adapter in post_adapters.items():
                patches = base6[:, 1:]
                candidate = torch.cat([
                    base6[:, :1], patches + args.alpha * adapter(patches)
                ], dim=1)
                output = downstream_from_layer(model, candidate, BLOCK - 1)
                corrected[name].extend((output.argmax(1) == labels).cpu().tolist())
            lora.enable(True)
            candidate = model.vit.layers[BLOCK - 1](input6, attention_mask=None)
            lora.enable(False)
            output = downstream_from_layer(model, candidate, BLOCK - 1)
            corrected["lora_qv"].extend((output.argmax(1) == labels).cpu().tolist())
    baseline = np.asarray(baseline, dtype=bool)
    corrected = {name: np.asarray(values, dtype=bool) for name, values in corrected.items()}
    result = {
        name: paired_comparison(
            baseline, values, seed + index, args.bootstrap_repetitions
        )
        for index, (name, values) in enumerate(corrected.items())
    }
    arrays = {"baseline": baseline} | corrected
    return result, arrays


def build_modules(model, seed, args, device):
    post_adapters = {
        "rank8_position": LowRankHiddenAdapter(8).to(device),
        "linear_rank9": BottleneckAdapter(9, "identity").to(device),
        "mlp_rank9": BottleneckAdapter(9, "gelu").to(device),
        "random_projection_rank14": RandomProjectionAdapter(14, seed + 59000).to(device),
    }
    lora = Block6AttentionLoRA(model, args.lora_rank, args.lora_alpha).to(device)
    return post_adapters, lora


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 59: parameter-matched Block-6 architecture controls"
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--classification-weight", type=float, default=0.05)
    parser.add_argument("--preservation-weight", type=float, default=0.05)
    parser.add_argument("--train-alpha", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=1.0)
    parser.add_argument("--lora-rank", type=int, default=5)
    parser.add_argument("--lora-alpha", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; use --resume")
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    training, validation, outcomes, parameter_counts = {}, {}, {}, None

    for seed in args.seeds:
        torch.manual_seed(seed + 5900)
        np.random.seed(seed + 5900)
        seed_dir = output_dir / f"seed_{seed}"
        record_path = seed_dir / "completed_record.json"
        if args.resume and record_path.exists():
            record = json.loads(record_path.read_text())
            training[f"seed_{seed}"] = record["training"]
            validation[f"seed_{seed}"] = record["validation"]
            counts = record["parameter_counts"]
            if parameter_counts is None:
                parameter_counts = counts
            elif counts != parameter_counts:
                raise RuntimeError("Resumed parameter counts differ across seeds")
            print(f"Skipping completed seed {seed}")
            continue
        seed_dir.mkdir(exist_ok=args.resume)
        model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
        model.requires_grad_(False)
        post_adapters, lora = build_modules(model, seed, args, device)
        counts = {name: trainable_count(module) for name, module in post_adapters.items()}
        counts["lora_qv"] = trainable_count(lora)
        if parameter_counts is None:
            parameter_counts = counts
            print(f"Trainable parameter counts: {counts}")
        elif counts != parameter_counts:
            raise RuntimeError("Parameter counts changed across seeds")
        train_loader = make_loader(
            "noise", splits[seed]["train"], seed, args.batch_size,
            args.num_workers, True, args.max_samples,
        )
        validation_loader = make_loader(
            "noise", splits[seed]["validation"], seed, args.batch_size,
            args.num_workers, False, args.max_samples,
        )
        training[f"seed_{seed}"] = train(
            model, lora, post_adapters, train_loader, validation_loader,
            device, args, seed_dir,
        )
        condition_results = {}
        condition_arrays = {}
        for condition, corruption in (("clean", "noise"), ("noise4", "noise"), ("blur4", "blur")):
            loader = make_loader(
                corruption, splits[seed]["validation"], seed, args.batch_size,
                args.num_workers, False, args.max_samples,
            )
            result, arrays = evaluate(
                model, lora, post_adapters, loader, condition, device, args,
                59000 + seed * 100 + len(condition_results) * 10,
            )
            condition_results[condition] = result
            for name, values in arrays.items():
                condition_arrays[f"{condition}_{name}"] = values
        validation[f"seed_{seed}"] = condition_results
        np.savez_compressed(seed_dir / "paired_outcomes.npz", **condition_arrays)
        (seed_dir / "summary.json").write_text(json.dumps(condition_results, indent=2))
        record_path.write_text(json.dumps({
            "parameter_counts": parameter_counts,
            "training": training[f"seed_{seed}"],
            "validation": validation[f"seed_{seed}"],
        }, indent=2))

    aggregate = {}
    for method in parameter_counts:
        aggregate[method] = {}
        for condition in ("clean", "noise4", "blur4"):
            changes = [
                validation[f"seed_{seed}"][condition][method]["accuracy_difference"]
                for seed in args.seeds
            ]
            aggregate[method][condition] = {
                "changes_by_seed": changes,
                "mean_change": float(np.mean(changes)),
            }
        aggregate[method]["trainable_parameters"] = parameter_counts[method]
        aggregate[method]["parameter_ratio_to_rank8"] = (
            parameter_counts[method] / parameter_counts["rank8_position"]
        )

    summary = {
        "configuration": vars(args) | {
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "block": BLOCK,
            "identity_weight": IDENTITY_WEIGHT,
            "frozen_components": ["ViT backbone"],
            "imageNetV2_accessed": False,
            "status": "leakage-free development architecture control",
        },
        "method_definitions": {
            "rank8_position": "Current post-Block-6 low-rank residual with learned rank-8 patch positions.",
            "linear_rank9": "Post-Block-6 linear bottleneck residual without activation or position embedding.",
            "mlp_rank9": "Post-Block-6 GELU bottleneck residual without position embedding.",
            "random_projection_rank14": "Fixed random orthogonal down-projection with learned low-rank positions and up-projection.",
            "lora_qv": "Rank-5 LoRA updates inside Block-6 attention query and value projections.",
        },
        "parameter_counts": parameter_counts,
        "development_splits": {str(seed): splits[seed] for seed in args.seeds},
        "training": training,
        "validation": validation,
        "aggregate": aggregate,
        "interpretation_rule": (
            "The mechanistically localized rank-8 architecture has architecture-specific support "
            "only if it consistently exceeds parameter-matched controls while preserving clean accuracy."
        ),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"parameter_counts": parameter_counts, "aggregate": aggregate}, indent=2))
    print(f"Saved Experiment 59 to {output_dir}")


if __name__ == "__main__":
    main()
