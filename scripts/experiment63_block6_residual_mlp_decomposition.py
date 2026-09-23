import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment61_block6_attention_content_patching import (
    CleanCorruptionPairs,
    attention_parts,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment63_block6_residual_mlp"
DEFAULT_DATASET = PROJECT_ROOT / "Dataset"
COMPONENTS = ("x", "a", "m")


def coalition_name(clean_components):
    if not clean_components:
        return "corrupted_baseline"
    return "factorial_clean_" + "".join(component for component in COMPONENTS if component in clean_components)


def attention_output(layer, normalized, block_shape):
    weights, values = attention_parts(layer.attention, normalized)
    attended = torch.matmul(weights, values)
    attended = attended.transpose(1, 2).reshape(block_shape).contiguous()
    return layer.dropout(layer.attention.o_proj(attended))


def mlp_output(layer, attention_residual):
    return layer.dropout(layer.mlp(layer.layernorm_after(attention_residual)))


def true_class_margin(logits, labels):
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    competing = logits.clone()
    competing.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - competing.max(1).values


def shapley_values(coalition_accuracies):
    values = {}
    total = len(COMPONENTS)
    for component in COMPONENTS:
        contribution = 0.0
        others = [item for item in COMPONENTS if item != component]
        for size in range(len(others) + 1):
            weight = math.factorial(size) * math.factorial(total - size - 1) / math.factorial(total)
            for subset in itertools.combinations(others, size):
                subset = frozenset(subset)
                with_component = subset | {component}
                contribution += weight * (
                    coalition_accuracies[with_component] - coalition_accuracies[subset]
                )
        values[component] = float(contribution)
    return values


def evaluate(model, loader, block, device, bootstrap_repetitions, statistical_seed):
    block_index = block - 1
    coalitions = [frozenset(items) for size in range(4) for items in itertools.combinations(COMPONENTS, size)]
    names = [coalition_name(coalition) for coalition in coalitions]
    extra_names = ["sequential_clean_a_recompute_m", "wrong_pair_full_block"]
    all_names = names + extra_names
    correct = {name: [] for name in all_names}
    margins = {name: [] for name in all_names}
    clean_correct = []
    reconstruction_max_error = 0.0
    layer = model.vit.layers[block_index]

    with torch.no_grad():
        for clean, corrupted, labels in tqdm(loader, desc="Block-6 residual/MLP decomposition"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, corrupted]).to(device),
                output_hidden_states=True,
            )
            clean_logits, corrupted_logits = outputs.logits.split(batch)
            clean_x, corrupted_x = outputs.hidden_states[block_index].split(batch)
            clean_z, corrupted_z = outputs.hidden_states[block].split(batch)

            clean_a = attention_output(layer, layer.layernorm_before(clean_x), clean_x.shape)
            corrupted_a = attention_output(
                layer, layer.layernorm_before(corrupted_x), corrupted_x.shape
            )
            clean_m = mlp_output(layer, clean_x + clean_a)
            corrupted_m = mlp_output(layer, corrupted_x + corrupted_a)
            reconstruction_max_error = max(
                reconstruction_max_error,
                float((corrupted_x + corrupted_a + corrupted_m - corrupted_z).abs().max()),
                float((clean_x + clean_a + clean_m - clean_z).abs().max()),
            )

            logits = {"corrupted_baseline": corrupted_logits}
            for coalition in coalitions[1:]:
                x = clean_x if "x" in coalition else corrupted_x
                a = clean_a if "a" in coalition else corrupted_a
                m = clean_m if "m" in coalition else corrupted_m
                logits[coalition_name(coalition)] = downstream_from_layer(
                    model, x + a + m, block_index
                )

            sequential_y = corrupted_x + clean_a
            sequential_z = sequential_y + mlp_output(layer, sequential_y)
            logits["sequential_clean_a_recompute_m"] = downstream_from_layer(
                model, sequential_z, block_index
            )
            logits["wrong_pair_full_block"] = downstream_from_layer(
                model, clean_z.roll(1, dims=0), block_index
            )

            clean_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            for name, condition_logits in logits.items():
                correct[name].extend((condition_logits.argmax(1) == labels).cpu().tolist())
                margins[name].extend(true_class_margin(condition_logits, labels).cpu().tolist())

    if reconstruction_max_error > 1e-3:
        raise RuntimeError(f"Block-6 additive reconstruction error is {reconstruction_max_error}")
    clean_correct = np.asarray(clean_correct, dtype=bool)
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    margins = {name: np.asarray(values, dtype=np.float32) for name, values in margins.items()}
    baseline = correct["corrupted_baseline"]
    failures = clean_correct & ~baseline

    results = {
        "clean_accuracy": float(clean_correct.mean()),
        "corrupted_baseline_accuracy": float(baseline.mean()),
        "clean_correct_corrupted_wrong_images": int(failures.sum()),
        "additive_reconstruction_max_absolute_error": reconstruction_max_error,
        "conditions": {},
    }
    for offset, name in enumerate(all_names[1:]):
        comparison = paired_comparison(
            baseline, correct[name], statistical_seed + offset, bootstrap_repetitions
        )
        comparison.update(
            {
                "mean_margin_change": float((margins[name] - margins["corrupted_baseline"]).mean()),
                "failure_recovery_count": int((correct[name] & failures).sum()),
                "failure_recovery_rate": (
                    float((correct[name] & failures).sum() / failures.sum())
                    if failures.any()
                    else None
                ),
            }
        )
        results["conditions"][name] = comparison

    coalition_accuracies = {
        coalition: float(correct[coalition_name(coalition)].mean()) for coalition in coalitions
    }
    results["factorial_shapley_accuracy_contributions"] = shapley_values(coalition_accuracies)
    results["factorial_coalition_accuracies"] = {
        coalition_name(coalition): accuracy for coalition, accuracy in coalition_accuracies.items()
    }
    arrays = {"clean_correct": clean_correct}
    arrays.update({f"correct_{name}": values for name, values in correct.items()})
    arrays.update({f"margin_{name}": values for name, values in margins.items()})
    return results, arrays


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 63: factorial Block-6 residual, attention, and MLP restoration"
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--corruption", choices=["noise", "blur"], default="noise")
    parser.add_argument("--block", type=int, choices=range(1, 7), default=6)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--start-index", type=int, default=47000)
    parser.add_argument("--corruption-seed", type=int, default=2063)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if args.start_index < 47000 or args.start_index + args.samples > 50000:
        raise ValueError("Experiment 63 is locked to analysis range [47000,50000)")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    dataset = CleanCorruptionPairs(
        args.dataset_dir,
        args.samples,
        args.start_index,
        args.corruption,
        args.corruption_seed,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)

    results, arrays = evaluate(
        model,
        loader,
        args.block,
        device,
        args.bootstrap_repetitions,
        args.corruption_seed + 6300 + args.block * 10,
    )
    np.savez_compressed(output_dir / "paired_outcomes.npz", **arrays)
    summary = {
        "configuration": vars(args)
        | {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "model_frozen": True,
            "block": args.block,
            "components": {
            "x": f"Block-{args.block} incoming residual stream",
            "a": f"Block-{args.block} attention output after o_proj/dropout",
            "m": f"Block-{args.block} MLP output after dropout",
            },
            "uses_paired_clean_oracle": True,
            "evaluation_split_status": "reused mechanistic analysis split",
            "imageNetV2_accessed": False,
            "imageNetA_accessed": False,
            "training_or_tuning": False,
        },
        "results": results,
        "guardrails": [
            "Factorial component swaps are additive path patches and can create hybrid off-manifold states.",
            "Shapley values average each component contribution over all factorial contexts.",
            "Sequential clean-attention patching recomputes the MLP to test a more on-path intervention.",
            "This is an oracle mechanistic analysis, not a deployment method.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved Experiment 63 to {output_dir}")


if __name__ == "__main__":
    main()
