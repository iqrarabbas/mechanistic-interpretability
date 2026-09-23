import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment61_block6_attention_content"
DEFAULT_DATASET = PROJECT_ROOT / "Dataset"
BLOCK = 6
BLOCK_INDEX = BLOCK - 1


class CleanCorruptionPairs(Dataset):
    def __init__(self, dataset_dir, samples, start_index, corruption, corruption_seed):
        self.clean = ImageNetDataset(dataset_dir, samples, start_index)
        self.corrupted = ImageNetDataset(
            dataset_dir,
            samples,
            start_index,
            corruption=corruption,
            blur_severity=4,
            noise_severity=4,
            corruption_seed=corruption_seed,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, clean_label = self.clean[index]
        corrupted, corrupted_label = self.corrupted[index]
        if clean_label != corrupted_label:
            raise RuntimeError("Paired labels differ")
        return clean, corrupted, clean_label


def attention_parts(attention, hidden_states):
    normalized = attention_input = hidden_states
    input_shape = normalized.shape[:-1]
    hidden_shape = (*input_shape, -1, attention.head_dim)
    query = attention.q_proj(attention_input).view(hidden_shape).transpose(1, 2)
    key = attention.k_proj(attention_input).view(hidden_shape).transpose(1, 2)
    value = attention.v_proj(attention_input).view(hidden_shape).transpose(1, 2)
    scores = torch.matmul(query, key.transpose(-1, -2)) * attention.scaling
    weights = torch.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    return weights, value


def block6_from_attention_parts(layer, block_input, attention_weights, values):
    attended = torch.matmul(attention_weights, values)
    attended = attended.transpose(1, 2).reshape(block_input.shape).contiguous()
    attended = layer.attention.o_proj(attended)
    after_attention = layer.dropout(attended) + block_input
    mlp_output = layer.mlp(layer.layernorm_after(after_attention))
    return layer.dropout(mlp_output) + after_attention


def true_class_margin(logits, labels):
    true_logits = logits.gather(1, labels[:, None]).squeeze(1)
    masked = logits.clone()
    masked.scatter_(1, labels[:, None], float("-inf"))
    return true_logits - masked.max(1).values


def evaluate(model, loader, device, bootstrap_repetitions, statistical_seed):
    names = [
        "corrupted_baseline",
        "clean_attention_corrupt_values",
        "corrupt_attention_clean_values",
        "clean_attention_clean_values",
        "wrong_pair_clean_attention_values",
        "clean_block6_output",
    ]
    correct = {name: [] for name in names}
    margins = {name: [] for name in names}
    clean_correct = []
    layer = model.vit.layers[BLOCK_INDEX]

    with torch.no_grad():
        for clean, corrupted, labels in tqdm(loader, desc="Block-6 attention/content patching"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, corrupted]).to(device),
                output_hidden_states=True,
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_input, noise_input = outputs.hidden_states[BLOCK_INDEX].split(batch)
            clean_block_output, _ = outputs.hidden_states[BLOCK].split(batch)

            clean_normalized = layer.layernorm_before(clean_input)
            noise_normalized = layer.layernorm_before(noise_input)
            clean_attention, clean_values = attention_parts(layer.attention, clean_normalized)
            noise_attention, noise_values = attention_parts(layer.attention, noise_normalized)

            wrong_attention = clean_attention.roll(1, dims=0)
            wrong_values = clean_values.roll(1, dims=0)
            block_outputs = {
                "clean_attention_corrupt_values": block6_from_attention_parts(
                    layer, noise_input, clean_attention, noise_values
                ),
                "corrupt_attention_clean_values": block6_from_attention_parts(
                    layer, noise_input, noise_attention, clean_values
                ),
                "clean_attention_clean_values": block6_from_attention_parts(
                    layer, noise_input, clean_attention, clean_values
                ),
                "wrong_pair_clean_attention_values": block6_from_attention_parts(
                    layer, noise_input, wrong_attention, wrong_values
                ),
                "clean_block6_output": clean_block_output,
            }
            logits = {"corrupted_baseline": noise_logits}
            logits.update(
                {
                    name: downstream_from_layer(model, hidden, BLOCK_INDEX)
                    for name, hidden in block_outputs.items()
                }
            )
            clean_correct.extend((clean_logits.argmax(1) == labels).cpu().tolist())
            for name, condition_logits in logits.items():
                correct[name].extend((condition_logits.argmax(1) == labels).cpu().tolist())
                margins[name].extend(true_class_margin(condition_logits, labels).cpu().tolist())

    clean_correct = np.asarray(clean_correct, dtype=bool)
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    margins = {name: np.asarray(values, dtype=np.float32) for name, values in margins.items()}
    baseline = correct["corrupted_baseline"]
    failures = clean_correct & ~baseline

    results = {
        "clean_accuracy": float(clean_correct.mean()),
        "corrupted_baseline_accuracy": float(baseline.mean()),
        "clean_correct_corrupted_wrong_images": int(failures.sum()),
        "conditions": {},
    }
    for offset, name in enumerate(names[1:]):
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

    full_gain = results["conditions"]["clean_block6_output"]["accuracy_difference"]
    for name in [
        "clean_attention_corrupt_values",
        "corrupt_attention_clean_values",
        "clean_attention_clean_values",
    ]:
        gain = results["conditions"][name]["accuracy_difference"]
        results["conditions"][name]["fraction_of_full_block6_accuracy_gain"] = (
            float(gain / full_gain) if full_gain != 0 else None
        )
    arrays = {"clean_correct": clean_correct}
    arrays.update({f"correct_{name}": values for name, values in correct.items()})
    arrays.update({f"margin_{name}": values for name, values in margins.items()})
    return results, arrays


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 61: causal routing-versus-value decomposition inside ViT Block 6"
    )
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--samples", type=int, default=3000)
    parser.add_argument("--start-index", type=int, default=47000)
    parser.add_argument("--corruption-seed", type=int, default=2061)
    parser.add_argument("--corruption", choices=["noise", "blur"], default="noise")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    if args.start_index < 47000 or args.start_index + args.samples > 50000:
        raise ValueError("Experiment 61 is locked to the previously unused range [47000,50000)")
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
    if len(dataset) != args.samples:
        raise ValueError(f"Requested {args.samples} images, found {len(dataset)}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
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
        device,
        args.bootstrap_repetitions,
        args.corruption_seed + 6100,
    )
    np.savez_compressed(output_dir / "paired_outcomes.npz", **arrays)
    summary = {
        "configuration": vars(args)
        | {
            "dataset_dir": str(args.dataset_dir.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "model_frozen": True,
            "corruption": f"{args.corruption.capitalize()}-4",
            "block": BLOCK,
            "patch_location": "per-head attention probabilities and value vectors before concatenation/o_proj",
            "evaluation_split_status": "previously unused mechanistic analysis split",
            "deployment_method": False,
            "uses_paired_clean_oracle": True,
            "imageNetV2_accessed": False,
            "imageNetA_accessed": False,
        },
        "results": results,
        "interpretation_guardrails": [
            "Clean activation patching is an oracle causal diagnostic, not an inference-time method.",
            "Attention probabilities isolate routing (Q/K-derived A); values isolate transported content (V).",
            "The full clean Block-6 output is an upper bound that also restores the residual stream and MLP output.",
            "The wrong-pair control tests whether arbitrary clean attention content causes recovery.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved Experiment 61 to {output_dir}")


if __name__ == "__main__":
    main()
