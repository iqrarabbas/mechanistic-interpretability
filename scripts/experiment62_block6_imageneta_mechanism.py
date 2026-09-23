import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment60_block6_imageneta_ood import (
    DEFAULT_IMAGE_ROOT,
    DEFAULT_META,
    ImageNetAOODDataset,
    predictions,
)
from scripts.experiment61_block6_attention_content_patching import (
    BLOCK,
    BLOCK_INDEX,
    attention_parts,
    block6_from_attention_parts,
)


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment62_block6_imageneta_mechanism"
CONDITIONS = [
    "corrupted_baseline",
    "clean_attention_corrupt_values",
    "corrupt_attention_clean_values",
    "clean_attention_clean_values",
    "wrong_pair_clean_attention_values",
    "clean_block6_output",
]


def masked_margin(logits, labels, allowed_ids):
    selected = logits[:, allowed_ids]
    local_labels = (allowed_ids[None, :] == labels[:, None]).long().argmax(1)
    true_logits = selected.gather(1, local_labels[:, None]).squeeze(1)
    competing = selected.clone()
    competing.scatter_(1, local_labels[:, None], float("-inf"))
    return true_logits - competing.max(1).values


def summarize(condition_arrays, clean_correct, bootstrap_repetitions, seed):
    results = {}
    for protocol_index, protocol in enumerate(["masked_200", "full_1k"]):
        results[protocol] = {}
        for corruption_index, corruption in enumerate(["noise4", "blur4"]):
            correct = {
                name: condition_arrays[f"{corruption}_{protocol}_{name}"]
                for name in CONDITIONS
            }
            baseline = correct["corrupted_baseline"]
            failures = clean_correct[protocol] & ~baseline
            condition_results = {
                "baseline_accuracy": float(baseline.mean()),
                "clean_correct_corrupted_wrong_images": int(failures.sum()),
                "conditions": {},
            }
            for condition_index, name in enumerate(CONDITIONS[1:]):
                comparison = paired_comparison(
                    baseline,
                    correct[name],
                    seed + protocol_index * 1000 + corruption_index * 100 + condition_index,
                    bootstrap_repetitions,
                )
                comparison["failure_recovery_count"] = int((correct[name] & failures).sum())
                comparison["failure_recovery_rate"] = (
                    float((correct[name] & failures).sum() / failures.sum())
                    if failures.any()
                    else None
                )
                condition_results["conditions"][name] = comparison
            full_gain = condition_results["conditions"]["clean_block6_output"][
                "accuracy_difference"
            ]
            for name in CONDITIONS[1:4]:
                gain = condition_results["conditions"][name]["accuracy_difference"]
                condition_results["conditions"][name][
                    "fraction_of_full_block6_accuracy_gain"
                ] = float(gain / full_gain) if full_gain else None
            results[protocol][corruption] = condition_results
    return results


def evaluate(model, loader, allowed_ids, device, bootstrap_repetitions, seed):
    arrays = {
        f"{corruption}_{protocol}_{condition}": []
        for corruption in ["noise4", "blur4"]
        for protocol in ["masked_200", "full_1k"]
        for condition in CONDITIONS
    }
    clean_correct = {"masked_200": [], "full_1k": []}
    margin_changes = {
        f"{corruption}_{condition}": []
        for corruption in ["noise4", "blur4"]
        for condition in CONDITIONS[1:]
    }
    layer = model.vit.layers[BLOCK_INDEX]

    with torch.no_grad():
        for native, noise, blur, labels in tqdm(loader, desc="ImageNet-A Block-6 mechanism"):
            batch = native.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([native, noise, blur]).to(device),
                output_hidden_states=True,
            )
            clean_logits, noise_logits, blur_logits = outputs.logits.split(batch)
            clean_input, noise_input, blur_input = outputs.hidden_states[BLOCK_INDEX].split(batch)
            clean_block_output, _, _ = outputs.hidden_states[BLOCK].split(batch)

            clean_attention, clean_values = attention_parts(
                layer.attention, layer.layernorm_before(clean_input)
            )
            clean_full, clean_masked = predictions(clean_logits, allowed_ids)
            clean_correct["full_1k"].extend((clean_full == labels).cpu().tolist())
            clean_correct["masked_200"].extend((clean_masked == labels).cpu().tolist())

            for corruption, corrupted_input, baseline_logits in [
                ("noise4", noise_input, noise_logits),
                ("blur4", blur_input, blur_logits),
            ]:
                corrupted_attention, corrupted_values = attention_parts(
                    layer.attention, layer.layernorm_before(corrupted_input)
                )
                logits = {"corrupted_baseline": baseline_logits}
                patch_specs = {
                    "clean_attention_corrupt_values": (clean_attention, corrupted_values),
                    "corrupt_attention_clean_values": (corrupted_attention, clean_values),
                    "clean_attention_clean_values": (clean_attention, clean_values),
                    "wrong_pair_clean_attention_values": (
                        clean_attention.roll(1, dims=0),
                        clean_values.roll(1, dims=0),
                    ),
                }
                for name, (attention, values) in patch_specs.items():
                    hidden = block6_from_attention_parts(
                        layer, corrupted_input, attention, values
                    )
                    logits[name] = downstream_from_layer(model, hidden, BLOCK_INDEX)
                logits["clean_block6_output"] = downstream_from_layer(
                    model, clean_block_output, BLOCK_INDEX
                )

                baseline_margin = masked_margin(baseline_logits, labels, allowed_ids)
                for name, values in logits.items():
                    full, masked = predictions(values, allowed_ids)
                    arrays[f"{corruption}_full_1k_{name}"].extend(
                        (full == labels).cpu().tolist()
                    )
                    arrays[f"{corruption}_masked_200_{name}"].extend(
                        (masked == labels).cpu().tolist()
                    )
                    if name != "corrupted_baseline":
                        margin_changes[f"{corruption}_{name}"].extend(
                            (masked_margin(values, labels, allowed_ids) - baseline_margin)
                            .cpu()
                            .tolist()
                        )

    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    clean_correct = {
        name: np.asarray(values, dtype=bool) for name, values in clean_correct.items()
    }
    results = summarize(arrays, clean_correct, bootstrap_repetitions, seed)
    for corruption in ["noise4", "blur4"]:
        for name in CONDITIONS[1:]:
            results["masked_200"][corruption]["conditions"][name][
                "mean_margin_change"
            ] = float(np.mean(margin_changes[f"{corruption}_{name}"]))
    return results, arrays, clean_correct


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 62: ImageNet-A replication of Block-6 routing/value causality"
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META)
    parser.add_argument("--samples", type=int, default=7500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--corruption-seed", type=int, default=2062)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    dataset = ImageNetAOODDataset(
        args.image_root, args.meta_path, args.corruption_seed, args.samples
    )
    if len(dataset) != args.samples:
        raise ValueError(f"Requested {args.samples} images, found {len(dataset)}")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ImageNet-A samples: {len(dataset)}")
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)

    results, arrays, clean_correct = evaluate(
        model,
        loader,
        dataset.allowed_model_ids.to(device),
        device,
        args.bootstrap_repetitions,
        args.corruption_seed + 6200,
    )
    np.savez_compressed(
        output_dir / "paired_outcomes.npz",
        **arrays,
        **{f"clean_{name}": values for name, values in clean_correct.items()},
    )
    summary = {
        "configuration": vars(args)
        | {
            "image_root": str(args.image_root.resolve()),
            "meta_path": str(args.meta_path.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "model_frozen": True,
            "dataset": "ImageNet-A",
            "primary_protocol": "masked_200",
            "corruptions": ["Noise-4", "Blur-4"],
            "patch_location": "per-head A/V before concatenation and o_proj",
            "uses_paired_clean_oracle": True,
            "training_or_tuning": False,
            "status": "frozen cross-dataset mechanistic replication",
        },
        "results": results,
        "guardrails": [
            "This is an oracle causal analysis, not a deployable intervention.",
            "ImageNet-A had already been used for frozen adapter evaluation before this analysis.",
            "No method parameter is selected or changed using these mechanistic results.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results["masked_200"], indent=2))
    print(f"Saved Experiment 62 to {output_dir}")


if __name__ == "__main__":
    main()
