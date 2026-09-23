import argparse
import csv
import json
from pathlib import Path

import torch
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment10_corruption_agnostic_sae_repair import load_fixed_sae
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment47_resource_audit"
DEFAULT_ADAPTER = (
    PROJECT_ROOT / "results" / "sae" / "experiment39_disjoint_adapter_training"
    / "full_disjoint_zero_init_3seed_v1" / "seed_0" / "classification_weight_0.05.pt"
)
PATCHES = 196
TOKENS = 197
WIDTH = 768
MLP_WIDTH = 3072
BLOCKS = 12


def file_sizes(path):
    size = path.stat().st_size if path is not None and path.exists() else None
    return {
        "checkpoint_path": str(path.resolve()) if path is not None else None,
        "checkpoint_bytes": size,
        "checkpoint_mb_decimal": size / 1e6 if size is not None else None,
        "checkpoint_mib": size / 2**20 if size is not None else None,
    }


def vit_macs():
    patch_embedding = PATCHES * WIDTH * (3 * 16 * 16)
    attention_projection = TOKENS * 4 * WIDTH * WIDTH
    attention_scores_and_values = 2 * TOKENS * TOKENS * WIDTH
    mlp = TOKENS * 2 * WIDTH * MLP_WIDTH
    classifier = WIDTH * 1000
    total = patch_embedding + BLOCKS * (
        attention_projection + attention_scores_and_values + mlp
    ) + classifier
    return {
        "patch_embedding": patch_embedding,
        "transformer_blocks": BLOCKS * (
            attention_projection + attention_scores_and_values + mlp
        ),
        "classifier": classifier,
        "total": total,
    }


def method_row(name, trainable, frozen_auxiliary, added_macs, inference_sae, checkpoint):
    return {
        "method": name,
        "trainable_parameters": trainable,
        "frozen_auxiliary_parameters": frozen_auxiliary,
        "added_macs_per_image": added_macs,
        "added_flops_per_image_two_per_mac": 2 * added_macs,
        "inference_requires_sae": inference_sae,
    } | file_sizes(checkpoint)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows, vit_parameters, backbone_macs):
    lines = [
        "# Frozen ViT Robustness Resource Audit",
        "",
        f"Frozen backbone: `{BASE_MODEL}` with {vit_parameters:,} parameters.",
        f"Analytical backbone compute: {backbone_macs / 1e9:.3f} GMAC/image "
        f"({2 * backbone_macs / 1e9:.3f} GFLOP using two FLOPs per MAC).",
        "",
        "| Method | Trainable params | Added params vs ViT | Added GMAC/image | Checkpoint MiB | SAE at inference |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in rows:
        checkpoint = row["checkpoint_mib"]
        checkpoint_text = f"{checkpoint:.3f}" if checkpoint is not None else "—"
        lines.append(
            f"| {row['method']} | {row['trainable_parameters']:,} | "
            f"{100 * row['trainable_parameters'] / vit_parameters:.4f}% | "
            f"{row['added_macs_per_image'] / 1e9:.4f} | "
            f"{checkpoint_text} | {'Yes' if row['inference_requires_sae'] else 'No'} |"
        )
    lines += [
        "",
        "## Conventions and limitations",
        "",
        "- A MAC is one multiply-accumulate. The FLOP column in JSON/CSV uses two FLOPs per MAC.",
        "- Analytical compute counts dominant dense matrix multiplications and attention products; normalization, activations, additions, and data preprocessing are omitted.",
        "- Adapter compute includes the 768-by-768 patch projection for 196 patch tokens; positional additions are omitted as non-MAC operations.",
        "- The 16-coordinate methods include raw-input projection and fixed-basis decoding.",
        "- SAE-gate compute includes a full 768-to-24,576 encoder because the existing implementation materializes the full SAE latent vector.",
        "- Exact CUDA latency is deliberately pending until Experiment 46 releases the GPU.",
        "- No test-time parameter updates are performed by these frozen correction methods.",
    ]
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="CPU-only parameter, checkpoint, and analytical compute audit")
    parser.add_argument("--adapter-checkpoint", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)

    model = ViTForImageClassification.from_pretrained(BASE_MODEL, local_files_only=True)
    vit_parameters = sum(parameter.numel() for parameter in model.parameters())
    adapter_parameters = sum(parameter.numel() for parameter in HiddenLinear().parameters())
    sae = load_fixed_sae("clean", torch.device("cpu"))
    sae_parameters = sum(parameter.numel() for parameter in sae.parameters())
    sae_checkpoint = PROJECT_ROOT / "checkpoints" / "sae" / "clean_vanilla" / "model.pt"
    if not sae_checkpoint.exists():
        candidates = sorted((PROJECT_ROOT / "checkpoints" / "sae").glob("clean*/model.pt"))
        sae_checkpoint = candidates[0] if candidates else PROJECT_ROOT / "checkpoints" / "sae" / "blur4_base_vanilla_paper" / "model.pt"

    full_adapter_macs = PATCHES * WIDTH * WIDTH
    rank16_input_output_macs = PATCHES * (WIDTH * 16 + 16 * WIDTH)
    sae_encoder_macs = PATCHES * WIDTH * sae.latent_dim
    gate_macs = sae_encoder_macs + PATCHES * 16
    rows = [
        method_row("Frozen ViT baseline", 0, 0, 0, False, None),
        method_row("Full 768D residual adapter", adapter_parameters, 0, full_adapter_macs, False, args.adapter_checkpoint),
        method_row(
            "16D raw-hidden repair", 3408, 0, rank16_input_output_macs, False,
            PROJECT_ROOT / "results" / "sae" / "experiment43_direct_latent_repair"
            / "full_3seed_matched_direct_repair_optimized_v2" / "seed_0" / "raw_hidden.pt",
        ),
        method_row(
            "16D shared-input subspace repair", 15440, 0, rank16_input_output_macs, False,
            PROJECT_ROOT / "results" / "sae" / "experiment45_sae_hidden_subspace"
            / "full_3seed_rank16_normalized_v1" / "seed_0" / "harmful_sae_decoder.pt",
        ),
        method_row("17-parameter SAE gate", 17, sae_parameters, gate_macs, True, sae_checkpoint),
    ]
    backbone = vit_macs()
    summary = {
        "configuration": {
            "backbone": BASE_MODEL,
            "input_resolution": 224,
            "patches": PATCHES,
            "tokens": TOKENS,
            "mac_definition": "one multiply-accumulate",
            "flop_conversion": "two FLOPs per MAC",
            "latency_status": "pending until Experiment 46 releases CUDA",
            "inference_rerun": False,
            "device": "cpu",
        },
        "backbone": {
            "parameters": vit_parameters,
            "analytical_macs": backbone,
            "analytical_flops_two_per_mac": 2 * backbone["total"],
        },
        "methods": rows,
        "limitations": [
            "Dominant dense MACs only; elementwise operations and preprocessing are excluded.",
            "Checkpoint size depends on serialization dtype and container overhead.",
            "Latency must be measured empirically on the same GPU, batch size, and warmup protocol.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    write_csv(output_dir / "resource_table.csv", rows)
    write_markdown(
        output_dir / "resource_table.md", rows, vit_parameters, backbone["total"]
    )
    print(json.dumps({
        "vit_parameters": vit_parameters,
        "backbone_gmac": backbone["total"] / 1e9,
        "methods": rows,
    }, indent=2))
    print(f"Saved resource audit to {output_dir}")


if __name__ == "__main__":
    main()
