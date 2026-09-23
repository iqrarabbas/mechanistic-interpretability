import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment52_block6_repair_propagation import make_loader
from scripts.experiment61_block6_attention_content_patching import attention_parts


ROOT = Path(__file__).parent.parent
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment71_downstream_circuit_recovery"
ADAPTER_ROOT = (
    ACTIVE_ROOT
    / "results/sae/experiment67_mixed_noise_blur_block6/"
    "full_3seed_noise_blur_identity0p2_v1"
)
BLOCK6 = 6
DOWNSTREAM_BLOCKS = tuple(range(7, 13))
EPSILON = 1e-12


def load_adapter(seed, device):
    path = ADAPTER_ROOT / f"seed_{seed}" / "mixed_identity0p2.pt"
    adapter = HiddenLinear().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval(), path


def attention_computation(layer, hidden):
    normalized = layer.layernorm_before(hidden)
    weights, values = attention_parts(layer.attention, normalized)
    head_content = torch.matmul(weights, values)
    joined = head_content.transpose(1, 2).reshape(hidden.shape).contiguous()
    attention_output = layer.dropout(layer.attention.o_proj(joined))
    after_attention = hidden + attention_output
    normalized_mlp = layer.layernorm_after(after_attention)
    mlp_activation = layer.mlp.activation_fn(layer.mlp.fc1(normalized_mlp))
    mlp_output = layer.dropout(layer.mlp.fc2(mlp_activation))
    return weights, head_content, mlp_activation, after_attention + mlp_output


def empty_accumulators():
    return {
        block: {
            "routing_noise_sse": torch.zeros(12, dtype=torch.float64),
            "routing_corrected_sse": torch.zeros(12, dtype=torch.float64),
            "cls_routing_noise_sse": torch.zeros(12, dtype=torch.float64),
            "cls_routing_corrected_sse": torch.zeros(12, dtype=torch.float64),
            "content_noise_sse": torch.zeros(12, dtype=torch.float64),
            "content_corrected_sse": torch.zeros(12, dtype=torch.float64),
            "mlp_noise_sse": torch.zeros(3072, dtype=torch.float64),
            "mlp_corrected_sse": torch.zeros(3072, dtype=torch.float64),
        }
        for block in DOWNSTREAM_BLOCKS
    }


def add_squared_error(target, candidate, dimensions):
    return (candidate - target).double().square().sum(dim=dimensions).cpu()


def trace_condition(model, adapter, loader, device):
    accumulators = empty_accumulators()
    replay_error = 0.0
    with torch.no_grad():
        for clean, corrupted, _ in tqdm(loader, desc="downstream circuit discovery"):
            batch = clean.shape[0]
            outputs = model(
                pixel_values=torch.cat([clean, corrupted]).to(device),
                output_hidden_states=True,
            )
            clean_hidden, corrupt_hidden = outputs.hidden_states[BLOCK6].split(batch)
            corrected_hidden = torch.cat(
                [
                    corrupt_hidden[:, :1],
                    corrupt_hidden[:, 1:] + adapter(corrupt_hidden[:, 1:]),
                ],
                dim=1,
            )
            for block in DOWNSTREAM_BLOCKS:
                layer = model.vit.layers[block - 1]
                clean_parts = attention_computation(layer, clean_hidden)
                corrupt_parts = attention_computation(layer, corrupt_hidden)
                corrected_parts = attention_computation(layer, corrected_hidden)
                clean_weights, clean_content, clean_mlp, clean_hidden = clean_parts
                corrupt_weights, corrupt_content, corrupt_mlp, corrupt_hidden = corrupt_parts
                corrected_weights, corrected_content, corrected_mlp, corrected_hidden = corrected_parts
                store = accumulators[block]
                store["routing_noise_sse"] += add_squared_error(
                    clean_weights, corrupt_weights, (0, 2, 3)
                )
                store["routing_corrected_sse"] += add_squared_error(
                    clean_weights, corrected_weights, (0, 2, 3)
                )
                store["cls_routing_noise_sse"] += add_squared_error(
                    clean_weights[:, :, 0], corrupt_weights[:, :, 0], (0, 2)
                )
                store["cls_routing_corrected_sse"] += add_squared_error(
                    clean_weights[:, :, 0], corrected_weights[:, :, 0], (0, 2)
                )
                store["content_noise_sse"] += add_squared_error(
                    clean_content, corrupt_content, (0, 2, 3)
                )
                store["content_corrected_sse"] += add_squared_error(
                    clean_content, corrected_content, (0, 2, 3)
                )
                store["mlp_noise_sse"] += add_squared_error(
                    clean_mlp, corrupt_mlp, (0, 1)
                )
                store["mlp_corrected_sse"] += add_squared_error(
                    clean_mlp, corrected_mlp, (0, 1)
                )
                expected_clean, expected_corrupt = outputs.hidden_states[block].split(batch)
                replay_error = max(
                    replay_error,
                    float((clean_hidden - expected_clean).abs().max()),
                    float((corrupt_hidden - expected_corrupt).abs().max()),
                )
    if replay_error > 1e-3:
        raise RuntimeError(f"Downstream replay mismatch: {replay_error}")
    return accumulators, replay_error


def recovery(noise_sse, corrected_sse):
    return 1.0 - corrected_sse.numpy() / np.maximum(noise_sse.numpy(), EPSILON)


def summarize(accumulators, top_heads, top_neurons):
    blocks = {}
    head_rows = []
    neuron_rows = []
    for block, store in accumulators.items():
        metrics = {}
        for prefix in ["routing", "cls_routing", "content", "mlp"]:
            values = recovery(store[f"{prefix}_noise_sse"], store[f"{prefix}_corrected_sse"])
            metrics[f"{prefix}_recovery"] = values.tolist()
            if prefix != "mlp":
                for index, value in enumerate(values):
                    head_rows.append(
                        {"block": block, "head": index, "metric": prefix, "recovery": float(value)}
                    )
            else:
                for index, value in enumerate(values):
                    neuron_rows.append(
                        {"block": block, "neuron": index, "recovery": float(value)}
                    )
        blocks[str(block)] = metrics
    return {
        "blocks": blocks,
        "top_heads": sorted(head_rows, key=lambda row: row["recovery"], reverse=True)[:top_heads],
        "top_mlp_neurons": sorted(
            neuron_rows, key=lambda row: row["recovery"], reverse=True
        )[:top_neurons],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corruptions", nargs="+", choices=["noise", "blur"], default=["noise", "blur"])
    parser.add_argument("--start-index", type=int, default=36000)
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--top-heads", type=int, default=30)
    parser.add_argument("--top-neurons", type=int, default=100)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start_index < 36000 or args.start_index + args.samples > 46000:
        raise ValueError("Discovery is locked to the mechanistic-development interval [36000,46000)")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)
    results = {}
    adapter_paths = {}
    replay_errors = {}
    for seed in args.seeds:
        adapter, path = load_adapter(seed, device)
        adapter_paths[str(seed)] = str(path)
        results[str(seed)] = {}
        for corruption in args.corruptions:
            split = {"start": args.start_index, "end": args.start_index + args.samples}
            loader = make_loader(
                corruption, split, seed, args.batch_size, args.workers, args.samples
            )
            accumulators, replay_error = trace_condition(
                model, adapter, loader, device
            )
            results[str(seed)][corruption] = summarize(
                accumulators, args.top_heads, args.top_neurons
            )
            replay_errors[f"seed{seed}_{corruption}"] = replay_error
    summary = {
        "configuration": vars(args)
        | {
            "model": BASE_MODEL,
            "adapter_paths": adapter_paths,
            "adapter_location": "after Block 6",
            "traced_blocks": DOWNSTREAM_BLOCKS,
            "vit_frozen": True,
            "adapters_frozen": True,
            "uses_paired_clean_oracle": True,
            "training_or_tuning": False,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "candidate circuit discovery only",
        },
        "replay_max_absolute_errors": replay_errors,
        "results": results,
        "guardrails": [
            "Because the adapter is inserted after Block 6, it cannot change computations already completed inside Block 6.",
            "The analysis ranks recovery in Blocks 7-12 and does not yet establish causal necessity.",
            "Candidate heads and neurons require confirmation on a disjoint split against random controls.",
            "Paired clean images are used only for oracle diagnosis and are not required by the deployed adapter.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
