import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment49_sae_jaccard_controls import bootstrap_difference
from scripts.experiment52_block6_repair_propagation import make_loader
from scripts.experiment72_downstream_circuit_necessity import load_adapter


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment75_block7_qkv_mediation"
BLOCK6 = 6
BLOCK7_INDEX = 6
CONDITIONS = (
    "full",
    "revert_q",
    "revert_k",
    "revert_v",
    "revert_qk",
    "revert_qkv",
    "revert_attention_output",
    "revert_mlp_output",
    "wrong_q",
    "wrong_k",
    "wrong_v",
    "wrong_qkv",
)


def qkv(attention, normalized):
    shape = (*normalized.shape[:-1], -1, attention.head_dim)
    query = attention.q_proj(normalized).view(shape).transpose(1, 2)
    key = attention.k_proj(normalized).view(shape).transpose(1, 2)
    value = attention.v_proj(normalized).view(shape).transpose(1, 2)
    return query, key, value


def attention_output(layer, hidden, query, key, value):
    weights = torch.softmax(
        torch.matmul(query, key.transpose(-1, -2)) * layer.attention.scaling,
        dim=-1,
        dtype=torch.float32,
    ).to(query.dtype)
    attended = torch.matmul(weights, value)
    joined = attended.transpose(1, 2).reshape(hidden.shape).contiguous()
    return layer.dropout(layer.attention.o_proj(joined))


def mlp_output(layer, after_attention):
    return layer.dropout(layer.mlp(layer.layernorm_after(after_attention)))


def block7_variants(layer, noisy_input, corrected_input):
    noisy_norm = layer.layernorm_before(noisy_input)
    corrected_norm = layer.layernorm_before(corrected_input)
    noisy_q, noisy_k, noisy_v = qkv(layer.attention, noisy_norm)
    corrected_q, corrected_k, corrected_v = qkv(layer.attention, corrected_norm)
    wrong_q = noisy_q.roll(1, dims=0)
    wrong_k = noisy_k.roll(1, dims=0)
    wrong_v = noisy_v.roll(1, dims=0)
    component_sets = {
        "full": (corrected_q, corrected_k, corrected_v),
        "revert_q": (noisy_q, corrected_k, corrected_v),
        "revert_k": (corrected_q, noisy_k, corrected_v),
        "revert_v": (corrected_q, corrected_k, noisy_v),
        "revert_qk": (noisy_q, noisy_k, corrected_v),
        "revert_qkv": (noisy_q, noisy_k, noisy_v),
        "wrong_q": (wrong_q, corrected_k, corrected_v),
        "wrong_k": (corrected_q, wrong_k, corrected_v),
        "wrong_v": (corrected_q, corrected_k, wrong_v),
        "wrong_qkv": (wrong_q, wrong_k, wrong_v),
    }
    noisy_attention = attention_output(layer, noisy_input, noisy_q, noisy_k, noisy_v)
    noisy_after_attention = noisy_input + noisy_attention
    noisy_mlp = mlp_output(layer, noisy_after_attention)
    outputs = {}
    for name, (query, key, value) in component_sets.items():
        attended = attention_output(layer, corrected_input, query, key, value)
        after_attention = corrected_input + attended
        outputs[name] = after_attention + mlp_output(layer, after_attention)
    corrected_attention = attention_output(
        layer, corrected_input, corrected_q, corrected_k, corrected_v
    )
    corrected_after_attention = corrected_input + corrected_attention
    hybrid_after_attention = corrected_input + noisy_attention
    outputs["revert_attention_output"] = (
        hybrid_after_attention + mlp_output(layer, hybrid_after_attention)
    )
    outputs["revert_mlp_output"] = corrected_after_attention + noisy_mlp
    return outputs


def evaluate(model, adapter, loader, device, args, statistical_seed):
    correct = {"baseline": [], **{name: [] for name in CONDITIONS}}
    margins = {"baseline": [], **{name: [] for name in CONDITIONS}}
    replay_error = 0.0
    layer = model.vit.layers[BLOCK7_INDEX]
    with torch.no_grad():
        for _, corrupted, labels in loader:
            labels = labels.to(device)
            outputs = model(pixel_values=corrupted.to(device), output_hidden_states=True)
            noisy6 = outputs.hidden_states[BLOCK6]
            corrected6 = torch.cat(
                [noisy6[:, :1], noisy6[:, 1:] + adapter(noisy6[:, 1:])], dim=1
            )
            variants7 = block7_variants(layer, noisy6, corrected6)
            logits = {
                name: downstream_from_layer(model, hidden7, BLOCK7_INDEX)
                for name, hidden7 in variants7.items()
            }
            expected_full = downstream_from_layer(model, corrected6, BLOCK6 - 1)
            replay_error = max(
                replay_error, float((logits["full"] - expected_full).abs().max())
            )
            all_logits = {"baseline": outputs.logits, **logits}
            for name, condition_logits in all_logits.items():
                correct[name].extend(
                    (condition_logits.argmax(1) == labels).cpu().tolist()
                )
                _, margin = classification_margin(condition_logits, labels)
                margins[name].extend(margin.cpu().tolist())
    if replay_error > 1e-3:
        raise RuntimeError(f"Full adapter replay mismatch: {replay_error}")
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    margins = {name: np.asarray(values, dtype=np.float64) for name, values in margins.items()}
    full = correct["full"]
    result = {
        "baseline_to_full": paired_comparison(
            correct["baseline"], full, statistical_seed, args.bootstrap
        ),
        "conditions": {},
        "full_replay_max_absolute_logit_error": replay_error,
    }
    for offset, name in enumerate(CONDITIONS[1:]):
        comparison = paired_comparison(
            full, correct[name], statistical_seed + offset + 1, args.bootstrap
        )
        comparison["adapter_gain_lost"] = -comparison["accuracy_difference"]
        comparison["margin_change_from_full"] = bootstrap_difference(
            margins[name] - margins["full"],
            np.zeros_like(margins["full"]),
            statistical_seed + 100 + offset,
            args.bootstrap,
        )
        result["conditions"][name] = comparison
    return result, correct, margins


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corruptions", nargs="+", choices=["noise", "blur"], default=["noise", "blur"])
    parser.add_argument("--start-index", type=int, default=38000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start_index < 38000 or args.start_index + args.samples > 39000:
        raise ValueError("Experiment 75 is locked to [38000,39000)")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)
    split = {"start": args.start_index, "end": args.start_index + args.samples}
    results = {}
    outcomes = {}
    adapter_paths = {}
    for seed in args.seeds:
        adapter, path = load_adapter(seed, device)
        adapter_paths[str(seed)] = str(path)
        results[str(seed)] = {}
        for corruption_index, corruption in enumerate(args.corruptions):
            loader = make_loader(
                corruption, split, seed, args.batch_size, args.workers, args.samples
            )
            condition, correct, margins = evaluate(
                model,
                adapter,
                loader,
                device,
                args,
                2075 + seed * 1000 + corruption_index * 100,
            )
            results[str(seed)][corruption] = condition
            for name, values in correct.items():
                outcomes[f"seed{seed}_{corruption}_correct_{name}"] = values
            for name, values in margins.items():
                outcomes[f"seed{seed}_{corruption}_margin_{name}"] = values
    summary = {
        "configuration": vars(args)
        | {
            "model": BASE_MODEL,
            "adapter_paths": adapter_paths,
            "adapter_location": "after Block 6",
            "decomposed_layer": 7,
            "confirmation_range": [args.start_index, args.start_index + args.samples],
            "vit_frozen": True,
            "adapters_frozen": True,
            "uses_clean_counterpart": False,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "causal computational-path mediation analysis",
        },
        "results": results,
        "guardrails": [
            "No paired clean activation is used; components are reverted to the corresponding noisy-pass values.",
            "Wrong-image substitutions test whether arbitrary noisy components reproduce a reversion effect.",
            "Component substitution creates hybrid states and tests causal mediation/necessity, not sufficiency.",
            "This is mechanistic development on previously used ImageNet validation images, not final independent evaluation.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
