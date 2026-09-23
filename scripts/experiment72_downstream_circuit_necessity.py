import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment52_block6_repair_propagation import make_loader
from scripts.experiment61_block6_attention_content_patching import attention_parts


ROOT = Path(__file__).parent.parent
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment72_downstream_circuit_necessity"
ADAPTER_ROOT = (
    ACTIVE_ROOT
    / "results/sae/experiment67_mixed_noise_blur_block6/"
    "full_3seed_noise_blur_identity0p2_v1"
)
BLOCK6 = 6
TARGET_HEADS = {7: (0, 11), 9: (4,)}
TARGET_NEURONS = {7: (2343, 2923, 3005), 8: (1840,), 9: (1775,)}
EPSILON = 1e-12


def load_adapter(seed, device):
    path = ADAPTER_ROOT / f"seed_{seed}" / "mixed_identity0p2.pt"
    adapter = HiddenLinear().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval(), path


def sample_controls(targets, population, controls, generator):
    excluded = set(targets)
    available = np.asarray([index for index in range(population) if index not in excluded])
    return [tuple(sorted(generator.choice(available, len(targets), replace=False).tolist())) for _ in range(controls)]


def build_variants(controls, seed):
    generator = np.random.default_rng(seed)
    random_heads = {
        block: sample_controls(indices, 12, controls, generator)
        for block, indices in TARGET_HEADS.items()
    }
    random_neurons = {
        block: sample_controls(indices, 3072, controls, generator)
        for block, indices in TARGET_NEURONS.items()
    }
    variants = [
        {"name": "full", "head": None, "neuron": None},
        {"name": "target_heads", "head": TARGET_HEADS, "neuron": None},
        {"name": "target_neurons", "head": None, "neuron": TARGET_NEURONS},
        {"name": "target_both", "head": TARGET_HEADS, "neuron": TARGET_NEURONS},
    ]
    for index in range(controls):
        head = {block: random_heads[block][index] for block in TARGET_HEADS}
        neuron = {block: random_neurons[block][index] for block in TARGET_NEURONS}
        variants.extend(
            [
                {"name": f"random_heads_{index}", "head": head, "neuron": None},
                {"name": f"random_neurons_{index}", "head": None, "neuron": neuron},
                {"name": f"random_both_{index}", "head": head, "neuron": neuron},
            ]
        )
    return variants


def index_mask(variants, key, block, width, device):
    candidate = torch.zeros(len(variants), width, dtype=torch.bool, device=device)
    target = torch.zeros_like(candidate)
    enabled = torch.zeros(len(variants), dtype=torch.bool, device=device)
    target_indices = TARGET_HEADS.get(block, ()) if key == "head" else TARGET_NEURONS.get(block, ())
    for variant_index, variant in enumerate(variants):
        selected = variant[key]
        if selected is None or block not in selected:
            continue
        enabled[variant_index] = True
        candidate[variant_index, list(selected[block])] = True
        target[variant_index, list(target_indices)] = True
    return candidate, target, enabled


def energy_matched_component(candidate, target, enabled):
    candidate_norm = candidate.norm(dim=-1, keepdim=True).clamp_min(EPSILON)
    target_norm = target.norm(dim=-1, keepdim=True)
    scale = target_norm / candidate_norm
    scale = scale * enabled[:, None, None, None].to(scale.dtype)
    return candidate * scale


def replay_layer(layer, hidden, variants, block):
    variant_count, batch, tokens, width = hidden.shape
    flat = hidden.reshape(variant_count * batch, tokens, width)
    normalized = layer.layernorm_before(flat)
    weights, values = attention_parts(layer.attention, normalized)
    content = torch.matmul(weights, values)
    joined = content.transpose(1, 2).reshape(flat.shape).contiguous()
    attention_output = layer.attention.o_proj(joined)

    candidate_mask, target_mask, enabled = index_mask(
        variants, "head", block, 12, hidden.device
    )
    if enabled.any():
        head_content = content.reshape(variant_count, batch, 12, tokens, 64)
        candidate_joined = (
            head_content * candidate_mask[:, None, :, None, None]
        ).transpose(2, 3).reshape(variant_count * batch, tokens, width)
        target_joined = (
            head_content * target_mask[:, None, :, None, None]
        ).transpose(2, 3).reshape(variant_count * batch, tokens, width)
        candidate_component = F.linear(candidate_joined, layer.attention.o_proj.weight)
        target_component = F.linear(target_joined, layer.attention.o_proj.weight)
        candidate_component = candidate_component.reshape(variant_count, batch, tokens, width)
        target_component = target_component.reshape(variant_count, batch, tokens, width)
        removal = energy_matched_component(candidate_component, target_component, enabled)
        attention_output = attention_output.reshape(variant_count, batch, tokens, width) - removal
        attention_output = attention_output.reshape(variant_count * batch, tokens, width)

    after_attention = flat + layer.dropout(attention_output)
    normalized_mlp = layer.layernorm_after(after_attention)
    activation = layer.mlp.activation_fn(layer.mlp.fc1(normalized_mlp))
    mlp_output = layer.mlp.fc2(activation)
    candidate_mask, target_mask, enabled = index_mask(
        variants, "neuron", block, 3072, hidden.device
    )
    if enabled.any():
        activation = activation.reshape(variant_count, batch, tokens, 3072)
        candidate_activation = activation * candidate_mask[:, None, None, :]
        target_activation = activation * target_mask[:, None, None, :]
        candidate_component = F.linear(
            candidate_activation.reshape(variant_count * batch, tokens, 3072),
            layer.mlp.fc2.weight,
        ).reshape(variant_count, batch, tokens, width)
        target_component = F.linear(
            target_activation.reshape(variant_count * batch, tokens, 3072),
            layer.mlp.fc2.weight,
        ).reshape(variant_count, batch, tokens, width)
        removal = energy_matched_component(candidate_component, target_component, enabled)
        mlp_output = mlp_output.reshape(variant_count, batch, tokens, width) - removal
        mlp_output = mlp_output.reshape(variant_count * batch, tokens, width)
    output = after_attention + layer.dropout(mlp_output)
    return output.reshape(variant_count, batch, tokens, width)


def evaluate(model, adapter, loader, variants, device, args, statistical_seed):
    correct = {"baseline": []}
    correct.update({variant["name"]: [] for variant in variants})
    replay_max_absolute_error = 0.0
    with torch.no_grad():
        for _, corrupted, labels in tqdm(loader, desc="circuit necessity"):
            labels = labels.to(device)
            outputs = model(pixel_values=corrupted.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK6]
            corrected = torch.cat(
                [hidden[:, :1], hidden[:, 1:] + adapter(hidden[:, 1:])], dim=1
            )
            branch_hidden = corrected.unsqueeze(0).expand(len(variants), -1, -1, -1).clone()
            for block in range(7, 13):
                branch_hidden = replay_layer(
                    model.vit.layers[block - 1], branch_hidden, variants, block
                )
            logits = model.classifier(model.vit.layernorm(branch_hidden)[:, :, 0])
            expected_full_logits = downstream_from_layer(model, corrected, BLOCK6 - 1)
            replay_max_absolute_error = max(
                replay_max_absolute_error,
                float((logits[0] - expected_full_logits).abs().max()),
            )
            correct["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for index, variant in enumerate(variants):
                correct[variant["name"]].extend(
                    (logits[index].argmax(1) == labels).cpu().tolist()
                )
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    if replay_max_absolute_error > 1e-3:
        raise RuntimeError(f"Unsuppressed adapter replay mismatch: {replay_max_absolute_error}")
    full = correct["full"]
    result = {
        "baseline_to_full": paired_comparison(
            correct["baseline"], full, statistical_seed, args.bootstrap
        ),
        "ablations": {},
        "unsuppressed_replay_max_absolute_logit_error": replay_max_absolute_error,
    }
    for offset, name in enumerate(correct):
        if name in {"baseline", "full"}:
            continue
        result["ablations"][name] = paired_comparison(
            full, correct[name], statistical_seed + offset + 1, args.bootstrap
        )
    for kind in ["heads", "neurons", "both"]:
        target_name = f"target_{kind}"
        target_cost = -result["ablations"][target_name]["accuracy_difference"]
        random_costs = np.asarray(
            [
                -result["ablations"][f"random_{kind}_{index}"]["accuracy_difference"]
                for index in range(args.random_controls)
            ]
        )
        result[f"{kind}_test"] = {
            "target_removal_cost": target_cost,
            "random_mean_removal_cost": float(random_costs.mean()),
            "random_costs": random_costs.tolist(),
            "target_exceeds_random": int(np.sum(target_cost > random_costs)),
            "empirical_one_sided_pvalue": float(
                (1 + np.sum(random_costs >= target_cost)) / (len(random_costs) + 1)
            ),
        }
    return result, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corruptions", nargs="+", choices=["noise", "blur"], default=["noise", "blur"])
    parser.add_argument("--start-index", type=int, default=46000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start_index < 46000 or args.start_index + args.samples > 47000:
        raise ValueError("Causal confirmation is locked to [46000,47000), disjoint from Experiment 71")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager"
    ).to(device).eval()
    model.requires_grad_(False)
    variants = build_variants(args.random_controls, 2072)
    results = {}
    outcomes = {}
    adapter_paths = {}
    split = {"start": args.start_index, "end": args.start_index + args.samples}
    for seed in args.seeds:
        adapter, path = load_adapter(seed, device)
        adapter_paths[str(seed)] = str(path)
        results[str(seed)] = {}
        for corruption_index, corruption in enumerate(args.corruptions):
            loader = make_loader(
                corruption, split, seed, args.batch_size, args.workers, args.samples
            )
            condition, arrays = evaluate(
                model,
                adapter,
                loader,
                variants,
                device,
                args,
                2072 + seed * 1000 + corruption_index * 100,
            )
            results[str(seed)][corruption] = condition
            for name, values in arrays.items():
                outcomes[f"seed{seed}_{corruption}_{name}"] = values
    summary = {
        "configuration": vars(args)
        | {
            "model": BASE_MODEL,
            "adapter_paths": adapter_paths,
            "adapter_location": "after Block 6",
            "target_heads": TARGET_HEADS,
            "target_neurons": TARGET_NEURONS,
            "candidate_source": "Experiment 71, [36000,38000)",
            "confirmation_split": [args.start_index, args.start_index + args.samples],
            "energy_matching": "per-token norm of removed projected output contribution",
            "vit_frozen": True,
            "adapters_frozen": True,
            "uses_clean_counterpart_for_intervention": False,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "disjoint causal circuit confirmation",
        },
        "results": results,
        "guardrails": [
            "Target circuits were fixed before confirmation-set evaluation.",
            "Random controls match the number and block location of suppressed components and their removed output norm per token.",
            "Ablation establishes necessity only if targeted removal consistently exceeds controls; it does not establish sufficiency or a complete circuit.",
            "The confirmation images appeared in older unrelated diagnostics, so this remains mechanistic development rather than an independent final benchmark.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
