import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

import scripts.experiment72_downstream_circuit_necessity as circuit
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment52_block6_repair_propagation import make_loader


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment73_seed_specific_attention"
DISCOVERY_SUMMARY = (
    ACTIVE_ROOT
    / "results/sae/experiment71_downstream_circuit_recovery/"
    "full_3seed_noise_blur_2000_v1/summary.json"
)
COMMON_HEADS = {7: (0, 11), 9: (4,)}
HEAD_METRICS = ("routing_recovery", "cls_routing_recovery", "content_recovery")


def select_seed_heads(discovery, seed):
    selected = {}
    for block, count in [(7, 2), (9, 1)]:
        scores = []
        for head in range(12):
            values = [
                discovery[str(seed)][corruption]["blocks"][str(block)][metric][head]
                for corruption in ["noise", "blur"]
                for metric in HEAD_METRICS
            ]
            scores.append((float(max(values)), head, values))
        winners = sorted(scores, reverse=True)[:count]
        selected[block] = tuple(sorted(row[1] for row in winners))
    return selected


def sample_random_heads(target, controls, seed):
    generator = np.random.default_rng(seed)
    sampled = []
    for _ in range(controls):
        candidate = {}
        for block, indices in target.items():
            available = [head for head in range(12) if head not in indices]
            candidate[block] = tuple(
                sorted(generator.choice(available, len(indices), replace=False).tolist())
            )
        sampled.append(candidate)
    return sampled


def build_variants(seed_heads, controls, seed):
    variants = [
        {"name": "full", "head": None, "neuron": None},
        {"name": "seed_specific", "head": seed_heads, "neuron": None},
        {"name": "common", "head": COMMON_HEADS, "neuron": None},
    ]
    variants.extend(
        {
            "name": f"random_{index}",
            "head": heads,
            "neuron": None,
        }
        for index, heads in enumerate(sample_random_heads(seed_heads, controls, seed))
    )
    return variants


def evaluate(model, adapter, loader, variants, device, args, statistical_seed):
    correct = {"baseline": []}
    correct.update({variant["name"]: [] for variant in variants})
    replay_error = 0.0
    with torch.no_grad():
        for _, corrupted, labels in tqdm(loader, desc="seed-specific circuit confirmation"):
            labels = labels.to(device)
            outputs = model(pixel_values=corrupted.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[circuit.BLOCK6]
            corrected = torch.cat(
                [hidden[:, :1], hidden[:, 1:] + adapter(hidden[:, 1:])], dim=1
            )
            branches = corrected.unsqueeze(0).expand(len(variants), -1, -1, -1).clone()
            for block in range(7, 13):
                branches = circuit.replay_layer(
                    model.vit.layers[block - 1], branches, variants, block
                )
            logits = model.classifier(model.vit.layernorm(branches)[:, :, 0])
            expected = downstream_from_layer(model, corrected, circuit.BLOCK6 - 1)
            replay_error = max(replay_error, float((logits[0] - expected).abs().max()))
            correct["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for index, variant in enumerate(variants):
                correct[variant["name"]].extend(
                    (logits[index].argmax(1) == labels).cpu().tolist()
                )
    if replay_error > 1e-3:
        raise RuntimeError(f"Replay mismatch: {replay_error}")
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    full = correct["full"]
    comparisons = {
        name: paired_comparison(full, values, statistical_seed + offset, args.bootstrap)
        for offset, (name, values) in enumerate(correct.items())
        if name not in {"baseline", "full"}
    }
    seed_cost = -comparisons["seed_specific"]["accuracy_difference"]
    common_cost = -comparisons["common"]["accuracy_difference"]
    random_costs = np.asarray(
        [-comparisons[f"random_{index}"]["accuracy_difference"] for index in range(args.random_controls)]
    )
    return {
        "baseline_to_full": paired_comparison(
            correct["baseline"], full, statistical_seed + 900, args.bootstrap
        ),
        "seed_specific_ablation": comparisons["seed_specific"],
        "common_ablation": comparisons["common"],
        "seed_specific_removal_cost": seed_cost,
        "common_removal_cost": common_cost,
        "seed_specific_minus_common_cost": seed_cost - common_cost,
        "random_mean_removal_cost": float(random_costs.mean()),
        "random_costs": random_costs.tolist(),
        "seed_specific_exceeds_random": int(np.sum(seed_cost > random_costs)),
        "common_exceeds_random": int(np.sum(common_cost > random_costs)),
        "seed_specific_empirical_pvalue": float(
            (1 + np.sum(random_costs >= seed_cost)) / (len(random_costs) + 1)
        ),
        "replay_max_absolute_logit_error": replay_error,
    }, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery-summary", type=Path, default=DISCOVERY_SUMMARY)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--corruptions", nargs="+", choices=["noise", "blur"], default=["noise", "blur"])
    parser.add_argument("--start-index", type=int, default=47000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.start_index < 47000 or args.start_index + args.samples > 48000:
        raise ValueError("Confirmation is locked to [47000,48000), disjoint from Experiments 71 and 72")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    discovery_summary = json.loads(args.discovery_summary.read_text())
    discovery = discovery_summary["results"]
    seed_heads = {seed: select_seed_heads(discovery, seed) for seed in args.seeds}
    print("Frozen seed-specific heads:", seed_heads)
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
        circuit.TARGET_HEADS = seed_heads[seed]
        circuit.TARGET_NEURONS = {}
        variants = build_variants(seed_heads[seed], args.random_controls, 2073 + seed)
        adapter, path = circuit.load_adapter(seed, device)
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
                2073 + seed * 1000 + corruption_index * 100,
            )
            results[str(seed)][corruption] = condition
            for name, values in arrays.items():
                outcomes[f"seed{seed}_{corruption}_{name}"] = values
    summary = {
        "configuration": vars(args)
        | {
            "discovery_summary": str(args.discovery_summary.resolve()),
            "model": BASE_MODEL,
            "adapter_paths": adapter_paths,
            "seed_specific_heads": seed_heads,
            "common_heads": COMMON_HEADS,
            "selection_rule": "Within each adapter seed, select two unique Block-7 heads and one unique Block-9 head by maximum recovery across routing, CLS routing, and content over Noise-4 and Blur-4 discovery data.",
            "candidate_source_range": [36000, 38000],
            "confirmation_range": [args.start_index, args.start_index + args.samples],
            "vit_frozen": True,
            "adapters_frozen": True,
            "uses_clean_counterpart_for_intervention": False,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "disjoint seed-specific causal circuit confirmation",
        },
        "results": results,
        "guardrails": [
            "Candidate selection was frozen before confirmation inference.",
            "Every random set matches the candidate block distribution and removed projected-output norm per token.",
            "This tests necessity, not sufficiency or circuit completeness.",
            "The ImageNet validation confirmation range was used in older unrelated diagnostics, so this is mechanistic development rather than a final independent benchmark.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
