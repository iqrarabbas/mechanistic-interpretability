import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment52_block6_repair_propagation import make_loader
from scripts.experiment75_block7_qkv_path_mediation import attention_output, mlp_output, qkv
from scripts.experiment124_frozen_noise_subspace_sketch import load_factors, folded_delta


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results/sae/experiment126_noise_distributed_block_circuit"
BLOCK6 = 6
DOWNSTREAM_BLOCKS = tuple(range(7, 13))


def definitions():
    variants = {"full": {"attention": set(), "mlp": set()}}
    for block in DOWNSTREAM_BLOCKS:
        variants[f"block{block}_attention"] = {"attention": {block}, "mlp": set()}
        variants[f"block{block}_mlp"] = {"attention": set(), "mlp": {block}}
        variants[f"block{block}_both"] = {"attention": {block}, "mlp": {block}}
    for start, end in ((7, 8), (9, 10), (11, 12)):
        blocks = set(range(start, end + 1))
        variants[f"blocks{start}_{end}_attention"] = {"attention": blocks, "mlp": set()}
        variants[f"blocks{start}_{end}_mlp"] = {"attention": set(), "mlp": blocks}
        variants[f"blocks{start}_{end}_both"] = {"attention": blocks, "mlp": blocks}
    all_blocks = set(DOWNSTREAM_BLOCKS)
    variants["blocks7_12_attention"] = {"attention": all_blocks, "mlp": set()}
    variants["blocks7_12_mlp"] = {"attention": set(), "mlp": all_blocks}
    variants["blocks7_12_both"] = {"attention": all_blocks, "mlp": all_blocks}
    return variants


def components(layer, hidden):
    normalized = layer.layernorm_before(hidden)
    query, key, value = qkv(layer.attention, normalized)
    attention = attention_output(layer, hidden, query, key, value)
    after_attention = hidden + attention
    mlp = mlp_output(layer, after_attention)
    return attention, mlp, after_attention + mlp


def propagate_variants(model, baseline6, corrected6, variants):
    names = list(variants)
    batch = baseline6.shape[0]
    states = corrected6.unsqueeze(0).expand(len(names), -1, -1, -1).clone()
    baseline = baseline6
    replay_error = 0.0
    for block in DOWNSTREAM_BLOCKS:
        layer = model.vit.layers[block - 1]
        baseline_attention, baseline_mlp, baseline_next = components(layer, baseline)
        flat_states = states.flatten(0, 1)
        normalized = layer.layernorm_before(flat_states)
        query, key, value = qkv(layer.attention, normalized)
        attentions = attention_output(layer, flat_states, query, key, value).reshape_as(states)
        for index, name in enumerate(names):
            if block in variants[name]["attention"]:
                attentions[index] = baseline_attention
        after_attention = states + attentions
        mlps = mlp_output(layer, after_attention.flatten(0, 1)).reshape_as(states)
        for index, name in enumerate(names):
            if block in variants[name]["mlp"]:
                mlps[index] = baseline_mlp
        states = after_attention + mlps
        normal_next = layer(baseline, attention_mask=None)
        replay_error = max(replay_error, float((baseline_next - normal_next).abs().max()))
        baseline = baseline_next
    logits = model.classifier(model.vit.layernorm(states)[:, :, 0])
    return {name: logits[index] for index, name in enumerate(names)}, baseline, replay_error


def evaluate(model, factors, loader, device, args, statistical_seed):
    variants = definitions()
    correct = {"baseline": [], **{name: [] for name in variants}}
    max_component_replay_error = 0.0
    max_baseline_logit_error = 0.0
    max_full_logit_error = 0.0
    with torch.inference_mode():
        for clean, noise, labels in loader:
            pixels = clean if args.condition == "clean" else noise
            labels = labels.to(device)
            outputs = model(pixel_values=pixels.to(device), output_hidden_states=True)
            baseline6 = outputs.hidden_states[BLOCK6]
            patches = baseline6[:, 1:]
            corrected6 = torch.cat(
                [baseline6[:, :1], patches + folded_delta(patches, factors)], dim=1
            )
            logits, baseline12, replay_error = propagate_variants(
                model, baseline6, corrected6, variants
            )
            max_component_replay_error = max(max_component_replay_error, replay_error)
            baseline_logits = model.classifier(model.vit.layernorm(baseline12)[:, 0])
            max_baseline_logit_error = max(
                max_baseline_logit_error,
                float((baseline_logits - outputs.logits).abs().max()),
            )
            expected_full = downstream_from_layer(model, corrected6, BLOCK6 - 1)
            max_full_logit_error = max(
                max_full_logit_error, float((logits["full"] - expected_full).abs().max())
            )
            correct["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())
            for name, condition_logits in logits.items():
                correct[name].extend((condition_logits.argmax(1) == labels).cpu().tolist())
    errors = {
        "component_replay": max_component_replay_error,
        "baseline_logit_replay": max_baseline_logit_error,
        "full_repair_logit_replay": max_full_logit_error,
    }
    if max(errors.values()) > 1e-3:
        raise RuntimeError(f"Replay validation failed: {errors}")
    correct = {name: np.asarray(values, dtype=bool) for name, values in correct.items()}
    full = correct["full"]
    results = {
        "baseline_to_full": paired_comparison(
            correct["baseline"], full, statistical_seed, args.bootstrap
        ),
        "reversions": {},
        "validation": errors,
    }
    for offset, name in enumerate(list(variants)[1:]):
        comparison = paired_comparison(
            full, correct[name], statistical_seed + offset + 1, args.bootstrap
        )
        comparison["repair_gain_lost"] = -comparison["accuracy_difference"]
        results["reversions"][name] = comparison
    return results, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", choices=["noise", "clean"], default=["noise", "clean"])
    parser.add_argument("--start-index", type=int, default=38000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.seeds != [0, 1, 2]:
        raise ValueError("Circuit confirmation is locked to seeds [0,1,2]")
    if not 38000 <= args.start_index < args.start_index + args.samples <= 39000:
        raise ValueError("Mechanistic split is locked to [38000,39000)")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    progress_path = output_dir / "progress.json"
    outcome_path = output_dir / "paired_outcomes_partial.npz"
    results = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    outcomes = {}
    if outcome_path.exists():
        with np.load(outcome_path) as saved:
            outcomes = {name: saved[name] for name in saved.files}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(
        BASE_MODEL, attn_implementation="eager", local_files_only=True
    ).to(device).eval()
    model.requires_grad_(False)
    factor_paths = {}
    for seed in args.seeds:
        factors, factor_path = load_factors(seed, "delta_pca", device)
        factor_paths[str(seed)] = str(factor_path)
        results.setdefault(str(seed), {})
        for condition_index, condition in enumerate(args.conditions):
            if condition in results[str(seed)]:
                print(f"Skipping completed seed {seed} {condition}", flush=True)
                continue
            loader = make_loader(
                "noise",
                {"start": args.start_index, "end": args.start_index + args.samples},
                seed, args.batch_size, args.workers, args.samples,
            )
            args.condition = condition
            result, arrays = evaluate(
                model, factors, loader, device, args,
                1260000 + seed * 1000 + condition_index * 500,
            )
            results[str(seed)][condition] = result
            for name, values in arrays.items():
                outcomes[f"seed{seed}_{condition}_{name}_correct"] = values
            progress_path.write_text(json.dumps(results, indent=2))
            np.savez_compressed(outcome_path, **outcomes)
            print(f"Completed seed {seed} {condition}", flush=True)
    aggregate = {}
    for condition in args.conditions:
        aggregate[condition] = {}
        for name in list(definitions())[1:]:
            losses = [
                results[str(seed)][condition]["reversions"][name]["repair_gain_lost"]
                for seed in args.seeds
            ]
            aggregate[condition][name] = {
                "gain_lost_by_seed": losses,
                "mean_gain_lost": float(np.mean(losses)),
            }
    summary = {
        "configuration": {key: value for key, value in vars(args).items() if key != "condition"} | {
            "model": BASE_MODEL,
            "repair": "frozen rank-128 adapter-delta PCA map after Block 6",
            "factor_paths": factor_paths,
            "vit_frozen": True,
            "repair_frozen": True,
            "uses_clean_counterpart_for_noise_intervention": False,
            "imageNetV2_accessed": False,
            "status": "mechanistic development analysis on a previously used range",
        },
        "variant_definitions": {
            name: {component: sorted(blocks) for component, blocks in definition.items()}
            for name, definition in definitions().items()
        },
        "results": results,
        "aggregate": aggregate,
        "limitations": [
            "Component reversion forms hybrid states and measures conditional causal dependence, not a unique complete circuit.",
            "Baseline components are taken from the same noisy or clean image; no paired clean activation repairs Noise.",
            "The [38000,39000) range was used in earlier mechanistic work and is not an untouched final benchmark.",
            "Attention and MLP effects are nonlinear and need not add to the total repair gain.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2), flush=True)
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
