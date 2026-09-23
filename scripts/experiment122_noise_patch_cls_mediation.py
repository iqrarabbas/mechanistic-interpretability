import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment49_sae_jaccard_controls import bootstrap_difference
from scripts.experiment52_block6_repair_propagation import make_loader
from scripts.experiment72_downstream_circuit_necessity import load_adapter


ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = ROOT / "results/sae/experiment122_noise_patch_cls_mediation"
STAGES = tuple(range(6, 13))
CONDITIONS = ("baseline", "full", "revert_cls", "revert_patches")


def final_logits(model, hidden):
    return model.classifier(model.vit.layernorm(hidden)[:, 0])


def evaluate(model, adapter, loader, device, bootstrap, statistical_seed):
    correct = {stage: {name: [] for name in CONDITIONS} for stage in STAGES}
    margins = {stage: {name: [] for name in CONDITIONS} for stage in STAGES}
    errors = {"baseline_replay": 0.0, "full_replay": 0.0,
              "stage6_cls_identity": 0.0, "stage6_patch_identity": 0.0}
    with torch.inference_mode():
        for _, corrupted, labels in loader:
            labels = labels.to(device)
            outputs = model(pixel_values=corrupted.to(device), output_hidden_states=True)
            baseline6 = outputs.hidden_states[6]
            corrected = torch.cat(
                [baseline6[:, :1], baseline6[:, 1:] + adapter(baseline6[:, 1:])], dim=1
            )
            for stage in STAGES:
                if stage > 6:
                    corrected = model.vit.layers[stage - 1](corrected, attention_mask=None)
                baseline = outputs.hidden_states[stage]
                revert_cls = torch.cat([baseline[:, :1], corrected[:, 1:]], dim=1)
                revert_patches = torch.cat([corrected[:, :1], baseline[:, 1:]], dim=1)
                states = {
                    "baseline": baseline,
                    "full": corrected,
                    "revert_cls": revert_cls,
                    "revert_patches": revert_patches,
                }
                for name, state in states.items():
                    logits = downstream_from_layer(model, state, stage - 1)
                    correct[stage][name].extend((logits.argmax(1) == labels).cpu().tolist())
                    _, margin = classification_margin(logits, labels)
                    margins[stage][name].extend(margin.cpu().tolist())
                    if stage == 12 and name == "baseline":
                        errors["baseline_replay"] = max(
                            errors["baseline_replay"], float((logits - outputs.logits).abs().max())
                        )
                    if stage == 12 and name == "full":
                        errors["full_replay"] = max(
                            errors["full_replay"], float((logits - final_logits(model, corrected)).abs().max())
                        )
                if stage == 6:
                    errors["stage6_cls_identity"] = max(
                        errors["stage6_cls_identity"], float((revert_cls - corrected).abs().max())
                    )
                    errors["stage6_patch_identity"] = max(
                        errors["stage6_patch_identity"], float((revert_patches - baseline).abs().max())
                    )
    if max(errors.values()) > 1e-3:
        raise RuntimeError(f"Replay/identity validation failed: {errors}")
    correct = {
        stage: {name: np.asarray(values, dtype=bool) for name, values in stage_values.items()}
        for stage, stage_values in correct.items()
    }
    margins = {
        stage: {name: np.asarray(values, dtype=np.float64) for name, values in stage_values.items()}
        for stage, stage_values in margins.items()
    }
    results = {}
    for stage in STAGES:
        full = correct[stage]["full"]
        results[str(stage)] = {
            "baseline_to_full": paired_comparison(
                correct[stage]["baseline"], full, statistical_seed + stage, bootstrap
            ),
            "reversions": {},
        }
        for offset, name in enumerate(("revert_cls", "revert_patches")):
            comparison = paired_comparison(
                full, correct[stage][name], statistical_seed + 100 + 10 * stage + offset, bootstrap
            )
            comparison["adapter_gain_lost"] = -comparison["accuracy_difference"]
            comparison["margin_change_from_full"] = bootstrap_difference(
                margins[stage][name] - margins[stage]["full"],
                np.zeros_like(margins[stage]["full"]),
                statistical_seed + 200 + 10 * stage + offset,
                bootstrap,
            )
            results[str(stage)]["reversions"][name] = comparison
    return results, correct, margins, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--conditions", nargs="+", choices=["noise", "clean"], default=["noise", "clean"])
    parser.add_argument("--start-index", type=int, default=38000)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if not 38000 <= args.start_index < args.start_index + args.samples <= 39000:
        raise ValueError("Mechanistic replication is locked to ImageNet validation [38000,39000)")
    if len(set(args.seeds)) != len(args.seeds) or any(seed not in (0, 1, 2) for seed in args.seeds):
        raise ValueError("Adapter seeds must be unique and drawn from {0,1,2}")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device, flush=True)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL, attn_implementation="eager").to(device).eval()
    model.requires_grad_(False)
    all_results = {}
    outcomes = {}
    adapter_paths = {}
    for seed in args.seeds:
        adapter, path = load_adapter(seed, device)
        adapter.requires_grad_(False)
        adapter_paths[str(seed)] = str(path)
        all_results[str(seed)] = {}
        for condition in args.conditions:
            loader = make_loader(
                "noise", {"start": args.start_index, "end": args.start_index + args.samples},
                seed, args.batch_size, args.workers, args.samples,
            )
            if condition == "clean":
                loader = ((clean, clean, labels) for clean, _, labels in loader)
            results, correct, margins, errors = evaluate(
                model, adapter, loader, device, args.bootstrap,
                122000 + seed * 1000 + (condition == "clean") * 500,
            )
            all_results[str(seed)][condition] = {"stages": results, "validation": errors}
            for stage in STAGES:
                for name in CONDITIONS:
                    prefix = f"seed{seed}_{condition}_stage{stage}_{name}"
                    outcomes[f"{prefix}_correct"] = correct[stage][name]
                    outcomes[f"{prefix}_margin"] = margins[stage][name]
            print(f"Completed seed {seed} {condition}", flush=True)
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "adapter_paths": adapter_paths,
            "intervention": "frozen mixed Noise+Blur adapter after Block 6",
            "image_range": [args.start_index, args.start_index + args.samples],
            "uses_clean_counterpart_for_intervention": False,
            "vit_frozen": True,
            "adapters_frozen": True,
        },
        "results": all_results,
        "limitations": [
            "The image range was used in earlier mechanistic work and is not a pristine final test set.",
            "Patch/CLS reversion creates hybrid states; it measures conditional causal dependence, not a unique circuit or additive mediation.",
            "Clean images are an accuracy-preservation control; no paired clean activation enters the Noise intervention.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir, flush=True)


if __name__ == "__main__":
    main()
