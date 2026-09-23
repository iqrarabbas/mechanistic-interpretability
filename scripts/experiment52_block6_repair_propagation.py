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
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment49_sae_jaccard_controls import bootstrap_difference


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment52_block6_repair_propagation"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment50_block6_clean_preservation"
    / "full_3seed_identity_sweep_v1"
)
INTERVENTION_BLOCK = 6
STAGES = list(range(6, 13))
EPSILON = 1e-8


def make_loader(corruption, split, seed, batch_size, workers, max_samples):
    samples = split["end"] - split["start"]
    if max_samples is not None:
        samples = min(samples, max_samples)
    return DataLoader(
        PairedCorruptionDataset(corruption, samples, split["start"], seed),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )


def load_adapter(path, device):
    adapter = HiddenLinear().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval()


def representation_metrics(clean_hidden, candidate_hidden):
    clean_patches = clean_hidden[:, 1:]
    candidate_patches = candidate_hidden[:, 1:]
    patch_cosine = F.cosine_similarity(
        clean_patches, candidate_patches, dim=-1, eps=EPSILON
    ).mean(dim=1)
    patch_relative_l2 = (
        (candidate_patches - clean_patches).flatten(1).norm(dim=1)
        / clean_patches.flatten(1).norm(dim=1).clamp_min(EPSILON)
    )
    cls_cosine = F.cosine_similarity(
        clean_hidden[:, 0], candidate_hidden[:, 0], dim=-1, eps=EPSILON
    )
    cls_relative_l2 = (
        (candidate_hidden[:, 0] - clean_hidden[:, 0]).norm(dim=1)
        / clean_hidden[:, 0].norm(dim=1).clamp_min(EPSILON)
    )
    return {
        "patch_cosine": patch_cosine,
        "patch_relative_l2": patch_relative_l2,
        "cls_cosine": cls_cosine,
        "cls_relative_l2": cls_relative_l2,
    }


def logit_lens(model, hidden, labels):
    logits = model.classifier(model.vit.layernorm(hidden)[:, 0])
    true_logit, margin = classification_margin(logits, labels)
    return logits, true_logit, margin


def append_values(store, values):
    for name, tensor in values.items():
        store[name].extend(tensor.detach().cpu().tolist())


def evaluate_seed(model, adapter, loader, device, alpha):
    metric_names = [
        "patch_cosine",
        "patch_relative_l2",
        "cls_cosine",
        "cls_relative_l2",
        "true_logit",
        "margin",
    ]
    stores = {
        stage: {
            condition: {metric: [] for metric in metric_names}
            for condition in ["clean", "noise", "corrected"]
        }
        for stage in STAGES
    }
    final_correct = {"clean": [], "noise": [], "corrected": []}
    replay_max_absolute_logit_error = 0.0

    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Block-6 repair propagation"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            clean_hidden6, noise_hidden6 = outputs.hidden_states[INTERVENTION_BLOCK].split(batch)
            corrected = torch.cat(
                [
                    noise_hidden6[:, :1],
                    noise_hidden6[:, 1:] + alpha * adapter(noise_hidden6[:, 1:]),
                ],
                dim=1,
            )
            noise_replayed = noise_hidden6.clone()

            for stage in STAGES:
                if stage > INTERVENTION_BLOCK:
                    corrected = model.vit.layers[stage - 1](corrected, attention_mask=None)
                    noise_replayed = model.vit.layers[stage - 1](
                        noise_replayed, attention_mask=None
                    )
                clean_hidden, noise_hidden = outputs.hidden_states[stage].split(batch)
                for condition, hidden in [
                    ("clean", clean_hidden),
                    ("noise", noise_hidden),
                    ("corrected", corrected),
                ]:
                    representation = representation_metrics(clean_hidden, hidden)
                    _, true_logit, margin = logit_lens(model, hidden, labels)
                    append_values(
                        stores[stage][condition],
                        representation | {"true_logit": true_logit, "margin": margin},
                    )

            corrected_logits, _, _ = logit_lens(model, corrected, labels)
            replayed_noise_logits, _, _ = logit_lens(model, noise_replayed, labels)
            replay_max_absolute_logit_error = max(
                replay_max_absolute_logit_error,
                float((replayed_noise_logits - noise_logits).abs().max().item()),
            )
            final_correct["clean"].extend((clean_logits.argmax(1) == labels).cpu().tolist())
            final_correct["noise"].extend((noise_logits.argmax(1) == labels).cpu().tolist())
            final_correct["corrected"].extend(
                (corrected_logits.argmax(1) == labels).cpu().tolist()
            )
    stores = {
        stage: {
            condition: {
                metric: np.asarray(values, dtype=np.float64)
                for metric, values in metrics.items()
            }
            for condition, metrics in conditions.items()
        }
        for stage, conditions in stores.items()
    }
    final_correct = {
        name: np.asarray(values, dtype=bool) for name, values in final_correct.items()
    }
    if replay_max_absolute_logit_error > 1e-4:
        raise RuntimeError(
            "Manual Block-6 downstream replay does not match normal ViT inference: "
            f"max absolute logit error={replay_max_absolute_logit_error:.8g}"
        )
    return stores, final_correct, replay_max_absolute_logit_error


def save_outcomes(path, stores, final_correct):
    arrays = {}
    for stage, conditions in stores.items():
        for condition, metrics in conditions.items():
            for metric, values in metrics.items():
                arrays[f"block_{stage}__{condition}__{metric}"] = values
    for condition, values in final_correct.items():
        arrays[f"final__{condition}__correct"] = values
    np.savez_compressed(path, **arrays)


def summarize_seed(stores, final_correct, seed, bootstrap_repetitions):
    stages = {}
    for stage in STAGES:
        conditions = stores[stage]
        aggregate = {
            condition: {
                metric: float(values.mean()) for metric, values in metrics.items()
            }
            for condition, metrics in conditions.items()
        }
        tests = {
            "patch_cosine_gain": bootstrap_difference(
                conditions["corrected"]["patch_cosine"],
                conditions["noise"]["patch_cosine"],
                seed + stage,
                bootstrap_repetitions,
            ),
            "patch_relative_l2_reduction": bootstrap_difference(
                conditions["noise"]["patch_relative_l2"],
                conditions["corrected"]["patch_relative_l2"],
                seed + 20 + stage,
                bootstrap_repetitions,
            ),
            "cls_cosine_gain": bootstrap_difference(
                conditions["corrected"]["cls_cosine"],
                conditions["noise"]["cls_cosine"],
                seed + 40 + stage,
                bootstrap_repetitions,
            ),
            "cls_relative_l2_reduction": bootstrap_difference(
                conditions["noise"]["cls_relative_l2"],
                conditions["corrected"]["cls_relative_l2"],
                seed + 60 + stage,
                bootstrap_repetitions,
            ),
            "margin_gain": bootstrap_difference(
                conditions["corrected"]["margin"],
                conditions["noise"]["margin"],
                seed + 80 + stage,
                bootstrap_repetitions,
            ),
            "true_logit_gain": bootstrap_difference(
                conditions["corrected"]["true_logit"],
                conditions["noise"]["true_logit"],
                seed + 100 + stage,
                bootstrap_repetitions,
            ),
        }
        stages[str(stage)] = {"means": aggregate, "corrected_minus_noise": tests}
    final = paired_comparison(
        final_correct["noise"], final_correct["corrected"],
        seed + 5200, bootstrap_repetitions,
    )
    return {
        "stages": stages,
        "final_classification": final,
        "clean_accuracy": float(final_correct["clean"].mean()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 52: track frozen Block-6 repair through downstream ViT blocks"
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--selected-variant", default="identity_0p2")
    parser.add_argument("--corruption", choices=["noise", "blur"], default="noise")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--image-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    source_summary = json.loads((args.source_run / "summary.json").read_text())
    if source_summary["selected_variant"] != args.selected_variant:
        raise ValueError("Requested variant does not match Experiment 50's frozen selection.")
    splits = locked_seed_splits(args.split_manifest, args.split_protocol, args.seeds)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Frozen adapter: Block {INTERVENTION_BLOCK}, {args.selected_variant}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    results = {}

    for seed in args.seeds:
        adapter = load_adapter(
            args.source_run / f"seed_{seed}" / f"{args.selected_variant}.pt", device
        )
        loader = make_loader(
            args.corruption, splits[seed]["validation"], seed, args.image_batch_size,
            args.num_workers, args.max_samples,
        )
        stores, final_correct, replay_error = evaluate_seed(
            model, adapter, loader, device, args.alpha
        )
        save_outcomes(output_dir / f"seed_{seed}_paired_outcomes.npz", stores, final_correct)
        results[f"seed_{seed}"] = summarize_seed(
            stores, final_correct, seed, args.bootstrap_repetitions
        )
        results[f"seed_{seed}"]["manual_replay_max_absolute_logit_error"] = replay_error
        (output_dir / f"seed_{seed}_summary.json").write_text(
            json.dumps(results[f"seed_{seed}"], indent=2)
        )

    trajectory = {}
    for stage in STAGES:
        trajectory[str(stage)] = {}
        for metric in [
            "patch_cosine_gain",
            "patch_relative_l2_reduction",
            "cls_cosine_gain",
            "cls_relative_l2_reduction",
            "margin_gain",
            "true_logit_gain",
        ]:
            values = [
                results[f"seed_{seed}"]["stages"][str(stage)]["corrected_minus_noise"][metric]["mean_paired_difference"]
                for seed in args.seeds
            ]
            trajectory[str(stage)][metric] = {
                "by_seed": values,
                "mean": float(np.mean(values)),
            }
    final_gains = [
        results[f"seed_{seed}"]["final_classification"]["accuracy_difference"]
        for seed in args.seeds
    ]
    summary = {
        "configuration": vars(args) | {
            "source_run": str(args.source_run.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "model_frozen": True,
            "adapter_frozen": True,
            "intervention_block": INTERVENTION_BLOCK,
            "tracked_blocks": STAGES,
            "logit_lens_caveat": "Intermediate classifier margins use the final ViT layernorm/classifier and are diagnostic, not native intermediate heads.",
            "imageNetV2_accessed": False,
            "status": "development mechanistic analysis",
        },
        "development_splits": {str(seed): splits[seed]["validation"] for seed in args.seeds},
        "per_seed": results,
        "mean_trajectory": trajectory,
        "final_corruption4_gains_by_seed": final_gains,
        "mean_final_corruption4_gain": float(np.mean(final_gains)),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "mean_trajectory": trajectory,
        "corruption": args.corruption,
        "final_corruption4_gains_by_seed": final_gains,
        "mean_final_corruption4_gain": float(np.mean(final_gains)),
    }, indent=2))
    print(f"Saved Experiment 52 to {output_dir}")


if __name__ == "__main__":
    main()
