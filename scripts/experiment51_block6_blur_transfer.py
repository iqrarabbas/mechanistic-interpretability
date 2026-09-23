import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment39_disjoint_adapter_training import DEFAULT_MANIFEST, locked_seed_splits
from scripts.experiment41_disjoint_gate_development import paired_comparison


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment51_block6_blur_transfer"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment50_block6_clean_preservation"
    / "full_3seed_identity_sweep_v1"
)
BLOCK = 6


def make_loader(split, seed, batch_size, workers, max_samples):
    samples = split["end"] - split["start"]
    if max_samples is not None:
        samples = min(samples, max_samples)
    dataset = PairedCorruptionDataset("blur", samples, split["start"], seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )


def load_adapter(path, device):
    adapter = HiddenLinear().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval()


def evaluate(model, adapters, loader, device, alpha, seed, bootstrap_repetitions):
    arrays = {"baseline_clean": [], "baseline_blur": []}
    for name in adapters:
        arrays[f"{name}_clean"] = []
        arrays[f"{name}_blur"] = []
    with torch.no_grad():
        for clean, blur, labels in tqdm(loader, desc="Frozen Block-6 Blur transfer"):
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, blur]).to(device), output_hidden_states=True
            )
            batch = clean.shape[0]
            clean_logits, blur_logits = outputs.logits.split(batch)
            arrays["baseline_clean"].extend((clean_logits.argmax(1) == labels).cpu().tolist())
            arrays["baseline_blur"].extend((blur_logits.argmax(1) == labels).cpu().tolist())
            clean_hidden, blur_hidden = outputs.hidden_states[BLOCK].split(batch)
            for name, adapter in adapters.items():
                for condition, hidden in (("clean", clean_hidden), ("blur", blur_hidden)):
                    patches = hidden[:, 1:]
                    candidate = torch.cat(
                        [hidden[:, :1], patches + alpha * adapter(patches)], dim=1
                    )
                    logits = downstream_from_layer(model, candidate, BLOCK - 1)
                    arrays[f"{name}_{condition}"].extend(
                        (logits.argmax(1) == labels).cpu().tolist()
                    )
    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    results = {
        "baseline_clean_accuracy": float(arrays["baseline_clean"].mean()),
        "baseline_blur4_accuracy": float(arrays["baseline_blur"].mean()),
        "variants": {},
    }
    for offset, name in enumerate(adapters):
        results["variants"][name] = {
            "clean_vs_baseline": paired_comparison(
                arrays["baseline_clean"], arrays[f"{name}_clean"],
                seed + offset, bootstrap_repetitions,
            ),
            "blur4_vs_baseline": paired_comparison(
                arrays["baseline_blur"], arrays[f"{name}_blur"],
                seed + 100 + offset, bootstrap_repetitions,
            ),
        }
    return results, arrays


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 51: frozen Noise-trained Block-6 adapter transfer to Blur-4"
    )
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-protocol", default="proposed_protocol")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--selected-variant", default="identity_0p2")
    parser.add_argument("--include-no-identity-control", action="store_true")
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
    print(f"Frozen selected variant: {args.selected_variant}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    validation = {}
    outcomes = {}

    for seed in args.seeds:
        names = [args.selected_variant]
        if args.include_no_identity_control:
            names.append("identity_0")
        adapters = {
            name: load_adapter(args.source_run / f"seed_{seed}" / f"{name}.pt", device)
            for name in names
        }
        loader = make_loader(
            splits[seed]["validation"], seed, args.image_batch_size,
            args.num_workers, args.max_samples,
        )
        result, arrays = evaluate(
            model, adapters, loader, device, args.alpha,
            seed + 5100, args.bootstrap_repetitions,
        )
        validation[f"seed_{seed}"] = result
        for name, values in arrays.items():
            outcomes[f"seed_{seed}_{name}"] = values

    aggregate = {}
    variant_names = list(validation[f"seed_{args.seeds[0]}"]["variants"])
    for name in variant_names:
        blur = [
            validation[f"seed_{seed}"]["variants"][name]["blur4_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        clean = [
            validation[f"seed_{seed}"]["variants"][name]["clean_vs_baseline"]["accuracy_difference"]
            for seed in args.seeds
        ]
        aggregate[name] = {
            "blur4_gains_by_seed": blur,
            "mean_blur4_gain": float(np.mean(blur)),
            "clean_gains_by_seed": clean,
            "mean_clean_gain": float(np.mean(clean)),
        }
    summary = {
        "configuration": vars(args) | {
            "source_run": str(args.source_run.resolve()),
            "split_manifest": str(args.split_manifest.resolve()),
            "device": str(device),
            "block": BLOCK,
            "model": BASE_MODEL,
            "model_frozen": True,
            "adapter_frozen": True,
            "adapter_training_corruption": "Noise-4",
            "evaluation_corruption": "Blur-4",
            "blur_used_for_selection": False,
            "imageNetV2_accessed": False,
            "status": "development transfer evaluation",
        },
        "development_splits": {str(seed): splits[seed]["validation"] for seed in args.seeds},
        "validation": validation,
        "aggregate": aggregate,
    }
    np.savez_compressed(output_dir / "paired_validation_outcomes.npz", **outcomes)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(aggregate, indent=2))
    print(f"Saved Experiment 51 to {output_dir}")


if __name__ == "__main__":
    main()
