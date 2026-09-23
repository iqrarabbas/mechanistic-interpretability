import argparse
import io
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoImageProcessor, ViTForImageClassification

from corruption.gaussian_blur import apply_gaussian_blur
from corruption.gaussian_noise import apply_gaussian_noise
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment41_disjoint_gate_development import paired_comparison


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment60_block6_imageneta_ood"
DEFAULT_IMAGE_ROOT = PROJECT_ROOT / "external_data" / "imagenet-a-hf"
DEFAULT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "sae"
    / "experiment50_block6_clean_preservation"
    / "full_3seed_identity_sweep_v1"
)
DEFAULT_META = PROJECT_ROOT / "Dataset" / "ILSVRC2012_devkit_t12" / "data" / "meta.mat"
BLOCK = 6


def imagenet_wnid_to_model_id(meta_path):
    synsets = loadmat(meta_path)["synsets"]
    wnids = []
    for index in range(len(synsets)):
        ilsvrc_id = int(synsets[index][0][0][0][0])
        if 1 <= ilsvrc_id <= 1000:
            wnids.append(synsets[index][0][1][0])
    return {wnid: model_id for model_id, wnid in enumerate(sorted(wnids))}


def imageneta_wnids(dataset_info_path):
    information = json.loads(dataset_info_path.read_text())
    dataset = next(iter(information.values()))
    return dataset["features"]["label"]["names"]


class ImageNetAOODDataset(Dataset):
    def __init__(self, root, meta_path, corruption_seed, max_samples=None):
        root = Path(root)
        files = sorted((root / "data").glob("*.parquet"))
        if len(files) != 2:
            raise FileNotFoundError(f"Expected two ImageNet-A Parquet shards under {root / 'data'}")
        self.table = pq.read_table(files, columns=["image", "label"]).combine_chunks()
        self.length = len(self.table) if max_samples is None else min(max_samples, len(self.table))
        self.processor = AutoImageProcessor.from_pretrained(BASE_MODEL)
        self.corruption_seed = corruption_seed

        wnid_to_model_id = imagenet_wnid_to_model_id(meta_path)
        local_wnids = imageneta_wnids(root / "dataset_infos.json")
        self.local_to_model = [wnid_to_model_id[wnid] for wnid in local_wnids]
        self.allowed_model_ids = torch.tensor(self.local_to_model, dtype=torch.long)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        row = self.table.slice(index, 1).to_pylist()[0]
        image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
        label = self.local_to_model[row["label"]]
        noise = apply_gaussian_noise(image, severity=4, seed=self.corruption_seed + index)
        blur = apply_gaussian_blur(image, severity=4)

        pixels = self.processor(images=[image, noise, blur], return_tensors="pt")["pixel_values"]
        return pixels[0], pixels[1], pixels[2], label


def load_adapter(path, device):
    adapter = HiddenLinear().to(device)
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    adapter.requires_grad_(False)
    return adapter.eval()


def predictions(logits, allowed_model_ids):
    full = logits.argmax(1)
    restricted = allowed_model_ids[logits[:, allowed_model_ids].argmax(1)]
    return full, restricted


def evaluate(model, adapters, loader, allowed_model_ids, device, alpha, bootstrap_repetitions, seed):
    conditions = ["native", "noise4", "blur4"]
    protocols = ["full_1k", "masked_200"]
    arrays = {
        f"baseline_{condition}_{protocol}": []
        for condition in conditions
        for protocol in protocols
    }
    for adapter_name in adapters:
        for condition in conditions:
            for protocol in protocols:
                arrays[f"{adapter_name}_{condition}_{protocol}"] = []

    with torch.no_grad():
        for native, noise, blur, labels in tqdm(loader, desc="ImageNet-A OOD evaluation"):
            batch = native.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([native, noise, blur]).to(device),
                output_hidden_states=True,
            )
            logits_by_condition = outputs.logits.split(batch)
            hidden_by_condition = outputs.hidden_states[BLOCK].split(batch)

            for condition, logits in zip(conditions, logits_by_condition):
                full, restricted = predictions(logits, allowed_model_ids)
                arrays[f"baseline_{condition}_full_1k"].extend((full == labels).cpu().tolist())
                arrays[f"baseline_{condition}_masked_200"].extend(
                    (restricted == labels).cpu().tolist()
                )

            for adapter_name, adapter in adapters.items():
                for condition, hidden in zip(conditions, hidden_by_condition):
                    patches = hidden[:, 1:]
                    corrected = torch.cat(
                        [hidden[:, :1], patches + alpha * adapter(patches)], dim=1
                    )
                    logits = downstream_from_layer(model, corrected, BLOCK - 1)
                    full, restricted = predictions(logits, allowed_model_ids)
                    arrays[f"{adapter_name}_{condition}_full_1k"].extend(
                        (full == labels).cpu().tolist()
                    )
                    arrays[f"{adapter_name}_{condition}_masked_200"].extend(
                        (restricted == labels).cpu().tolist()
                    )

    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    results = {protocol: {} for protocol in protocols}
    for protocol_index, protocol in enumerate(protocols):
        for condition_index, condition in enumerate(conditions):
            reference = arrays[f"baseline_{condition}_{protocol}"]
            condition_results = {
                "baseline_accuracy": float(reference.mean()),
                "adapters": {},
            }
            for adapter_index, adapter_name in enumerate(adapters):
                candidate = arrays[f"{adapter_name}_{condition}_{protocol}"]
                condition_results["adapters"][adapter_name] = paired_comparison(
                    reference,
                    candidate,
                    seed + protocol_index * 1000 + condition_index * 100 + adapter_index,
                    bootstrap_repetitions,
                )
            gains = [
                condition_results["adapters"][name]["accuracy_difference"]
                for name in adapters
            ]
            condition_results["aggregate"] = {
                "gains_by_seed": gains,
                "mean_gain": float(np.mean(gains)),
                "all_seed_gains_positive": bool(np.all(np.asarray(gains) > 0)),
            }
            results[protocol][condition] = condition_results
    return results, arrays


def main():
    parser = argparse.ArgumentParser(
        description="Experiment 60: frozen Block-6 adapter evaluation on ImageNet-A OOD data"
    )
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT)
    parser.add_argument("--meta-path", type=Path, default=DEFAULT_META)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--selected-variant", default="identity_0p2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--corruption-seed", type=int, default=2060)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()

    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    source_summary = json.loads((args.source_run / "summary.json").read_text())
    if source_summary["selected_variant"] != args.selected_variant:
        raise ValueError("Requested variant does not match Experiment 50's frozen selection")

    dataset = ImageNetAOODDataset(
        args.image_root, args.meta_path, args.corruption_seed, args.max_samples
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ImageNet-A samples: {len(dataset)}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    adapters = {
        f"seed_{seed}": load_adapter(
            args.source_run / f"seed_{seed}" / f"{args.selected_variant}.pt", device
        )
        for seed in args.seeds
    }
    results, arrays = evaluate(
        model,
        adapters,
        loader,
        dataset.allowed_model_ids.to(device),
        device,
        args.alpha,
        args.bootstrap_repetitions,
        args.corruption_seed + 6000,
    )
    np.savez_compressed(output_dir / "paired_outcomes.npz", **arrays)

    summary = {
        "configuration": vars(args)
        | {
            "image_root": str(args.image_root.resolve()),
            "meta_path": str(args.meta_path.resolve()),
            "source_run": str(args.source_run.resolve()),
            "device": str(device),
            "model": BASE_MODEL,
            "vit_frozen": True,
            "adapters_frozen": True,
            "adapter_training_corruption": "Noise-4",
            "adapter_training_dataset": "ImageNet-1K validation development splits",
            "evaluation_dataset": "ImageNet-A",
            "evaluation_samples": len(dataset),
            "evaluation_used_for_selection_or_tuning": False,
            "deployment_uses_clean_counterpart": False,
            "standard_primary_protocol": "masked_200",
            "secondary_protocol": "full_1k",
            "status": "independent frozen OOD evaluation",
        },
        "results": results,
        "limitations": [
            "ImageNet-A contains 200 ImageNet classes and naturally difficult examples.",
            "Masked-200 accuracy is the standard primary protocol; full-1K accuracy is also reported.",
            "No hyperparameter, adapter, seed, or threshold may be selected using these results.",
        ],
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(results, indent=2))
    print(f"Saved Experiment 60 to {output_dir}")


if __name__ == "__main__":
    main()
