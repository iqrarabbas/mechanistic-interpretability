import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.experiment24_classification_aware_residual_predictor import HiddenLinear
from scripts.experiment41_disjoint_gate_development import paired_comparison
from scripts.experiment68_unseen_online_corruptions import CORRUPTIONS, OnlineDataset


ROOT = Path(__file__).parent.parent
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment70_shared_transfer_subspace"
ADAPTER_ROOT = (
    ACTIVE_ROOT
    / "results/sae/experiment67_mixed_noise_blur_block6/"
    "full_3seed_noise_blur_identity0p2_v1"
)
BLOCK = 6


def dataset(condition, samples, start, corruption_seed):
    common = dict(dataset_dir=ROOT / "Dataset", max_samples=samples, start_index=start)
    if condition == "clean":
        return ImageNetDataset(**common)
    if condition == "noise":
        return ImageNetDataset(
            **common,
            corruption="noise",
            noise_severity=4,
            corruption_seed=corruption_seed,
        )
    if condition == "blur":
        return ImageNetDataset(**common, corruption="blur", blur_severity=4)
    return OnlineDataset(
        ROOT / "Dataset",
        samples,
        start,
        corruption_name=condition,
        severity=4,
        seed=corruption_seed,
    )


def load_adapter(seed, device):
    adapter = HiddenLinear().to(device)
    path = ADAPTER_ROOT / f"seed_{seed}" / "mixed_identity0p2.pt"
    adapter.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    return adapter.eval(), path


def residual_second_moment(model, adapter, loader, device):
    moment = torch.zeros(768, 768, dtype=torch.float64, device=device)
    count = 0
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="shared-subspace discovery", leave=False):
            hidden = model(
                pixel_values=images.to(device), output_hidden_states=True
            ).hidden_states[BLOCK][:, 1:]
            residual = adapter(hidden).reshape(-1, 768).double()
            moment += residual.T @ residual
            count += residual.shape[0]
    return moment / count


def shared_basis(noise_moment, blur_moment, pca_rank, shared_rank):
    _, noise_vectors = torch.linalg.eigh(noise_moment)
    _, blur_vectors = torch.linalg.eigh(blur_moment)
    noise_top = noise_vectors[:, -pca_rank:]
    blur_top = blur_vectors[:, -pca_rank:]
    left, singular_values, right_t = torch.linalg.svd(noise_top.T @ blur_top)
    vectors = noise_top @ left[:, :shared_rank] + blur_top @ right_t.T[:, :shared_rank]
    basis, _ = torch.linalg.qr(vectors, mode="reduced")
    return basis.float(), singular_values[:shared_rank].float()


def random_bases(shared, controls, seed):
    generator = torch.Generator(device=shared.device).manual_seed(seed)
    bases = []
    for _ in range(controls):
        candidate = torch.randn(
            shared.shape, generator=generator, device=shared.device
        )
        candidate -= shared @ (shared.T @ candidate)
        basis, _ = torch.linalg.qr(candidate, mode="reduced")
        bases.append(basis)
    return bases


def remove_projection(residual, basis):
    return residual - (residual @ basis) @ basis.T


def energy_matched_random_removal(residual, shared, random_basis):
    shared_component = (residual @ shared) @ shared.T
    random_component = (residual @ random_basis) @ random_basis.T
    shared_norm = shared_component.norm(dim=-1, keepdim=True)
    random_norm = random_component.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return residual - random_component * (shared_norm / random_norm)


def evaluate_condition(model, adapter, shared, random, loader, device, args, seed):
    arrays = {"baseline": [], "full": [], "shared_removed": []}
    arrays.update({f"random_{index}": [] for index in range(len(random))})
    energy = {"shared": 0.0, "total": 0.0, "tokens": 0}
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="causal transfer evaluation", leave=False):
            labels = labels.to(device)
            outputs = model(pixel_values=images.to(device), output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            residual = adapter(hidden[:, 1:])
            shared_component = (residual @ shared) @ shared.T
            energy["shared"] += float(shared_component.square().sum())
            energy["total"] += float(residual.square().sum())
            energy["tokens"] += residual.shape[0] * residual.shape[1]
            arrays["baseline"].extend((outputs.logits.argmax(1) == labels).cpu().tolist())

            variants = {"full": residual, "shared_removed": residual - shared_component}
            variants.update(
                {
                    f"random_{index}": energy_matched_random_removal(
                        residual, shared, basis
                    )
                    for index, basis in enumerate(random)
                }
            )
            for name, corrected_residual in variants.items():
                candidate = torch.cat(
                    [hidden[:, :1], hidden[:, 1:] + corrected_residual], dim=1
                )
                logits = downstream_from_layer(model, candidate, BLOCK - 1)
                arrays[name].extend((logits.argmax(1) == labels).cpu().tolist())

    arrays = {name: np.asarray(values, dtype=bool) for name, values in arrays.items()}
    comparisons = {
        name: paired_comparison(arrays["full"], values, seed + index, args.bootstrap)
        for index, (name, values) in enumerate(arrays.items())
        if name not in {"baseline", "full"}
    }
    baseline_full = paired_comparison(
        arrays["baseline"], arrays["full"], seed + 900, args.bootstrap
    )
    harmful_cost = -comparisons["shared_removed"]["accuracy_difference"]
    random_costs = np.asarray(
        [-comparisons[f"random_{index}"]["accuracy_difference"] for index in range(len(random))]
    )
    summary = {
        "baseline_to_full": baseline_full,
        "shared_removal": comparisons["shared_removed"],
        "shared_removal_cost": harmful_cost,
        "random_removal_costs": random_costs.tolist(),
        "random_mean_removal_cost": float(random_costs.mean()),
        "shared_exceeds_random_controls": int(np.sum(harmful_cost > random_costs)),
        "empirical_one_sided_pvalue": float(
            (1 + np.sum(random_costs >= harmful_cost)) / (len(random_costs) + 1)
        ),
        "shared_residual_energy_fraction": energy["shared"] / energy["total"],
    }
    return summary, arrays


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--discovery-start", type=int, default=36000)
    parser.add_argument("--discovery-samples", type=int, default=1000)
    parser.add_argument("--evaluation-start", type=int, default=49000)
    parser.add_argument("--evaluation-samples", type=int, default=1000)
    parser.add_argument("--pca-rank", type=int, default=64)
    parser.add_argument("--shared-rank", type=int, default=16)
    parser.add_argument("--random-controls", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=2070)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--max-conditions", type=int)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if args.discovery_start + args.discovery_samples > args.evaluation_start:
        raise ValueError("Discovery and evaluation intervals must be disjoint")
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    conditions = ["clean", "noise", "blur", *CORRUPTIONS]
    if args.max_conditions:
        conditions = conditions[: args.max_conditions]

    results = {}
    saved = {}
    adapter_paths = {}
    for adapter_seed in args.seeds:
        adapter, adapter_path = load_adapter(adapter_seed, device)
        adapter_paths[str(adapter_seed)] = str(adapter_path)
        moments = {}
        for condition in ["noise", "blur"]:
            loader = DataLoader(
                dataset(
                    condition,
                    args.discovery_samples,
                    args.discovery_start,
                    args.corruption_seed,
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
            moments[condition] = residual_second_moment(
                model, adapter, loader, device
            )
        shared, similarities = shared_basis(
            moments["noise"], moments["blur"], args.pca_rank, args.shared_rank
        )
        random = random_bases(shared, args.random_controls, args.corruption_seed + adapter_seed)
        results[str(adapter_seed)] = {
            "principal_angle_cosines": similarities.cpu().tolist(),
            "conditions": {},
        }
        for condition_index, condition in enumerate(conditions):
            loader = DataLoader(
                dataset(
                    condition,
                    args.evaluation_samples,
                    args.evaluation_start,
                    args.corruption_seed,
                ),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
            )
            condition_result, arrays = evaluate_condition(
                model,
                adapter,
                shared,
                random,
                loader,
                device,
                args,
                args.corruption_seed + adapter_seed * 1000 + condition_index * 50,
            )
            results[str(adapter_seed)]["conditions"][condition] = condition_result
            for name, values in arrays.items():
                saved[f"seed{adapter_seed}_{condition}_{name}"] = values

    summary = {
        "configuration": vars(args)
        | {
            "model": BASE_MODEL,
            "block": BLOCK,
            "adapter_paths": adapter_paths,
            "vit_frozen": True,
            "adapters_frozen": True,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "status": "disjoint ImageNet-validation mechanism development",
            "random_control": "orthogonal random rank-matched removal scaled per token to equal shared-component norm",
        },
        "results": results,
        "limitations": [
            "ImageNet validation images in these ranges appeared in earlier diagnostics; discovery and evaluation are disjoint within this experiment, which is mechanistic development rather than final independent confirmation.",
            "Unseen online corruption implementations are controlled approximations, not official ImageNet-C.",
        ],
    }
    np.savez_compressed(output_dir / "paired_outcomes.npz", **saved)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
