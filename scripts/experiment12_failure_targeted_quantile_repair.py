import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment10_corruption_agnostic_sae_repair import CONDITIONS, DATASET_DIR, encode, load_fixed_sae, make_dataset
from scripts.experiment11_quantile_sae_repair import calibrate_quantiles, evaluate_condition


OUTPUT_ROOT = Path(__file__).parent.parent / "results" / "sae" / "experiment12_failure_targeted_repair"
EPSILON = 1e-8


class PairedCorruptionDataset(Dataset):
    def __init__(self, corruption, samples, start_index, seed):
        common = dict(dataset_dir=DATASET_DIR, max_samples=samples, start_index=start_index)
        self.clean = ImageNetDataset(**common)
        kwargs = {"corruption": corruption, "corruption_seed": seed}
        kwargs[f"{corruption}_severity"] = 4
        self.corrupt = ImageNetDataset(**common, **kwargs)

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, label = self.clean[index]
        corrupt, corrupt_label = self.corrupt[index]
        if label != corrupt_label:
            raise RuntimeError(f"Label mismatch at {index}")
        return clean, corrupt, label


def discover(model, sae, loader, device):
    deltas, margin_deltas, failure = [], [], []
    with torch.no_grad():
        for clean, corrupt, labels in tqdm(loader, desc="Failure-feature discovery"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(pixel_values=torch.cat([clean, corrupt]).to(device), output_hidden_states=True)
            clean_logits, corrupt_logits = outputs.logits.split(batch)
            clean_hidden, corrupt_hidden = outputs.hidden_states[-2].split(batch)
            clean_z = sae.encode(clean_hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)
            corrupt_z = sae.encode(corrupt_hidden[:, 1:].flatten(0, 1)).reshape(batch, 196, -1)
            deltas.append((corrupt_z.mean(1) - clean_z.mean(1)).cpu())
            clean_margin = classification_margin(clean_logits, labels)[1]
            corrupt_margin = classification_margin(corrupt_logits, labels)[1]
            margin_deltas.append((corrupt_margin - clean_margin).cpu())
            failure.append(((clean_logits.argmax(1) == labels) & (corrupt_logits.argmax(1) != labels)).cpu())
    delta = torch.cat(deltas).float()
    margin = torch.cat(margin_deltas).float()
    failures = torch.cat(failure)
    centered_delta = delta - delta.mean(0)
    centered_margin = margin - margin.mean()
    correlation = (centered_delta * centered_margin[:, None]).mean(0) / (
        centered_delta.std(0, unbiased=False) * centered_margin.std(unbiased=False) + EPSILON
    )
    mean_change = delta.mean(0)
    consistency = (delta > 0).float().mean(0)
    stable = ~failures
    failure_difference = delta[failures].mean(0) - delta[stable].mean(0) if failures.any() else torch.zeros_like(mean_change)
    standardized = mean_change / (delta.std(0, unbiased=False) + EPSILON)
    score = (-correlation).clamp_min(0) * standardized.clamp_min(0) * failure_difference.clamp_min(0) * consistency
    eligible = (mean_change > 0) & (correlation < 0) & (failure_difference > 0) & (consistency >= 0.55)
    score[~eligible] = 0
    order = torch.argsort(score, descending=True)
    return {
        "order": order.tolist(),
        "score": score.tolist(),
        "mean_change": mean_change.tolist(),
        "margin_correlation": correlation.tolist(),
        "failure_difference": failure_difference.tolist(),
        "consistency": consistency.tolist(),
        "failure_images": int(failures.sum()),
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 12: failure-targeted clean-SAE quantile repair")
    parser.add_argument("--discovery-samples", type=int, default=5000)
    parser.add_argument("--calibration-samples", type=int, default=2000)
    parser.add_argument("--validation-samples", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=5000)
    parser.add_argument("--discovery-start", type=int, default=25000)
    parser.add_argument("--calibration-start", type=int, default=30000)
    parser.add_argument("--validation-start", type=int, default=32000)
    parser.add_argument("--evaluation-start", type=int, default=35000)
    parser.add_argument("--top-k", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--quantile", type=float, default=0.99)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = load_fixed_sae("clean", device)

    discoveries = {}
    for corruption in ["blur", "noise"]:
        loader = DataLoader(PairedCorruptionDataset(corruption, args.discovery_samples, args.discovery_start, args.seed), batch_size=args.batch_size)
        discoveries[corruption] = discover(model, sae, loader, device)
    combined_score = np.asarray(discoveries["blur"]["score"]) + np.asarray(discoveries["noise"]["score"])
    combined_order = np.argsort(-combined_score).tolist()

    calibration_loader = DataLoader(make_dataset(("clean", None, 0), args.calibration_samples, args.calibration_start, args.seed), batch_size=args.batch_size)
    calibration = calibrate_quantiles(model, sae, calibration_loader, device, 4, [args.quantile], args.seed)
    thresholds = {key: value.to(device) for key, value in calibration["quantiles"].items()}
    expected_rate = calibration["expected_rates"][str(args.quantile)]
    candidates = {}
    for top_k in args.top_k:
        features = combined_order[:top_k]
        for alpha in args.alphas:
            candidates[f"top{top_k}_alpha{alpha:g}"] = {"quantile": args.quantile, "expected_rate": expected_rate, "alpha_max": alpha, "adaptive_width": 1.0, "adaptive": False, "features": features, "residual_intervention": True}
    validation = {}
    for condition in CONDITIONS:
        loader = DataLoader(make_dataset(condition, args.validation_samples, args.validation_start, args.seed), batch_size=args.batch_size)
        validation[condition[0]], _ = evaluate_condition(model, sae, loader, device, thresholds, candidates)
    corruptions = [name for name, _, _ in CONDITIONS if name != "clean"]
    clean_base = validation["clean"]["original_vit"]["accuracy"]
    def score(name):
        corrupt_accuracy = np.mean([validation[item][name]["accuracy"] for item in corruptions])
        clean_loss = max(0, clean_base - validation["clean"][name]["accuracy"])
        return corrupt_accuracy - 2 * clean_loss
    selected_name = max(candidates, key=score)
    selected = candidates[selected_name]
    generator = np.random.default_rng(args.seed)
    random_features = generator.choice(sae.latent_dim, len(selected["features"]), replace=False).tolist()
    configurations = {
        "targeted": selected,
        "targeted_confidence_safe": selected | {"confidence_safe": True},
        "random_control": selected | {"features": random_features},
    }
    evaluation = {}
    for condition in CONDITIONS:
        loader = DataLoader(make_dataset(condition, args.evaluation_samples, args.evaluation_start, args.seed), batch_size=args.batch_size)
        evaluation[condition[0]], _ = evaluate_condition(model, sae, loader, device, thresholds, configurations)
    summary = {"configuration": vars(args), "selected": {"name": selected_name} | selected, "discoveries": discoveries, "combined_top_features": combined_order[:max(args.top_k)], "validation": validation, "evaluation": evaluation}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"selected": summary["selected"], "evaluation": evaluation}, indent=2))
    print(f"Saved Experiment 12 to {output_dir}")


if __name__ == "__main__":
    main()
