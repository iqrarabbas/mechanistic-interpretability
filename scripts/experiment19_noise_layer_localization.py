import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import binomtest
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment2_sae_strength_vs_classification import classification_margin
from scripts.experiment12_failure_targeted_quantile_repair import PairedCorruptionDataset


PROJECT_ROOT = Path(__file__).parent.parent
OUTPUT_ROOT = PROJECT_ROOT / "results" / "sae" / "experiment19_noise_layer_localization"


def downstream_from_layer(model, hidden, layer_index):
    for index in range(layer_index + 1, len(model.vit.layers)):
        hidden = model.vit.layers[index](hidden, attention_mask=None)
    hidden = model.vit.layernorm(hidden)
    return model.classifier(hidden[:, 0])


def summarize(original_correct, correct, margin_change, true_logit_change):
    recovered = int((~original_correct & correct).sum())
    damaged = int((original_correct & ~correct).sum())
    return {
        "accuracy": float(correct.mean()),
        "accuracy_gain": float((correct.astype(float) - original_correct.astype(float)).mean()),
        "mean_margin_change": float(margin_change.mean()),
        "mean_true_logit_change": float(true_logit_change.mean()),
        "predictions_recovered": recovered,
        "originally_correct_damaged": damaged,
        "mcnemar_exact_pvalue": float(binomtest(recovered, recovered + damaged, 0.5).pvalue) if recovered + damaged else 1.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Experiment 19: causal Noise localization across ViT layers")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--start-index", type=int, default=40000)
    parser.add_argument("--layers", type=int, nargs="+", default=[0, 2, 5, 8, 10, 11])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    output_dir = OUTPUT_ROOT / args.run_name
    output_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    loader = DataLoader(
        PairedCorruptionDataset("noise", args.samples, args.start_index, args.seed),
        batch_size=args.batch_size,
        shuffle=False,
    )
    definitions = {}
    for layer in args.layers:
        for alpha in args.alphas:
            definitions[f"layer{layer + 1}_full_alpha{alpha:g}"] = (layer, alpha, "full")
        definitions[f"layer{layer + 1}_patch_only"] = (layer, 1.0, "patch")
        definitions[f"layer{layer + 1}_cls_only"] = (layer, 1.0, "cls")
        definitions[f"layer{layer + 1}_reverse"] = (layer, -1.0, "full")
        definitions[f"layer{layer + 1}_shuffled"] = (layer, 1.0, "shuffled")
    stores = {
        name: {key: [] for key in ["correct", "margin_change", "true_logit_change"]}
        for name in definitions
    }
    original_correct, original_margin, original_true_logit = [], [], []
    representation = {
        layer: {"patch_cosine": [], "relative_l2": [], "cls_cosine": []}
        for layer in args.layers
    }
    with torch.no_grad():
        for clean, noise, labels in tqdm(loader, desc="Layer localization"):
            batch = clean.shape[0]
            labels = labels.to(device)
            outputs = model(
                pixel_values=torch.cat([clean, noise]).to(device), output_hidden_states=True
            )
            clean_logits, noise_logits = outputs.logits.split(batch)
            noise_true, noise_margin = classification_margin(noise_logits, labels)
            original_correct.extend((noise_logits.argmax(1) == labels).cpu().tolist())
            original_margin.extend(noise_margin.cpu().tolist())
            original_true_logit.extend(noise_true.cpu().tolist())
            for layer in args.layers:
                clean_hidden, noise_hidden = outputs.hidden_states[layer + 1].split(batch)
                delta = clean_hidden - noise_hidden
                patch_cosine = F.cosine_similarity(clean_hidden[:, 1:], noise_hidden[:, 1:], dim=-1).mean(1)
                relative_l2 = delta[:, 1:].flatten(1).norm(dim=1) / clean_hidden[:, 1:].flatten(1).norm(dim=1).clamp_min(1e-8)
                cls_cosine = F.cosine_similarity(clean_hidden[:, 0], noise_hidden[:, 0], dim=-1)
                representation[layer]["patch_cosine"].extend(patch_cosine.cpu().tolist())
                representation[layer]["relative_l2"].extend(relative_l2.cpu().tolist())
                representation[layer]["cls_cosine"].extend(cls_cosine.cpu().tolist())
                for name, (definition_layer, alpha, scope) in definitions.items():
                    if definition_layer != layer:
                        continue
                    applied = delta
                    if scope == "patch":
                        applied = torch.cat([torch.zeros_like(delta[:, :1]), delta[:, 1:]], dim=1)
                    elif scope == "cls":
                        applied = torch.cat([delta[:, :1], torch.zeros_like(delta[:, 1:])], dim=1)
                    elif scope == "shuffled":
                        applied = delta.roll(1, dims=0)
                    logits = downstream_from_layer(model, noise_hidden + alpha * applied, layer)
                    true_logit, margin = classification_margin(logits, labels)
                    stores[name]["correct"].extend((logits.argmax(1) == labels).cpu().tolist())
                    stores[name]["margin_change"].extend((margin - noise_margin).cpu().tolist())
                    stores[name]["true_logit_change"].extend((true_logit - noise_true).cpu().tolist())
    original_correct = np.asarray(original_correct, dtype=bool)
    original_margin = np.asarray(original_margin)
    original_true_logit = np.asarray(original_true_logit)
    results = {
        "original_noise4": {
            "accuracy": float(original_correct.mean()),
            "mean_margin": float(original_margin.mean()),
            "mean_true_logit": float(original_true_logit.mean()),
        }
    }
    for name, values in stores.items():
        results[name] = summarize(
            original_correct,
            np.asarray(values["correct"], dtype=bool),
            np.asarray(values["margin_change"]),
            np.asarray(values["true_logit_change"]),
        )
    representation_summary = {
        f"layer{layer + 1}": {
            key: float(np.mean(values)) for key, values in metrics.items()
        }
        for layer, metrics in representation.items()
    }
    full_restore = [
        {"layer": layer + 1} | results[f"layer{layer + 1}_full_alpha1"]
        for layer in args.layers
    ]
    partial_alphas = [alpha for alpha in args.alphas if 0 < alpha < 1]
    diagnostic_alpha = min(partial_alphas) if partial_alphas else min(args.alphas)
    partial_restore = [
        {"layer": layer + 1} | results[f"layer{layer + 1}_full_alpha{diagnostic_alpha:g}"]
        for layer in args.layers
    ]
    best = max(partial_restore, key=lambda row: (row["accuracy_gain"], row["mean_margin_change"]))
    summary = {
        "configuration": vars(args) | {
            "device": str(device),
            "status": "oracle paired analysis only; not an inference-time method",
        },
        "representation_change": representation_summary,
        "results": results,
        "diagnostic_partial_alpha": diagnostic_alpha,
        "partial_clean_restoration_by_layer": partial_restore,
        "full_clean_restoration_by_layer": full_restore,
        "best_layer_by_partial_recovery": best,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "original": results["original_noise4"],
        "representation_change": representation_summary,
        "diagnostic_partial_alpha": diagnostic_alpha,
        "partial_clean_restoration_by_layer": partial_restore,
        "full_clean_restoration_by_layer": full_restore,
        "best_layer": best,
    }, indent=2))
    print(f"Saved Experiment 19 to {output_dir}")


if __name__ == "__main__":
    main()
