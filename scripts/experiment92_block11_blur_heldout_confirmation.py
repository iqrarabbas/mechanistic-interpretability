import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import ViTForImageClassification

from interpretability.sae import BatchTopKSAE
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
import scripts.experiment89_sae_affine_clean_prior as affine
import scripts.experiment90_sae_affine_heldout_confirmation as confirmation


ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment92_block11_blur_heldout_confirmation"
SAE_CHECKPOINT = ACTIVE_ROOT / "results/sae/experiment91_batchtopk_layer_screen/block11_seed0_1000train_200val_3epoch_v1/batchtopk_lambda_1em03/model.pt"
DIAGNOSTIC_SUMMARY = ACTIVE_ROOT / "results/sae/experiment85_block6_batchtopk_diagnostics/layer_screen_block11_seed0_noise4_blur4_v1/summary.json"
BLOCK = 11
ALPHA = 0.5


def atomic_json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description="Experiment 92: frozen Block-11 Blur SAE repair confirmation")
    parser.add_argument("--fit-start", type=int, default=47000)
    parser.add_argument("--fit-samples", type=int, default=500)
    parser.add_argument("--evaluation-start", type=int, default=48000)
    parser.add_argument("--evaluation-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--corruption-seed", type=int, default=0)
    parser.add_argument("--sae-seed", type=int, default=0)
    parser.add_argument("--sae-checkpoint", type=Path, default=SAE_CHECKPOINT)
    parser.add_argument("--diagnostic-summary", type=Path, default=DIAGNOSTIC_SUMMARY)
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if (args.fit_start, args.fit_start + args.fit_samples) != (47000, 47500):
        raise ValueError("Mapping fit is locked to [47000,47500)")
    if (args.evaluation_start, args.evaluation_start + args.evaluation_samples) != (48000, 49000):
        raise ValueError("Confirmation is locked to [48000,49000)")
    output_dir = OUTPUT_ROOT / args.run_name
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    sae = BatchTopKSAE(expansion_factor=32, k=32, input_unit_norm=True, n_batches_to_dead=5)
    sae.load_state_dict(torch.load(args.sae_checkpoint, map_location="cpu", weights_only=True))
    sae = sae.to(device).eval()
    diagnostic = json.loads(args.diagnostic_summary.read_text())["results"]["blur"]
    groups = {
        "strengthened": [row["feature"] for row in diagnostic["top_strengthened"][:100]],
        "weakened": [row["feature"] for row in diagnostic["top_weakened"][:100]],
    }
    features = groups["strengthened"] + groups["weakened"]
    affine.BLOCK = BLOCK
    confirmation.BLOCK = BLOCK
    slope, intercept, fit_patches = affine.fit_affine(
        model, sae, device, features, args.fit_start, args.fit_samples,
        "blur", args.batch_size, args.workers, args.corruption_seed,
    )
    blur = confirmation.evaluate(
        model, sae, device, features, (slope, intercept), args.evaluation_start,
        args.evaluation_samples, "blur", args.batch_size, args.workers,
        args.corruption_seed,
    )
    clean = confirmation.evaluate(
        model, sae, device, features, (slope, intercept), args.evaluation_start,
        args.evaluation_samples, None, args.batch_size, args.workers,
        args.corruption_seed,
    )
    np.savez_compressed(
        output_dir / "paired_outcomes.npz",
        **{f"blur_{key}": value for key, value in blur["arrays"].items()},
        **{f"clean_{key}": value for key, value in clean["arrays"].items()},
    )
    summary = {
        "configuration": vars(args) | {
            "model": BASE_MODEL,
            "block": BLOCK,
            "alpha_frozen": ALPHA,
            "sae_checkpoint": str(args.sae_checkpoint.resolve()),
            "diagnostic_summary": str(args.diagnostic_summary.resolve()),
            "top_strengthened": 100,
            "top_weakened": 100,
        },
        "features": groups,
        "fit_patches": fit_patches,
        "mapping": {"slope": slope.cpu().tolist(), "intercept": intercept.cpu().tolist()},
        "blur": blur["metrics"],
        "clean": clean["metrics"],
        "guardrails": [
            "Block 11, Blur features, and alpha=0.5 were frozen before accessing [48000,49000).",
            "The affine mapping is fitted without labels on paired [47000,47500) representations.",
            "Inference uses only the corrupted image and oracle Blur-family identity.",
            "Only residual decoder deltas are added; the full SAE reconstruction is never inserted.",
            "Per-image paired outcomes, exact McNemar p-values, and bootstrap intervals are saved.",
            "[49000,50000) remains untouched.",
        ],
        "status": f"frozen SAE-seed-{args.sae_seed} held-out confirmation",
    }
    atomic_json_write(output_dir / "summary.json", summary)
    print(json.dumps({"blur": summary["blur"], "clean": summary["clean"]}, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
