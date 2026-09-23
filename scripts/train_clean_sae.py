import argparse
import json
from pathlib import Path
import re
import sys

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.imagenet_dataset import ImageNetDataset
from interpretability.sae import VanillaReLUSAE, extract_penultimate_patch_activations, sae_loss
from scripts.train_sae_level4 import BASE_MODEL, DATASET_DIR, MemoryEfficientAdamW


OUTPUT_ROOT = PROJECT_ROOT / "checkpoints" / "sae"


def run_epoch(sae, vit, loader, device, l1_coefficient, optimizer=None):
    training = optimizer is not None
    sae.train(training)
    totals = {"loss": 0.0, "reconstruction": 0.0, "sparsity": 0.0, "active_features": 0.0}
    samples = 0
    for images, _ in tqdm(loader, leave=False):
        activations = extract_penultimate_patch_activations(vit, images.to(device)).flatten(0, 1)
        if training:
            optimizer.zero_grad(set_to_none=True)
        reconstruction, latent = sae(activations)
        loss, reconstruction_loss, sparsity_loss = sae_loss(
            activations,
            reconstruction,
            latent,
            l1_coefficient=l1_coefficient,
        )
        count = activations.shape[0]
        metrics = {
            "loss": loss.item(),
            "reconstruction": reconstruction_loss.item(),
            "sparsity": sparsity_loss.item(),
            "active_features": (latent > 0).sum(-1).float().mean().item(),
        }
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            sae.normalize_decoder()
        for name, value in metrics.items():
            totals[name] += value * count
        samples += count
    return {name: value / samples for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train a clean-only Vanilla SAE for ViT-B/16")
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l1-coefficient", type=float, default=1e-3)
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    if args.output_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_tag):
        parser.error("--output-tag may contain only letters, numbers, dot, dash, and underscore")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_data = ImageNetDataset(DATASET_DIR, max_samples=args.train_samples, start_index=0)
    validation_data = ImageNetDataset(DATASET_DIR, max_samples=args.val_samples, start_index=10000)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    vit = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    vit.requires_grad_(False)
    sae = VanillaReLUSAE(expansion_factor=args.expansion_factor).to(device)
    sae.normalize_decoder()
    optimizer = MemoryEfficientAdamW(sae.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    suffix = "clean_base_vanilla_paper"
    if args.output_tag:
        suffix += f"_{args.output_tag}"
    output_dir = OUTPUT_ROOT / suffix
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    best_loss = float("inf")
    stale_epochs = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            sae, vit, train_loader, device, args.l1_coefficient, optimizer
        )
        validation_metrics = run_epoch(
            sae, vit, validation_loader, device, args.l1_coefficient
        )
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": validation_metrics})
        print(
            f"Epoch {epoch}: train={train_metrics['loss']:.6f}, "
            f"validation={validation_metrics['loss']:.6f}"
        )
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            stale_epochs = 0
            torch.save(sae.state_dict(), output_dir / "model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping triggered")
                break

    metadata = vars(args) | {
        "model": "base",
        "sae_type": "vanilla",
        "training_distribution": "clean_only",
        "train_start_index": 0,
        "validation_start_index": 10000,
        "implementation": "paper_vanilla_literal_l2_v3_clean_only",
        "best_validation_loss": best_loss,
        "history": history,
    }
    (output_dir / "training.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved clean-only SAE to {output_dir}")


if __name__ == "__main__":
    main()
