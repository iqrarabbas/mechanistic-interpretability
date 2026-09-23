import argparse
import json
import math
from pathlib import Path
import re
import sys

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import ViTForImageClassification

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.imagenet_dataset import ImageNetDataset
from interpretability.sae import (
    BatchTopKSAE,
    VanillaReLUSAE,
    extract_penultimate_patch_activations,
    sae_loss,
)

DATASET_DIR = PROJECT_ROOT / "Dataset"
OUTPUT_ROOT = PROJECT_ROOT / "checkpoints" / "sae"
BASE_MODEL = "google/vit-base-patch16-224"


class MemoryEfficientAdamW(Optimizer):
    """AdamW that reuses gradients as the denominator workspace.

    PyTorch's standard AdamW allocates an additional full-size tensor for
    ``sqrt(exp_avg_sq)``.  The SAE's largest parameter is about 72 MiB, which
    exceeds the remaining memory on the available 12 GB GPU.  Once the first
    and second moments have been updated, the gradient is no longer needed for
    that step, so it can safely be used for this temporary denominator.  This
    implements the same AdamW update (betas, bias correction, epsilon, and
    decoupled weight decay) without that extra allocation.
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("MemoryEfficientAdamW does not support sparse gradients.")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                if weight_decay:
                    parameter.mul_(1 - lr * weight_decay)
                exp_avg.lerp_(gradient, 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                # Reuse gradient storage rather than allocating sqrt(exp_avg_sq).
                gradient.copy_(exp_avg_sq).sqrt_()
                gradient.div_(math.sqrt(1 - beta2**step)).add_(eps)
                step_size = lr / (1 - beta1**step)
                parameter.addcdiv_(exp_avg, gradient, value=-step_size)
        return loss


class MixedPairDataset(Dataset):
    def __init__(self, corruption, samples, start_index, seed):
        common = dict(
            dataset_dir=DATASET_DIR,
            max_samples=samples,
            start_index=start_index,
        )
        self.clean = ImageNetDataset(**common)
        self.corrupted = ImageNetDataset(
            **common,
            corruption=corruption,
            blur_severity=4,
            noise_severity=4,
            corruption_seed=seed,
        )

    def __len__(self):
        return len(self.clean)

    def __getitem__(self, index):
        clean, _ = self.clean[index]
        corrupted, _ = self.corrupted[index]
        return clean, corrupted


def build_sae(args):
    if args.sae_type == "vanilla":
        return VanillaReLUSAE(expansion_factor=args.expansion_factor)
    return BatchTopKSAE(
        expansion_factor=args.expansion_factor,
        k=args.k,
        input_unit_norm=args.input_unit_norm,
        n_batches_to_dead=args.n_batches_to_dead,
    )


def model_path(args):
    if args.model == "base":
        return BASE_MODEL
    return PROJECT_ROOT / "checkpoints" / f"vit_{args.corruption}4_best"


def flatten_mixed_activations(vit, clean, corrupted, device, patches_per_image):
    images = torch.cat((clean, corrupted), dim=0).to(device)
    patches = extract_penultimate_patch_activations(vit, images)
    if patches_per_image and patches_per_image < patches.shape[1]:
        indices = torch.randperm(patches.shape[1], device=device)[:patches_per_image]
        patches = patches[:, indices]
    return patches.reshape(-1, patches.shape[-1])


def run_epoch(sae, vit, loader, device, args, optimizer=None):
    training = optimizer is not None
    sae.train(training)
    totals = {
        "loss": 0.0,
        "reconstruction": 0.0,
        "sparsity": 0.0,
        "active_features": 0.0,
        "aux": 0.0,
    }
    samples = 0

    for clean, corrupted in tqdm(loader, leave=False):
        activations = flatten_mixed_activations(
            vit, clean, corrupted, device, args.patches_per_image
        )
        if training:
            optimizer.zero_grad(set_to_none=True)
        if isinstance(sae, BatchTopKSAE):
            (
                loss,
                reconstruction_loss,
                sparsity_loss,
                auxiliary_loss,
            ) = sae.compute_loss(
                activations,
                l1_coefficient=args.l1_coefficient,
                aux_coefficient=args.aux_coefficient,
                aux_k=args.aux_k,
                update_activity=training,
                return_outputs=False,
            )
        else:
            reconstruction, latent = sae(activations)
            loss, reconstruction_loss, sparsity_loss = sae_loss(
                activations,
                reconstruction,
                latent,
                l1_coefficient=args.l1_coefficient,
            )
            auxiliary_loss = activations.new_zeros(())
        count = activations.shape[0]
        active_features = (
            float(args.k)
            if isinstance(sae, BatchTopKSAE)
            else (latent > 0).sum(dim=-1).float().mean().item()
        )
        if training:
            loss.backward()
            if isinstance(sae, BatchTopKSAE):
                sae.project_decoder_gradient()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            # Metrics have already been copied to CPU below.  Release the
            # full BatchTopK activation graph before AdamW creates its update
            # temporary tensors on a 12 GB GPU.
            metric_values = (
                loss.item(),
                reconstruction_loss.item(),
                sparsity_loss.item(),
                auxiliary_loss.item(),
            )
            del loss, reconstruction_loss, sparsity_loss, auxiliary_loss, activations
            torch.cuda.empty_cache()
            optimizer.step()
            sae.normalize_decoder()
        else:
            metric_values = (
                loss.item(),
                reconstruction_loss.item(),
                sparsity_loss.item(),
                auxiliary_loss.item(),
            )

        totals["loss"] += metric_values[0] * count
        totals["reconstruction"] += metric_values[1] * count
        totals["sparsity"] += metric_values[2] * count
        totals["active_features"] += active_features * count
        totals["aux"] += metric_values[3] * count
        samples += count

    return {name: value / samples for name, value in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="Train a level-4 ViT SAE")
    parser.add_argument("--corruption", choices=["blur", "noise"], required=True)
    parser.add_argument("--model", choices=["base", "fine_tuned"], required=True)
    parser.add_argument("--sae-type", choices=["vanilla", "batchtopk"], required=True)
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--val-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--patches-per-image", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--l1-coefficient", type=float, default=1e-3)
    parser.add_argument("--aux-coefficient", type=float, default=1 / 32)
    parser.add_argument("--aux-k", type=int, default=512)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--input-unit-norm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--n-batches-to-dead", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-tag", default="")
    args = parser.parse_args()

    if args.output_tag and not re.fullmatch(r"[A-Za-z0-9_.-]+", args.output_tag):
        parser.error("--output-tag may contain only letters, numbers, dot, dash, and underscore")

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Reference BatchTopK retains a large activation matrix for the encoder,
    # batch-wide TopK, reconstruction, and auxiliary dead-feature loss.  With
    # 196 ViT patches per image, two image pairs require more than a 12 GB GPU
    # can safely hold during backward.  Reducing the *image* batch to one does
    # not discard patches or examples; it simply performs more optimizer steps.
    if (
        args.sae_type == "batchtopk"
        and device.type == "cuda"
        and args.batch_size > 1
        and torch.cuda.get_device_properties(device).total_memory <= 12 * 1024**3
    ):
        print(
            "BatchTopK memory safeguard: using --batch-size 1 on this <=12 GB GPU "
            "(all clean and corrupted image patches are still used)."
        )
        args.batch_size = 1

    train_data = MixedPairDataset(args.corruption, args.train_samples, 0, args.seed)
    val_data = MixedPairDataset(args.corruption, args.val_samples, 10000, args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    vit = ViTForImageClassification.from_pretrained(model_path(args)).to(device)
    vit.eval()
    vit.requires_grad_(False)
    sae = build_sae(args).to(device)
    sae.normalize_decoder()
    optimizer = MemoryEfficientAdamW(sae.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    suffix = args.sae_type
    if args.sae_type == "batchtopk":
        suffix = "batchtopk_reference"
    elif args.sae_type == "vanilla":
        suffix = "vanilla_paper"
    if args.output_tag:
        suffix = f"{suffix}_{args.output_tag}"
    output_dir = OUTPUT_ROOT / f"{args.corruption}4_{args.model}_{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    stale_epochs = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(sae, vit, train_loader, device, args, optimizer)
        val_metrics = run_epoch(sae, vit, val_loader, device, args)
        scheduler.step()
        history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
        print(
            f"Epoch {epoch}: train={train_metrics['loss']:.6f}, "
            f"validation={val_metrics['loss']:.6f}"
        )
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            stale_epochs = 0
            torch.save(sae.state_dict(), output_dir / "model.pt")
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print("Early stopping triggered")
                break

    metadata = vars(args) | {
        "implementation": (
            "reference_batchtopk_v1"
            if args.sae_type == "batchtopk"
            else "paper_vanilla_literal_l2_v3"
        ),
        "best_validation_loss": best_loss,
        "history": history,
    }
    with open(output_dir / "training.json", "w") as output_file:
        json.dump(metadata, output_file, indent=2)
    print(f"Saved best SAE to {output_dir}")


if __name__ == "__main__":
    main()
