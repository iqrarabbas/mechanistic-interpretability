import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import ViTForImageClassification

from data.imagenet_dataset import ImageNetDataset
from interpretability.sae import BatchTopKSAE, VanillaReLUSAE, sae_loss
from scripts.experiment1_base_blur4_sae_identity_strength import BASE_MODEL
from scripts.experiment19_noise_layer_localization import downstream_from_layer
from scripts.train_sae_level4 import MemoryEfficientAdamW


ROOT = Path(__file__).parent.parent
ACTIVE_ROOT = Path("/media/dr-yougart/Iqrar/vit_mi")
OUTPUT_ROOT = ACTIVE_ROOT / "results/sae/experiment84_block6_clean_sae_calibration"
BLOCK = 6
INPUT_DIM = 768
TARGET_ACTIVE = 32


def atomic_json_write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def atomic_torch_save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def build_sae(kind, expansion_factor, k):
    if kind == "vanilla":
        return VanillaReLUSAE(
            input_dim=INPUT_DIM, expansion_factor=expansion_factor
        )
    if kind == "batchtopk":
        return BatchTopKSAE(
            input_dim=INPUT_DIM,
            expansion_factor=expansion_factor,
            k=k,
            input_unit_norm=True,
            n_batches_to_dead=5,
        )
    raise ValueError(kind)


def make_loader(start, samples, batch_size, shuffle, workers, seed):
    dataset = ImageNetDataset(ROOT / "Dataset", samples, start)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        generator=generator if shuffle else None,
    )


def reconstruction_and_loss(sae, activations, kind, l1_coefficient, training):
    if kind == "batchtopk":
        (
            loss,
            reconstruction_loss,
            sparsity_loss,
            auxiliary_loss,
            reconstruction,
            latent,
        ) = sae.compute_loss(
            activations,
            l1_coefficient=l1_coefficient,
            aux_coefficient=1 / 32,
            aux_k=512,
            update_activity=training,
            return_outputs=True,
        )
    else:
        reconstruction, latent = sae(activations)
        loss, reconstruction_loss, sparsity_loss = sae_loss(
            activations,
            reconstruction,
            latent,
            l1_coefficient=l1_coefficient,
        )
        auxiliary_loss = activations.new_zeros(())
    return (
        loss,
        reconstruction_loss,
        sparsity_loss,
        auxiliary_loss,
        reconstruction,
        latent,
    )


def cache_block_activations(model, data_loader, device, cache_path):
    if cache_path.exists():
        print(f"Loading cached activations: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)
    activations = []
    labels = []
    for images, batch_labels in tqdm(data_loader, desc="Caching Block-6 activations"):
        with torch.no_grad():
            outputs = model(
                pixel_values=images.to(device), output_hidden_states=True
            )
        activations.append(outputs.hidden_states[BLOCK][:, 1:].cpu())
        labels.append(batch_labels.cpu())
    payload = {
        "activations": torch.cat(activations),
        "labels": torch.cat(labels),
        "block": BLOCK,
        "model": BASE_MODEL,
        "dtype": "float32",
    }
    atomic_torch_save(cache_path, payload)
    return payload


def run_cached_epoch(
    sae,
    activations,
    device,
    l1_coefficient,
    optimizer=None,
    activation_batch_size=16,
    seed=0,
):
    training = optimizer is not None
    sae.train(training)
    flat = activations.flatten(0, 1)
    order = torch.randperm(flat.shape[0], generator=torch.Generator().manual_seed(seed)) if training else torch.arange(flat.shape[0])
    totals = {name: 0.0 for name in ("objective", "training_reconstruction_term", "sparsity_term", "auxiliary_term", "active_features")}
    input_sum = input_square_sum = error_sum = error_square_sum = 0.0
    scalar_count = patches_seen = 0
    active_any = torch.zeros(sae.latent_dim, dtype=torch.bool, device=device)
    for indices in tqdm(order.split(activation_batch_size), desc="Cached SAE epoch", leave=False):
        activation_chunk = flat[indices].to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        values = reconstruction_and_loss(sae, activation_chunk, "batchtopk", l1_coefficient, training)
        loss, reconstruction_term, sparsity_term, auxiliary_term, reconstruction, latent = values
        if training:
            loss.backward()
            sae.project_decoder_gradient(chunk_size=1024)
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0, foreach=False)
            optimizer.step()
            sae.normalize_decoder()
        with torch.no_grad():
            error = reconstruction.float() - activation_chunk.float()
            count = activation_chunk.shape[0]
            scalar_count += activation_chunk.numel()
            patches_seen += count
            input_sum += float(activation_chunk.float().sum())
            input_square_sum += float(activation_chunk.float().square().sum())
            error_sum += float(error.sum())
            error_square_sum += float(error.square().sum())
            active_count = (latent > 0).sum(dim=1)
            active_any |= (latent > 0).any(dim=0)
            for name, value in zip(totals, (loss, reconstruction_term, sparsity_term, auxiliary_term, active_count.float().mean())):
                totals[name] += float(value.detach()) * count
        del values, loss, reconstruction_term, sparsity_term, auxiliary_term, reconstruction, latent, error, active_count, activation_chunk
    input_variance_sum = input_square_sum - input_sum * input_sum / scalar_count
    error_variance_sum = error_square_sum - error_sum * error_sum / scalar_count
    metrics = {name: value / patches_seen for name, value in totals.items()}
    metrics.update({
        "normalized_mse": error_square_sum / max(input_square_sum, 1e-12),
        "explained_variance": 1 - error_variance_sum / max(input_variance_sum, 1e-12),
        "features_active_at_least_once": int(active_any.sum()),
        "features_never_active": int((~active_any).sum()),
        "patches": patches_seen,
    })
    return metrics


def run_epoch(
    model,
    sae,
    data_loader,
    device,
    kind,
    l1_coefficient,
    optimizer=None,
    measure_downstream=False,
    activation_batch_size=32,
):
    training = optimizer is not None
    sae.train(training)
    totals = {
        "objective": 0.0,
        "training_reconstruction_term": 0.0,
        "sparsity_term": 0.0,
        "auxiliary_term": 0.0,
        "active_features": 0.0,
    }
    patches_seen = 0
    scalar_count = 0
    input_sum = 0.0
    input_square_sum = 0.0
    error_sum = 0.0
    error_square_sum = 0.0
    active_any = torch.zeros(sae.latent_dim, dtype=torch.bool, device=device)
    baseline_correct = 0
    reconstructed_correct = 0
    images_seen = 0
    for images, labels in tqdm(data_loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        with torch.no_grad():
            outputs = model(pixel_values=images, output_hidden_states=True)
            hidden = outputs.hidden_states[BLOCK]
            patches = hidden[:, 1:]
            activations = patches.flatten(0, 1)
        chunks = (
            activations.split(activation_batch_size)
            if kind == "batchtopk"
            else (activations,)
        )
        reconstructed_chunks = []
        for activation_chunk in chunks:
            if training:
                optimizer.zero_grad(set_to_none=True)
            values = reconstruction_and_loss(
                sae, activation_chunk, kind, l1_coefficient, training
            )
            (
                loss,
                reconstruction_term,
                sparsity_term,
                auxiliary_term,
                reconstruction,
                latent,
            ) = values
            if training:
                loss.backward()
                if kind == "batchtopk":
                    sae.project_decoder_gradient()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            with torch.no_grad():
                error = reconstruction.float() - activation_chunk.float()
                scalar_count += activation_chunk.numel()
                input_sum += float(activation_chunk.float().sum())
                input_square_sum += float(activation_chunk.float().square().sum())
                error_sum += float(error.sum())
                error_square_sum += float(error.square().sum())
                active_count = (latent > 0).sum(dim=1)
                active_any |= (latent > 0).any(dim=0)
                count = activation_chunk.shape[0]
                detached_metrics = (
                    float(loss.detach()),
                    float(reconstruction_term.detach()),
                    float(sparsity_term.detach()),
                    float(auxiliary_term.detach()),
                    float(active_count.float().mean()),
                )
                for name, value in zip(totals, detached_metrics):
                    totals[name] += value * count
                patches_seen += count
                if measure_downstream:
                    reconstructed_chunks.append(reconstruction.detach())
            del (
                values,
                loss,
                reconstruction_term,
                sparsity_term,
                auxiliary_term,
                reconstruction,
                latent,
                activation_chunk,
                error,
                active_count,
                detached_metrics,
            )
            if training:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                optimizer.step()
                sae.normalize_decoder()
                optimizer.zero_grad(set_to_none=True)
        if measure_downstream:
            reconstructed_patches = torch.cat(reconstructed_chunks).reshape_as(patches)
            reconstructed_hidden = torch.cat(
                (hidden[:, :1], reconstructed_patches), dim=1
            )
            reconstructed_logits = downstream_from_layer(
                model, reconstructed_hidden, BLOCK - 1
            )
            baseline_correct += int((outputs.logits.argmax(1) == labels).sum())
            reconstructed_correct += int(
                (reconstructed_logits.argmax(1) == labels).sum()
            )
            images_seen += labels.shape[0]
            del reconstructed_chunks, reconstructed_patches, reconstructed_hidden, reconstructed_logits
        del activations
        del images, labels, outputs, hidden, patches
    input_variance_sum = input_square_sum - input_sum * input_sum / scalar_count
    error_variance_sum = error_square_sum - error_sum * error_sum / scalar_count
    metrics = {name: value / patches_seen for name, value in totals.items()}
    metrics.update(
        {
            "normalized_mse": error_square_sum / max(input_square_sum, 1e-12),
            "explained_variance": 1 - error_variance_sum / max(input_variance_sum, 1e-12),
            "features_active_at_least_once": int(active_any.sum().item()),
            "features_never_active": int((~active_any).sum().item()),
            "patches": patches_seen,
        }
    )
    if measure_downstream:
        metrics.update(
            {
                "baseline_clean_accuracy": baseline_correct / images_seen,
                "full_reconstruction_clean_accuracy": reconstructed_correct / images_seen,
                "full_reconstruction_clean_accuracy_change": (
                    reconstructed_correct - baseline_correct
                )
                / images_seen,
                "images": images_seen,
            }
        )
    return metrics


def candidate_name(kind, l1_coefficient):
    value = f"{l1_coefficient:.0e}".replace("-", "m").replace("+", "")
    return f"{kind}_lambda_{value}"


def train_candidate(
    candidate,
    model,
    device,
    args,
    output_dir,
    candidate_index,
    cached_train=None,
    cached_validation=None,
):
    kind = candidate["kind"]
    l1_coefficient = candidate["l1_coefficient"]
    name = candidate_name(kind, l1_coefficient)
    candidate_dir = output_dir / name
    candidate_dir.mkdir(exist_ok=True)
    record_path = candidate_dir / "training.json"
    best_path = candidate_dir / "model.pt"
    latest_path = candidate_dir / "latest.pt"
    if args.resume and record_path.exists():
        record = json.loads(record_path.read_text())
        if record.get("status") == "complete" and best_path.exists():
            print(f"Skipping completed candidate {name}")
            return record
    initialization_seed = args.seed + candidate_index * 1000
    torch.manual_seed(initialization_seed)
    np.random.seed(initialization_seed)
    sae = build_sae(kind, args.expansion_factor, args.k).to(device)
    sae.normalize_decoder()
    optimizer = MemoryEfficientAdamW(sae.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    history = []
    best_normalized_mse = float("inf")
    stale = 0
    start_epoch = 1
    record = {
        "status": "running",
        "candidate": candidate,
        "candidate_name": name,
        "initialization_seed": initialization_seed,
        "history": history,
        "best_validation_normalized_mse": best_normalized_mse,
        "best_checkpoint": str(best_path.resolve()),
        "latest_checkpoint": str(latest_path.resolve()),
    }
    if args.resume and latest_path.exists():
        payload = torch.load(latest_path, map_location=device, weights_only=False)
        sae.load_state_dict(payload["sae"])
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        history = payload["history"]
        best_normalized_mse = payload["best_normalized_mse"]
        stale = payload["stale"]
        start_epoch = payload["epoch"] + 1
        print(f"Resuming {name} at epoch {start_epoch}")
    batch_size = 1 if kind == "batchtopk" else args.batch_size
    for epoch in range(start_epoch, args.epochs + 1):
        if kind == "batchtopk" and cached_train is not None:
            train_metrics = run_cached_epoch(
                sae,
                cached_train["activations"],
                device,
                l1_coefficient,
                optimizer,
                args.activation_batch_size,
                initialization_seed + epoch,
            )
            with torch.no_grad():
                validation_metrics = run_cached_epoch(
                    sae,
                    cached_validation["activations"],
                    device,
                    l1_coefficient,
                    None,
                    args.activation_batch_size,
                    initialization_seed,
                )
            validation_metrics.update(
                {
                    "baseline_clean_accuracy": None,
                    "full_reconstruction_clean_accuracy": None,
                    "full_reconstruction_clean_accuracy_change": None,
                    "images": int(cached_validation["labels"].shape[0]),
                    "downstream_accuracy_deferred_until_optimizer_unloaded": True,
                }
            )
        else:
            train_loader = make_loader(
                args.train_start,
                args.train_samples,
                batch_size,
                True,
                args.workers,
                initialization_seed + epoch,
            )
            validation_loader = make_loader(
                args.validation_start,
                args.validation_samples,
                batch_size,
                False,
                args.workers,
                initialization_seed,
            )
            train_metrics = run_epoch(
                model,
                sae,
                train_loader,
                device,
                kind,
                l1_coefficient,
                optimizer,
                False,
                args.activation_batch_size,
            )
            with torch.no_grad():
                validation_metrics = run_epoch(
                    model,
                    sae,
                    validation_loader,
                    device,
                    kind,
                    l1_coefficient,
                    None,
                    True,
                    args.activation_batch_size,
                )
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "train": train_metrics,
                "validation": validation_metrics,
            }
        )
        improved = validation_metrics["normalized_mse"] < best_normalized_mse
        if improved:
            best_normalized_mse = validation_metrics["normalized_mse"]
            stale = 0
            atomic_torch_save(best_path, sae.state_dict())
        else:
            stale += 1
        payload = {
            "epoch": epoch,
            "sae": sae.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_normalized_mse": best_normalized_mse,
            "stale": stale,
        }
        atomic_torch_save(latest_path, payload)
        record = {
            "status": "running",
            "candidate": candidate,
            "candidate_name": name,
            "initialization_seed": initialization_seed,
            "history": history,
            "best_validation_normalized_mse": best_normalized_mse,
            "best_checkpoint": str(best_path.resolve()),
            "latest_checkpoint": str(latest_path.resolve()),
        }
        atomic_json_write(record_path, record)
        print(
            f"{name} epoch={epoch} "
            f"val_nmse={validation_metrics['normalized_mse']:.6f} "
            f"val_ev={validation_metrics['explained_variance']:.6f} "
            f"active={validation_metrics['active_features']:.2f} "
            f"clean_delta={validation_metrics['full_reconstruction_clean_accuracy_change']}"
        )
        if stale >= args.patience:
            print(f"Early stopping {name}")
            break
    if kind == "batchtopk" and cached_validation is not None:
        del optimizer, scheduler
        sae.zero_grad(set_to_none=True)
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        sae.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        evaluation_model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
        evaluation_model.requires_grad_(False)
        validation_loader = make_loader(
            args.validation_start,
            args.validation_samples,
            1,
            False,
            args.workers,
            initialization_seed,
        )
        with torch.no_grad():
            downstream_metrics = run_epoch(
                evaluation_model,
                sae,
                validation_loader,
                device,
                kind,
                l1_coefficient,
                None,
                True,
                args.activation_batch_size,
            )
        best_history_item = min(
            history, key=lambda item: item["validation"]["normalized_mse"]
        )
        for key in (
            "baseline_clean_accuracy",
            "full_reconstruction_clean_accuracy",
            "full_reconstruction_clean_accuracy_change",
            "images",
        ):
            best_history_item["validation"][key] = downstream_metrics[key]
        best_history_item["validation"]["downstream_accuracy_deferred_until_optimizer_unloaded"] = False
        del evaluation_model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    record["history"] = history
    record["best_validation_normalized_mse"] = best_normalized_mse
    record["status"] = "complete"
    record["completed_epochs"] = len(history)
    atomic_json_write(record_path, record)
    return record


def select_candidates(records):
    ranking = []
    for record in records:
        best_epoch = min(
            record["history"],
            key=lambda item: item["validation"]["normalized_mse"],
        )
        metrics = best_epoch["validation"]
        active = metrics["active_features"]
        eligible = 16 <= active <= 64
        ranking.append(
            {
                "candidate_name": record["candidate_name"],
                "kind": record["candidate"]["kind"],
                "l1_coefficient": record["candidate"]["l1_coefficient"],
                "best_epoch": best_epoch["epoch"],
                "eligible_sparsity_band_16_to_64": eligible,
                "distance_from_target_active_32": abs(active - TARGET_ACTIVE),
                **metrics,
                "best_checkpoint": record["best_checkpoint"],
            }
        )
    eligible = [item for item in ranking if item["eligible_sparsity_band_16_to_64"]]
    if eligible:
        selected = min(eligible, key=lambda item: item["normalized_mse"])
        rule = "lowest validation normalized MSE among candidates with 16-64 active features"
    else:
        selected = min(
            ranking,
            key=lambda item: (
                item["distance_from_target_active_32"], item["normalized_mse"]
            ),
        )
        rule = "closest validation active-feature count to 32, then normalized MSE"
    return sorted(
        ranking,
        key=lambda item: (
            not item["eligible_sparsity_band_16_to_64"],
            item["normalized_mse"] if item["eligible_sparsity_band_16_to_64"] else item["distance_from_target_active_32"],
        ),
    ), selected, rule


def main():
    global BLOCK
    parser = argparse.ArgumentParser(
        description="Leakage-safe clean-only Block-6 SAE calibration"
    )
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--block", type=int, default=6)
    parser.add_argument("--train-samples", type=int, default=1000)
    parser.add_argument("--validation-start", type=int, default=10000)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--activation-batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--expansion-factor", type=int, default=32)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument(
        "--vanilla-lambdas",
        type=float,
        nargs="+",
        default=[1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0],
    )
    parser.add_argument("--batchtopk-lambda", type=float, default=1e-3)
    parser.add_argument("--batchtopk-only", action="store_true")
    parser.add_argument("--activation-cache-root", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-name", required=True)
    args = parser.parse_args()
    if not 1 <= args.block <= 12:
        raise ValueError("--block must be between 1 and 12")
    BLOCK = args.block
    if args.train_start != 0 or args.train_start + args.train_samples > 10000:
        raise ValueError("Calibration training is locked inside [0,10000)")
    if (
        args.validation_start < 10000
        or args.validation_start + args.validation_samples > 11000
    ):
        raise ValueError("Calibration validation is locked inside [10000,11000)")
    output_dir = args.output_root / args.run_name
    if output_dir.exists() and not args.resume:
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True, exist_ok=args.resume)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    model = ViTForImageClassification.from_pretrained(BASE_MODEL).to(device).eval()
    model.requires_grad_(False)
    candidates = [
        {"kind": "vanilla", "l1_coefficient": value}
        for value in args.vanilla_lambdas
    ] + [{"kind": "batchtopk", "l1_coefficient": args.batchtopk_lambda}]
    if args.batchtopk_only:
        candidates = [candidates[-1]]
    records = []
    for index, candidate in enumerate(candidates[:-1]):
        records.append(
            train_candidate(candidate, model, device, args, output_dir, index)
        )
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    cache_dir = args.activation_cache_root or (output_dir / "activation_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_train = cache_block_activations(
        model,
        make_loader(args.train_start, args.train_samples, args.batch_size, False, args.workers, args.seed),
        device,
        cache_dir / f"train_{args.train_start}_{args.train_samples}_block{BLOCK}.pt",
    )
    cached_validation = cache_block_activations(
        model,
        make_loader(args.validation_start, args.validation_samples, args.batch_size, False, args.workers, args.seed),
        device,
        cache_dir / f"validation_{args.validation_start}_{args.validation_samples}_block{BLOCK}.pt",
    )
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    records.append(
        train_candidate(
            candidates[-1],
            None,
            device,
            args,
            output_dir,
            len(args.vanilla_lambdas),
            cached_train,
            cached_validation,
        )
    )
    ranking, selected, rule = select_candidates(records)
    summary = {
        "configuration": vars(args)
        | {
            "output_root": str(args.output_root.resolve()),
            "model": BASE_MODEL,
            "block": BLOCK,
            "hidden_dimension": INPUT_DIM,
            "latent_dimension": INPUT_DIM * args.expansion_factor,
            "vit_frozen": True,
            "training_distribution": "clean only",
            "all_196_patch_tokens_used": True,
            "imageNetV2_accessed": False,
            "imageNetSketch_accessed": False,
            "corrupted_images_accessed": False,
            "status": "SAE hyperparameter calibration; not corruption evaluation",
        },
        "selection_rule_frozen_before_training": rule,
        "ranking": ranking,
        "selected_candidate": selected,
        "next_stage": "Retrain the selected configuration on [0,10000), validate on [10000,11000), using three independent SAE seeds.",
        "leakage_guardrails": [
            "Only clean feature-development images are used.",
            "No adapter/gate training, validation, OOD, ImageNetV2, or ImageNet-Sketch images are accessed.",
            "No corruption accuracy is used for SAE hyperparameter selection.",
            "Vanilla and BatchTopK are compared using common original-hidden-space metrics rather than their differently scaled training objectives.",
        ],
        "limitations": [
            "The paper does not report a numerical lambda, so the Vanilla coefficient is calibrated explicitly.",
            "Full SAE reconstruction accuracy is a diagnostic only; later interventions must use residual decoder deltas rather than hidden-state replacement.",
            "Calibration is a small development run and does not establish SAE feature stability.",
        ],
    }
    atomic_json_write(output_dir / "summary.json", summary)
    print(json.dumps({"selected_candidate": selected, "selection_rule": rule}, indent=2))
    print("Saved", output_dir)


if __name__ == "__main__":
    main()
