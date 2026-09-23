import json
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm


def standardized_features(features):
    return torch.tanh(F.layer_norm(features, (features.shape[-1],)))


def classifier_outputs(model, images):
    hidden = model.vit(pixel_values=images).last_hidden_state[:, 0]
    return model.classifier(hidden), hidden


def collect_layernorm_parameters(model):
    parameters = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            for parameter_name, parameter in module.named_parameters(recurse=False):
                parameter.requires_grad_(True)
                parameters.append(parameter)
                names.append(f"{name}.{parameter_name}")
    selected = {id(parameter) for parameter in parameters}
    for parameter in model.parameters():
        if id(parameter) not in selected:
            parameter.requires_grad_(False)
    return parameters, names


def calculate_source_statistics(model, loader, device, output_path, max_moment=3):
    feature_dim = model.config.hidden_size
    classes = model.config.num_labels
    total = 0
    class_counts = torch.zeros(classes, dtype=torch.float64)
    global_sum = torch.zeros(feature_dim, dtype=torch.float64)
    class_sum = torch.zeros(classes, feature_dim, dtype=torch.float64)
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="CFA source means"):
            labels = labels.to(device)
            _, features = classifier_outputs(model, images.to(device))
            features = standardized_features(features).double().cpu()
            labels_cpu = labels.cpu()
            global_sum += features.sum(0)
            class_sum.index_add_(0, labels_cpu, features)
            class_counts += torch.bincount(labels_cpu, minlength=classes).double()
            total += labels.numel()
    global_mean = global_sum / total
    class_mean = class_sum / class_counts.clamp_min(1)[:, None]
    global_moments = [global_mean]
    higher_sums = [torch.zeros(feature_dim, dtype=torch.float64) for _ in range(1, max_moment)]
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="CFA source moments"):
            _, features = classifier_outputs(model, images.to(device))
            centered = standardized_features(features).double().cpu() - global_mean
            for order in range(2, max_moment + 1):
                higher_sums[order - 2] += centered.pow(order).sum(0)
    global_moments.extend(moment / total for moment in higher_sums)
    statistics = {
        "global_moments": [moment.float() for moment in global_moments],
        "class_mean": class_mean.float(),
        "class_counts": class_counts,
        "source_samples": total,
        "max_moment": max_moment,
    }
    torch.save(statistics, output_path)
    metadata = {
        "source_samples": total,
        "classes_with_samples": int((class_counts > 0).sum()),
        "minimum_class_samples": int(class_counts.min()),
        "maximum_class_samples": int(class_counts.max()),
        "max_moment": max_moment,
    }
    Path(output_path).with_suffix(".json").write_text(json.dumps(metadata, indent=2))
    return statistics


def load_source_statistics(path, device):
    statistics = torch.load(path, map_location="cpu", weights_only=True)
    return {
        "global_moments": [moment.to(device) for moment in statistics["global_moments"]],
        "class_mean": statistics["class_mean"].to(device),
        "class_counts": statistics["class_counts"],
        "source_samples": statistics["source_samples"],
        "max_moment": statistics["max_moment"],
    }


def cfa_loss(logits, features, statistics, full_max_moment=3, class_max_moment=1):
    standardized = standardized_features(features)
    target_global = statistics["global_moments"]
    batch_mean = standardized.mean(0)
    global_loss = (batch_mean - target_global[0]).square().sum().sqrt() / 2.0
    for order in range(2, full_max_moment + 1):
        batch_moment = (standardized - batch_mean).pow(order).mean(0)
        global_loss = global_loss + (
            (batch_moment - target_global[order - 1]).square().sum().sqrt()
            / (2.0**order)
        )
    class_loss = standardized.new_zeros(())
    if class_max_moment:
        predictions = logits.argmax(1)
        active_classes = predictions.unique()
        distances = []
        for class_index in active_classes:
            class_features = standardized[predictions == class_index]
            target = statistics["class_mean"][class_index]
            distances.append(
                (class_features.mean(0) - target).square().sum().sqrt() / 2.0
            )
        class_loss = torch.stack(distances).mean()
    return global_loss + class_loss, {
        "global_loss": global_loss.detach(),
        "class_loss": class_loss.detach(),
    }


def cfa_online_step(
    model,
    optimizer,
    images,
    statistics,
    full_max_moment=3,
    class_max_moment=1,
    max_grad_norm=1.0,
):
    optimizer.zero_grad(set_to_none=True)
    logits, features = classifier_outputs(model, images)
    loss, parts = cfa_loss(
        logits, features, statistics, full_max_moment, class_max_moment
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return logits.detach(), float(loss.detach()), {
        key: float(value) for key, value in parts.items()
    }
