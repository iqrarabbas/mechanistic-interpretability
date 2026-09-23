import torch
import torch.nn.functional as F


def attention_entropy(attention, epsilon=1e-12):
    """Mean attention entropy over the batch, heads, and query tokens."""
    probabilities = attention.clamp_min(epsilon)
    entropy = -(probabilities * probabilities.log()).sum(dim=-1)
    return entropy.mean()


def compare_attention_tensors(clean_attention, corrupted_attention):
    """Calculate the paper's paired attention metrics for one layer."""
    if clean_attention.shape != corrupted_attention.shape:
        raise ValueError(
            "Clean and corrupted attention tensors must have the same shape; "
            f"got {clean_attention.shape} and {corrupted_attention.shape}."
        )

    clean_entropy = attention_entropy(clean_attention)
    corrupted_entropy = attention_entropy(corrupted_attention)
    squared_difference = (corrupted_attention - clean_attention).square().mean()
    cosine_similarity = F.cosine_similarity(
        clean_attention,
        corrupted_attention,
        dim=-1,
        eps=1e-8,
    ).mean()

    return {
        "clean_entropy": clean_entropy,
        "corrupted_entropy": corrupted_entropy,
        "entropy_difference": corrupted_entropy - clean_entropy,
        "mean_squared_difference": squared_difference,
        "cosine_similarity": cosine_similarity,
    }


def get_attention_metrics_by_layer(model, clean_images, corrupted_images):
    """Run paired inputs through a ViT and return metrics for every layer."""
    model.eval()
    with torch.no_grad():
        clean_outputs = model(
            pixel_values=clean_images,
            output_attentions=True,
        )
        corrupted_outputs = model(
            pixel_values=corrupted_images,
            output_attentions=True,
        )

    if clean_outputs.attentions is None or corrupted_outputs.attentions is None:
        raise RuntimeError(
            "The model did not return attention tensors. Load it with "
            "attn_implementation='eager'."
        )

    if len(clean_outputs.attentions) != len(corrupted_outputs.attentions):
        raise RuntimeError("Clean and corrupted runs returned different layer counts.")

    return [
        compare_attention_tensors(clean_layer, corrupted_layer)
        for clean_layer, corrupted_layer in zip(
            clean_outputs.attentions,
            corrupted_outputs.attentions,
        )
    ]
