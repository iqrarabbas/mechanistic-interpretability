import torch


def get_layerwise_class_outputs(model, images, labels):
    """
    Apply the ViT classifier head to the CLS token after every encoder layer.

    Returns tensors shaped [num_layers, batch_size] containing the correct-class
    probability and top-1 prediction. The embedding output (hidden_states[0]) is
    intentionally excluded because it is not a transformer layer.
    """

    model.eval()

    with torch.no_grad():
        outputs = model(
            pixel_values=images,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states[1:]

    probs_by_layer = []
    predictions_by_layer = []

    for layer_hidden in hidden_states:
        cls_token = layer_hidden[:, 0, :]

        logits = model.classifier(cls_token)
        probs = torch.softmax(logits, dim=-1)

        correct_probs = probs[
            torch.arange(labels.size(0), device=labels.device),
            labels,
        ]

        probs_by_layer.append(correct_probs.detach().cpu())
        predictions_by_layer.append(logits.argmax(dim=-1).detach().cpu())

    probs_by_layer = torch.stack(probs_by_layer, dim=0)
    predictions_by_layer = torch.stack(predictions_by_layer, dim=0)

    return probs_by_layer, predictions_by_layer


def get_correct_class_probs_by_layer(model, images, labels):
    """Return correct-class probabilities after each transformer layer."""
    probs_by_layer, _ = get_layerwise_class_outputs(model, images, labels)
    return probs_by_layer


def get_first_correct_prediction_layers(predictions_by_layer, labels):
    """
    Find the first layer where the correct class is the top-1 prediction.

    Layer numbers are one-based to match the ViT's 12 encoder layers. Samples
    for which the correct class never becomes top-1 are assigned -1, allowing
    callers to apply the paper's restriction to successful samples explicitly.
    """
    labels = labels.detach().cpu()
    correct_by_layer = predictions_by_layer.eq(labels.unsqueeze(0))
    appears = correct_by_layer.any(dim=0)
    first_layers = correct_by_layer.float().argmax(dim=0).to(torch.long) + 1
    first_layers[~appears] = -1
    return first_layers
