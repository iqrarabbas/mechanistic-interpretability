import torch


def get_correct_class_probs_by_layer(model, images, labels):
    """
    Logit Lens for ViT.

    For each hidden layer:
    - take CLS token
    - apply model classifier head
    - compute probability of correct class
    """

    model.eval()

    with torch.no_grad():
        outputs = model(
            pixel_values=images,
            output_hidden_states=True,
        )

    hidden_states = outputs.hidden_states

    probs_by_layer = []

    for layer_hidden in hidden_states:
        cls_token = layer_hidden[:, 0, :]

        logits = model.classifier(cls_token)
        probs = torch.softmax(logits, dim=-1)

        correct_probs = probs[
            torch.arange(labels.size(0), device=labels.device),
            labels,
        ]

        probs_by_layer.append(correct_probs.detach().cpu())

    probs_by_layer = torch.stack(probs_by_layer, dim=0)

    return probs_by_layer