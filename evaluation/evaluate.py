import torch
from tqdm import tqdm


def evaluate_model(model, dataloader, device, num_ece_bins=15):
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total = 0
    all_confidences = []
    all_correct = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=1)

            _, top10_preds = torch.topk(logits, k=10, dim=1)

            correct_top1 += (top10_preds[:, 0] == labels).sum().item()
            correct_top5 += (top10_preds[:, :5] == labels.unsqueeze(1)).any(dim=1).sum().item()
            correct_top10 += (top10_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)
            confidences, predictions = probabilities.max(dim=1)
            all_confidences.append(confidences.cpu())
            all_correct.append(predictions.eq(labels).float().cpu())

    confidences = torch.cat(all_confidences)
    correct = torch.cat(all_correct)
    bin_boundaries = torch.linspace(0, 1, num_ece_bins + 1)
    ece = torch.tensor(0.0)

    for lower, upper in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        # Include confidence 0 in the first bin and confidence 1 in the last.
        in_bin = (confidences > lower) & (confidences <= upper)
        if lower == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        if in_bin.any():
            bin_accuracy = correct[in_bin].mean()
            bin_confidence = confidences[in_bin].mean()
            ece += in_bin.float().mean() * (bin_accuracy - bin_confidence).abs()

    return {
        "top1": correct_top1 / total,
        "top5": correct_top5 / total,
        "top10": correct_top10 / total,
        "ece": ece.item(),
    }
