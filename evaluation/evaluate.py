import torch
from tqdm import tqdm


def evaluate_model(model, dataloader, device):
    model.eval()

    correct_top1 = 0
    correct_top5 = 0
    correct_top10 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(pixel_values=images)
            logits = outputs.logits

            _, top10_preds = torch.topk(logits, k=10, dim=1)

            correct_top1 += (top10_preds[:, 0] == labels).sum().item()
            correct_top5 += (top10_preds[:, :5] == labels.unsqueeze(1)).any(dim=1).sum().item()
            correct_top10 += (top10_preds == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    return {
        "top1": correct_top1 / total,
        "top5": correct_top5 / total,
        "top10": correct_top10 / total,
    }