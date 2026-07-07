from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

metrics = ["Clean Top-1", "Clean Top-5", "Clean Top-10",
           "Blur Top-1", "Blur Top-5", "Blur Top-10"]

base = [80.6, 96.4, 98.3, 65.0, 84.8, 89.9]
finetuned = [79.2, 95.7, 97.0, 74.2, 92.5, 95.5]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(12, 6))
plt.bar(x - width / 2, base, width, label="Base ViT")
plt.bar(x + width / 2, finetuned, width, label="Fine-tuned ViT (1 epoch)")

plt.ylabel("Accuracy (%)")
plt.title("Base vs Fine-tuned ViT on Clean and Gaussian Blur Level 4")
plt.xticks(x, metrics, rotation=30, ha="right")
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

output_path = RESULTS_DIR / "base_vs_finetuned_1epoch_accuracy.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved graph to: {output_path}")