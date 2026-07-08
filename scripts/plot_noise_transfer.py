from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

metrics = ["Top-1", "Top-5", "Top-10"]

base_noise = [69.3, 88.1, 92.6]
blur_finetuned_noise = [65.6, 87.1, 91.5]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    base_noise,
    width,
    label="Base ViT on Noise-4",
)

plt.bar(
    x + width / 2,
    blur_finetuned_noise,
    width,
    label="Blur-4 Fine-tuned ViT on Noise-4",
)

plt.ylabel("Accuracy (%)")
plt.title("Does Blur-4 Fine-tuning Transfer to Gaussian Noise Level 4?")
plt.xticks(x, metrics)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

output_path = RESULTS_DIR / "blur_finetuning_transfer_to_noise4.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved graph to: {output_path}")


# Extra graph: negative transfer/drop
drop = [
    blur_finetuned_noise[i] - base_noise[i]
    for i in range(len(metrics))
]

plt.figure(figsize=(8, 5))
plt.bar(metrics, drop)

plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Change in Accuracy (%)")
plt.title("Noise-4 Accuracy Change After Blur-4 Fine-tuning")
plt.ylim(min(drop) - 2, 2)
plt.grid(axis="y", linestyle="--", alpha=0.5)

output_path = RESULTS_DIR / "noise4_drop_after_blur_finetuning.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved graph to: {output_path}")