from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

metrics = ["Top-1", "Top-5", "Top-10"]

clean = [79.8, 94.6, 97.1]
noise = [69.3, 88.1, 92.6]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(x - width / 2, clean, width, label="Clean")
plt.bar(x + width / 2, noise, width, label="Gaussian Noise Level 4")

plt.ylabel("Accuracy (%)")
plt.title("Base ViT Accuracy on Clean vs Gaussian Noise Level 4")
plt.xticks(x, metrics)
plt.ylim(0, 100)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

output_path = RESULTS_DIR / "base_vit_clean_vs_noise4_accuracy.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved graph to: {output_path}")