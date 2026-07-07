from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

metrics = ["Top-1", "Top-5", "Top-10"]

base_clean = [80.6, 96.4, 98.3]
base_blur = [65.0, 84.8, 89.9]

ft_clean = [79.0, 93.9, 96.7]
ft_blur = [72.4, 91.2, 94.9]


def save_grouped_bar(title, base_values, ft_values, filename):
    x = np.arange(len(metrics))
    width = 0.35

    plt.figure(figsize=(8, 5))
    plt.bar(x - width / 2, base_values, width, label="Base ViT")
    plt.bar(x + width / 2, ft_values, width, label="Fine-tuned ViT")

    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xticks(x, metrics)
    plt.ylim(0, 100)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# Graph 1: Clean comparison
save_grouped_bar(
    "Clean ImageNet Accuracy: Base vs Fine-tuned ViT",
    base_clean,
    ft_clean,
    "clean_base_vs_finetuned.png",
)

# Graph 2: Blur comparison
save_grouped_bar(
    "Gaussian Blur Level 4 Accuracy: Base vs Fine-tuned ViT",
    base_blur,
    ft_blur,
    "blur_base_vs_finetuned.png",
)

# Graph 3: Improvement on blur
improvement = [ft_blur[i] - base_blur[i] for i in range(len(metrics))]

plt.figure(figsize=(8, 5))
plt.bar(metrics, improvement)
plt.ylabel("Improvement (%)")
plt.title("Accuracy Improvement After Blur-4 Fine-tuning")
plt.ylim(0, max(improvement) + 3)
plt.grid(axis="y", linestyle="--", alpha=0.5)

output_path = RESULTS_DIR / "blur_improvement_after_finetuning.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")