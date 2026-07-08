from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

metrics = ["Top-1", "Top-5", "Top-10"]

# Same-corruption robustness
base_blur = [65.0, 84.8, 89.9]
blur_ft_on_blur = [72.4, 91.2, 94.9]

base_noise = [69.3, 88.1, 92.6]
noise_ft_on_noise = [73.0, 92.4, 95.2]

# Cross-corruption transfer
blur_ft_on_noise = [65.6, 87.1, 91.5]
noise_ft_on_blur = [61.8, 84.5, 89.2]


def grouped_bar(title, series_dict, filename, y_min=55, y_max=100):
    x = np.arange(len(metrics))
    width = 0.8 / len(series_dict)

    plt.figure(figsize=(10, 6))

    for i, (label, values) in enumerate(series_dict.items()):
        offset = (i - (len(series_dict) - 1) / 2) * width
        plt.bar(x + offset, values, width, label=label)

    plt.ylabel("Accuracy (%)")
    plt.title(title)
    plt.xticks(x, metrics)
    plt.ylim(y_min, y_max)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    output_path = RESULTS_DIR / filename
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {output_path}")


# Graph 1: same-corruption improvement
grouped_bar(
    title="Same-Corruption Robustness Improvement",
    series_dict={
        "Base on Blur-4": base_blur,
        "Blur-4 FT on Blur-4": blur_ft_on_blur,
        "Base on Noise-4": base_noise,
        "Noise-4 FT on Noise-4": noise_ft_on_noise,
    },
    filename="same_corruption_robustness_improvement.png",
)

# Graph 2: cross-corruption transfer
grouped_bar(
    title="Cross-Corruption Transfer: Fine-tuned Models on Unseen Corruption Type",
    series_dict={
        "Base on Noise-4": base_noise,
        "Blur-4 FT on Noise-4": blur_ft_on_noise,
        "Base on Blur-4": base_blur,
        "Noise-4 FT on Blur-4": noise_ft_on_blur,
    },
    filename="cross_corruption_transfer_results.png",
)

# Graph 3: transfer change relative to base
blur_to_noise_change = [
    blur_ft_on_noise[i] - base_noise[i]
    for i in range(len(metrics))
]

noise_to_blur_change = [
    noise_ft_on_blur[i] - base_blur[i]
    for i in range(len(metrics))
]

x = np.arange(len(metrics))
width = 0.35

plt.figure(figsize=(9, 5))

plt.bar(
    x - width / 2,
    blur_to_noise_change,
    width,
    label="Blur-4 FT tested on Noise-4",
)

plt.bar(
    x + width / 2,
    noise_to_blur_change,
    width,
    label="Noise-4 FT tested on Blur-4",
)

plt.axhline(0, linestyle="--", linewidth=1)
plt.ylabel("Change from Base Accuracy (%)")
plt.title("Cross-Corruption Transfer Change Relative to Base Model")
plt.xticks(x, metrics)
plt.ylim(-5, 2)
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

output_path = RESULTS_DIR / "cross_corruption_transfer_change.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved: {output_path}")