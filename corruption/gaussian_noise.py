import numpy as np
from PIL import Image


def apply_gaussian_noise(image, severity=4):
    """
    Apply Gaussian noise.

    Severity 4 is a strong noise level.
    This is used to test whether blur fine-tuning transfers to noise.
    """

    image_array = np.array(image).astype(np.float32) / 255.0

    noise_std_by_severity = {
        1: 0.08,
        2: 0.12,
        3: 0.18,
        4: 0.26,
        5: 0.38,
    }

    std = noise_std_by_severity[severity]

    noise = np.random.normal(
        loc=0.0,
        scale=std,
        size=image_array.shape,
    )

    noisy_image = image_array + noise
    noisy_image = np.clip(noisy_image, 0.0, 1.0)

    noisy_image = (noisy_image * 255).astype(np.uint8)

    return Image.fromarray(noisy_image)