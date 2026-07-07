from PIL import ImageFilter


def apply_gaussian_blur(image, severity=4):
    """
    Apply Gaussian Blur like the paper.

    In the paper:
    blur severity level = standard deviation of Gaussian blur kernel.

    So:
    severity=1 -> sigma/radius 1
    severity=2 -> sigma/radius 2
    severity=4 -> sigma/radius 4
    """
    return image.filter(ImageFilter.GaussianBlur(radius=severity))