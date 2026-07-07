from transformers import AutoImageProcessor
from transformers import ViTForImageClassification

MODEL_NAME = "google/vit-base-patch16-224"


def load_model():
    """
    Loads the pretrained Vision Transformer and its image processor.

    Returns:
        processor : AutoImageProcessor
        model     : ViTForImageClassification
    """

    print("Loading pretrained Vision Transformer...")

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME)

    print("Model loaded successfully.\n")

    return processor, model


if __name__ == "__main__":
    processor, model = load_model()
    print(model)