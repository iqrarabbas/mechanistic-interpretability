from PIL import Image
import torch
#from transformers import AutoImageProcessor, ViTForImageClassification

from load_model import load_model

processor, model = load_model()

IMAGE_PATH = "images/Polar.jpg"


image = Image.open(IMAGE_PATH).convert("RGB")

inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

# Convert logits into probabilities
probabilities = torch.softmax(logits, dim=-1)

# Get the Top-5 predictions
top5_probabilities, top5_class_ids = torch.topk(probabilities, k=5)

print("\nTop 5 Predictions")
print("-" * 40)

for probability, class_id in zip(top5_probabilities[0], top5_class_ids[0]):
    label = model.config.id2label[class_id.item()]
    print(f"{label:30} {probability.item()*100:.2f}%")