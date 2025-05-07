from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch
import requests

# Model ve görüntü işleyiciyi yükle
model = ViTForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
processor = ViTImageProcessor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")

# Örnek bir görsel yükle (isteğe bağlı)
image = Image.open(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/cifar10-airplane.png", stream=True).raw)

# Görseli işle
inputs = processor(images=image, return_tensors="pt")

# Tahmin yap
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()

print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")
