import streamlit as st
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image
import torch

st.title("CIFAR-10 Görsel Sınıflandırma (ViT Modeli)")

# Türkçe sınıf adları
label_map_tr = {
    "airplane": "Uçak",
    "automobile": "Otomobil",
    "bird": "Kuş",
    "cat": "Kedi",
    "deer": "Geyik",
    "dog": "Köpek",
    "frog": "Kurbağa",
    "horse": "At",
    "ship": "Gemi",
    "truck": "Kamyon"
}

@st.cache_resource
def load_model():
    model = ViTForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    processor = ViTImageProcessor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")
    return model, processor

model, processor = load_model()

uploaded_file = st.file_uploader("Bir görsel yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_idx = logits.argmax(-1).item()
        label_en = model.config.id2label[predicted_idx]
        label_tr = label_map_tr.get(label_en, "Bilinmeyen")

    st.success(f"Tahmin Edilen Sınıf: **{label_tr}**")
