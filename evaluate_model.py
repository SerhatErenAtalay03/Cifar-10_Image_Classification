import torch
from torchvision import datasets, transforms
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.metrics import classification_report
from tqdm import tqdm

def main():
    # Cihaz seçimi
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nCihaz: {device}")

    # CIFAR-10 test verisi hazırlanıyor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),   # PIL Image olarak gelir
        transforms.ToTensor()            # [0, 1] aralığında Tensor
    ])

    testset = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # Hugging Face'ten model ve processor yükleniyor
    print("\nModel ve işlemci yükleniyor...")
    model = ViTForImageClassification.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10").to(device)
    processor = ViTImageProcessor.from_pretrained("aaraki/vit-base-patch16-224-in21k-finetuned-cifar10")

    y_true = []
    y_pred = []

    print("\nTest verisi üzerinde değerlendirme başlatıldı...\n")
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc="Tahmin ediliyor"):
            # images: Tensor [B, C, H, W] -> Tek tek listeye ayır
            images = [transforms.ToPILImage()(img) for img in images]

            inputs = processor(images=images, return_tensors="pt").to(device)
            outputs = model(**inputs)
            preds = outputs.logits.argmax(dim=-1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # Performans metrikleri
    print("\n📊 Sınıflandırma Raporu (CIFAR-10 Test Seti):\n")
    print(classification_report(y_true, y_pred, target_names=testset.classes))

if __name__ == "__main__":
    main()
