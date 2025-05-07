# CIFAR-10 Görsel Sınıflandırma Projesi

Bu proje,hazır ,önceden eğitilmiş bir model olan Vision Transformer (ViT) mimarisi kullanılarak CIFAR-10 veri seti üzerinde görsel sınıflandırma gerçekleştiren bir derin öğrenme uygulamasıdır.

## Proje Özeti

Bu uygulama, modern derin öğrenme tekniklerinden biri olan Vision Transformer (ViT) mimarisini kullanarak, CIFAR-10 veri setindeki 10 farklı nesne kategorisini sınıflandırmaktadır. Uygulama, kullanıcı dostu bir arayüz sunarak, yüklenen görüntülerin gerçek zamanlı sınıflandırılmasını sağlamaktadır. 

## Teknik Detaylar

- **Model Mimarisi**: Vision Transformer (ViT) base modeli
- **Veri Seti**: CIFAR-10 (60,000 eğitim, 10,000 test görüntüsü)
- **Sınıf Sayısı**: 10 farklı nesne kategorisi
- **Görüntü Boyutu**: 224x224 piksel
- **Framework**: PyTorch ve Hugging Face Transformers

## Sınıf Kategorileri

1. Uçak (Airplane)
2. Otomobil (Automobile)
3. Kuş (Bird)
4. Kedi (Cat)
5. Geyik (Deer)
6. Köpek (Dog)
7. Kurbağa (Frog)
8. At (Horse)
9. Gemi (Ship)
10. Kamyon (Truck)

## Kurulum

1. Gerekli paketlerin yüklenmesi:
```bash
pip install -r requirements.txt
```
2. Modelin yüklenmesi
```bash
python load_model.py

```
3. Uygulamanın çalıştırılması:
```bash
streamlit run app.py
```

## Model Değerlendirme

Model performansını değerlendirmek için aşağıdaki komutu çalıştırabilirsiniz:
```bash
python evaluate_model.py
```

Bu komut, modelin accuracy, precision, recall ve F1-score gibi performans metriklerini hesaplayacaktır.

## Teknik Gereksinimler

- Python 3.7+
- PyTorch
- Transformers
- Streamlit
- Pillow
- scikit-learn

## Dipnot
Bu projede arayüz ve sonuçlar klasöründe programın arayüzüne ve programın sağlam çalışıp çalışmadığını test etmek için bu cifar-10 verisetindeki 10 tane kategorilerin her biri ile ilgili örneğin at kategorisi ile ilgili rasgele 5 tane görsel yükledik uygulamamıza bu 5 görselin 5 ini doğru tahmin etti yani kısacası cifar-10 verisetindeki 10 tane kategorilerin her bir kategori ile ilgili 5 tane rastgele görsel yükledik 5 ini doğru bir şekilde programımız tahmin etti. Bu ilgili testlerin sonucuna ulaşmak için ise sonuçlar klasöründen ulaşabilirsiniz.Modelimizin accuracy,presicion,recall,f1-score değerlerini görüntülemek için ise modelimizin accuracy,presicion,recall,f1-score değerleri adlı klasörden ulaşabilirsiniz. Bu programdaki .py dosyalarını çalıştırmadan önce gerekli paketlerin yüklenmesi gerekmektedir.

Ayrıca Uygulama Python 3.10 veya 3.11 ile en iyi performansı göstermektedir ve programın daha iyi ve daha sağlıklı bir şekilde kurulması için .venv klasörü oluşturulup sonrasında bu klasörün içinde bir sanal ortamın oluşturulup sırası ile aşağıdaki kodları kullanarak aktif edilmelidir. 

python -m venv venv
venv\Scripts\activate       # Windows işletim sistemi için
source venv/bin/activate   # macOS/Linux işletim sistemi için

Sanal ortam(venv), bir Python projesinde kullanacağınız programları ve kütüphaneleri, bilgisayarının geri kalanından ayrı bir klasörde saklayan özel bir alandır. Böylece her proje, kendi ihtiyacı olan sürümleri kullanır ve diğer projelerle karışmaz. Böylece programımızın kurulumunda ve programımızın iyi bir şekilde çalışmasına yardımcı olur.

Son olarak bu ViT modeli kullanılan Cifar-10 görsel sınıflandırıcı uygulamasının demo videosuna ulaşmak için demo klasörü içerisinde yer alan txt dosyası içerisindeki drive linkinden ilgili demo videosuna ulaşabilirsiniz.

## Lisans

Bu proje MIT lisansı altında lisanslanmıştır. 