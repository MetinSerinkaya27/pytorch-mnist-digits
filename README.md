# 🧠 PyTorch MNIST Digit Recognizer

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0-red)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B)
![License](https://img.shields.io/badge/License-MIT-green)

Bu proje, el yazısı rakamları (0-9) tanımak için geliştirilmiş uçtan uca bir **Derin Öğrenme (Deep Learning)** uygulamasıdır. PyTorch kullanılarak eğitilen Yapay Sinir Ağı (ANN), **Streamlit** arayüzü üzerinden kullanıcıların çizimlerini anlık olarak tahmin eder.

## 📸 Ekran Görüntüsü

![Demo Uygulama](https://raw.githubusercontent.com/27MetinSerinkaya/pytorch-mnist-digits/main/screenshots/demo.png)


## 🚀 Özellikler

* **Yapay Sinir Ağı (ANN/MLP):** PyTorch ile sıfırdan oluşturulmuş, 3 katmanlı özelleştirilmiş mimari.
* **İnteraktif Arayüz:** `streamlit-drawable-canvas` kütüphanesi ile tarayıcı üzerinde gerçek zamanlı çizim imkanı.
* **CUDA Hızlandırma:** NVIDIA GPU (RTX Serisi) desteği ile yüksek performanslı eğitim ve çıkarım (Inference).
* **Görselleştirme:** Eğitim kaybı (Loss) grafikleri ve tahmin olasılık dağılımları.
* **Model Kaydı:** Eğitilen model `.pth` formatında kaydedilir ve tekrar kullanılabilir.

## 🛠️ Teknoloji Yığını

* **Dil:** Python
* **Core AI:** PyTorch (Torch & Torchvision)
* **Web Framework:** Streamlit
* **Veri İşleme & Görselleştirme:** NumPy, Matplotlib, PIL
* **Veri Seti:** MNIST (60.000 Eğitim / 10.000 Test Verisi)

## 🏗️ Model Mimarisi

Projede kullanılan model (MLP) şu katmanlardan oluşur:

1.  **Input Layer:** 28x28 (784) Piksel (Flatten)
2.  **Hidden Layer 1:** 128 Nöron + ReLU Aktivasyonu
3.  **Hidden Layer 2:** 64 Nöron + ReLU Aktivasyonu
4.  **Output Layer:** 10 Nöron (Softmax öncesi Logits)

## ⚙️ Kurulum ve Çalıştırma

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyin:

### 1. Projeyi Klonlayın
```bash
git clone [https://github.com/27MetinSerinkaya/pytorch-mnist-digits.git](https://github.com/27MetinSerinkaya/pytorch-mnist-digits.git)
cd pytorch-mnist-digits