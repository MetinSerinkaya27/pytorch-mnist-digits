import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- 1. Model Mimarisi (Aynısını buraya da koyuyoruz ki yükleyebilsin) ---
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# --- 2. Ayarlar ve Modeli Yükleme ---
# Önbellek (Cache) kullanıyoruz ki her tıklamada modeli tekrar yüklemesin
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNetwork().to(device)
    # Eğer model dosyan varsa burayı aç:
    try:
        model.load_state_dict(torch.load("mnist_model.pth", map_location=device))
        model.eval()
    except FileNotFoundError:
        st.error("Model dosyası (mnist_model.pth) bulunamadı! Lütfen önce eğitimi tamamla.")
    return model, device

model, device = load_model()

# --- 3. Sayfa Tasarımı ---
st.title("🧠 Yapay Zeka Rakam Tanıma")
st.write("Aşağıdaki siyah alana 0-9 arası bir rakam çiz ve tahmin etmesini bekle.")

# İki sütun yapalım: Solda Çizim, Sağda Sonuç
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Çizim Alanı")
    # Çizim Tahtası (Canvas)
    canvas_result = st_canvas(
        fill_color="#000000",  # Dolgu rengi
        stroke_width=30,       # Kalem kalınlığı (Biraz kalın olsun)
        stroke_color="#FFFFFF",# Kalem rengi (Beyaz)
        background_color="#000000", # Arka plan (Siyah - MNIST formatına uygun)
        height=280,            # Yükseklik
        width=280,             # Genişlik (28x28'in 10 katı)
        drawing_mode="freedraw",
        key="canvas",
    )

# --- 4. Tahmin İşlemi ---
if canvas_result.image_data is not None:
    # Çizilen resmi al
    img_data = canvas_result.image_data
    
    # Resmi PIL formatına ve Grayscale'e çevir
    image = Image.fromarray(img_data.astype('uint8')).convert('L')
    
    # Resmi 28x28 boyutuna küçült (Modelimiz böyle istiyor)
    image = image.resize((28, 28))
    
    # Görselleştirmek için (İsteğe bağlı, backend ne görüyor diye)
    with col2:
        st.subheader("Yapay Zekanın Gördüğü")
        st.image(image, caption="28x28 Piksel Hali", width=150)
        
        # --- 5. Tahmin Butonu ---
        if st.button('Tahmin Et! 🚀'):
            # Görüntüyü Tensor'a çevir
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            # Modele verilecek hale getir (Batch boyutu ekle: [1, 1, 28, 28])
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            # Modele sor
            with torch.no_grad():
                output = model(input_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Sonucu yazdır
            st.success(f"Bu Sayı: {predicted.item()}")
            st.info(f"Eminlik Oranı: %{confidence.item()*100:.2f}")
            
            # Tüm ihtimalleri grafik olarak göster (Bar Chart)
            st.bar_chart(probabilities.cpu().numpy()[0])