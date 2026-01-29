import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np 

# 1. Cihaz Belirleme
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Cihaz: {device}")

# 2. Veri Yükleyiciler
def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_set = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

train_loader, test_loader = get_data_loaders()

# 3. Model Mimarisi
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

# Modeli GPU'ya gönder
model = NeuralNetwork().to(device)

# 4. Optimizer ve Loss Tanımlama (Lambda'yı burada çağırıyoruz!)
define_loss_and_optimizer = lambda model: (
    nn.CrossEntropyLoss(), 
    optim.Adam(model.parameters(), lr=0.001)
)
# İşte burası eksikti, değişkenleri oluşturuyoruz:
criterion, optimizer = define_loss_and_optimizer(model)

# 5. Eğitim Fonksiyonu
def train_model(model, train_loader, criterion, optimizer, epochs=15):
    model.train()
    train_losses = []
    
    print("Eğitim Başlıyor...")
    
    for epoch in range(epochs):
        total_loss = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()           # Temizle
            predictions = model(images)     # Tahmin et
            loss = criterion(predictions, labels) # Hatayı ölç
            loss.backward()                 # Hatayı yay
            optimizer.step()                # Öğren
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        
    print("Eğitim Tamamlandı!")
    return train_losses

# 6. Eğitimi Başlat (Fonksiyonun dışında çağırıyoruz)
loss_history = train_model(model, train_loader, criterion, optimizer, epochs=15)

# 7. Sonuç Grafiğini Çiz (Eğitim bittikten sonra)
plt.figure(figsize=(10, 5))
plt.plot(loss_history, marker='o', linestyle='-', color='b', label='Eğitim Kaybı')
plt.title('Epoch Bazlı Kayıp (Loss) Grafiği')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

def test_model(model,test_loader):
    model.eval()
    corect = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            corect += (predicted == labels).sum().item()
    print(f'Test Setindeki Doğruluk: %{100 * corect / total:.3f}')
test_model(model,test_loader)
# --- Modeli Kaydetme (Çok Önemli) ---
torch.save(model.state_dict(), "mnist_model.pth")
print("✅ Model 'mnist_model.pth' olarak kaydedildi!")