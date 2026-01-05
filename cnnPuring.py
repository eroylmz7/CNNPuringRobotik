import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.utils.prune as prune
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# --- 1. AYARLAR VE SABİTLER ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS_TRAIN = 20    # Stochastic Depth için yeterli süre
EPOCHS_FT = 10       # Fine-Tuning için süre
LEARNING_RATE = 0.1
PRUNING_AMOUNT = 0.5 # %50 Budama Hedefi
SEED = 42

# Tekrarlanabilirlik (Reproducibility)
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

print(f"Çalışma Ortamı: {DEVICE} | Seed: {SEED}")

# --- 2. MODEL TANIMLAMASI (ResNet + Stochastic Depth) ---
class StochasticBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, survival_prob=1.0):
        super().__init__()
        self.survival_prob = survival_prob
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # Stochastic Depth: Eğitimde ve olasılık < 1 ise bloğu rastgele kapat
        if self.training and self.survival_prob < 1.0:
            mask = torch.empty((x.shape[0], 1, 1, 1), device=x.device).bernoulli_(self.survival_prob)
            out = (out / self.survival_prob) * mask # Scaling

        out += identity
        return self.relu(out)

class ResNetCifar(nn.Module):
    def __init__(self, depth=20, p_final=0.5):
        super().__init__()
        num_blocks = (depth - 2) // 6
        self.in_channels = 16
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layers = nn.ModuleList()

        # Lineer Azalma Kuralı
        total_blocks = num_blocks * 3
        current_block_idx = 0
        for stage_idx, out_channels in enumerate([16, 32, 64]):
            stride = 2 if stage_idx > 0 else 1
            for b in range(num_blocks):
                p_l = 1.0 - (current_block_idx / total_blocks) * (1.0 - p_final)
                strd = stride if b == 0 else 1
                self.layers.append(StochasticBlock(self.in_channels, out_channels, strd, p_l))
                self.in_channels = out_channels
                current_block_idx += 1

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)

# --- 3. YARDIMCI ARAÇLAR ---
def get_data():
    print("Veri Seti (CIFAR-10) Hazırlanıyor...")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    return DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True), DataLoader(testset, batch_size=100, shuffle=False)

def measure_size_mb(model):
    """Modelin diskteki boyutunu ölçer"""
    torch.save(model.state_dict(), "temp.pth")
    size_mb = os.path.getsize("temp.pth") / 1e6
    os.remove("temp.pth")
    return size_mb

def measure_inference_ms(model, device):
    """Ortalama çıkarım süresini (ms) ölçer"""
    model.eval()
    dummy = torch.randn(1, 3, 32, 32).to(device)
    # Isınma turları
    for _ in range(10): _ = model(dummy)

    start = time.time()
    with torch.no_grad():
        for _ in range(100): _ = model(dummy)
    return ((time.time() - start) / 100) * 1000

# --- 4. ANA ÇALIŞMA AKIŞI ---
def main():
    train_loader, test_loader = get_data()
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model Başlatma
    model = ResNetCifar(depth=20, p_final=0.5).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS_TRAIN)
    criterion = nn.CrossEntropyLoss()

    history = {'train_acc': [], 'test_acc': []}

    # --- FAZ 1: EĞİTİM ---
    print("\n>>> FAZ 1: Stochastic Depth Eğitimi Başlıyor...")
    start_train = time.time()

    for epoch in range(EPOCHS_TRAIN):
        model.train()
        correct, total = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        train_acc = 100. * correct / total

        # Test
        model.eval()
        correct_t, total_t = 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_t += targets.size(0)
                correct_t += predicted.eq(targets).sum().item()
        test_acc = 100. * correct_t / total_t

        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        print(f"Epoch {epoch+1}/{EPOCHS_TRAIN} | Train: %{train_acc:.2f} | Test: %{test_acc:.2f}")

    print(f"Eğitim Tamamlandı ({time.time()-start_train:.1f} sn). Orijinal Acc: %{test_acc:.2f}")

    # Orijinal Metrikler
    orig_size = measure_size_mb(model)
    orig_inf = measure_inference_ms(model, DEVICE)
    torch.save(model.state_dict(), "best_model_original.pth")

    # --- FAZ 2: BUDAMA ---
    print(f"\n>>> FAZ 2: %{PRUNING_AMOUNT*100} Budama (Pruning) Uygulanıyor...")
    parameters_to_prune = []
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))

    prune.global_unstructured(parameters_to_prune, pruning_method=prune.L1Unstructured, amount=PRUNING_AMOUNT)

    # Budanmış Acc
    model.eval()
    correct_p, total_p = 0, 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total_p += targets.size(0)
            correct_p += predicted.eq(targets).sum().item()
    pruned_acc_initial = 100. * correct_p / total_p
    print(f"Budama Sonrası (Fine-Tune Öncesi) Acc: %{pruned_acc_initial:.2f}")

    # --- FAZ 3: FINE-TUNING ---
    print("\n>>> FAZ 3: Fine-Tuning (İyileştirme) Başlıyor...")
    optimizer_ft = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    ft_history = []
    all_preds = []
    all_targets = []

    for epoch in range(EPOCHS_FT):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer_ft.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer_ft.step()

        # Test
        model.eval()
        correct_f, total_f = 0, 0
        epoch_preds = []
        epoch_targets = []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total_f += targets.size(0)
                correct_f += predicted.eq(targets).sum().item()

                # Son epoch için confusion matrix verisi topla
                if epoch == EPOCHS_FT - 1:
                    epoch_preds.extend(predicted.cpu().numpy())
                    epoch_targets.extend(targets.cpu().numpy())

        ft_acc = 100. * correct_f / total_f
        ft_history.append(ft_acc)
        print(f"FT Epoch {epoch+1}/{EPOCHS_FT} | Test Acc: %{ft_acc:.2f}")

    # Pruning'i Kalıcı Yap
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')

    final_size = measure_size_mb(model) # Sparse tensor kullanılmadığı için boyut aynı görünebilir (zip ile fark anlaşılır)
    final_inf = measure_inference_ms(model, DEVICE)
    torch.save(model.state_dict(), "best_model_pruned.pth")

    # --- RAPORLAMA VE GÖRSELLEŞTİRME ---
    print("\n=== SONUÇ RAPORU ===")
    print(f"Orijinal Model -> Boyut: {orig_size:.2f}MB | Hız: {orig_inf:.2f}ms | Acc: %{test_acc:.2f}")
    print(f"Budanmış Model -> Boyut: {final_size:.2f}MB | Hız: {final_inf:.2f}ms | Acc: %{ft_history[-1]:.2f}")

    # Grafikler
    plt.figure(figsize=(15, 5))

    # 1. Başarı Eğrisi
    plt.subplot(1, 3, 1)
    full_history = history['test_acc'] + ft_history
    plt.plot(full_history, label='Test Accuracy', linewidth=2)
    plt.axvline(x=EPOCHS_TRAIN-1, color='red', linestyle='--', label='Pruning')
    plt.title('Eğitim ve Budama Süreci')
    plt.xlabel('Epoch')
    plt.legend()

    # 2. Confusion Matrix
    plt.subplot(1, 3, 2)
    cm = confusion_matrix(epoch_targets, epoch_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix (Final)')
    plt.xticks(rotation=45)

    # 3. Bar Chart
    plt.subplot(1, 3, 3)
    x = ['Orijinal', 'Budanmış']
    y = [test_acc, ft_history[-1]]
    plt.bar(x, y, color=['gray', 'green'])
    plt.ylim(min(y)-5, max(y)+2)
    plt.title('Final Doğruluk Karşılaştırması')

    plt.tight_layout()
    plt.savefig('project_results.png')
    print("Grafikler 'project_results.png' olarak kaydedildi.")

if __name__ == "__main__":
    main()