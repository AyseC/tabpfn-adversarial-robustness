"""İlk gerçek deney - Wine dataset ile tüm modelleri test"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper

print("="*60)
print("İLK DENEY: Wine Dataset - Model Karşılaştırması")
print("="*60)

# Veri yükle
print("\nVeri yükleniyor...")
data = load_wine()
X, y = data.data, data.target

# Binary classification için sadece 2 sınıf
mask = y < 2
X, y = X[mask], y[mask]

print(f"Toplam örnek: {len(X)}")
print(f"Özellik sayısı: {X.shape[1]}")
print(f"Sınıf dağılımı: Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

print(f"\nTrain set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")

# Modelleri tanımla
models = {
    'TabPFN': TabPFNWrapper(device='cpu'),
    'XGBoost': GBDTWrapper(model_type='xgboost'),
    'LightGBM': GBDTWrapper(model_type='lightgbm')
}

print("\n" + "="*60)
print("MODEL EĞİTİMİ VE DEĞERLENDİRME")
print("="*60)

results = {}

for name, model in models.items():
    print(f"\n{name}:")
    print("-" * 40)
    
    # Eğit
    model.fit(X_train, y_train)
    
    # Değerlendir
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    
    # Basit robustness testi
    noise_level = 0.1
    X_noisy = X_test + np.random.randn(*X_test.shape) * noise_level
    noisy_acc = model.score(X_noisy, y_test)
    
    print(f"  Noisy Accuracy (ε={noise_level}): {noisy_acc:.4f}")
    print(f"  Accuracy Drop:  {test_acc - noisy_acc:.4f}")
    
    results[name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'noisy_acc': noisy_acc,
        'drop': test_acc - noisy_acc
    }

# Özet
print("\n" + "="*60)
print("ÖZET SONUÇLAR")
print("="*60)
print(f"\n{'Model':<12} {'Test Acc':<12} {'Noisy Acc':<12} {'Drop':<10}")
print("-" * 50)

for name, res in results.items():
    print(f"{name:<12} {res['test_acc']:<12.4f} {res['noisy_acc']:<12.4f} {res['drop']:<10.4f}")

# En iyi model
best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✓ En yüksek test accuracy: {best_model[0]} ({best_model[1]['test_acc']:.4f})")

most_robust = min(results.items(), key=lambda x: x[1]['drop'])
print(f"✓ En robust (en az düşüş): {most_robust[0]} (düşüş: {most_robust[1]['drop']:.4f})")

print("\n" + "="*60)
print("DENEY TAMAMLANDI! 🎉")
print("="*60)
