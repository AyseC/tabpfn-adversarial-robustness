"""Test improved NES Attack"""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from src.models.gbdt_wrapper import GBDTWrapper
from src.models.tabpfn_wrapper import TabPFNWrapper
from src.attacks.nes_attack import NESAttack

print("Testing Improved NES Attack...")

# Load data
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Test on TabPFN (usually more vulnerable)
print("\nTesting on TabPFN...")
model = TabPFNWrapper(device='cpu')
model.fit(X_train, y_train)

print(f"Model accuracy: {model.score(X_test, y_test):.3f}")

# More aggressive parameters
attack = NESAttack(
    model, 
    max_iterations=200,
    n_samples=100,
    learning_rate=0.2,
    sigma=0.2,
    verbose=False
)

successes = 0
total = 0

for i in range(10):
    x_orig = X_test[i]
    y_true = y_test[i]
    
    if model.predict(x_orig.reshape(1, -1))[0] != y_true:
        continue
    
    total += 1
    x_adv, success, queries, pert = attack.attack(x_orig, y_true)
    
    if success:
        successes += 1
        y_adv = model.predict(x_adv.reshape(1, -1))[0]
        print(f"  [{i+1}] ✓ {y_true}→{y_adv}, pert={pert:.3f}, q={queries}")
    else:
        print(f"  [{i+1}] Failed (pert={pert:.3f})")

print(f"\nTabPFN Success Rate: {successes}/{total} ({100*successes/total if total>0 else 0:.0f}%)")
