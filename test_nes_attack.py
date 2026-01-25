"""Test NES Attack - improved parameters"""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.nes_attack import NESAttack

print("Testing NES Attack - Multiple Samples...")

# Load data
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = GBDTWrapper(model_type='xgboost')
model.fit(X_train, y_train)

print(f"Model accuracy: {model.score(X_test, y_test):.3f}")

# Try multiple samples with better parameters
attack = NESAttack(
    model, 
    max_iterations=100,      # More iterations
    n_samples=30,            # More samples for gradient
    learning_rate=0.05,      # Larger step
    sigma=0.05,              # Larger perturbation
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
        print(f"  [{i+1}] Failed")

print(f"\nSuccess Rate: {successes}/{total} ({100*successes/total:.0f}%)")

if successes > 0:
    print("✓ NES Attack working!")
else:
    print("⚠️ NES needs parameter tuning")
