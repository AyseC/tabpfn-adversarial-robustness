"""Quick NES test - faster parameters"""
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from src.models.tabpfn_wrapper import TabPFNWrapper
from src.attacks.nes_attack import NESAttack

print("Quick NES Test...")

# Load data
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# TabPFN
model = TabPFNWrapper(device='cpu')
model.fit(X_train, y_train)
print(f"Model accuracy: {model.score(X_test, y_test):.3f}")

# FASTER parameters
attack = NESAttack(
    model, 
    max_iterations=50,      # 200 → 50 (4x faster)
    n_samples=30,           # 100 → 30 (3x faster)
    learning_rate=0.3,      # More aggressive
    sigma=0.3,
    max_queries=2000,       # Hard limit
    verbose=True            # Show progress
)

print("\nTesting 3 samples...")

successes = 0
for i in range(3):  # Only 3 samples
    x_orig = X_test[i]
    y_true = y_test[i]
    
    if model.predict(x_orig.reshape(1, -1))[0] != y_true:
        continue
    
    print(f"\nSample {i+1}:")
    x_adv, success, queries, pert = attack.attack(x_orig, y_true)
    
    if success:
        successes += 1
        y_adv = model.predict(x_adv.reshape(1, -1))[0]
        print(f"  ✓ SUCCESS: {y_true}→{y_adv}, pert={pert:.3f}, q={queries}")
    else:
        print(f"  ✗ Failed (pert={pert:.3f}, q={queries})")

print(f"\nResult: {successes}/3 successful")

if successes > 0:
    print("✓ NES Attack is working!")
else:
    print("⚠️ NES still not working - consider using Boundary only")
