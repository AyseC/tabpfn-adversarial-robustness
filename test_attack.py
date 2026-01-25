"""Test boundary attack"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack

print("Testing Boundary Attack...")

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

# Attack
attack = BoundaryAttack(model, max_iterations=50, verbose=True)

x_orig = X_test[0]
y_true = y_test[0]
y_pred = model.predict(x_orig.reshape(1, -1))[0]

print(f"\nOriginal label: {y_true}, Predicted: {y_pred}")

if y_pred == y_true:
    x_adv, success, queries, pert = attack.attack(x_orig, y_true)
    y_adv = model.predict(x_adv.reshape(1, -1))[0]
    
    print(f"\nAttack {'SUCCESS' if success else 'FAILED'}!")
    print(f"Adversarial prediction: {y_adv}")
    print(f"Perturbation: {pert:.4f}")
    print(f"Queries used: {queries}")
else:
    print("Already misclassified!")
