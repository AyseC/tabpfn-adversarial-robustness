"""Test boundary attack on multiple samples"""
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from src.models.gbdt_wrapper import GBDTWrapper
from src.models.tabpfn_wrapper import TabPFNWrapper
from src.attacks.boundary_attack import BoundaryAttack

print("="*60)
print("ADVERSARIAL ATTACK TEST - Multiple Samples")
print("="*60)

# Load data
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Test multiple models
models = {
    'XGBoost': GBDTWrapper(model_type='xgboost'),
    'TabPFN': TabPFNWrapper(device='cpu')
}

for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")
    
    # Train
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model accuracy: {acc:.3f}")
    
    # Attack with more tries
    attack = BoundaryAttack(model, max_iterations=200, epsilon=0.5, verbose=False)
    
    successes = 0
    total_queries = 0
    perturbations = []
    
    n_samples = 10  # Test 10 samples
    
    print(f"\nAttacking {n_samples} samples...")
    
    for i in range(min(n_samples, len(X_test))):
        x_orig = X_test[i]
        y_true = y_test[i]
        y_pred = model.predict(x_orig.reshape(1, -1))[0]
        
        if y_pred != y_true:
            continue  # Skip misclassified
        
        # Improved initialization - try different scales
        x_adv, success, queries, pert = attack.attack(x_orig, y_true)
        
        if success:
            successes += 1
            perturbations.append(pert)
            total_queries += queries
            y_adv = model.predict(x_adv.reshape(1, -1))[0]
            print(f"  Sample {i}: SUCCESS! {y_true}→{y_adv}, pert={pert:.3f}, queries={queries}")
        else:
            print(f"  Sample {i}: Failed")
    
    # Results
    print(f"\n{'-'*60}")
    if successes > 0:
        print(f"Attack Success Rate: {successes}/{n_samples} ({100*successes/n_samples:.1f}%)")
        print(f"Average Perturbation: {np.mean(perturbations):.4f}")
        print(f"Average Queries: {total_queries/successes:.0f}")
    else:
        print("No successful attacks!")
        print("Tip: Model might be very robust or initialization needs improvement")

print("\n" + "="*60)
print("TEST COMPLETE!")
print("="*60)
