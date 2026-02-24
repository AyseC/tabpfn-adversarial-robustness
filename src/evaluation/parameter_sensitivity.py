"""Parameter Sensitivity Analysis for Attack Methods"""
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

np.random.seed(42)

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack
from src.attacks.nes_attack import NESAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

print("="*80)
print("PARAMETER SENSITIVITY ANALYSIS")
print("="*80)

# Load Wine dataset (representative)
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
print("\nTraining models...")
models = {
    'TabPFN': TabPFNWrapper(device='cpu'),
    'XGBoost': GBDTWrapper(model_type='xgboost'),
    'LightGBM': GBDTWrapper(model_type='lightgbm')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = np.mean(model.predict(X_test) == y_test)
    print(f"  {name}: {acc:.2%}")

n_samples = 10  # Reduced for faster testing
results = {'boundary': {}, 'nes': {}}

# ============================================
# 1. BOUNDARY ATTACK - max_iterations
# ============================================
print("\n" + "="*80)
print("1. BOUNDARY ATTACK - max_iterations")
print("="*80)

max_iterations_values = [50, 100, 200, 500]

for max_iter in max_iterations_values:
    print(f"\n  Testing max_iterations={max_iter}...")
    results['boundary'][f'max_iter_{max_iter}'] = {}
    
    for model_name, model in models.items():
        attack = BoundaryAttack(model, max_iterations=max_iter, epsilon=0.5, verbose=False)
        
        successes = 0
        total = 0
        perturbations = []
        
        for i in range(min(n_samples, len(X_test))):
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred = model.predict(x_orig.reshape(1, -1))[0]
            
            if y_pred != y_true:
                continue
            
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            total += 1
            if success:
                successes += 1
                perturbations.append(pert)
        
        asr = successes / total if total > 0 else 0
        avg_pert = np.mean(perturbations) if perturbations else 0
        
        results['boundary'][f'max_iter_{max_iter}'][model_name] = {
            'asr': asr,
            'avg_perturbation': avg_pert
        }
        print(f"    {model_name}: ASR={asr:.2%}, Pert={avg_pert:.3f}")

# ============================================
# 2. BOUNDARY ATTACK - epsilon
# ============================================
print("\n" + "="*80)
print("2. BOUNDARY ATTACK - epsilon (step size)")
print("="*80)

epsilon_values = [0.1, 0.3, 0.5, 1.0]

for eps in epsilon_values:
    print(f"\n  Testing epsilon={eps}...")
    results['boundary'][f'epsilon_{eps}'] = {}
    
    for model_name, model in models.items():
        attack = BoundaryAttack(model, max_iterations=200, epsilon=eps, verbose=False)
        
        successes = 0
        total = 0
        perturbations = []
        
        for i in range(min(n_samples, len(X_test))):
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred = model.predict(x_orig.reshape(1, -1))[0]
            
            if y_pred != y_true:
                continue
            
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            total += 1
            if success:
                successes += 1
                perturbations.append(pert)
        
        asr = successes / total if total > 0 else 0
        avg_pert = np.mean(perturbations) if perturbations else 0
        
        results['boundary'][f'epsilon_{eps}'][model_name] = {
            'asr': asr,
            'avg_perturbation': avg_pert
        }
        print(f"    {model_name}: ASR={asr:.2%}, Pert={avg_pert:.3f}")

# ============================================
# 3. NES ATTACK - learning_rate
# ============================================
print("\n" + "="*80)
print("3. NES ATTACK - learning_rate")
print("="*80)

lr_values = [0.1, 0.3, 0.5, 1.0]

for lr in lr_values:
    print(f"\n  Testing learning_rate={lr}...")
    results['nes'][f'lr_{lr}'] = {}
    
    for model_name, model in models.items():
        attack = NESAttack(model, max_iterations=200, n_samples=30, 
                          learning_rate=lr, sigma=0.3, verbose=False)
        
        successes = 0
        total = 0
        perturbations = []
        
        for i in range(min(n_samples, len(X_test))):
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred = model.predict(x_orig.reshape(1, -1))[0]
            
            if y_pred != y_true:
                continue
            
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            total += 1
            if success:
                successes += 1
                perturbations.append(pert)
        
        asr = successes / total if total > 0 else 0
        avg_pert = np.mean(perturbations) if perturbations else 0
        
        results['nes'][f'lr_{lr}'][model_name] = {
            'asr': asr,
            'avg_perturbation': avg_pert
        }
        print(f"    {model_name}: ASR={asr:.2%}, Pert={avg_pert:.3f}")

# ============================================
# 4. NES ATTACK - sigma (noise scale)
# ============================================
print("\n" + "="*80)
print("4. NES ATTACK - sigma (noise scale)")
print("="*80)

sigma_values = [0.1, 0.3, 0.5, 1.0]

for sigma in sigma_values:
    print(f"\n  Testing sigma={sigma}...")
    results['nes'][f'sigma_{sigma}'] = {}
    
    for model_name, model in models.items():
        attack = NESAttack(model, max_iterations=200, n_samples=30,
                          learning_rate=0.3, sigma=sigma, verbose=False)
        
        successes = 0
        total = 0
        perturbations = []
        
        for i in range(min(n_samples, len(X_test))):
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred = model.predict(x_orig.reshape(1, -1))[0]
            
            if y_pred != y_true:
                continue
            
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            total += 1
            if success:
                successes += 1
                perturbations.append(pert)
        
        asr = successes / total if total > 0 else 0
        avg_pert = np.mean(perturbations) if perturbations else 0
        
        results['nes'][f'sigma_{sigma}'][model_name] = {
            'asr': asr,
            'avg_perturbation': avg_pert
        }
        print(f"    {model_name}: ASR={asr:.2%}, Pert={avg_pert:.3f}")

# ============================================
# 5. NES ATTACK - n_samples (population size)
# ============================================
print("\n" + "="*80)
print("5. NES ATTACK - n_samples (population size)")
print("="*80)

n_samples_values = [10, 30, 50, 100]

for n_pop in n_samples_values:
    print(f"\n  Testing n_samples={n_pop}...")
    results['nes'][f'n_samples_{n_pop}'] = {}
    
    for model_name, model in models.items():
        attack = NESAttack(model, max_iterations=200, n_samples=n_pop,
                          learning_rate=0.3, sigma=0.3, verbose=False)
        
        successes = 0
        total = 0
        perturbations = []
        
        for i in range(min(n_samples, len(X_test))):
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred = model.predict(x_orig.reshape(1, -1))[0]
            
            if y_pred != y_true:
                continue
            
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            total += 1
            if success:
                successes += 1
                perturbations.append(pert)
        
        asr = successes / total if total > 0 else 0
        avg_pert = np.mean(perturbations) if perturbations else 0
        
        results['nes'][f'n_samples_{n_pop}'][model_name] = {
            'asr': asr,
            'avg_perturbation': avg_pert
        }
        print(f"    {model_name}: ASR={asr:.2%}, Pert={avg_pert:.3f}")

# Save results
Path("results").mkdir(exist_ok=True)
with open('results/parameter_sensitivity.json', 'w') as f:
    json.dump(results, f, indent=2)

# Summary
print("\n" + "="*80)
print("PARAMETER SENSITIVITY SUMMARY")
print("="*80)

print("\nBOUNDARY ATTACK:")
print("-"*50)
print("max_iterations: Higher → More iterations → Higher ASR")
print("epsilon: Higher → Larger steps → Faster but noisier")

print("\nNES ATTACK:")
print("-"*50)
print("learning_rate: Trade-off between speed and precision")
print("sigma: Higher → More exploration → May escape local optima")
print("n_samples: Higher → Better gradient estimate → More queries")

print(f"\n✓ Saved: results/parameter_sensitivity.json")
