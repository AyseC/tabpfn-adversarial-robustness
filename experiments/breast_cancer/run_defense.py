"""Defense Mechanisms Experiment - Breast Cancer Dataset"""
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack

print("="*70)
print("DEFENSE MECHANISMS - BREAST CANCER DATASET")
print("="*70)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

print(f"\nDataset: Breast Cancer")
print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

# Standardize
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train TabPFN
print("\nTraining TabPFN...")
model = TabPFNWrapper(device='cpu')
model.fit(X_train, y_train)
clean_acc = np.mean(model.predict(X_test) == y_test)
print(f"  Clean Accuracy: {clean_acc:.2%}")

n_samples = 15
results = {}

# 1. No Defense (Baseline)
print("\n" + "-"*70)
print("1. NO DEFENSE (Baseline)")
print("-"*70)

attack = BoundaryAttack(model, max_iterations=200, epsilon=0.5, verbose=False)
baseline_success = 0
baseline_total = 0

for i in range(min(n_samples, len(X_test))):
    x_orig = X_test[i]
    y_true = y_test[i]
    y_pred = model.predict(x_orig.reshape(1, -1))[0]
    
    if y_pred != y_true:
        continue
    
    x_adv, success, _, pert = attack.attack(x_orig, y_true)
    baseline_total += 1
    if success:
        baseline_success += 1

baseline_asr = baseline_success / baseline_total if baseline_total > 0 else 0
print(f"  ASR: {baseline_asr:.2%} ({baseline_success}/{baseline_total})")
results['no_defense'] = {'asr': baseline_asr, 'n_samples': baseline_total}

# 2. Gaussian Noise Defense
print("\n" + "-"*70)
print("2. GAUSSIAN NOISE DEFENSE")
print("-"*70)

noise_levels = [0.01, 0.03, 0.05, 0.1]

for noise_std in noise_levels:
    success = 0
    total = 0
    
    for i in range(min(n_samples, len(X_test))):
        x_orig = X_test[i]
        y_true = y_test[i]
        y_pred = model.predict(x_orig.reshape(1, -1))[0]
        
        if y_pred != y_true:
            continue
        
        x_adv, attack_success, _, _ = attack.attack(x_orig, y_true)
        
        if attack_success:
            x_defended = x_adv + np.random.normal(0, noise_std, x_adv.shape)
            y_defended = model.predict(x_defended.reshape(1, -1))[0]
            
            if y_defended != y_true:
                success += 1
        
        total += 1
    
    defended_asr = success / total if total > 0 else 0
    recovery = baseline_asr - defended_asr if baseline_asr > 0 else 0
    
    print(f"  σ={noise_std}: ASR={defended_asr:.2%}, Recovery={recovery:.2%}")
    results[f'gaussian_noise_{noise_std}'] = {
        'asr': defended_asr,
        'recovery': recovery,
        'noise_std': noise_std
    }

# 3. Feature Squeezing Defense
print("\n" + "-"*70)
print("3. FEATURE SQUEEZING DEFENSE")
print("-"*70)

bit_depths = [4, 8, 16]

for bits in bit_depths:
    success = 0
    total = 0
    
    for i in range(min(n_samples, len(X_test))):
        x_orig = X_test[i]
        y_true = y_test[i]
        y_pred = model.predict(x_orig.reshape(1, -1))[0]
        
        if y_pred != y_true:
            continue
        
        x_adv, attack_success, _, _ = attack.attack(x_orig, y_true)
        
        if attack_success:
            x_min, x_max = X_train.min(), X_train.max()
            x_normalized = (x_adv - x_min) / (x_max - x_min + 1e-8)
            x_squeezed = np.round(x_normalized * (2**bits - 1)) / (2**bits - 1)
            x_defended = x_squeezed * (x_max - x_min) + x_min
            
            y_defended = model.predict(x_defended.reshape(1, -1))[0]
            
            if y_defended != y_true:
                success += 1
        
        total += 1
    
    defended_asr = success / total if total > 0 else 0
    recovery = baseline_asr - defended_asr if baseline_asr > 0 else 0
    
    print(f"  {bits}-bit: ASR={defended_asr:.2%}, Recovery={recovery:.2%}")
    results[f'feature_squeezing_{bits}bit'] = {
        'asr': defended_asr,
        'recovery': recovery,
        'bit_depth': bits
    }

# 4. Ensemble Voting Defense
print("\n" + "-"*70)
print("4. ENSEMBLE VOTING DEFENSE")
print("-"*70)

xgb = GBDTWrapper(model_type='xgboost')
lgb = GBDTWrapper(model_type='lightgbm')
xgb.fit(X_train, y_train)
lgb.fit(X_train, y_train)

success = 0
total = 0

for i in range(min(n_samples, len(X_test))):
    x_orig = X_test[i]
    y_true = y_test[i]
    y_pred = model.predict(x_orig.reshape(1, -1))[0]
    
    if y_pred != y_true:
        continue
    
    x_adv, attack_success, _, _ = attack.attack(x_orig, y_true)
    
    if attack_success:
        pred_tabpfn = model.predict(x_adv.reshape(1, -1))[0]
        pred_xgb = xgb.predict(x_adv.reshape(1, -1))[0]
        pred_lgb = lgb.predict(x_adv.reshape(1, -1))[0]
        
        votes = [pred_tabpfn, pred_xgb, pred_lgb]
        ensemble_pred = max(set(votes), key=votes.count)
        
        if ensemble_pred != y_true:
            success += 1
    
    total += 1

defended_asr = success / total if total > 0 else 0
recovery = baseline_asr - defended_asr if baseline_asr > 0 else 0

print(f"  Ensemble: ASR={defended_asr:.2%}, Recovery={recovery:.2%}")
results['ensemble_voting'] = {'asr': defended_asr, 'recovery': recovery}

# Save
Path("results").mkdir(exist_ok=True)
with open('results/breast_cancer_defense_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*70)
print("SUMMARY - BREAST CANCER DEFENSE")
print("="*70)
print(f"\n{'Defense':<25} {'ASR':<12} {'Recovery':<12}")
print("-"*50)
print(f"{'No Defense':<25} {results['no_defense']['asr']:<12.2%} {'-':<12}")
for key, val in results.items():
    if key != 'no_defense' and 'recovery' in val:
        print(f"{key:<25} {val['asr']:<12.2%} {val['recovery']:<12.2%}")

print(f"\n✓ Saved: results/breast_cancer_defense_results.json")
