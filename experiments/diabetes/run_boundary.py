"""Diabetes Dataset Experiment - Boundary Attack"""
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

print("="*70)
print("ADVERSARIAL ROBUSTNESS: DIABETES DATASET")
print("="*70)

# Load Diabetes dataset - SAME as NES experiment
print("\nLoading Diabetes dataset from OpenML...")
try:
    diabetes = fetch_openml('diabetes', version=1, as_frame=False, parser='auto')
    X, y = diabetes.data, diabetes.target
    
    # Convert target to int
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(int)
    
    if len(np.unique(y)) > 2:
        y = (y > 0).astype(int)
    
    print("  ✓ Loaded from OpenML (Pima Indians Diabetes)")
except Exception as e:
    print(f"  OpenML failed: {e}")
    print("  Using sklearn diabetes dataset with binary conversion...")
    from sklearn.datasets import load_diabetes as load_diabetes_reg
    diabetes_reg = load_diabetes_reg()
    X = diabetes_reg.data
    y = (diabetes_reg.target > np.median(diabetes_reg.target)).astype(int)

print(f"\nDataset: Diabetes")
print(f"  Samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")

# Standardize
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

n_samples = 15

# Models
models = {
    'XGBoost': GBDTWrapper(model_type='xgboost'),
    'LightGBM': GBDTWrapper(model_type='lightgbm'),
    'TabPFN': TabPFNWrapper(device='cpu')
}

all_results = {}

for model_name, model in models.items():
    print(f"\n{'-'*70}")
    print(f"Model: {model_name}")
    print(f"{'-'*70}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    clean_acc = np.mean(y_pred == y_test)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    
    print(f"\nAttacking {n_samples} samples...")
    attack = BoundaryAttack(model, max_iterations=200, epsilon=0.5, verbose=False)
    
    results = []
    for i in range(min(n_samples, len(X_test))):
        x_orig = X_test[i]
        y_true = y_test[i]
        y_pred_i = model.predict(x_orig.reshape(1, -1))[0]
        
        if y_pred_i != y_true:
            continue
        
        x_adv, success, queries, pert = attack.attack(x_orig, y_true)
        y_adv = model.predict(x_adv.reshape(1, -1))[0]
        
        result = AttackResult(
            original_label=y_true,
            predicted_label=y_pred_i,
            adversarial_label=y_adv,
            success=success,
            perturbation=pert,
            queries=queries,
            original_sample=x_orig,
            adversarial_sample=x_adv
        )
        results.append(result)
        
        if success:
            print(f"  [{i+1}] ✓ {y_true}→{y_adv}, pert={pert:.2f}, q={queries}")
    
    print(f"\nSuccessful attacks: {sum(1 for r in results if r.success)}/{len(results)}")
    
    metrics = RobustnessMetrics.compute_all(results, y_test[:n_samples], y_pred[:n_samples])
    
    all_results[model_name] = {
        'clean_accuracy': clean_acc,
        'attack_success_rate': metrics['attack_success_rate'],
        'adversarial_accuracy': metrics['adversarial_accuracy'],
        'avg_perturbation': metrics['avg_perturbation'],
        'avg_queries': metrics['avg_queries'],
        'robustness_score': metrics['robustness_score']
    }
    
    print(f"\n{model_name} Metrics:")
    print(f"  ASR: {metrics['attack_success_rate']:.2%}")
    print(f"  Adversarial Accuracy: {metrics['adversarial_accuracy']:.2%}")

# Save results
Path("results").mkdir(exist_ok=True)
with open('results/diabetes_experiment.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*70}")
print("RESULTS - DIABETES DATASET")
print(f"{'='*70}")
print(f"\n{'Model':<12} {'ASR':<10} {'Adv Acc':<12} {'Robustness':<12}")
print("-"*50)
for model_name, res in all_results.items():
    print(f"{model_name:<12} {res['attack_success_rate']:<10.2%} "
          f"{res['adversarial_accuracy']:<12.2%} {res['robustness_score']:<12.4f}")

print(f"\n✓ Saved: results/diabetes_experiment.json")
