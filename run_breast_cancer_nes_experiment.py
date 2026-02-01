"""NES Attack Experiment - Breast Cancer Dataset"""
import numpy as np
import json
import warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.nes_attack import NESAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

print("="*70)
print("NES ATTACK EXPERIMENT - BREAST CANCER DATASET")
print("="*70)

# Load data
data = load_breast_cancer()
X, y = data.data, data.target

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nDataset: Breast Cancer")
print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

n_samples = 15
all_results = {}

models = {
    'TabPFN': TabPFNWrapper(device='cpu'),
    'XGBoost': GBDTWrapper(model_type='xgboost'),
    'LightGBM': GBDTWrapper(model_type='lightgbm')
}

for model_name, model in models.items():
    print(f"\n{'-'*70}")
    print(f"Model: {model_name}")
    print(f"{'-'*70}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    clean_acc = np.mean(y_pred == y_test)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    
    print(f"\nNES Attack on {n_samples} samples...")
    attack = NESAttack(model, max_iterations=200, n_samples=30, 
                       learning_rate=0.3, sigma=0.3, verbose=False)
    
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
with open('results/breast_cancer_nes_experiment.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*70}")
print("RESULTS")
print(f"{'='*70}")
print(f"\n{'Model':<12} {'ASR':<10} {'Adv Acc':<12} {'Robustness':<12}")
print("-"*50)
for model_name, res in all_results.items():
    print(f"{model_name:<12} {res['attack_success_rate']:<10.2%} "
          f"{res['adversarial_accuracy']:<12.2%} {res['robustness_score']:<12.4f}")

print(f"\n✓ Saved: results/breast_cancer_nes_experiment.json")
