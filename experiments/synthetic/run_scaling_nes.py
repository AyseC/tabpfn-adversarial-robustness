"""NES Attack - Synthetic Feature Scaling Experiment"""
import numpy as np
import torch
import json
import warnings
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.nes_attack import NESAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

def get_stratified_attack_indices(y_test, y_pred, n_samples, random_state=42):
    """Select n_samples indices stratified by class from correctly classified samples."""
    correct_indices = np.where(y_pred == y_test)[0]
    if len(correct_indices) <= n_samples:
        return correct_indices
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=random_state)
        _, sel = next(sss.split(correct_indices.reshape(-1, 1), y_test[correct_indices]))
        return correct_indices[sel]
    except ValueError:
        rng = np.random.RandomState(random_state)
        sel = rng.choice(len(correct_indices), n_samples, replace=False)
        return correct_indices[sel]


print("="*80)
print("NES ATTACK - SYNTHETIC FEATURE SCALING EXPERIMENT")
print("="*80)

feature_counts = [5, 10, 15, 20]
n_samples_per_dataset = 200
n_test_attacks = 15

print(f"\nExperimental Design:")
print(f"  Feature counts: {feature_counts}")
print(f"  Samples: {n_samples_per_dataset}")
print(f"  Attacks: {n_test_attacks}")

all_results = {'TabPFN': {}, 'XGBoost': {}, 'LightGBM': {}}

for n_features in feature_counts:
    print(f"\n{'='*80}")
    print(f"TESTING: {n_features} FEATURES")
    print(f"{'='*80}")
    
    X, y = make_classification(
        n_samples=n_samples_per_dataset,
        n_features=n_features,
        n_informative=max(3, n_features // 2),
        n_redundant=max(1, n_features // 5),
        n_classes=2,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    models = {
        'TabPFN': TabPFNWrapper(device='cpu'),
        'XGBoost': GBDTWrapper(model_type='xgboost'),
        'LightGBM': GBDTWrapper(model_type='lightgbm')
    }
    
    for model_name, model in models.items():
        print(f"\n  Testing {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        clean_acc = np.mean(y_pred == y_test)
        attack_indices = get_stratified_attack_indices(y_test, y_pred, n_test_attacks)
        print(f"    Clean Accuracy: {clean_acc:.4f}")
        
        attack = NESAttack(model, max_iterations=200, n_samples=30,
                          learning_rate=0.3, sigma=0.3, verbose=False)
        
        results = []
        for i in attack_indices:
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred_i = model.predict(x_orig.reshape(1, -1))[0]
            
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            y_adv = model.predict(x_adv.reshape(1, -1))[0]
            
            result = AttackResult(
                original_label=y_true, predicted_label=y_pred_i,
                adversarial_label=y_adv, success=success,
                perturbation=pert, queries=queries,
                original_sample=x_orig, adversarial_sample=x_adv
            )
            results.append(result)
        
        metrics = RobustnessMetrics.compute_all(results, y_test[attack_indices], y_pred[attack_indices])
        
        all_results[model_name][str(n_features)] = {
            'n_features': n_features,
            'clean_accuracy': clean_acc,
            'attack_success_rate': metrics['attack_success_rate'],
            'adversarial_accuracy': metrics['adversarial_accuracy'],
            'avg_perturbation': metrics['avg_perturbation'],
            'robustness_score': metrics['robustness_score'],
            'n_samples_tested': len(results)
        }
        print(f"    ASR: {metrics['attack_success_rate']:.2%}")

# Save
Path("results").mkdir(exist_ok=True)
with open('results/synthetic_scaling_nes_experiment.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"\n{'Features':<10} {'Model':<12} {'ASR':<10} {'Adv Acc':<12}")
print("-"*50)
for n_feat in feature_counts:
    for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
        res = all_results[model_name][str(n_feat)]
        print(f"{n_feat:<10} {model_name:<12} {res['attack_success_rate']:<10.2%} {res['adversarial_accuracy']:<12.2%}")
    print("-"*50)

print(f"\n✓ Saved: results/synthetic_scaling_nes_experiment.json")
