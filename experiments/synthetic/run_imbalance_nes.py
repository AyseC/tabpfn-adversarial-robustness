"""NES Attack - Synthetic Class Imbalance Experiment"""
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.nes_attack import NESAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

print("="*80)
print("NES ATTACK - SYNTHETIC CLASS IMBALANCE EXPERIMENT")
print("="*80)

imbalance_ratios = [0.5, 0.3, 0.2, 0.1]
n_samples_per_dataset = 200
n_test_attacks = 15
n_features = 10

print(f"\nExperimental Design:")
print(f"  Imbalance ratios: {imbalance_ratios}")
print(f"  Features: {n_features}")
print(f"  Samples: {n_samples_per_dataset}")
print(f"  Attacks: {n_test_attacks}")

all_results = {'TabPFN': {}, 'XGBoost': {}, 'LightGBM': {}}

for ratio in imbalance_ratios:
    print(f"\n{'='*80}")
    print(f"TESTING: {int((1-ratio)*100)}/{int(ratio*100)} CLASS RATIO")
    print(f"{'='*80}")
    
    X, y = make_classification(
        n_samples=n_samples_per_dataset,
        n_features=n_features,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        weights=[1-ratio, ratio],
        random_state=42,
        flip_y=0.01
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
        print(f"    Clean Accuracy: {clean_acc:.4f}")
        
        attack = NESAttack(model, max_iterations=200, n_samples=30,
                          learning_rate=0.3, sigma=0.3, verbose=False)
        
        results = []
        for i in range(min(n_test_attacks, len(X_test))):
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred_i = model.predict(x_orig.reshape(1, -1))[0]
            
            if y_pred_i != y_true:
                continue
            
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            y_adv = model.predict(x_adv.reshape(1, -1))[0]
            
            result = AttackResult(
                original_label=y_true, predicted_label=y_pred_i,
                adversarial_label=y_adv, success=success,
                perturbation=pert, queries=queries,
                original_sample=x_orig, adversarial_sample=x_adv
            )
            results.append(result)
        
        metrics = RobustnessMetrics.compute_all(results, y_test[:n_test_attacks], y_pred[:n_test_attacks])
        
        ratio_key = f"{int((1-ratio)*100)}/{int(ratio*100)}"
        all_results[model_name][ratio_key] = {
            'imbalance_ratio': ratio,
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
with open('results/synthetic_imbalance_nes_experiment.json', 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*80}")
print("RESULTS")
print(f"{'='*80}")
print(f"\n{'Ratio':<12} {'Model':<12} {'ASR':<10} {'Adv Acc':<12}")
print("-"*50)
for ratio in imbalance_ratios:
    ratio_key = f"{int((1-ratio)*100)}/{int(ratio*100)}"
    for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
        res = all_results[model_name][ratio_key]
        print(f"{ratio_key:<12} {model_name:<12} {res['attack_success_rate']:<10.2%} {res['adversarial_accuracy']:<12.2%}")
    print("-"*50)

print(f"\n✓ Saved: results/synthetic_imbalance_nes_experiment.json")
