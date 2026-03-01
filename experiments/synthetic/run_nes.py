"""NES Attack Experiment - Synthetic Dataset (Feature Scaling)"""
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

def get_common_attack_indices(y_test, all_preds, n_samples, random_state=42):
    """Select n_samples from samples correctly classified by ALL models, stratified by class."""
    correct_sets = [set(np.where(preds == y_test)[0]) for preds in all_preds.values()]
    common_correct = np.array(sorted(set.intersection(*correct_sets)))
    if len(common_correct) <= n_samples:
        return common_correct
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=random_state)
        _, sel = next(sss.split(common_correct.reshape(-1, 1), y_test[common_correct]))
        return common_correct[sel]
    except ValueError:
        rng = np.random.RandomState(random_state)
        sel = rng.choice(len(common_correct), n_samples, replace=False)
        return common_correct[sel]


print("="*80)
print("NES ATTACK EXPERIMENT - SYNTHETIC FEATURE SCALING")
print("="*80)

# Feature counts to test
feature_counts = [5, 10, 15, 20]
n_samples_per_dataset = 200
n_test_attacks = 15

print(f"\nExperimental Design:")
print(f"  Feature counts: {feature_counts}")
print(f"  Samples per dataset: {n_samples_per_dataset}")
print(f"  Attacks per model: {n_test_attacks}")

all_results = {
    'TabPFN': {},
    'XGBoost': {},
    'LightGBM': {}
}

for n_features in feature_counts:
    print(f"\n{'='*80}")
    print(f"TESTING: {n_features} FEATURES")
    print(f"{'='*80}")
    
    # Generate synthetic dataset
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
    
    print(f"  Generated: {len(X)} samples, {n_features} features")
    
    models = {
        'TabPFN': TabPFNWrapper(device='cpu'),
        'XGBoost': GBDTWrapper(model_type='xgboost'),
        'LightGBM': GBDTWrapper(model_type='lightgbm')
    }
    

    # Pre-train all models and collect predictions for common indices
    all_preds = {}
    for _name, _model in models.items():
        _model.fit(X_train, y_train)
        all_preds[_name] = _model.predict(X_test)

    attack_indices = get_common_attack_indices(y_test, all_preds, n_test_attacks)

    for model_name, model in models.items():
        print(f"\n  Testing {model_name}...")
        
        y_pred = all_preds[model_name]
        clean_acc = np.mean(y_pred == y_test)
        print(f"    Clean Accuracy: {clean_acc:.4f}")
        
        attack = NESAttack(model, max_iterations=200, n_samples=30,
                          learning_rate=0.3, sigma=0.3, verbose=False)
        
        results = []
        successful = 0
        
        for i in attack_indices:
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred_i = model.predict(x_orig.reshape(1, -1))[0]
            
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
                successful += 1
        
        print(f"    Successful: {successful}/{len(results)}")
        
        metrics = RobustnessMetrics.compute_all(results, y_test[attack_indices],
                                                y_pred[attack_indices])
        
        all_results[model_name][str(n_features)] = {
            'clean_accuracy': clean_acc,
            'attack_success_rate': metrics['attack_success_rate'],
            'adversarial_accuracy': metrics['adversarial_accuracy'],
            'avg_perturbation': metrics['avg_perturbation'],
            'avg_queries': metrics['avg_queries'],
            'robustness_score': metrics['robustness_score'],
            'n_samples_tested': len(results)
        }
        
        print(f"    ASR: {metrics['attack_success_rate']:.2%}")
        print(f"    Adv Acc: {metrics['adversarial_accuracy']:.2%}")

# Summary
print(f"\n{'='*80}")
print("NES ATTACK RESULTS - SYNTHETIC FEATURE SCALING")
print(f"{'='*80}")

print(f"\n{'Features':<10} {'Model':<12} {'ASR':<10} {'Adv Acc':<12} {'Robustness':<12}")
print("-"*60)

for n_feat in feature_counts:
    for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
        res = all_results[model_name][str(n_feat)]
        print(f"{n_feat:<10} {model_name:<12} "
              f"{res['attack_success_rate']:<10.2%} "
              f"{res['adversarial_accuracy']:<12.2%} "
              f"{res['robustness_score']:<12.4f}")
    print("-"*60)

# Correlation analysis
print(f"\n{'='*80}")
print("FEATURE COUNT vs ASR CORRELATION (NES Attack)")
print(f"{'='*80}")

for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    asrs = [all_results[model_name][str(f)]['attack_success_rate'] for f in feature_counts]
    corr = np.corrcoef(feature_counts, asrs)[0, 1]
    print(f"  {model_name}: r = {corr:+.3f}")

# Save
Path("results").mkdir(exist_ok=True)
with open('results/synthetic_nes_experiment.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)
print(f"\n✓ Saved: results/synthetic_nes_experiment.json")

print(f"\n{'='*80}")
print("SYNTHETIC NES EXPERIMENT COMPLETE!")
print(f"{'='*80}")
