"""Synthetic Dataset - Class Imbalance Experiment
Tests TabPFN vulnerability across different class imbalance ratios
"""
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
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
from src.attacks.boundary_attack import BoundaryAttack
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
print("SYNTHETIC CLASS IMBALANCE EXPERIMENT")
print("Testing vulnerability across different class imbalance ratios")
print("="*80)

# Class imbalance ratios to test (minority class weight)
imbalance_ratios = [0.5, 0.3, 0.2, 0.1]  # 50/50, 70/30, 80/20, 90/10
n_samples_per_dataset = 200
n_test_attacks = 15
n_features = 10  # Fixed feature count

print(f"\nExperimental Design:")
print(f"  Imbalance ratios: {imbalance_ratios} (minority class)")
print(f"  Features: {n_features} (fixed)")
print(f"  Samples per dataset: {n_samples_per_dataset}")
print(f"  Attacks per model: {n_test_attacks}")

# Store all results
all_results = {
    'TabPFN': {},
    'XGBoost': {},
    'LightGBM': {}
}

for ratio in imbalance_ratios:
    print(f"\n{'='*80}")
    print(f"TESTING: {int((1-ratio)*100)}/{int(ratio*100)} CLASS RATIO")
    print(f"{'='*80}")
    
    # Generate synthetic dataset with imbalance
    print(f"\n[1/3] Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=n_samples_per_dataset,
        n_features=n_features,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        weights=[1-ratio, ratio],  # Class imbalance
        random_state=42,
        flip_y=0.01
    )
    
    class_counts = np.bincount(y)
    print(f"  ✓ Generated: {len(X)} samples, Class 0: {class_counts[0]}, Class 1: {class_counts[1]}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Test each model
    models = {
        'TabPFN': TabPFNWrapper(device='cpu'),
        'XGBoost': GBDTWrapper(model_type='xgboost'),
        'LightGBM': GBDTWrapper(model_type='lightgbm')
    }
    
    for model_name, model in models.items():
        print(f"\n[2/3] Testing {model_name}...")
        
        # Train
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        clean_acc = np.mean(y_pred == y_test)
        attack_indices = get_stratified_attack_indices(y_test, y_pred, n_test_attacks)
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        
        # Attack
        print(f"  Attacking {n_test_attacks} samples...")
        attack = BoundaryAttack(model, max_iterations=100, epsilon=0.5, verbose=False)
        
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
        
        print(f"  Successful attacks: {successful}/{len(results)}")
        
        # Compute metrics
        metrics = RobustnessMetrics.compute_all(results, y_test[attack_indices], y_pred[attack_indices])
        
        # Store results
        ratio_key = f"{int((1-ratio)*100)}/{int(ratio*100)}"
        all_results[model_name][ratio_key] = {
            'imbalance_ratio': ratio,
            'majority_pct': 1-ratio,
            'clean_accuracy': clean_acc,
            'attack_success_rate': metrics['attack_success_rate'],
            'adversarial_accuracy': metrics['adversarial_accuracy'],
            'avg_perturbation': metrics['avg_perturbation'],
            'robustness_score': metrics['robustness_score'],
            'n_samples_tested': len(results)
        }
        
        print(f"  ASR: {metrics['attack_success_rate']:.2%}")
        print(f"  Adversarial Accuracy: {metrics['adversarial_accuracy']:.2%}")

# Print summary table
print(f"\n{'='*80}")
print("CLASS IMBALANCE EXPERIMENT RESULTS")
print(f"{'='*80}")

print(f"\n{'Ratio':<12} {'Model':<12} {'Clean Acc':<12} {'ASR':<10} {'Adv Acc':<12} {'Robustness':<12}")
print("-"*70)

for ratio in imbalance_ratios:
    ratio_key = f"{int((1-ratio)*100)}/{int(ratio*100)}"
    for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
        res = all_results[model_name][ratio_key]
        print(f"{ratio_key:<12} {model_name:<12} "
              f"{res['clean_accuracy']:<12.2%} "
              f"{res['attack_success_rate']:<10.2%} "
              f"{res['adversarial_accuracy']:<12.2%} "
              f"{res['robustness_score']:<12.4f}")
    print("-"*70)

# Analysis
print(f"\n{'='*80}")
print("CLASS IMBALANCE IMPACT ANALYSIS")
print(f"{'='*80}")

majority_pcts = [1-r for r in imbalance_ratios]

for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    print(f"\n{model_name}:")
    
    asrs = [all_results[model_name][f"{int((1-r)*100)}/{int(r*100)}"]['attack_success_rate'] 
            for r in imbalance_ratios]
    clean_accs = [all_results[model_name][f"{int((1-r)*100)}/{int(r*100)}"]['clean_accuracy'] 
                  for r in imbalance_ratios]
    
    # Correlation with imbalance
    corr_asr = np.corrcoef(majority_pcts, asrs)[0, 1]
    corr_acc = np.corrcoef(majority_pcts, clean_accs)[0, 1]
    
    print(f"  ASR (50/50 → 90/10): {asrs[0]*100:.1f}% → {asrs[-1]*100:.1f}%")
    print(f"  Clean Acc (50/50 → 90/10): {clean_accs[0]*100:.1f}% → {clean_accs[-1]*100:.1f}%")
    print(f"  ASR-Imbalance correlation: r = {corr_asr:+.3f}")

# Key findings
print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}")

print("\nTabPFN vs Best GBDT (ASR Ratio):")
for ratio in imbalance_ratios:
    ratio_key = f"{int((1-ratio)*100)}/{int(ratio*100)}"
    tabpfn_asr = all_results['TabPFN'][ratio_key]['attack_success_rate']
    xgb_asr = all_results['XGBoost'][ratio_key]['attack_success_rate']
    lgb_asr = all_results['LightGBM'][ratio_key]['attack_success_rate']
    best_gbdt = min(xgb_asr, lgb_asr)
    
    if best_gbdt > 0:
        vuln_ratio = tabpfn_asr / best_gbdt
        print(f"  {ratio_key}: {vuln_ratio:.2f}x {'(TabPFN worse)' if vuln_ratio > 1 else '(TabPFN better)'}")
    else:
        print(f"  {ratio_key}: N/A")

# Save results
Path("results").mkdir(exist_ok=True)

with open('results/synthetic_imbalance_experiment.json', 'w') as f:
    json.dump(all_results, f, indent=2, default=lambda x: x.item() if hasattr(x, 'item') else x)
print(f"\n✓ Saved: results/synthetic_imbalance_experiment.json")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ratio_labels = [f"{int((1-r)*100)}/{int(r*100)}" for r in imbalance_ratios]

# Plot 1: ASR vs Imbalance
ax1 = axes[0]
for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    asrs = [all_results[model_name][f"{int((1-r)*100)}/{int(r*100)}"]['attack_success_rate']*100 
            for r in imbalance_ratios]
    ax1.plot(ratio_labels, asrs, marker='o', linewidth=2, markersize=8, label=model_name)

ax1.set_xlabel('Class Ratio (Majority/Minority)', fontweight='bold')
ax1.set_ylabel('Attack Success Rate (%)', fontweight='bold')
ax1.set_title('ASR vs Class Imbalance', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Clean Accuracy vs Imbalance
ax2 = axes[1]
for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    accs = [all_results[model_name][f"{int((1-r)*100)}/{int(r*100)}"]['clean_accuracy']*100 
            for r in imbalance_ratios]
    ax2.plot(ratio_labels, accs, marker='s', linewidth=2, markersize=8, label=model_name)

ax2.set_xlabel('Class Ratio (Majority/Minority)', fontweight='bold')
ax2.set_ylabel('Clean Accuracy (%)', fontweight='bold')
ax2.set_title('Clean Accuracy vs Class Imbalance', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Adversarial Accuracy vs Imbalance
ax3 = axes[2]
for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    adv_accs = [all_results[model_name][f"{int((1-r)*100)}/{int(r*100)}"]['adversarial_accuracy']*100 
                for r in imbalance_ratios]
    ax3.plot(ratio_labels, adv_accs, marker='^', linewidth=2, markersize=8, label=model_name)

ax3.set_xlabel('Class Ratio (Majority/Minority)', fontweight='bold')
ax3.set_ylabel('Adversarial Accuracy (%)', fontweight='bold')
ax3.set_title('Adversarial Accuracy vs Class Imbalance', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

plt.suptitle('Synthetic Class Imbalance Experiment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/synthetic_imbalance_experiment.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: results/synthetic_imbalance_experiment.png")

print(f"\n{'='*80}")
print("CLASS IMBALANCE EXPERIMENT COMPLETE!")
print(f"{'='*80}")
