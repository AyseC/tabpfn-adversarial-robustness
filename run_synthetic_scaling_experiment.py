"""Synthetic Dataset - Feature Scaling Experiment
Tests TabPFN vulnerability across different feature dimensionalities
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Suppress all warnings
warnings.filterwarnings('ignore')

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

print("="*80)
print("SYNTHETIC FEATURE SCALING EXPERIMENT")
print("Testing vulnerability across different feature dimensionalities")
print("="*80)

# Feature counts to test
feature_counts = [5, 10, 15, 20]
n_samples_per_dataset = 200
n_test_attacks = 10

print(f"\nExperimental Design:")
print(f"  Feature counts: {feature_counts}")
print(f"  Samples per dataset: {n_samples_per_dataset}")
print(f"  Attacks per model: {n_test_attacks}")
print(f"  Total experiments: {len(feature_counts)} × 3 models = {len(feature_counts)*3}")

# Store all results
all_results = {
    'TabPFN': {},
    'XGBoost': {},
    'LightGBM': {}
}

# Run experiments for each feature count
for n_features in feature_counts:
    print(f"\n{'='*80}")
    print(f"TESTING: {n_features} FEATURES")
    print(f"{'='*80}")
    
    # Generate synthetic dataset
    print(f"\n[1/3] Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=n_samples_per_dataset,
        n_features=n_features,
        n_informative=max(3, n_features // 2),
        n_redundant=min(2, n_features // 3),
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    
    print(f"  ✓ Generated: {len(X)} samples, {n_features} features")
    print(f"  Class distribution: {sum(y==0)}/{sum(y==1)}")
    
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
        print(f"  Clean Accuracy: {clean_acc:.4f}")
        
        # Attack
        print(f"  Attacking {n_test_attacks} samples...")
        attack = BoundaryAttack(model, max_iterations=100, epsilon=0.5, verbose=False)
        
        results = []
        successful = 0
        
        for i in range(min(n_test_attacks, len(X_test))):
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
                successful += 1
        
        print(f"  Successful attacks: {successful}/{len(results)}")
        
        # Compute metrics
        metrics = RobustnessMetrics.compute_all(results, y_test[:n_test_attacks], 
                                                y_pred[:n_test_attacks])
        
        # Store results
        all_results[model_name][n_features] = {
            'clean_accuracy': clean_acc,
            'attack_success_rate': metrics['attack_success_rate'],
            'avg_perturbation': metrics['avg_perturbation'],
            'robustness_score': metrics['robustness_score'],
            'n_samples_tested': len(results)
        }
        
        print(f"  ASR: {metrics['attack_success_rate']:.2%}")
        print(f"  Robustness: {metrics['robustness_score']:.4f}")

# Print summary table
print(f"\n{'='*80}")
print("FEATURE SCALING EXPERIMENT RESULTS")
print(f"{'='*80}")

print(f"\n{'Features':<12} {'Model':<12} {'Clean Acc':<12} {'ASR':<10} {'Robustness':<12}")
print("-"*70)

for n_features in feature_counts:
    for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
        res = all_results[model_name][n_features]
        print(f"{n_features:<12} {model_name:<12} "
              f"{res['clean_accuracy']:<12.2%} "
              f"{res['attack_success_rate']:<10.2%} "
              f"{res['robustness_score']:<12.4f}")
    print("-"*70)

# Analysis: Feature count correlation
print(f"\n{'='*80}")
print("FEATURE COMPLEXITY ANALYSIS")
print(f"{'='*80}")

for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    print(f"\n{model_name}:")
    
    asrs = [all_results[model_name][nf]['attack_success_rate'] 
            for nf in feature_counts]
    
    # Correlation with feature count
    corr = np.corrcoef(feature_counts, asrs)[0, 1]
    
    print(f"  ASR Range: {min(asrs):.1%} → {max(asrs):.1%}")
    print(f"  Correlation with feature count: {corr:+.3f}")
    
    if abs(corr) > 0.7:
        print(f"  → {'STRONG' if abs(corr) > 0.9 else 'MODERATE'} "
              f"{'POSITIVE' if corr > 0 else 'NEGATIVE'} correlation")

# TabPFN vs GBDT comparison
print(f"\n{'='*80}")
print("TABPFN vs GBDT VULNERABILITY RATIO")
print(f"{'='*80}")

print(f"\n{'Features':<12} {'Ratio (TabPFN/LightGBM)':<30} {'Interpretation':<30}")
print("-"*80)

for n_features in feature_counts:
    tabpfn_asr = all_results['TabPFN'][n_features]['attack_success_rate']
    lightgbm_asr = all_results['LightGBM'][n_features]['attack_success_rate']
    
    if lightgbm_asr > 0:
        ratio = tabpfn_asr / lightgbm_asr
        interpretation = f"{ratio:.2f}x {'MORE' if ratio > 1 else 'LESS'} vulnerable"
    else:
        ratio = 0
        interpretation = "N/A"
    
    print(f"{n_features:<12} {ratio:<30.2f} {interpretation:<30}")

# VISUALIZATION
print(f"\n[3/3] Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: ASR vs Feature Count
ax1 = axes[0, 0]
for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    asrs = [all_results[model_name][nf]['attack_success_rate'] * 100 
            for nf in feature_counts]
    ax1.plot(feature_counts, asrs, marker='o', linewidth=2, markersize=8,
            label=model_name, alpha=0.8)

ax1.set_xlabel('Number of Features', fontweight='bold', fontsize=12)
ax1.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
ax1.set_title('Vulnerability vs Feature Complexity', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
ax1.set_xticks(feature_counts)

# Plot 2: Robustness Score vs Feature Count
ax2 = axes[0, 1]
for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    robs = [all_results[model_name][nf]['robustness_score'] 
            for nf in feature_counts]
    ax2.plot(feature_counts, robs, marker='s', linewidth=2, markersize=8,
            label=model_name, alpha=0.8)

ax2.set_xlabel('Number of Features', fontweight='bold', fontsize=12)
ax2.set_ylabel('Robustness Score', fontweight='bold', fontsize=12)
ax2.set_title('Robustness vs Feature Complexity', fontweight='bold', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_xticks(feature_counts)

# Plot 3: TabPFN/GBDT Ratio
ax3 = axes[1, 0]
ratios = []
for n_features in feature_counts:
    tabpfn_asr = all_results['TabPFN'][n_features]['attack_success_rate']
    lightgbm_asr = all_results['LightGBM'][n_features]['attack_success_rate']
    ratio = tabpfn_asr / lightgbm_asr if lightgbm_asr > 0 else 1.0
    ratios.append(ratio)

colors = ['#e74c3c' if r > 1 else '#2ecc71' for r in ratios]
bars = ax3.bar(feature_counts, ratios, color=colors, alpha=0.7, 
               edgecolor='black', linewidth=2)

ax3.axhline(y=1.0, color='black', linestyle='--', linewidth=2, 
           label='Equal vulnerability')
ax3.set_xlabel('Number of Features', fontweight='bold', fontsize=12)
ax3.set_ylabel('Vulnerability Ratio\n(TabPFN / LightGBM)', fontweight='bold', fontsize=12)
ax3.set_title('TabPFN Relative Vulnerability', fontweight='bold', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(axis='y', alpha=0.3)
ax3.set_xticks(feature_counts)

for bar, val in zip(bars, ratios):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.2f}x', ha='center', fontweight='bold', fontsize=10)

# Plot 4: Summary Statistics
ax4 = axes[1, 1]
ax4.axis('off')

summary_text = f"""
FEATURE SCALING EXPERIMENT SUMMARY

Datasets Tested: {len(feature_counts)}
Feature Range: {min(feature_counts)} → {max(feature_counts)}
Samples per Dataset: {n_samples_per_dataset}
Attacks per Model: {n_test_attacks}

KEY FINDINGS:

1. FEATURE COMPLEXITY CORRELATION:
"""

for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    asrs = [all_results[model_name][nf]['attack_success_rate'] 
            for nf in feature_counts]
    corr = np.corrcoef(feature_counts, asrs)[0, 1]
    summary_text += f"\n   {model_name}: r = {corr:+.3f}"

summary_text += f"""

2. TABPFN VULNERABILITY PATTERN:
   • Low features ({feature_counts[0]}): {ratios[0]:.2f}x vs GBDT
   • High features ({feature_counts[-1]}): {ratios[-1]:.2f}x vs GBDT
   • Trend: {'Increasing' if ratios[-1] > ratios[0] else 'Decreasing'}

3. CONTROLLED EXPERIMENT CONCLUSION:
   When other factors are constant,
   feature complexity {'DOES' if abs(np.corrcoef(feature_counts, ratios)[0,1]) > 0.5 else 'does NOT'}
   significantly impact TabPFN
   relative vulnerability.
"""

ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
         fontsize=11, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, pad=1))

plt.suptitle('Synthetic Feature Scaling Experiment: Controlled Analysis',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()

# Save
Path("results").mkdir(exist_ok=True)
plt.savefig('results/synthetic_feature_scaling.png', dpi=300, bbox_inches='tight')
print("  ✓ Saved: results/synthetic_feature_scaling.png")
plt.close()

# Save JSON results
with open('results/synthetic_feature_scaling.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print("  ✓ Saved: results/synthetic_feature_scaling.json")

print(f"\n{'='*80}")
print("SYNTHETIC EXPERIMENT COMPLETE!")
print(f"{'='*80}")
print("""
KEY TAKEAWAY:
This controlled experiment isolates the effect of feature dimensionality
on adversarial robustness. By keeping all other factors constant
(data distribution, sample size, class balance), we can definitively
assess whether feature count alone predicts vulnerability.
""")
print(f"{'='*80}")
