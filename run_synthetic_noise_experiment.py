"""Synthetic Dataset - Noise Level Experiment
Tests TabPFN vulnerability across different label noise levels
"""
import numpy as np
import json
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

print("="*80)
print("SYNTHETIC NOISE LEVEL EXPERIMENT")
print("Testing vulnerability across different label noise levels")
print("="*80)

# Noise levels to test
noise_levels = [0.0, 0.05, 0.10, 0.20]
n_samples_per_dataset = 200
n_test_attacks = 15
n_features = 10  # Fixed feature count

print(f"\nExperimental Design:")
print(f"  Noise levels: {noise_levels}")
print(f"  Features: {n_features} (fixed)")
print(f"  Samples per dataset: {n_samples_per_dataset}")
print(f"  Attacks per model: {n_test_attacks}")

# Store all results
all_results = {
    'TabPFN': {},
    'XGBoost': {},
    'LightGBM': {}
}

for noise in noise_levels:
    print(f"\n{'='*80}")
    print(f"TESTING: {noise*100:.0f}% LABEL NOISE")
    print(f"{'='*80}")
    
    # Generate synthetic dataset with noise
    print(f"\n[1/3] Generating synthetic dataset...")
    X, y = make_classification(
        n_samples=n_samples_per_dataset,
        n_features=n_features,
        n_informative=7,
        n_redundant=2,
        n_classes=2,
        random_state=42,
        flip_y=noise  # This is the label noise parameter
    )
    
    print(f"  ✓ Generated: {len(X)} samples, {n_features} features, {noise*100:.0f}% noise")
    
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
        noise_key = f"{noise*100:.0f}%"
        all_results[model_name][noise_key] = {
            'noise_level': noise,
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
print("NOISE LEVEL EXPERIMENT RESULTS")
print(f"{'='*80}")

print(f"\n{'Noise':<10} {'Model':<12} {'Clean Acc':<12} {'ASR':<10} {'Adv Acc':<12} {'Robustness':<12}")
print("-"*70)

for noise in noise_levels:
    noise_key = f"{noise*100:.0f}%"
    for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
        res = all_results[model_name][noise_key]
        print(f"{noise_key:<10} {model_name:<12} "
              f"{res['clean_accuracy']:<12.2%} "
              f"{res['attack_success_rate']:<10.2%} "
              f"{res['adversarial_accuracy']:<12.2%} "
              f"{res['robustness_score']:<12.4f}")
    print("-"*70)

# Analysis
print(f"\n{'='*80}")
print("NOISE IMPACT ANALYSIS")
print(f"{'='*80}")

for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    print(f"\n{model_name}:")
    
    asrs = [all_results[model_name][f"{n*100:.0f}%"]['attack_success_rate'] 
            for n in noise_levels]
    clean_accs = [all_results[model_name][f"{n*100:.0f}%"]['clean_accuracy'] 
                  for n in noise_levels]
    
    # Correlation with noise level
    corr_asr = np.corrcoef(noise_levels, asrs)[0, 1]
    corr_acc = np.corrcoef(noise_levels, clean_accs)[0, 1]
    
    print(f"  ASR change: {asrs[0]*100:.1f}% → {asrs[-1]*100:.1f}%")
    print(f"  Clean Acc change: {clean_accs[0]*100:.1f}% → {clean_accs[-1]*100:.1f}%")
    print(f"  ASR-Noise correlation: r = {corr_asr:+.3f}")

# Key findings
print(f"\n{'='*80}")
print("KEY FINDINGS")
print(f"{'='*80}")

# Compare TabPFN vs GBDT at different noise levels
print("\nTabPFN vs Best GBDT (ASR Ratio):")
for noise in noise_levels:
    noise_key = f"{noise*100:.0f}%"
    tabpfn_asr = all_results['TabPFN'][noise_key]['attack_success_rate']
    xgb_asr = all_results['XGBoost'][noise_key]['attack_success_rate']
    lgb_asr = all_results['LightGBM'][noise_key]['attack_success_rate']
    best_gbdt = min(xgb_asr, lgb_asr)
    
    if best_gbdt > 0:
        ratio = tabpfn_asr / best_gbdt
        print(f"  {noise_key} noise: {ratio:.2f}x {'(TabPFN worse)' if ratio > 1 else '(TabPFN better)'}")
    else:
        print(f"  {noise_key} noise: N/A")

# Save results
Path("results").mkdir(exist_ok=True)

with open('results/synthetic_noise_experiment.json', 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ Saved: results/synthetic_noise_experiment.json")

# Visualization
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

noise_pcts = [n*100 for n in noise_levels]

# Plot 1: ASR vs Noise
ax1 = axes[0]
for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    asrs = [all_results[model_name][f"{n*100:.0f}%"]['attack_success_rate']*100 
            for n in noise_levels]
    ax1.plot(noise_pcts, asrs, marker='o', linewidth=2, markersize=8, label=model_name)

ax1.set_xlabel('Label Noise (%)', fontweight='bold')
ax1.set_ylabel('Attack Success Rate (%)', fontweight='bold')
ax1.set_title('ASR vs Label Noise', fontweight='bold')
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xticks(noise_pcts)

# Plot 2: Clean Accuracy vs Noise
ax2 = axes[1]
for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    accs = [all_results[model_name][f"{n*100:.0f}%"]['clean_accuracy']*100 
            for n in noise_levels]
    ax2.plot(noise_pcts, accs, marker='s', linewidth=2, markersize=8, label=model_name)

ax2.set_xlabel('Label Noise (%)', fontweight='bold')
ax2.set_ylabel('Clean Accuracy (%)', fontweight='bold')
ax2.set_title('Clean Accuracy vs Label Noise', fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xticks(noise_pcts)

# Plot 3: Adversarial Accuracy vs Noise
ax3 = axes[2]
for model_name in ['TabPFN', 'XGBoost', 'LightGBM']:
    adv_accs = [all_results[model_name][f"{n*100:.0f}%"]['adversarial_accuracy']*100 
                for n in noise_levels]
    ax3.plot(noise_pcts, adv_accs, marker='^', linewidth=2, markersize=8, label=model_name)

ax3.set_xlabel('Label Noise (%)', fontweight='bold')
ax3.set_ylabel('Adversarial Accuracy (%)', fontweight='bold')
ax3.set_title('Adversarial Accuracy vs Label Noise', fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)
ax3.set_xticks(noise_pcts)

plt.suptitle('Synthetic Noise Level Experiment', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/synthetic_noise_experiment.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: results/synthetic_noise_experiment.png")

print(f"\n{'='*80}")
print("NOISE LEVEL EXPERIMENT COMPLETE!")
print(f"{'='*80}")
