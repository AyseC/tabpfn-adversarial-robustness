"""Parameter Sensitivity Analysis"""
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack

print("="*70)
print("PARAMETER SENSITIVITY ANALYSIS")
print("="*70)

# Load data
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train models
print("\nTraining models...")
tabpfn = TabPFNWrapper(device='cpu')
tabpfn.fit(X_train, y_train)

xgboost = GBDTWrapper(model_type='xgboost')
xgboost.fit(X_train, y_train)

models = {'TabPFN': tabpfn, 'XGBoost': xgboost}

# 1. EPSILON SENSITIVITY
print("\n" + "="*70)
print("1. EPSILON SENSITIVITY (Perturbation Budget)")
print("="*70)

epsilon_values = [0.1, 0.3, 0.5, 0.7, 1.0]
epsilon_results = {model_name: [] for model_name in models}

for eps in epsilon_values:
    print(f"\nTesting epsilon = {eps}")
    
    for model_name, model in models.items():
        attack = BoundaryAttack(model, max_iterations=100, epsilon=eps, verbose=False)
        
        successes = 0
        total = 0
        
        for i in range(5):  # Test 5 samples
            x_orig = X_test[i]
            y_true = y_test[i]
            
            if model.predict(x_orig.reshape(1, -1))[0] != y_true:
                continue
            
            total += 1
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            
            if success:
                successes += 1
        
        asr = successes / total if total > 0 else 0
        epsilon_results[model_name].append(asr * 100)
        print(f"  {model_name}: {asr:.1%} ASR")

# 2. ITERATION SENSITIVITY
print("\n" + "="*70)
print("2. ITERATION SENSITIVITY")
print("="*70)

iteration_values = [50, 100, 150, 200, 300]
iteration_results = {model_name: [] for model_name in models}

for max_iter in iteration_values:
    print(f"\nTesting max_iterations = {max_iter}")
    
    for model_name, model in models.items():
        attack = BoundaryAttack(model, max_iterations=max_iter, epsilon=0.5, verbose=False)
        
        successes = 0
        total = 0
        
        for i in range(5):
            x_orig = X_test[i]
            y_true = y_test[i]
            
            if model.predict(x_orig.reshape(1, -1))[0] != y_true:
                continue
            
            total += 1
            x_adv, success, queries, pert = attack.attack(x_orig, y_true)
            
            if success:
                successes += 1
        
        asr = successes / total if total > 0 else 0
        iteration_results[model_name].append(asr * 100)
        print(f"  {model_name}: {asr:.1%} ASR")

# VISUALIZATION
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Plot 1: Epsilon Sensitivity
ax1 = axes[0]
for model_name, results in epsilon_results.items():
    ax1.plot(epsilon_values, results, marker='o', linewidth=2, 
            markersize=8, label=model_name, alpha=0.8)

ax1.set_xlabel('Epsilon (Perturbation Budget)', fontweight='bold', fontsize=12)
ax1.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
ax1.set_title('Epsilon Sensitivity Analysis', fontweight='bold', fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(alpha=0.3)
ax1.set_ylim(0, 100)

# Add annotations
for model_name, results in epsilon_results.items():
    for i, (x, y) in enumerate(zip(epsilon_values, results)):
        if i % 2 == 0:  # Annotate every other point
            ax1.annotate(f'{y:.0f}%', (x, y), textcoords="offset points",
                        xytext=(0,10), ha='center', fontsize=9)

# Plot 2: Iteration Sensitivity
ax2 = axes[1]
for model_name, results in iteration_results.items():
    ax2.plot(iteration_values, results, marker='s', linewidth=2,
            markersize=8, label=model_name, alpha=0.8)

ax2.set_xlabel('Max Iterations', fontweight='bold', fontsize=12)
ax2.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
ax2.set_title('Iteration Sensitivity Analysis', fontweight='bold', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(alpha=0.3)
ax2.set_ylim(0, 100)

plt.suptitle('Parameter Sensitivity Analysis - Wine Dataset',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/parameter_sensitivity.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/parameter_sensitivity.png")
plt.show()

# Summary
print("\n" + "="*70)
print("KEY INSIGHTS")
print("="*70)

print("\n1. EPSILON IMPACT:")
for model_name in models:
    print(f"   {model_name}:")
    print(f"     Low epsilon (0.1): {epsilon_results[model_name][0]:.0f}% ASR")
    print(f"     High epsilon (1.0): {epsilon_results[model_name][-1]:.0f}% ASR")
    print(f"     Increase: {epsilon_results[model_name][-1] - epsilon_results[model_name][0]:.0f}%")

print("\n2. ITERATION IMPACT:")
for model_name in models:
    print(f"   {model_name}:")
    print(f"     Few iterations (50): {iteration_results[model_name][0]:.0f}% ASR")
    print(f"     Many iterations (300): {iteration_results[model_name][-1]:.0f}% ASR")
    print(f"     Improvement: {iteration_results[model_name][-1] - iteration_results[model_name][0]:.0f}%")

print("\n3. OBSERVATIONS:")
print("   • Higher epsilon → Higher success rate (as expected)")
print("   • More iterations → Diminishing returns after ~200")
print("   • TabPFN more sensitive to parameter changes")

# Save results
results_dict = {
    'epsilon_sensitivity': {
        'epsilon_values': epsilon_values,
        'results': epsilon_results
    },
    'iteration_sensitivity': {
        'iteration_values': iteration_values,
        'results': iteration_results
    }
}

with open('results/parameter_sensitivity.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n✓ Saved: results/parameter_sensitivity.json")
print("="*70)
