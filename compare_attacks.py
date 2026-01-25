"""Compare Boundary vs NES attacks"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load results
with open('results/wine_experiment.json', 'r') as f:
    boundary = json.load(f)

with open('results/wine_nes_experiment.json', 'r') as f:
    nes = json.load(f)

print("="*70)
print("BOUNDARY vs NES ATTACK COMPARISON - WINE DATASET")
print("="*70)

models = ['XGBoost', 'LightGBM', 'TabPFN']

print(f"\n{'Model':<12} {'Boundary ASR':<15} {'NES ASR':<15} {'Difference':<15}")
print("-"*70)

for model in models:
    b_asr = boundary[model]['attack_success_rate'] * 100
    n_asr = nes[model]['attack_success_rate'] * 100
    diff = b_asr - n_asr
    
    print(f"{model:<12} {b_asr:<15.1f}% {n_asr:<15.1f}% {diff:+.1f}%")

print("\n" + "="*70)
print("KEY FINDINGS:")
print("="*70)
print("\n1. Boundary Attack is more effective (higher ASR)")
print("2. Both attacks show same pattern: TabPFN most vulnerable")
print("3. TabPFN:")
print(f"   - Boundary: {boundary['TabPFN']['attack_success_rate']*100:.1f}% ASR")
print(f"   - NES: {nes['TabPFN']['attack_success_rate']*100:.1f}% ASR")
print(f"   - Consistently vulnerable across attack types!")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(len(models))
width = 0.35

# ASR comparison
b_asr_vals = [boundary[m]['attack_success_rate']*100 for m in models]
n_asr_vals = [nes[m]['attack_success_rate']*100 for m in models]

bars1 = ax1.bar(x - width/2, b_asr_vals, width, label='Boundary', 
                color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax1.bar(x + width/2, n_asr_vals, width, label='NES',
                color='#3498db', alpha=0.8, edgecolor='black')

ax1.set_ylabel('Attack Success Rate (%)', fontweight='bold', fontsize=12)
ax1.set_title('Attack Effectiveness Comparison', fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{height:.0f}%', ha='center', fontsize=10)

# Robustness Score
b_rob = [boundary[m]['robustness_score'] for m in models]
n_rob = [nes[m]['robustness_score'] for m in models]

bars1 = ax2.bar(x - width/2, b_rob, width, label='Boundary',
                color='#e74c3c', alpha=0.8, edgecolor='black')
bars2 = ax2.bar(x + width/2, n_rob, width, label='NES',
                color='#3498db', alpha=0.8, edgecolor='black')

ax2.set_ylabel('Robustness Score', fontweight='bold', fontsize=12)
ax2.set_title('Robustness Under Different Attacks', fontweight='bold', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.set_ylim(0, 0.8)
ax2.grid(axis='y', alpha=0.3)

plt.suptitle('Boundary vs NES Attack - Wine Dataset', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/attack_comparison.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/attack_comparison.png")
plt.show()
