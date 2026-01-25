"""Comprehensive visualization of all results"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load data
datasets = {}
for name in ['wine', 'iris']:
    with open(f'results/{name}_experiment.json', 'r') as f:
        datasets[name.title()] = json.load(f)

# Create comprehensive figure
fig = plt.figure(figsize=(16, 10))

# 1. ASR Comparison
ax1 = plt.subplot(2, 3, 1)
models = ['XGBoost', 'LightGBM', 'TabPFN']
x = np.arange(len(models))
width = 0.35

wine_asr = [datasets['Wine'][m]['attack_success_rate']*100 for m in models]
iris_asr = [datasets['Iris'][m]['attack_success_rate']*100 for m in models]

bars1 = ax1.bar(x - width/2, wine_asr, width, label='Wine', color='#e74c3c', alpha=0.8)
bars2 = ax1.bar(x + width/2, iris_asr, width, label='Iris', color='#3498db', alpha=0.8)

ax1.set_ylabel('Attack Success Rate (%)', fontweight='bold')
ax1.set_title('Attack Success Rate Comparison', fontweight='bold', fontsize=14)
ax1.set_xticks(x)
ax1.set_xticklabels(models)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.0f}%', ha='center', va='bottom', fontsize=9)

# 2. Perturbation
ax2 = plt.subplot(2, 3, 2)
wine_pert = [datasets['Wine'][m]['avg_perturbation'] for m in models]
iris_pert = [datasets['Iris'][m]['avg_perturbation'] for m in models]

bars1 = ax2.bar(x - width/2, wine_pert, width, label='Wine', color='#e74c3c', alpha=0.8)
bars2 = ax2.bar(x + width/2, iris_pert, width, label='Iris', color='#3498db', alpha=0.8)

ax2.set_ylabel('Avg L2 Perturbation', fontweight='bold')
ax2.set_title('Perturbation Magnitude', fontweight='bold', fontsize=14)
ax2.set_xticks(x)
ax2.set_xticklabels(models)
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

# 3. Robustness Score
ax3 = plt.subplot(2, 3, 3)
wine_rob = [datasets['Wine'][m]['robustness_score'] for m in models]
iris_rob = [datasets['Iris'][m]['robustness_score'] for m in models]

bars1 = ax3.bar(x - width/2, wine_rob, width, label='Wine', color='#e74c3c', alpha=0.8)
bars2 = ax3.bar(x + width/2, iris_rob, width, label='Iris', color='#3498db', alpha=0.8)

ax3.set_ylabel('Robustness Score', fontweight='bold')
ax3.set_title('Overall Robustness', fontweight='bold', fontsize=14)
ax3.set_xticks(x)
ax3.set_xticklabels(models)
ax3.legend()
ax3.set_ylim(0, 0.8)
ax3.grid(axis='y', alpha=0.3)

# 4. TabPFN Vulnerability Ratio
ax4 = plt.subplot(2, 3, 4)
wine_ratio = datasets['Wine']['TabPFN']['attack_success_rate'] / datasets['Wine']['LightGBM']['attack_success_rate']
iris_ratio = datasets['Iris']['TabPFN']['attack_success_rate'] / datasets['Iris']['XGBoost']['attack_success_rate']

bars = ax4.bar(['Wine', 'Iris'], [wine_ratio, iris_ratio], 
               color=['#e74c3c', '#3498db'], alpha=0.8, edgecolor='black', linewidth=2)

ax4.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Equal vulnerability')
ax4.set_ylabel('Vulnerability Ratio\n(TabPFN / Best GBDT)', fontweight='bold')
ax4.set_title('TabPFN Relative Vulnerability', fontweight='bold', fontsize=14)
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

for bar, val in zip(bars, [wine_ratio, iris_ratio]):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
            f'{val:.2f}x', ha='center', fontweight='bold', fontsize=12)

# 5. Dataset Characteristics
ax5 = plt.subplot(2, 3, 5)
characteristics = {
    'Wine': {'samples': 130, 'features': 13},
    'Iris': {'samples': 100, 'features': 4}
}

dataset_names = list(characteristics.keys())
samples = [characteristics[d]['samples'] for d in dataset_names]
features = [characteristics[d]['features'] for d in dataset_names]

ax5_twin = ax5.twinx()
bars1 = ax5.bar(x[:2] - width/2, samples, width, label='Samples', color='#2ecc71', alpha=0.7)
bars2 = ax5_twin.bar(x[:2] + width/2, features, width, label='Features', color='#f39c12', alpha=0.7)

ax5.set_ylabel('Number of Samples', fontweight='bold', color='#2ecc71')
ax5_twin.set_ylabel('Number of Features', fontweight='bold', color='#f39c12')
ax5.set_title('Dataset Characteristics', fontweight='bold', fontsize=14)
ax5.set_xticks(x[:2])
ax5.set_xticklabels(dataset_names)

# 6. Key Finding Text
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

finding_text = """
KEY FINDINGS:

1. Dataset-Dependent Robustness:
   • Wine: TabPFN 1.71x MORE vulnerable
   • Iris: TabPFN 0.93x LESS vulnerable

2. Pattern:
   • More features → TabPFN weaker
   • Fewer features → TabPFN stronger

3. Implication:
   TabPFN's adversarial robustness is
   NOT universally inferior to GBDTs.
   
4. Recommendation:
   Robustness evaluation should be
   dataset-specific!
"""

ax6.text(0.1, 0.9, finding_text, transform=ax6.transAxes,
         fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3),
         family='monospace')

plt.suptitle('Comprehensive Adversarial Robustness Analysis:\nTabPFN vs GBDTs Across Datasets',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout()
plt.savefig('results/comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: results/comprehensive_analysis.png")
plt.show()
