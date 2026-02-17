import matplotlib.pyplot as plt
import json

# Load results
with open('results/wine_experiment.json', 'r') as f:
    results = json.load(f)

models = list(results.keys())
asr = [results[m]['attack_success_rate'] * 100 for m in models]
pert = [results[m]['avg_perturbation'] for m in models]
robust = [results[m]['robustness_score'] for m in models]

# Create figure
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# Colors
colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red

# 1. Attack Success Rate
bars1 = ax1.bar(models, asr, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Attack Success Rate (%)', fontsize=12, fontweight='bold')
ax1.set_title('Adversarial Vulnerability\n(Lower is Better)', fontsize=13, fontweight='bold')
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, v) in enumerate(zip(bars1, asr)):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 3, 
             f'{v:.1f}%', ha='center', fontweight='bold', fontsize=11)

# 2. Average Perturbation
bars2 = ax2.bar(models, pert, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Average L2 Perturbation', fontsize=12, fontweight='bold')
ax2.set_title('Perturbation Needed\n(Higher is Better)', fontsize=13, fontweight='bold')
ax2.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, v) in enumerate(zip(bars2, pert)):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.15, 
             f'{v:.2f}', ha='center', fontweight='bold', fontsize=11)

# 3. Robustness Score
bars3 = ax3.bar(models, robust, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Robustness Score', fontsize=12, fontweight='bold')
ax3.set_title('Overall Robustness\n(Higher is Better)', fontsize=13, fontweight='bold')
ax3.set_ylim(0, 1)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
for i, (bar, v) in enumerate(zip(bars3, robust)):
    ax3.text(bar.get_x() + bar.get_width()/2, v + 0.03, 
             f'{v:.3f}', ha='center', fontweight='bold', fontsize=11)

plt.suptitle('Adversarial Robustness Comparison: Wine Dataset', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('results/wine_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Graph saved: results/wine_comparison.png")
plt.show()
