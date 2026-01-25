import matplotlib.pyplot as plt
import numpy as np

# Results
models = ['XGBoost', 'TabPFN']
attack_success = [40, 80]
avg_pert = [2.04, 2.20]
avg_queries = [362, 272]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# Attack Success Rate
ax1.bar(models, attack_success, color=['green', 'red'], alpha=0.7)
ax1.set_ylabel('Attack Success Rate (%)')
ax1.set_title('Adversarial Vulnerability')
ax1.set_ylim(0, 100)
for i, v in enumerate(attack_success):
    ax1.text(i, v+3, f'{v}%', ha='center', fontweight='bold')

# Perturbation
ax2.bar(models, avg_pert, color=['blue', 'orange'], alpha=0.7)
ax2.set_ylabel('Average L2 Perturbation')
ax2.set_title('Perturbation Magnitude')
for i, v in enumerate(avg_pert):
    ax2.text(i, v+0.1, f'{v:.2f}', ha='center')

# Queries
ax3.bar(models, avg_queries, color=['purple', 'brown'], alpha=0.7)
ax3.set_ylabel('Average Queries')
ax3.set_title('Query Efficiency')
for i, v in enumerate(avg_queries):
    ax3.text(i, v+10, f'{v:.0f}', ha='center')

plt.suptitle('Adversarial Robustness: XGBoost vs TabPFN', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('results/first_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Grafik kaydedildi: results/first_comparison.png")
plt.show()
