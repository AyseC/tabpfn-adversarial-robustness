"""Boundary Attack Results - Table and Plots (Real Datasets)"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Load data ──────────────────────────────────────────────────────────────────
datasets = ['iris', 'wine', 'breast_cancer', 'diabetes', 'heart']
dataset_labels = ['Iris', 'Wine', 'Breast\nCancer', 'Diabetes', 'Heart']
models = ['TabPFN', 'XGBoost', 'LightGBM']

files = {ds: f'results/{ds}_experiment.json' for ds in datasets}

data = {}
for ds, path in files.items():
    with open(path) as f:
        data[ds] = json.load(f)

# ── Console table ──────────────────────────────────────────────────────────────
print("=" * 90)
print("BOUNDARY ATTACK RESULTS — REAL DATASETS")
print("=" * 90)
print(f"\n{'Dataset':<16} {'Model':<10} {'Clean Acc':>10} {'ASR':>10} {'Adv Acc':>10} {'Rob Score':>11} {'Avg Pert':>10}")
print("-" * 90)

for ds, label in zip(datasets, ['Iris', 'Wine', 'Breast Cancer', 'Diabetes', 'Heart']):
    for i, model in enumerate(models):
        r = data[ds][model]
        prefix = label if i == 0 else ''
        print(f"{prefix:<16} {model:<10} "
              f"{r['clean_accuracy']:>10.2%} "
              f"{r['attack_success_rate']:>10.2%} "
              f"{r['adversarial_accuracy']:>10.2%} "
              f"{r['robustness_score']:>11.4f} "
              f"{r['avg_perturbation']:>10.3f}")
    print("-" * 90)

# ── Best model per dataset ─────────────────────────────────────────────────────
print("\nMost Robust Model per Dataset (lowest ASR):")
for ds, label in zip(datasets, ['Iris', 'Wine', 'Breast Cancer', 'Diabetes', 'Heart']):
    asrs = {m: data[ds][m]['attack_success_rate'] for m in models}
    best = min(asrs, key=asrs.get)
    print(f"  {label:<14}: {best} (ASR={asrs[best]:.2%})")

# ── Prepare arrays for plotting ────────────────────────────────────────────────
metrics = {
    'clean_accuracy':    {'label': 'Clean Accuracy',       'fmt': '{:.0%}'},
    'attack_success_rate': {'label': 'Attack Success Rate (ASR)', 'fmt': '{:.0%}'},
    'adversarial_accuracy': {'label': 'Adversarial Accuracy',  'fmt': '{:.0%}'},
    'robustness_score':  {'label': 'Robustness Score',     'fmt': '{:.3f}'},
}

values = {metric: {model: [] for model in models} for metric in metrics}
for ds in datasets:
    for model in models:
        for metric in metrics:
            values[metric][model].append(data[ds][model][metric])

# ── Colors ─────────────────────────────────────────────────────────────────────
colors = {'TabPFN': '#2196F3', 'XGBoost': '#FF9800', 'LightGBM': '#4CAF50'}

# ── Figure: 2×2 subplots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Boundary Attack Results — Real Datasets', fontsize=15, fontweight='bold', y=1.01)

x = np.arange(len(datasets))
width = 0.26
offsets = [-width, 0, width]

metric_keys = list(metrics.keys())

for ax_idx, (ax, metric) in enumerate(zip(axes.flat, metric_keys)):
    for model, offset in zip(models, offsets):
        vals = values[metric][model]
        bars = ax.bar(x + offset, vals, width, label=model,
                      color=colors[model], alpha=0.85, edgecolor='white', linewidth=0.5)
        # Value labels on bars
        for bar, v in zip(bars, vals):
            if metric == 'robustness_score':
                label = f'{v:.3f}'
                fontsize = 6.5
            else:
                label = f'{v:.0%}'
                fontsize = 6.5
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    label, ha='center', va='bottom', fontsize=fontsize, rotation=90)

    ax.set_title(metrics[metric]['label'], fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['Iris', 'Wine', 'Breast\nCancer', 'Diabetes', 'Heart'], fontsize=9)
    ax.set_ylim(0, 1.25 if metric != 'robustness_score' else 0.85)
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f'{v:.0%}') if metric != 'robustness_score'
        else plt.FuncFormatter(lambda v, _: f'{v:.2f}')
    )
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.spines[['top', 'right']].set_visible(False)

plt.tight_layout()
Path("results/figures").mkdir(parents=True, exist_ok=True)
fig.savefig('results/figures/boundary_all_datasets.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved: results/figures/boundary_all_datasets.png")

# ── Figure 2: ASR heatmap-style summary ───────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(10, 4))

asr_matrix = np.array([[data[ds][m]['attack_success_rate'] for ds in datasets] for m in models])

im = ax2.imshow(asr_matrix, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')

ax2.set_xticks(range(len(datasets)))
ax2.set_xticklabels(['Iris', 'Wine', 'Breast Cancer', 'Diabetes', 'Heart'], fontsize=10)
ax2.set_yticks(range(len(models)))
ax2.set_yticklabels(models, fontsize=10)
ax2.set_title('Attack Success Rate (ASR) — Boundary Attack\n(green = robust, red = vulnerable)',
              fontsize=11, fontweight='bold')

for i in range(len(models)):
    for j in range(len(datasets)):
        val = asr_matrix[i, j]
        color = 'white' if val > 0.75 or val < 0.25 else 'black'
        ax2.text(j, i, f'{val:.0%}', ha='center', va='center',
                 fontsize=12, fontweight='bold', color=color)

plt.colorbar(im, ax=ax2, label='ASR', fraction=0.03, pad=0.04)
plt.tight_layout()
fig2.savefig('results/figures/boundary_asr_heatmap.png', dpi=150, bbox_inches='tight')
print("✓ Saved: results/figures/boundary_asr_heatmap.png")

plt.show()
