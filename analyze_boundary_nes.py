"""Statistical comparison of Boundary vs NES attacks"""
import json
import numpy as np
from scipy import stats

print("="*80)
print("BOUNDARY vs NES ATTACK - STATISTICAL ANALYSIS")
print("="*80)

datasets = ['wine', 'iris', 'diabetes', 'heart', 'breast_cancer']
models = ['TabPFN', 'XGBoost', 'LightGBM']

boundary_asrs = []
nes_asrs = []

print("\n1. RAW DATA")
print("-"*70)
print(f"{'Dataset':<15} {'Model':<12} {'Boundary':<12} {'NES':<12} {'Diff':<12}")
print("-"*70)

for dataset in datasets:
    try:
        with open(f'results/{dataset}_experiment.json', 'r') as f:
            boundary = json.load(f)
        with open(f'results/{dataset}_nes_experiment.json', 'r') as f:
            nes = json.load(f)
        
        for model in models:
            b_asr = boundary[model]['attack_success_rate']
            n_asr = nes[model]['attack_success_rate']
            diff = n_asr - b_asr
            
            boundary_asrs.append(b_asr)
            nes_asrs.append(n_asr)
            
            print(f"{dataset:<15} {model:<12} {b_asr:<12.2%} {n_asr:<12.2%} {diff:<+12.2%}")
    except Exception as e:
        print(f"{dataset}: Error - {e}")

print("-"*70)

# Statistical tests
print("\n2. STATISTICAL TESTS")
print("-"*70)

# Paired t-test
t_stat, t_p = stats.ttest_rel(boundary_asrs, nes_asrs)
print(f"\nPaired t-test:")
print(f"  t = {t_stat:.3f}")
print(f"  p = {t_p:.6f}")
print(f"  Significant (p<0.05)? {'✓ YES' if t_p < 0.05 else '✗ NO'}")

# Wilcoxon signed-rank test
w_stat, w_p = stats.wilcoxon(boundary_asrs, nes_asrs)
print(f"\nWilcoxon signed-rank test:")
print(f"  W = {w_stat:.3f}")
print(f"  p = {w_p:.6f}")
print(f"  Significant (p<0.05)? {'✓ YES' if w_p < 0.05 else '✗ NO'}")

# Effect size (Cohen's d)
diff = np.array(boundary_asrs) - np.array(nes_asrs)
cohens_d = np.mean(diff) / np.std(diff, ddof=1)
print(f"\nEffect size (Cohen's d): {cohens_d:.3f}")
print(f"  Interpretation: {'Small' if abs(cohens_d) < 0.5 else 'Medium' if abs(cohens_d) < 0.8 else 'Large'}")

# Summary statistics
print("\n3. SUMMARY STATISTICS")
print("-"*70)
print(f"Boundary Attack ASR: Mean={np.mean(boundary_asrs):.2%}, Std={np.std(boundary_asrs):.2%}")
print(f"NES Attack ASR:      Mean={np.mean(nes_asrs):.2%}, Std={np.std(nes_asrs):.2%}")
print(f"Difference:          {np.mean(boundary_asrs) - np.mean(nes_asrs):.2%} (Boundary higher)")

# By model
print("\n4. BY MODEL ANALYSIS")
print("-"*70)

for model in models:
    b_model = [boundary_asrs[i] for i in range(len(boundary_asrs)) if i % 3 == models.index(model)]
    n_model = [nes_asrs[i] for i in range(len(nes_asrs)) if i % 3 == models.index(model)]
    
    t_stat, t_p = stats.ttest_rel(b_model, n_model)
    print(f"\n{model}:")
    print(f"  Boundary mean: {np.mean(b_model):.2%}")
    print(f"  NES mean:      {np.mean(n_model):.2%}")
    print(f"  t-test p:      {t_p:.4f} {'✓' if t_p < 0.05 else ''}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)
if t_p < 0.05:
    print(f"""
✓ Boundary Attack is SIGNIFICANTLY more effective than NES Attack
  - p = {t_p:.6f} (highly significant)
  - Boundary ASR {np.mean(boundary_asrs) - np.mean(nes_asrs):.1%} higher on average
  - This is UNEXPECTED - NES should be stronger (uses gradients)
  
POSSIBLE EXPLANATIONS:
1. NES parameters not optimized for these datasets
2. Boundary attack better suited for tabular data
3. TabPFN/GBDT probability outputs may be noisy for NES
""")
else:
    print("✗ No significant difference between attack methods")
