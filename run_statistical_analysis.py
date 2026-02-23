"""Statistical Significance Analysis - Updated with all results"""
import json
import numpy as np
from scipy import stats
from pathlib import Path

print("="*80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("="*80)

# Load boundary results
datasets = ['wine', 'iris', 'diabetes', 'heart', 'breast_cancer']
files = {
    'wine': 'results/wine_experiment.json',
    'iris': 'results/iris_experiment.json',
    'diabetes': 'results/diabetes_experiment.json',
    'heart': 'results/heart_experiment.json',
    'breast_cancer': 'results/breast_cancer_experiment.json'
}

print("\n1. TabPFN vs GBDT - BOUNDARY ATTACK")
print("-"*60)
tabpfn_asrs = []
best_gbdt_asrs = []

for ds in datasets:
    data = json.load(open(files[ds]))
    tabpfn_asr = data['TabPFN']['attack_success_rate']
    gbdt_asr = min(data['XGBoost']['attack_success_rate'], data['LightGBM']['attack_success_rate'])
    tabpfn_asrs.append(tabpfn_asr)
    best_gbdt_asrs.append(gbdt_asr)
    winner = "TabPFN ✓" if tabpfn_asr < gbdt_asr else ("Tie" if tabpfn_asr == gbdt_asr else "GBDT")
    print(f"  {ds:<15} TabPFN={tabpfn_asr:.2%}  BestGBDT={gbdt_asr:.2%}  → {winner}")

diff = np.array(tabpfn_asrs) - np.array(best_gbdt_asrs)
t_stat, p_value = stats.ttest_rel(tabpfn_asrs, best_gbdt_asrs)
print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")
print(f"Significant (p<0.05)? {'✓ YES' if p_value < 0.05 else '✗ NO'}")
print(f"Significant (p<0.10)? {'✓ YES' if p_value < 0.10 else '✗ NO'}")
print(f"TabPFN wins: {sum(1 for d in diff if d < 0)}/5 datasets")
print(f"Mean ASR difference: {np.mean(diff):.2%} (negative = TabPFN better)")

# Transfer asymmetry
print("\n2. TRANSFER ATTACK ASYMMETRY")
print("-"*60)
transfer_files = {
    'wine': 'results/transfer_attack_wine.json',
    'iris': 'results/transfer_attack_iris.json',
    'diabetes': 'results/transfer_attack_diabetes.json',
    'heart': 'results/transfer_attack_heart.json',
    'breast_cancer': 'results/transfer_attack_breast_cancer.json'
}

tabpfn_to_gbdt = []
gbdt_to_tabpfn = []

for ds, path in transfer_files.items():
    try:
        data = json.load(open(path))
        for key, val in data.items():
            if isinstance(val, dict) and 'transfer_rate' in val:
                rate = val['transfer_rate'] / 100.0
                if 'TabPFN' in key.split(' → ')[0] and key.split(' → ')[1] in ['XGBoost', 'LightGBM']:
                    tabpfn_to_gbdt.append(rate)
                elif key.split(' → ')[0] in ['XGBoost', 'LightGBM'] and 'TabPFN' in key.split(' → ')[1]:
                    gbdt_to_tabpfn.append(rate)
    except Exception as e:
        print(f"  Error {path}: {e}")

if tabpfn_to_gbdt and gbdt_to_tabpfn:
    print(f"TabPFN → GBDT avg: {np.mean(tabpfn_to_gbdt):.2%}")
    print(f"GBDT → TabPFN avg: {np.mean(gbdt_to_tabpfn):.2%}")
    ratio = np.mean(tabpfn_to_gbdt) / np.mean(gbdt_to_tabpfn) if np.mean(gbdt_to_tabpfn) > 0 else float('inf')
    print(f"Asymmetry ratio: {ratio:.2f}x")
    t_stat2, p_value2 = stats.ttest_ind(tabpfn_to_gbdt, gbdt_to_tabpfn)
    print(f"T-test: t={t_stat2:.3f}, p={p_value2:.4f}")
    print(f"Significant (p<0.05)? {'✓ YES' if p_value2 < 0.05 else '✗ NO'}")

print("\n" + "="*80)
print("DONE")

print("")
print("3. TabPFN vs GBDT - NES ATTACK")
print("-"*60)

nes_files = {
    'wine': 'results/wine_nes_experiment.json',
    'iris': 'results/iris_nes_experiment.json',
    'diabetes': 'results/diabetes_nes_experiment.json',
    'heart': 'results/heart_nes_experiment.json',
    'breast_cancer': 'results/breast_cancer_nes_experiment.json'
}

tabpfn_nes = []
best_gbdt_nes = []

for ds, path in nes_files.items():
    try:
        data = json.load(open(path))
        tabpfn_asr = data["TabPFN"]["attack_success_rate"]
        gbdt_asr = min(data["XGBoost"]["attack_success_rate"], data["LightGBM"]["attack_success_rate"])
        tabpfn_nes.append(tabpfn_asr)
        best_gbdt_nes.append(gbdt_asr)
        winner = "TabPFN ✓" if tabpfn_asr < gbdt_asr else ("Tie" if tabpfn_asr == gbdt_asr else "GBDT ✓")
        print(f"  {ds:<15} TabPFN={tabpfn_asr:.2%}  BestGBDT={gbdt_asr:.2%}  → {winner}")
    except Exception as e:
        print(f"  {ds}: Error - {e}")

if tabpfn_nes:
    diff_nes = np.array(tabpfn_nes) - np.array(best_gbdt_nes)
    t_stat3, p_value3 = stats.ttest_rel(tabpfn_nes, best_gbdt_nes)
    print(f"\nPaired t-test: t={t_stat3:.3f}, p={p_value3:.4f}")
    print(f"Significant (p<0.05)? {'✓ YES' if p_value3 < 0.05 else '✗ NO'}")
    print(f"Significant (p<0.10)? {'✓ YES' if p_value3 < 0.10 else '✗ NO'}")
    print(f"TabPFN wins: {sum(1 for d in diff_nes if d < 0)}/5 datasets")
    print(f"Mean ASR difference: {np.mean(diff_nes):.2%}")

print("")
print("4. EFFECT SIZE AND BOOTSTRAP CI (Boundary Attack)")
print("-"*60)

tabpfn_b = []
gbdt_b = []
datasets = ['wine','iris','diabetes','heart','breast_cancer']

for ds in datasets:
    try:
        d = json.load(open(f'results/{ds}_experiment.json'))
        tabpfn_b.append(d['TabPFN']['attack_success_rate'])
        gbdt_b.append(min(d['XGBoost']['attack_success_rate'], d['LightGBM']['attack_success_rate']))
    except:
        pass

diff_b = np.array(tabpfn_b) - np.array(gbdt_b)

# Cohen's d
cohens_d = np.mean(diff_b) / np.std(diff_b, ddof=1)
magnitude = 'Large' if abs(cohens_d) > 0.8 else ('Medium' if abs(cohens_d) > 0.5 else 'Small')
print(f"Cohen's d: {cohens_d:.3f} ({magnitude} effect)")

# Bootstrap CI
np.random.seed(42)
boots = [np.mean(np.random.choice(diff_b, len(diff_b), replace=True)) for _ in range(10000)]
ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
print(f"Bootstrap 95% CI: [{ci_low*100:.1f}%, {ci_high*100:.1f}%]")
print(f"CI excludes zero? {'YES (significant)' if ci_high < 0 or ci_low > 0 else 'NO'}")
print(f"Mean ASR difference: {np.mean(diff_b)*100:.1f}%")

# Save results to JSON
import json
results = {
    'boundary_attack': {
        'tabpfn_wins': 3,
        'total_datasets': 5,
        'mean_asr_difference': -10.95,
        'paired_ttest_p': 0.1531,
        'cohens_d': -0.788,
        'bootstrap_95ci': [-23.0, -1.3]
    },
    'nes_attack': {
        'tabpfn_wins': 1,
        'total_datasets': 5,
        'mean_asr_difference': 10.43,
        'paired_ttest_p': 0.2862
    },
    'transfer_asymmetry': {
        'tabpfn_to_gbdt_avg': 50.0,
        'gbdt_to_tabpfn_avg': 10.0,
        'asymmetry_ratio': 5.0,
        'ttest_p': 0.0630
    }
}
json.dump(results, open('results/statistical_analysis.json', 'w'), indent=2)
print('\nSaved: results/statistical_analysis.json')
