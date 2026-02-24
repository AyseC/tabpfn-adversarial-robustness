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
    try:
        with open(files[ds]) as f:
            data = json.load(f)
        tabpfn_asr = data['TabPFN']['attack_success_rate']
        gbdt_asr = min(data['XGBoost']['attack_success_rate'], data['LightGBM']['attack_success_rate'])
    except Exception as e:
        print(f"  Error {files[ds]}: {e}")
        continue
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
        with open(path) as f:
            data = json.load(f)
        for key, val in data.items():
            if isinstance(val, dict) and 'transfer_rate' in val:
                # Wine-style flat format: {"XGBoost → TabPFN": {"transfer_rate": 0.5}}
                rate = val['transfer_rate']
                parts = key.split(' → ')
                if len(parts) == 2:
                    src, tgt = parts[0], parts[1]
                    if src == tgt:
                        continue
                    if 'TabPFN' in src and tgt in ['XGBoost', 'LightGBM']:
                        tabpfn_to_gbdt.append(rate)
                    elif src in ['XGBoost', 'LightGBM'] and 'TabPFN' in tgt:
                        gbdt_to_tabpfn.append(rate)
            elif isinstance(val, dict):
                # Nested matrix format: {"TabPFN": {"XGBoost": {"transfer_rate": 0.5}}}
                for tgt, tgt_data in val.items():
                    if not isinstance(tgt_data, dict) or 'transfer_rate' not in tgt_data:
                        continue
                    if key == tgt:
                        continue  # Skip self-transfer
                    rate = tgt_data['transfer_rate']
                    src = key
                    if src == 'TabPFN' and tgt in ['XGBoost', 'LightGBM']:
                        tabpfn_to_gbdt.append(rate)
                    elif src in ['XGBoost', 'LightGBM'] and tgt == 'TabPFN':
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
        with open(path) as f:
            data = json.load(f)
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
        with open(f'results/{ds}_experiment.json') as f:
            d = json.load(f)
        tabpfn_b.append(d['TabPFN']['attack_success_rate'])
        gbdt_b.append(min(d['XGBoost']['attack_success_rate'], d['LightGBM']['attack_success_rate']))
    except Exception:
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

# Save results to JSON (dynamically from computed values)
results = {}

if tabpfn_b:
    results['boundary_attack'] = {
        'tabpfn_wins': int(sum(1 for d in diff_b if d < 0)),
        'total_datasets': len(tabpfn_b),
        'mean_asr_difference': float(round(np.mean(diff_b) * 100, 2)),
        'paired_ttest_p': float(round(p_value, 4)),
        'cohens_d': float(round(cohens_d, 3)),
        'bootstrap_95ci': [float(round(ci_low * 100, 1)), float(round(ci_high * 100, 1))]
    }

if tabpfn_nes:
    results['nes_attack'] = {
        'tabpfn_wins': int(sum(1 for d in diff_nes if d < 0)),
        'total_datasets': len(tabpfn_nes),
        'mean_asr_difference': float(round(np.mean(diff_nes) * 100, 2)),
        'paired_ttest_p': float(round(p_value3, 4))
    }

if tabpfn_to_gbdt and gbdt_to_tabpfn:
    results['transfer_asymmetry'] = {
        'tabpfn_to_gbdt_avg': float(round(np.mean(tabpfn_to_gbdt) * 100, 1)),
        'gbdt_to_tabpfn_avg': float(round(np.mean(gbdt_to_tabpfn) * 100, 1)),
        'asymmetry_ratio': float(round(ratio, 2)),
        'ttest_p': float(round(p_value2, 4))
    }

with open('results/statistical_analysis.json', 'w') as f:
    json.dump(results, f, indent=2)
print('\nSaved: results/statistical_analysis.json')
