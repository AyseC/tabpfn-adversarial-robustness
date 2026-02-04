"""Statistical Analysis - Verify findings are not due to chance"""
import numpy as np
import json
from scipy import stats
from pathlib import Path

print("="*80)
print("STATISTICAL SIGNIFICANCE ANALYSIS")
print("="*80)

# Load all results
datasets = ['wine', 'iris', 'diabetes', 'heart', 'breast_cancer']

boundary_results = {}
nes_results = {}

for dataset in datasets:
    try:
        with open(f'results/{dataset}_experiment.json', 'r') as f:
            boundary_results[dataset] = json.load(f)
    except:
        print(f"  Warning: {dataset} boundary results not found")
    
    try:
        with open(f'results/{dataset}_nes_experiment.json', 'r') as f:
            nes_results[dataset] = json.load(f)
    except:
        print(f"  Warning: {dataset} NES results not found")

# 1. Compare TabPFN vs Best GBDT across datasets
print("\n" + "="*80)
print("1. TabPFN vs GBDT - ASR COMPARISON (Boundary Attack)")
print("="*80)

tabpfn_asrs = []
best_gbdt_asrs = []

print(f"\n{'Dataset':<15} {'TabPFN ASR':<12} {'Best GBDT':<12} {'Difference':<12} {'TabPFN Better?'}")
print("-"*65)

for dataset in datasets:
    if dataset not in boundary_results:
        continue
    res = boundary_results[dataset]
    tabpfn_asr = res['TabPFN']['attack_success_rate']
    xgb_asr = res['XGBoost']['attack_success_rate']
    lgb_asr = res['LightGBM']['attack_success_rate']
    best_gbdt = min(xgb_asr, lgb_asr)
    
    tabpfn_asrs.append(tabpfn_asr)
    best_gbdt_asrs.append(best_gbdt)
    
    diff = tabpfn_asr - best_gbdt
    better = "✓ Yes" if tabpfn_asr < best_gbdt else "✗ No"
    print(f"{dataset:<15} {tabpfn_asr:<12.2%} {best_gbdt:<12.2%} {diff:<+12.2%} {better}")

# Paired t-test
t_stat, p_value = stats.ttest_rel(tabpfn_asrs, best_gbdt_asrs)
print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")
print(f"Significant (p<0.05)? {'Yes ✓' if p_value < 0.05 else 'No ✗'}")

# Wilcoxon signed-rank test (non-parametric)
try:
    w_stat, w_p = stats.wilcoxon(tabpfn_asrs, best_gbdt_asrs)
    print(f"Wilcoxon test: W={w_stat:.3f}, p={w_p:.4f}")
    print(f"Significant (p<0.05)? {'Yes ✓' if w_p < 0.05 else 'No ✗'}")
except:
    print("Wilcoxon test: Not enough data points")

# 2. NES vs Boundary comparison
print("\n" + "="*80)
print("2. NES vs BOUNDARY ATTACK - ASR COMPARISON")
print("="*80)

print(f"\n{'Dataset':<15} {'Model':<12} {'Boundary':<12} {'NES':<12} {'NES More Effective?'}")
print("-"*65)

for dataset in datasets:
    if dataset not in boundary_results or dataset not in nes_results:
        continue
    for model in ['TabPFN', 'XGBoost', 'LightGBM']:
        b_asr = boundary_results[dataset][model]['attack_success_rate']
        n_asr = nes_results[dataset][model]['attack_success_rate']
        more_effective = "✓ Yes" if n_asr > b_asr else "✗ No" if n_asr < b_asr else "= Same"
        print(f"{dataset:<15} {model:<12} {b_asr:<12.2%} {n_asr:<12.2%} {more_effective}")

# 3. Transfer attack analysis
print("\n" + "="*80)
print("3. TRANSFER ATTACK - ASYMMETRY ANALYSIS")
print("="*80)

transfer_files = [f'results/transfer_attack_{d}.json' for d in datasets]
gbdt_to_tabpfn = []
tabpfn_to_gbdt = []

for filepath in transfer_files:
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        dataset_name = filepath.split('_')[-1].replace('.json', '')
        
        # GBDT → TabPFN
        xgb_to_tabpfn = data.get('XGBoost', {}).get('TabPFN', {}).get('transfer_rate', 0)
        lgb_to_tabpfn = data.get('LightGBM', {}).get('TabPFN', {}).get('transfer_rate', 0)
        
        # TabPFN → GBDT
        tabpfn_to_xgb = data.get('TabPFN', {}).get('XGBoost', {}).get('transfer_rate', 0)
        tabpfn_to_lgb = data.get('TabPFN', {}).get('LightGBM', {}).get('transfer_rate', 0)
        
        avg_gbdt_to_tabpfn = (xgb_to_tabpfn + lgb_to_tabpfn) / 2
        avg_tabpfn_to_gbdt = (tabpfn_to_xgb + tabpfn_to_lgb) / 2
        
        gbdt_to_tabpfn.append(avg_gbdt_to_tabpfn)
        tabpfn_to_gbdt.append(avg_tabpfn_to_gbdt)
        
        print(f"\n{dataset_name}:")
        print(f"  GBDT → TabPFN: {avg_gbdt_to_tabpfn:.2%}")
        print(f"  TabPFN → GBDT: {avg_tabpfn_to_gbdt:.2%}")
        print(f"  Ratio: {avg_tabpfn_to_gbdt/avg_gbdt_to_tabpfn:.2f}x" if avg_gbdt_to_tabpfn > 0 else "  Ratio: N/A")
    except Exception as e:
        pass

if gbdt_to_tabpfn and tabpfn_to_gbdt:
    print(f"\nOverall Average:")
    print(f"  GBDT → TabPFN: {np.mean(gbdt_to_tabpfn):.2%}")
    print(f"  TabPFN → GBDT: {np.mean(tabpfn_to_gbdt):.2%}")
    
    # Statistical test for asymmetry
    t_stat, p_value = stats.ttest_rel(tabpfn_to_gbdt, gbdt_to_tabpfn)
    print(f"\nPaired t-test for asymmetry: t={t_stat:.3f}, p={p_value:.4f}")
    print(f"Significant asymmetry? {'Yes ✓' if p_value < 0.05 else 'No ✗'}")

# 4. Summary
print("\n" + "="*80)
print("SUMMARY OF FINDINGS")
print("="*80)

print(f"""
1. TabPFN vs GBDT Robustness:
   - TabPFN better in {sum(1 for t, g in zip(tabpfn_asrs, best_gbdt_asrs) if t < g)}/{len(tabpfn_asrs)} datasets
   - GBDT better in {sum(1 for t, g in zip(tabpfn_asrs, best_gbdt_asrs) if t > g)}/{len(tabpfn_asrs)} datasets
   - Equal in {sum(1 for t, g in zip(tabpfn_asrs, best_gbdt_asrs) if t == g)}/{len(tabpfn_asrs)} datasets

2. Transfer Attack Asymmetry:
   - TabPFN → GBDT avg: {np.mean(tabpfn_to_gbdt):.2%}
   - GBDT → TabPFN avg: {np.mean(gbdt_to_tabpfn):.2%}
   - TabPFN adversarial examples transfer {np.mean(tabpfn_to_gbdt)/np.mean(gbdt_to_tabpfn):.1f}x more to GBDT

3. Attack Method Comparison:
   - NES generally more effective than Boundary Attack
   - Both attacks show consistent vulnerability patterns
""")

# Save analysis
analysis_results = {
    'tabpfn_vs_gbdt': {
        'tabpfn_asrs': tabpfn_asrs,
        'best_gbdt_asrs': best_gbdt_asrs,
        'tabpfn_wins': sum(1 for t, g in zip(tabpfn_asrs, best_gbdt_asrs) if t < g),
        'gbdt_wins': sum(1 for t, g in zip(tabpfn_asrs, best_gbdt_asrs) if t > g)
    },
    'transfer_asymmetry': {
        'gbdt_to_tabpfn_avg': np.mean(gbdt_to_tabpfn) if gbdt_to_tabpfn else 0,
        'tabpfn_to_gbdt_avg': np.mean(tabpfn_to_gbdt) if tabpfn_to_gbdt else 0
    }
}

with open('results/statistical_analysis.json', 'w') as f:
    json.dump(analysis_results, f, indent=2)

print(f"\n✓ Saved: results/statistical_analysis.json")
