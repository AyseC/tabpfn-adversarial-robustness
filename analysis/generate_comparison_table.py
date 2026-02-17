"""Generate comprehensive comparison table for thesis"""
import json
import pandas as pd
from pathlib import Path

print("="*80)
print("COMPREHENSIVE COMPARISON TABLE - ALL EXPERIMENTS")
print("="*80)

# Load all results
datasets = {
    'Iris': {'features': 4, 'file': 'iris_experiment.json'},
    'Diabetes': {'features': 10, 'file': 'diabetes_experiment.json'},
    'Wine': {'features': 13, 'file': 'wine_experiment.json'},
    'Heart': {'features': 13, 'file': 'heart_experiment.json'},
    'Breast Cancer': {'features': 30, 'file': 'breast_cancer_experiment.json'}
}

results = []

for dataset_name, info in datasets.items():
    try:
        with open(f"results/{info['file']}", 'r') as f:
            data = json.load(f)
        
        for model_name, metrics in data.items():
            results.append({
                'Dataset': dataset_name,
                'Features': info['features'],
                'Model': model_name,
                'Clean_Acc': metrics['clean_accuracy'],
                'ASR': metrics['attack_success_rate'],
                'Adv_Acc': metrics.get('adversarial_accuracy', 
                           metrics['clean_accuracy'] * (1 - metrics['attack_success_rate'])),
                'Avg_Pert': metrics['avg_perturbation'],
                'Robustness': metrics['robustness_score']
            })
        print(f"✓ Loaded {dataset_name}")
    except Exception as e:
        print(f"✗ Error loading {dataset_name}: {e}")

# Create DataFrame
df = pd.DataFrame(results)

# Format percentages
df['Clean_Acc_pct'] = (df['Clean_Acc'] * 100).round(1).astype(str) + '%'
df['ASR_pct'] = (df['ASR'] * 100).round(1).astype(str) + '%'
df['Adv_Acc_pct'] = (df['Adv_Acc'] * 100).round(1).astype(str) + '%'

# Table 1: Main Results
print("\n" + "="*80)
print("TABLE 1: ATTACK SUCCESS RATES BY DATASET AND MODEL")
print("="*80)

pivot_asr = df.pivot_table(values='ASR', index='Dataset', columns='Model', aggfunc='first')
pivot_asr = pivot_asr[['TabPFN', 'XGBoost', 'LightGBM']]  # Order columns
pivot_asr = pivot_asr.reindex(['Iris', 'Diabetes', 'Wine', 'Heart', 'Breast Cancer'])

print("\n(Lower ASR = More Robust)")
print("-"*80)
print(f"{'Dataset':<15} {'Features':<10} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12} {'Most Robust':<15}")
print("-"*80)

for dataset in ['Iris', 'Diabetes', 'Wine', 'Heart', 'Breast Cancer']:
    feat = datasets[dataset]['features']
    row = df[df['Dataset'] == dataset]
    
    tabpfn_asr = row[row['Model'] == 'TabPFN']['ASR'].values[0] * 100
    xgb_asr = row[row['Model'] == 'XGBoost']['ASR'].values[0] * 100
    lgb_asr = row[row['Model'] == 'LightGBM']['ASR'].values[0] * 100
    
    asrs = {'TabPFN': tabpfn_asr, 'XGBoost': xgb_asr, 'LightGBM': lgb_asr}
    most_robust = min(asrs, key=asrs.get)
    
    print(f"{dataset:<15} {feat:<10} {tabpfn_asr:<12.1f}% {xgb_asr:<12.1f}% {lgb_asr:<12.1f}% {most_robust:<15}")

print("-"*80)

# Table 2: Adversarial Accuracy
print("\n" + "="*80)
print("TABLE 2: ADVERSARIAL ACCURACY (Clean Acc × (1-ASR))")
print("="*80)
print("\n(Higher Adv Acc = More Robust)")
print("-"*80)
print(f"{'Dataset':<15} {'Features':<10} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12} {'Most Robust':<15}")
print("-"*80)

for dataset in ['Iris', 'Diabetes', 'Wine', 'Heart', 'Breast Cancer']:
    feat = datasets[dataset]['features']
    row = df[df['Dataset'] == dataset]
    
    tabpfn_adv = row[row['Model'] == 'TabPFN']['Adv_Acc'].values[0] * 100
    xgb_adv = row[row['Model'] == 'XGBoost']['Adv_Acc'].values[0] * 100
    lgb_adv = row[row['Model'] == 'LightGBM']['Adv_Acc'].values[0] * 100
    
    advs = {'TabPFN': tabpfn_adv, 'XGBoost': xgb_adv, 'LightGBM': lgb_adv}
    most_robust = max(advs, key=advs.get)
    
    print(f"{dataset:<15} {feat:<10} {tabpfn_adv:<12.1f}% {xgb_adv:<12.1f}% {lgb_adv:<12.1f}% {most_robust:<15}")

print("-"*80)

# Table 3: TabPFN vs GBDT Comparison
print("\n" + "="*80)
print("TABLE 3: TABPFN VULNERABILITY RATIO vs BEST GBDT")
print("="*80)
print("-"*80)
print(f"{'Dataset':<15} {'Features':<10} {'TabPFN ASR':<12} {'Best GBDT':<12} {'Ratio':<10} {'Interpretation':<20}")
print("-"*80)

for dataset in ['Iris', 'Diabetes', 'Wine', 'Heart', 'Breast Cancer']:
    feat = datasets[dataset]['features']
    row = df[df['Dataset'] == dataset]
    
    tabpfn_asr = row[row['Model'] == 'TabPFN']['ASR'].values[0]
    xgb_asr = row[row['Model'] == 'XGBoost']['ASR'].values[0]
    lgb_asr = row[row['Model'] == 'LightGBM']['ASR'].values[0]
    
    best_gbdt_asr = min(xgb_asr, lgb_asr)
    best_gbdt_name = 'XGBoost' if xgb_asr <= lgb_asr else 'LightGBM'
    
    if best_gbdt_asr > 0:
        ratio = tabpfn_asr / best_gbdt_asr
    else:
        ratio = 0
    
    if ratio < 0.8:
        interp = "TabPFN MORE robust"
    elif ratio > 1.2:
        interp = "GBDT MORE robust"
    else:
        interp = "Comparable"
    
    print(f"{dataset:<15} {feat:<10} {tabpfn_asr*100:<12.1f}% {best_gbdt_asr*100:<12.1f}% {ratio:<10.2f}x {interp:<20}")

print("-"*80)

# Summary Statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

tabpfn_wins = 0
gbdt_wins = 0
ties = 0

for dataset in ['Iris', 'Diabetes', 'Wine', 'Heart', 'Breast Cancer']:
    row = df[df['Dataset'] == dataset]
    tabpfn_asr = row[row['Model'] == 'TabPFN']['ASR'].values[0]
    best_gbdt_asr = min(
        row[row['Model'] == 'XGBoost']['ASR'].values[0],
        row[row['Model'] == 'LightGBM']['ASR'].values[0]
    )
    
    ratio = tabpfn_asr / best_gbdt_asr if best_gbdt_asr > 0 else 0
    
    if ratio < 0.8:
        tabpfn_wins += 1
    elif ratio > 1.2:
        gbdt_wins += 1
    else:
        ties += 1

print(f"\nTabPFN more robust: {tabpfn_wins}/5 datasets")
print(f"GBDT more robust:   {gbdt_wins}/5 datasets")
print(f"Comparable:         {ties}/5 datasets")

print("\n" + "="*80)
print("KEY FINDING: Dataset-Dependent Vulnerability")
print("="*80)
print("""
TabPFN's adversarial robustness varies significantly across datasets:
- More robust on: Iris, Heart
- Less robust on: Wine, Breast Cancer  
- Comparable on: Diabetes

Note: Wine and Heart both have 13 features but opposite results!
This confirms that feature count alone does not predict robustness.
""")

# Save to CSV
df_export = df[['Dataset', 'Features', 'Model', 'Clean_Acc_pct', 'ASR_pct', 
                'Adv_Acc_pct', 'Avg_Pert', 'Robustness']].copy()
df_export.columns = ['Dataset', 'Features', 'Model', 'Clean Acc', 'ASR', 
                     'Adv Acc', 'Avg Pert', 'Robustness Score']
df_export = df_export.round(4)
df_export.to_csv('results/comparison_table.csv', index=False)
print("\n✓ Saved: results/comparison_table.csv")

# Save summary JSON
summary = {
    'tabpfn_wins': tabpfn_wins,
    'gbdt_wins': gbdt_wins,
    'ties': ties,
    'datasets_tested': 5,
    'models_tested': 3,
    'samples_per_experiment': 15
}

with open('results/experiment_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
print("✓ Saved: results/experiment_summary.json")

print("\n" + "="*80)
print("COMPARISON TABLE GENERATION COMPLETE!")
print("="*80)
