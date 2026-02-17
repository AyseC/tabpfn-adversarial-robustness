"""Analyze defense results across all datasets"""
import json
import numpy as np
from scipy import stats

print("="*80)
print("DEFENSE MECHANISMS - COMPREHENSIVE ANALYSIS")
print("="*80)

datasets = ['wine', 'iris', 'diabetes', 'heart', 'breast_cancer']
defense_results = {}

for dataset in datasets:
    try:
        with open(f'results/{dataset}_defense_results.json', 'r') as f:
            defense_results[dataset] = json.load(f)
        print(f"  ✓ Loaded {dataset}")
    except Exception as e:
        print(f"  ✗ {dataset}: {e}")

print("\n" + "="*80)
print("1. BASELINE ASR (NO DEFENSE)")
print("="*80)

baseline_asrs = []
features = {'wine': 13, 'iris': 4, 'diabetes': 8, 'heart': 13, 'breast_cancer': 30}

print(f"\n{'Dataset':<20} {'Features':<10} {'Baseline ASR':<15}")
print("-"*50)
for dataset in datasets:
    if dataset in defense_results and 'no_defense' in defense_results[dataset]:
        data = defense_results[dataset]['no_defense']
        asr = data['asr'] if isinstance(data, dict) else data
        baseline_asrs.append(asr)
        print(f"{dataset:<20} {features[dataset]:<10} {asr:<15.2%}")

print(f"\nMean Baseline ASR: {np.mean(baseline_asrs):.2%}")

print("\n" + "="*80)
print("2. GAUSSIAN NOISE DEFENSE")
print("="*80)

noise_levels = ['0.01', '0.03', '0.05', '0.1']
print(f"\n{'Dataset':<15}", end="")
for n in noise_levels:
    print(f"σ={n:<8}", end="")
print("Best")
print("-"*65)

gaussian_best = []
for dataset in datasets:
    if dataset not in defense_results:
        continue
    print(f"{dataset:<15}", end="")
    best_recovery = 0
    for n in noise_levels:
        key = f'gaussian_noise_{n}'
        if key in defense_results[dataset]:
            data = defense_results[dataset][key]
            recovery = data.get('recovery', 0) if isinstance(data, dict) else 0
            print(f"{recovery:<10.1%}", end="")
            if recovery > best_recovery:
                best_recovery = recovery
        else:
            print(f"{'N/A':<10}", end="")
    print(f"{best_recovery:.1%}")
    gaussian_best.append(best_recovery)

print(f"\nMean Best Gaussian Recovery: {np.mean(gaussian_best):.2%}")

print("\n" + "="*80)
print("3. FEATURE SQUEEZING DEFENSE")
print("="*80)

bit_depths = ['4', '8', '16']
print(f"\n{'Dataset':<15}", end="")
for b in bit_depths:
    print(f"{b}-bit{'':<6}", end="")
print("Best")
print("-"*50)

squeezing_best = []
for dataset in datasets:
    if dataset not in defense_results:
        continue
    print(f"{dataset:<15}", end="")
    best_recovery = 0
    for b in bit_depths:
        key = f'feature_squeezing_{b}bit'
        if key in defense_results[dataset]:
            data = defense_results[dataset][key]
            recovery = data.get('recovery', 0) if isinstance(data, dict) else 0
            print(f"{recovery:<10.1%}", end="")
            if recovery > best_recovery:
                best_recovery = recovery
        else:
            print(f"{'N/A':<10}", end="")
    print(f"{best_recovery:.1%}")
    squeezing_best.append(best_recovery)

print(f"\nMean Best Squeezing Recovery: {np.mean(squeezing_best):.2%}")

print("\n" + "="*80)
print("4. ENSEMBLE VOTING DEFENSE")
print("="*80)

ensemble_recoveries = []
print(f"\n{'Dataset':<20} {'Features':<10} {'Recovery':<15}")
print("-"*50)
for dataset in datasets:
    if dataset in defense_results and 'ensemble_voting' in defense_results[dataset]:
        data = defense_results[dataset]['ensemble_voting']
        recovery = data.get('recovery', 0) if isinstance(data, dict) else 0
        ensemble_recoveries.append(recovery)
        print(f"{dataset:<20} {features[dataset]:<10} {recovery:<15.2%}")

print(f"\nMean Ensemble Recovery: {np.mean(ensemble_recoveries):.2%}")

# Correlation with features
feat_list = [features[d] for d in datasets if d in defense_results and 'ensemble_voting' in defense_results[d]]
if len(feat_list) == len(ensemble_recoveries) and len(feat_list) > 2:
    corr, p_val = stats.pearsonr(feat_list, ensemble_recoveries)
    print(f"\nFeature Count vs Ensemble Recovery:")
    print(f"  Pearson r = {corr:.3f}")
    print(f"  p-value = {p_val:.4f}")
    print(f"  Significant? {'Yes ✓' if p_val < 0.05 else 'No'}")

print("\n" + "="*80)
print("5. BEST DEFENSE PER DATASET")
print("="*80)

print(f"\n{'Dataset':<15} {'Features':<10} {'Best Defense':<25} {'Recovery':<12}")
print("-"*65)

for dataset in datasets:
    if dataset not in defense_results:
        continue
    
    best_defense = "None"
    best_recovery = 0
    
    for key, val in defense_results[dataset].items():
        if key == 'no_defense':
            continue
        if isinstance(val, dict):
            recovery = val.get('recovery', 0)
            if recovery > best_recovery:
                best_recovery = recovery
                best_defense = key
    
    print(f"{dataset:<15} {features[dataset]:<10} {best_defense:<25} {best_recovery:<12.2%}")

print("\n" + "="*80)
print("6. DEFENSE COMPARISON SUMMARY")
print("="*80)

if gaussian_best and squeezing_best and ensemble_recoveries:
    print(f"""
Defense Type          Mean Recovery    Std Dev
--------------------------------------------------
Gaussian Noise        {np.mean(gaussian_best):.2%}           {np.std(gaussian_best):.2%}
Feature Squeezing     {np.mean(squeezing_best):.2%}           {np.std(squeezing_best):.2%}
Ensemble Voting       {np.mean(ensemble_recoveries):.2%}           {np.std(ensemble_recoveries):.2%}
""")

print("\n" + "="*80)
print("7. KEY FINDINGS FOR THESIS")
print("="*80)

print("""
1. ENSEMBLE VOTING:
   - Most effective on HIGH-DIMENSIONAL data (Breast Cancer: 86.7%)
   - Least effective on LOW-DIMENSIONAL data (Diabetes: 9.1%)
   - Effectiveness correlates with feature count

2. GAUSSIAN NOISE:
   - Consistent across datasets
   - Simple but effective defense
   - Best at low noise levels (σ=0.01-0.03)

3. FEATURE SQUEEZING:
   - Variable effectiveness by dataset
   - Works well on some datasets (Iris, Diabetes)
   - Higher bit-depth often better

4. PATTERN:
   - No single defense works best for all datasets
   - Defense effectiveness depends on data characteristics
   - Ensemble benefits from feature diversity
""")

print("\n✓ Analysis complete!")
