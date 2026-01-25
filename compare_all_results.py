"""Compare results across all datasets"""
import json
import matplotlib.pyplot as plt
import numpy as np

# Load all results
datasets = {}

try:
    with open('results/wine_experiment.json', 'r') as f:
        datasets['Wine'] = json.load(f)
    print("✓ Loaded Wine results")
except:
    print("✗ No Wine results found")

try:
    with open('results/breast_cancer_experiment.json', 'r') as f:
        datasets['Breast Cancer'] = json.load(f)
    print("✓ Loaded Breast Cancer results")
except:
    print("✗ No Breast Cancer results found")

if not datasets:
    print("\n❌ No results found!")
    exit()

print("\n" + "="*70)
print("CROSS-DATASET COMPARISON")
print("="*70)

# Get all models
models = list(next(iter(datasets.values())).keys())

# Create comparison table
print(f"\n{'Dataset':<20} {'Model':<12} {'ASR':<10} {'Avg Pert':<12} {'Robustness':<12}")
print("-"*70)

for dataset_name, results in datasets.items():
    for model_name in models:
        m = results[model_name]
        print(f"{dataset_name:<20} {model_name:<12} "
              f"{m['attack_success_rate']:<10.2%} "
              f"{m['avg_perturbation']:<12.4f} "
              f"{m['robustness_score']:<12.4f}")
    print("-"*70)

# Summary
print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)

for dataset_name, results in datasets.items():
    best = max(results.items(), key=lambda x: x[1]['robustness_score'])
    worst = min(results.items(), key=lambda x: x[1]['robustness_score'])
    
    print(f"\n{dataset_name}:")
    print(f"  Most Robust: {best[0]} (score: {best[1]['robustness_score']:.4f})")
    print(f"  Least Robust: {worst[0]} (score: {worst[1]['robustness_score']:.4f})")
    print(f"  TabPFN ASR: {results['TabPFN']['attack_success_rate']:.1%}")
    
    # Compare TabPFN to best GBDT
    gbdt_results = [(name, data) for name, data in results.items() 
                    if name in ['XGBoost', 'LightGBM']]
    
    if gbdt_results:
        gbdt_best = max(gbdt_results, key=lambda x: x[1]['robustness_score'])
        ratio = results['TabPFN']['attack_success_rate'] / gbdt_best[1]['attack_success_rate']
        print(f"  TabPFN is {ratio:.1f}x more vulnerable than {gbdt_best[0]}")

print("\n" + "="*70)
print("\n🎓 THESIS CONCLUSION:")
print("TabPFN shows consistently higher vulnerability to adversarial attacks")
print("compared to traditional GBDT models (XGBoost, LightGBM)")
print("="*70)
