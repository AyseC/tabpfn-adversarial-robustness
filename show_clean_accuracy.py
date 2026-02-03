"""Show clean accuracy for all experiments"""
import json
from pathlib import Path

print("="*80)
print("CLEAN ACCURACY COMPARISON - ALL EXPERIMENTS")
print("="*80)

# Boundary Attack Results
print("\n" + "="*80)
print("BOUNDARY ATTACK - CLEAN ACCURACY")
print("="*80)
print(f"\n{'Dataset':<20} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12}")
print("-"*60)

boundary_files = {
    'Wine': 'wine_experiment.json',
    'Iris': 'iris_experiment.json',
    'Diabetes': 'diabetes_experiment.json',
    'Heart': 'heart_experiment.json',
    'Breast Cancer': 'breast_cancer_experiment.json'
}

for dataset, filename in boundary_files.items():
    try:
        with open(f'results/{filename}', 'r') as f:
            data = json.load(f)
        tabpfn = data.get('TabPFN', {}).get('clean_accuracy', 0) * 100
        xgb = data.get('XGBoost', {}).get('clean_accuracy', 0) * 100
        lgb = data.get('LightGBM', {}).get('clean_accuracy', 0) * 100
        print(f"{dataset:<20} {tabpfn:<12.1f}% {xgb:<12.1f}% {lgb:<12.1f}%")
    except Exception as e:
        print(f"{dataset:<20} Error: {e}")

# NES Attack Results
print("\n" + "="*80)
print("NES ATTACK - CLEAN ACCURACY")
print("="*80)
print(f"\n{'Dataset':<20} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12}")
print("-"*60)

nes_files = {
    'Wine': 'wine_nes_experiment.json',
    'Iris': 'iris_nes_experiment.json',
    'Diabetes': 'diabetes_nes_experiment.json',
    'Heart': 'heart_nes_experiment.json',
    'Breast Cancer': 'breast_cancer_nes_experiment.json'
}

for dataset, filename in nes_files.items():
    try:
        with open(f'results/{filename}', 'r') as f:
            data = json.load(f)
        tabpfn = data.get('TabPFN', {}).get('clean_accuracy', 0) * 100
        xgb = data.get('XGBoost', {}).get('clean_accuracy', 0) * 100
        lgb = data.get('LightGBM', {}).get('clean_accuracy', 0) * 100
        print(f"{dataset:<20} {tabpfn:<12.1f}% {xgb:<12.1f}% {lgb:<12.1f}%")
    except Exception as e:
        print(f"{dataset:<20} Error: {e}")

# Comparison Table
print("\n" + "="*80)
print("BOUNDARY vs NES - CLEAN ACCURACY MATCH CHECK")
print("="*80)
print(f"\n{'Dataset':<20} {'Boundary TabPFN':<18} {'NES TabPFN':<15} {'Match?':<10}")
print("-"*70)

for dataset in boundary_files.keys():
    try:
        with open(f'results/{boundary_files[dataset]}', 'r') as f:
            b_data = json.load(f)
        with open(f'results/{nes_files[dataset]}', 'r') as f:
            n_data = json.load(f)
        
        b_acc = b_data.get('TabPFN', {}).get('clean_accuracy', 0) * 100
        n_acc = n_data.get('TabPFN', {}).get('clean_accuracy', 0) * 100
        match = "✅" if abs(b_acc - n_acc) < 1 else "❌"
        print(f"{dataset:<20} {b_acc:<18.1f}% {n_acc:<15.1f}% {match:<10}")
    except Exception as e:
        print(f"{dataset:<20} Error: {e}")

print("\n" + "="*80)
