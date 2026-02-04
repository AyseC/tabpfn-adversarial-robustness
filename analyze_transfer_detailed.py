"""Detailed transfer analysis by dataset characteristics"""
import json
import numpy as np

print("="*80)
print("TRANSFER ASYMMETRY - DETAILED ANALYSIS")
print("="*80)

# Dataset characteristics
dataset_info = {
    'wine': {'features': 13, 'samples': 130, 'complexity': 'medium'},
    'iris': {'features': 4, 'samples': 100, 'complexity': 'low'},
    'diabetes': {'features': 8, 'samples': 768, 'complexity': 'high'},
    'heart': {'features': 13, 'samples': 297, 'complexity': 'medium'},
    'breast_cancer': {'features': 30, 'samples': 569, 'complexity': 'high'}
}

print("\n1. TRANSFER RATES BY DATASET")
print("-"*60)

results = []
for dataset in ['wine', 'iris', 'diabetes', 'heart', 'breast_cancer']:
    try:
        with open(f'results/transfer_attack_{dataset}.json', 'r') as f:
            data = json.load(f)
        
        # Calculate averages
        xgb_to_tabpfn = data.get('XGBoost', {}).get('TabPFN', {}).get('transfer_rate', 0)
        lgb_to_tabpfn = data.get('LightGBM', {}).get('TabPFN', {}).get('transfer_rate', 0)
        tabpfn_to_xgb = data.get('TabPFN', {}).get('XGBoost', {}).get('transfer_rate', 0)
        tabpfn_to_lgb = data.get('TabPFN', {}).get('LightGBM', {}).get('transfer_rate', 0)
        
        gbdt_to_tabpfn = (xgb_to_tabpfn + lgb_to_tabpfn) / 2
        tabpfn_to_gbdt = (tabpfn_to_xgb + tabpfn_to_lgb) / 2
        
        info = dataset_info[dataset]
        direction = "TabPFN→GBDT" if tabpfn_to_gbdt > gbdt_to_tabpfn else "GBDT→TabPFN"
        
        results.append({
            'dataset': dataset,
            'features': info['features'],
            'gbdt_to_tabpfn': gbdt_to_tabpfn,
            'tabpfn_to_gbdt': tabpfn_to_gbdt,
            'direction': direction
        })
        
        print(f"\n{dataset.upper()} ({info['features']} features):")
        print(f"  GBDT → TabPFN: {gbdt_to_tabpfn:.1%}")
        print(f"  TabPFN → GBDT: {tabpfn_to_gbdt:.1%}")
        print(f"  Dominant: {direction}")
    except Exception as e:
        print(f"  Error: {e}")

# Analyze by feature count
print("\n" + "="*80)
print("2. PATTERN ANALYSIS")
print("="*80)

low_feat = [r for r in results if r['features'] <= 10]
high_feat = [r for r in results if r['features'] > 10]

print("\nLow feature datasets (≤10 features):")
for r in low_feat:
    print(f"  {r['dataset']}: {r['direction']}")

print("\nHigh feature datasets (>10 features):")
for r in high_feat:
    print(f"  {r['dataset']}: {r['direction']}")

# Hypothesis
print("\n" + "="*80)
print("3. HYPOTHESIS")
print("="*80)
print("""
OBSERVATION:
- Low feature datasets (Iris, Diabetes): TabPFN → GBDT higher
- High feature datasets (Heart, Breast Cancer): GBDT → TabPFN higher
- Wine (13 features): No transfer at all

POSSIBLE EXPLANATION:
- TabPFN learns smoother decision boundaries on low-dim data
- These smooth perturbations transfer well to GBDT's axis-aligned splits
- On high-dim data, GBDT's perturbations exploit TabPFN's complexity

THESIS IMPLICATION:
- Report as "dataset-dependent asymmetry" not "universal asymmetry"
- Feature dimensionality may be a key factor
- This is actually a MORE INTERESTING finding than simple asymmetry!
""")
