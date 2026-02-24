"""Heart Disease Dataset Experiment"""
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

print("="*70)
print("ADVERSARIAL ROBUSTNESS: HEART DISEASE DATASET")
print("="*70)

from sklearn.datasets import fetch_openml

print("\nLoading Heart Disease dataset from OpenML...")
try:
    heart = fetch_openml('heart-statlog', version=1, as_frame=False, parser='auto')
    X, y = heart.data, heart.target.astype(int)
    
    if len(np.unique(y)) > 2:
        y = (y > 0).astype(int)
    
    print(f"  ✓ Loaded from OpenML")
    
except Exception as e:
    print(f"  OpenML failed: {e}")
    print("  Loading from UCI repository...")
    
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    
    try:
        df = pd.read_csv(url, names=columns, na_values='?')
        df = df.dropna()
        
        X = df.drop('target', axis=1).values
        y = df['target'].values
        y = (y > 0).astype(int)
        
        print(f"  ✓ Loaded from UCI")
    
    except Exception:
        print("  ✗ All methods failed. Using synthetic data as fallback...")
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=300, n_features=13, n_informative=10,
                                   n_redundant=3, random_state=42)

print(f"\nDataset: Heart Disease")
print(f"  Samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

models = {
    'XGBoost': GBDTWrapper(model_type='xgboost'),
    'LightGBM': GBDTWrapper(model_type='lightgbm'),
    'TabPFN': TabPFNWrapper(device='cpu')
}

all_results = {}
n_samples = 15

for model_name, model in models.items():
    print(f"\n{'-'*70}")
    print(f"Model: {model_name}")
    print(f"{'-'*70}")
    
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    clean_acc = np.mean(y_pred == y_test)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    
    print(f"\nAttacking {n_samples} samples...")
    attack = BoundaryAttack(model, max_iterations=200, epsilon=0.5, verbose=False)
    
    results = []
    successful = 0
    
    for i in range(min(n_samples, len(X_test))):
        x_orig = X_test[i]
        y_true = y_test[i]
        y_pred_i = model.predict(x_orig.reshape(1, -1))[0]
        
        if y_pred_i != y_true:
            continue
        
        x_adv, success, queries, pert = attack.attack(x_orig, y_true)
        y_adv = model.predict(x_adv.reshape(1, -1))[0]
        
        result = AttackResult(
            original_label=y_true,
            predicted_label=y_pred_i,
            adversarial_label=y_adv,
            success=success,
            perturbation=pert,
            queries=queries,
            original_sample=x_orig,
            adversarial_sample=x_adv
        )
        results.append(result)
        
        if success:
            successful += 1
            print(f"  [{i+1}] ✓ {y_true}→{y_adv}, pert={pert:.2f}, q={queries}")
    
    print(f"\nSuccessful: {successful}/{len(results)}")
    
    metrics = RobustnessMetrics.compute_all(results, y_test[:n_samples], y_pred[:n_samples])
    
    print(f"\n{model_name} Metrics:")
    print(f"  ASR: {metrics['attack_success_rate']:.2%}")
    print(f"  Avg Pert: {metrics['avg_perturbation']:.4f}")
    print(f"  Robustness: {metrics['robustness_score']:.4f}")
    
    all_results[model_name] = metrics
    all_results[model_name]["clean_accuracy"] = clean_acc

print("\n" + "="*70)
print("RESULTS - HEART DISEASE DATASET")
print("="*70)
print(f"\n{'Model':<12} {'ASR':<10} {'Avg Pert':<12} {'Robustness':<12}")
print("-"*70)

for model_name, m in all_results.items():
    print(f"{model_name:<12} {m['attack_success_rate']:<10.2%} "
          f"{m['avg_perturbation']:<12.4f} {m['robustness_score']:<12.4f}")

best = max(all_results.items(), key=lambda x: x[1]['robustness_score'])
worst = min(all_results.items(), key=lambda x: x[1]['robustness_score'])

print(f"\n✓ Most Robust: {best[0]}")
print(f"✗ Least Robust: {worst[0]}")

if 'TabPFN' in all_results:
    gbdt_best = max([('XGBoost', all_results['XGBoost']), 
                     ('LightGBM', all_results['LightGBM'])],
                    key=lambda x: x[1]['robustness_score'])

    gbdt_asr = gbdt_best[1]['attack_success_rate']
    if gbdt_asr > 0:
        ratio = all_results['TabPFN']['attack_success_rate'] / gbdt_asr
        print(f"\n TabPFN is {ratio:.2f}x vulnerability vs {gbdt_best[0]}")
    else:
        print(f"\n TabPFN vulnerability ratio vs {gbdt_best[0]}: undefined (GBDT ASR=0)")

Path("results").mkdir(exist_ok=True)
with open("results/heart_experiment.json", 'w') as f:
    save_data = {k: {kk: float(vv) for kk, vv in v.items()} 
                 for k, v in all_results.items()}
    json.dump(save_data, f, indent=2)

print("\n✓ Saved: results/heart_experiment.json")
print("="*70)
