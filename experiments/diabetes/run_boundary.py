"""Diabetes Dataset Experiment - Boundary Attack"""
import numpy as np
import torch
import json
import warnings
from pathlib import Path
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

def get_common_attack_indices(y_test, all_preds, n_samples, random_state=42):
    """Select n_samples from samples correctly classified by ALL models, stratified by class."""
    correct_sets = [set(np.where(preds == y_test)[0]) for preds in all_preds.values()]
    common_correct = np.array(sorted(set.intersection(*correct_sets)))
    if len(common_correct) <= n_samples:
        return common_correct
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=n_samples, random_state=random_state)
        _, sel = next(sss.split(common_correct.reshape(-1, 1), y_test[common_correct]))
        return common_correct[sel]
    except ValueError:
        rng = np.random.RandomState(random_state)
        sel = rng.choice(len(common_correct), n_samples, replace=False)
        return common_correct[sel]


print("="*70)
print("ADVERSARIAL ROBUSTNESS: DIABETES DATASET")
print("="*70)

# Load Diabetes dataset - SAME as NES experiment
print("\nLoading Diabetes dataset from OpenML...")
try:
    diabetes = fetch_openml('diabetes', version=1, as_frame=False, parser='auto')
    X, y = diabetes.data, diabetes.target
    
    # Convert target to int
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(int)
    
    if len(np.unique(y)) > 2:
        y = (y > 0).astype(int)
    
    print("  ✓ Loaded from OpenML (Pima Indians Diabetes)")
except Exception as e:
    print(f"  OpenML failed: {e}")
    print("  Using sklearn diabetes dataset with binary conversion...")
    from sklearn.datasets import load_diabetes as load_diabetes_reg
    diabetes_reg = load_diabetes_reg()
    X = diabetes_reg.data
    y = (diabetes_reg.target > np.median(diabetes_reg.target)).astype(int)

print(f"\nDataset: Diabetes")
print(f"  Samples: {len(X)}")
print(f"  Features: {X.shape[1]}")
print(f"  Class 0: {sum(y==0)}, Class 1: {sum(y==1)}")

# Standardize
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

n_samples = 15

# Models
models = {
    'XGBoost': GBDTWrapper(model_type='xgboost'),
    'LightGBM': GBDTWrapper(model_type='lightgbm'),
    'TabPFN': TabPFNWrapper(device='cpu')
}

all_results = {}

# Pre-train all models and collect predictions for common indices
print("\nPre-training all models...")
all_preds = {}
for _name, _model in models.items():
    _model.fit(X_train, y_train)
    all_preds[_name] = _model.predict(X_test)
    print(f"  {_name} trained")

attack_indices = get_common_attack_indices(y_test, all_preds, n_samples)

for model_name, model in models.items():
    print(f"\n{'-'*70}")
    print(f"Model: {model_name}")
    print(f"{'-'*70}")
    
    y_pred = all_preds[model_name]
    clean_acc = np.mean(y_pred == y_test)
    print(f"Clean Accuracy: {clean_acc:.4f}")
    
    print(f"\nAttacking {n_samples} samples...")
    attack = BoundaryAttack(model, max_iterations=200, epsilon=0.5, verbose=False)
    
    results = []
    for i in attack_indices:
        x_orig = X_test[i]
        y_true = y_test[i]
        y_pred_i = model.predict(x_orig.reshape(1, -1))[0]
        
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
            print(f"  [{i+1}] ✓ {y_true}→{y_adv}, pert={pert:.2f}, q={queries}")
    
    print(f"\nSuccessful attacks: {sum(1 for r in results if r.success)}/{len(results)}")
    
    metrics = RobustnessMetrics.compute_all(results, y_test[attack_indices], y_pred[attack_indices])
    
    all_results[model_name] = metrics
    all_results[model_name]["clean_accuracy"] = clean_acc

    print(f"\n{model_name} Metrics:")
    print(f"  ASR: {metrics['attack_success_rate']:.2%}")
    print(f"  Adversarial Accuracy: {metrics['adversarial_accuracy']:.2%}")

# Save results
Path("results").mkdir(exist_ok=True)
with open('results/diabetes_experiment.json', 'w') as f:
    save_data = {k: {kk: float(vv) for kk, vv in v.items()}
                 for k, v in all_results.items()}
    json.dump(save_data, f, indent=2)

print(f"\n{'='*70}")
print("RESULTS - DIABETES DATASET")
print(f"{'='*70}")
print(f"\n{'Model':<12} {'ASR':<10} {'Adv Acc':<12} {'Robustness':<12}")
print("-"*50)
for model_name, res in all_results.items():
    print(f"{model_name:<12} {res['attack_success_rate']:<10.2%} "
          f"{res['adversarial_accuracy']:<12.2%} {res['robustness_score']:<12.4f}")

print(f"\n✓ Saved: results/diabetes_experiment.json")
