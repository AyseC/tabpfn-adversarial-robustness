"""Transfer Attack Experiment - Breast Cancer Dataset"""
import numpy as np
import torch
import json
import warnings
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# Reproducibility
np.random.seed(42)
torch.manual_seed(42)

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack

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


print("="*80)
print("TRANSFER ATTACK EXPERIMENT - BREAST CANCER DATASET")
print("="*80)

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

print(f"\nDataset: Breast Cancer")
print(f"  Samples: {len(X)}, Features: {X.shape[1]}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
print("\n[1/4] Training models...")
models = {
    'TabPFN': TabPFNWrapper(device='cpu'),
    'XGBoost': GBDTWrapper(model_type='xgboost'),
    'LightGBM': GBDTWrapper(model_type='lightgbm')
}

for name, model in models.items():
    model.fit(X_train, y_train)
    acc = np.mean(model.predict(X_test) == y_test)
    print(f"  ✓ {name}: {acc:.2%}")

# Generate adversarial examples
n_samples = 15
transfer_results = {}

# Pre-compute common attack indices (samples correct for ALL models)
all_preds_for_common = {name: model.predict(X_test) for name, model in models.items()}
attack_indices = get_common_attack_indices(y_test, all_preds_for_common, n_samples)

print(f"\n[2/4] Generating adversarial examples...")

for source_name, source_model in models.items():
    print(f"\n  Source: {source_name}")
    attack = BoundaryAttack(source_model, max_iterations=200, epsilon=0.5, verbose=False)
    
    adversarial_examples = []
    for i in attack_indices:
        x_orig = X_test[i]
        y_true = y_test[i]
        
        x_adv, success, queries, pert = attack.attack(x_orig, y_true)
        if success:
            adversarial_examples.append({
                'x_orig': x_orig,
                'x_adv': x_adv,
                'y_true': y_true,
                'perturbation': pert
            })
    
    print(f"    Generated {len(adversarial_examples)} adversarial examples")
    
    # Test transfer
    transfer_results[source_name] = {}
    for target_name, target_model in models.items():
        if target_name == source_name:
            success_rate = 1.0
        else:
            successes = 0
            for ex in adversarial_examples:
                y_pred_target = target_model.predict(ex['x_adv'].reshape(1, -1))[0]
                if y_pred_target != ex['y_true']:
                    successes += 1
            success_rate = successes / len(adversarial_examples) if adversarial_examples else 0
        
        transfer_results[source_name][target_name] = {
            'transfer_rate': success_rate,
            'n_samples': len(adversarial_examples)
        }
        print(f"    → {target_name}: {success_rate:.2%}")

# Summary
print(f"\n{'='*80}")
print("TRANSFER ATTACK RESULTS - BREAST CANCER")
print(f"{'='*80}")

print(f"\n{'Source→Target':<20}", end="")
for target in models.keys():
    print(f"{target:<12}", end="")
print()
print("-"*60)

for source in models.keys():
    print(f"{source:<20}", end="")
    for target in models.keys():
        rate = transfer_results[source][target]['transfer_rate']
        print(f"{rate:<12.2%}", end="")
    print()

# Save
Path("results").mkdir(exist_ok=True)
with open('results/transfer_attack_breast_cancer.json', 'w') as f:
    json.dump(transfer_results, f, indent=2)
print(f"\n✓ Saved: results/transfer_attack_breast_cancer.json")
