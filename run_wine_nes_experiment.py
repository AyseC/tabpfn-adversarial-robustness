"""Wine dataset with NES Attack"""
import numpy as np
import json
import warnings
from pathlib import Path
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.nes_attack import NESAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult

print("="*70)
print("NES ATTACK: WINE DATASET")
print("="*70)

# Load
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Models
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
    
    print(f"\nNES Attack on {n_samples} samples...")
    attack = NESAttack(
        model,
        max_iterations=50,
        n_samples=30,
        learning_rate=0.3,
        sigma=0.3,
        max_queries=2000,
        verbose=False
    )
    
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
    print(f"  Avg Queries: {metrics['avg_queries']:.0f}")
    print(f"  Robustness: {metrics['robustness_score']:.4f}")
    
    all_results[model_name] = metrics

# Summary
print("\n" + "="*70)
print("NES ATTACK RESULTS")
print("="*70)
print(f"\n{'Model':<12} {'ASR':<10} {'Avg Pert':<12} {'Avg Queries':<12} {'Robustness':<12}")
print("-"*70)

for model_name, m in all_results.items():
    print(f"{model_name:<12} {m['attack_success_rate']:<10.2%} "
          f"{m['avg_perturbation']:<12.4f} {m['avg_queries']:<12.0f} "
          f"{m['robustness_score']:<12.4f}")

# Save
Path("results").mkdir(exist_ok=True)
with open("results/wine_nes_experiment.json", 'w') as f:
    save_data = {k: {kk: float(vv) for kk, vv in v.items()} 
                 for k, v in all_results.items()}
    json.dump(save_data, f, indent=2)

print("\n✓ Saved: results/wine_nes_experiment.json")
print("="*70)
