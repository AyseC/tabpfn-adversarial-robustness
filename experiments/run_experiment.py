"""Complete adversarial robustness experiment"""
import numpy as np
import json
from pathlib import Path
from sklearn.datasets import load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack
from src.evaluation.metrics import RobustnessMetrics, AttackResult


def run_experiment(dataset_name='wine', n_samples=20, max_iterations=200):
    """Run complete experiment"""
    
    print("="*70)
    print(f"ADVERSARIAL ROBUSTNESS EXPERIMENT: {dataset_name.upper()}")
    print("="*70)
    
    # Load dataset
    if dataset_name == 'wine':
        data = load_wine()
        X, y = data.data, data.target
        mask = y < 2  # Binary classification
        X, y = X[mask], y[mask]
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
        X, y = data.data, data.target
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    print(f"\nDataset: {dataset_name}")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Split
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
    
    for model_name, model in models.items():
        print(f"\n{'-'*70}")
        print(f"Model: {model_name}")
        print(f"{'-'*70}")
        
        # Train
        print("Training...")
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        clean_acc = np.mean(y_pred == y_test)
        print(f"Clean Accuracy: {clean_acc:.4f}")
        
        # Attack
        print(f"\nAttacking {n_samples} samples...")
        attack = BoundaryAttack(
            model, 
            max_iterations=max_iterations,
            epsilon=0.5,
            verbose=False
        )
        
        results = []
        successful = 0
        
        for i in range(min(n_samples, len(X_test))):
            x_orig = X_test[i]
            y_true = y_test[i]
            y_pred_i = model.predict(x_orig.reshape(1, -1))[0]
            
            if y_pred_i != y_true:
                continue  # Skip misclassified
            
            # Attack
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
                print(f"  [{i+1}/{n_samples}] SUCCESS: {y_true}→{y_adv}, "
                      f"pert={pert:.3f}, queries={queries}")
            else:
                print(f"  [{i+1}/{n_samples}] Failed")
        
        # Compute metrics
        metrics = RobustnessMetrics.compute_all(results, y_test[:n_samples], y_pred[:n_samples])
        
        print(f"\n{model_name} Results:")
        print(f"  Attack Success Rate: {metrics['attack_success_rate']:.2%}")
        print(f"  Avg Perturbation: {metrics['avg_perturbation']:.4f}")
        print(f"  Avg Queries: {metrics['avg_queries']:.0f}")
        print(f"  Robustness Score: {metrics['robustness_score']:.4f}")
        
        all_results[model_name] = metrics
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Model':<12} {'Clean Acc':<12} {'ASR':<12} {'Avg Pert':<12} {'Robust Score':<12}")
    print("-"*70)
    
    for model_name, metrics in all_results.items():
        print(f"{model_name:<12} "
              f"{metrics['clean_accuracy']:<12.4f} "
              f"{metrics['attack_success_rate']:<12.4f} "
              f"{metrics['avg_perturbation']:<12.4f} "
              f"{metrics['robustness_score']:<12.4f}")
    
    # Best model
    best_robust = max(all_results.items(), key=lambda x: x[1]['robustness_score'])
    print(f"\n✓ Most Robust Model: {best_robust[0]} (score: {best_robust[1]['robustness_score']:.4f})")
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / f"experiment_{dataset_name}.json"
    
    # Convert to serializable format
    save_data = {}
    for model_name, metrics in all_results.items():
        save_data[model_name] = {k: float(v) for k, v in metrics.items()}
    
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    import sys
    
    dataset = sys.argv[1] if len(sys.argv) > 1 else 'wine'
    n_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    
    run_experiment(dataset_name=dataset, n_samples=n_samples)
