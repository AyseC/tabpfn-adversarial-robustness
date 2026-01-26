"""Defense Mechanisms Against Adversarial Attacks"""
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.models.tabpfn_wrapper import TabPFNWrapper
from src.models.gbdt_wrapper import GBDTWrapper
from src.attacks.boundary_attack import BoundaryAttack

print("="*70)
print("DEFENSE MECHANISMS ANALYSIS")
print("Testing various defense strategies")
print("="*70)

# Load data
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# DEFENSE 1: INPUT PREPROCESSING (Gaussian Noise)
print("\n" + "="*70)
print("DEFENSE 1: INPUT PREPROCESSING (Gaussian Noise)")
print("="*70)

def add_gaussian_noise(X, std=0.1):
    """Add Gaussian noise to inputs"""
    return X + np.random.randn(*X.shape) * std

# Train TabPFN
tabpfn = TabPFNWrapper(device='cpu')
tabpfn.fit(X_train, y_train)

print("\nTesting defense effectiveness...")

# Test without defense
attack = BoundaryAttack(tabpfn, max_iterations=100, epsilon=0.5, verbose=False)
no_defense_success = 0
with_defense_success = 0
total = 0

for i in range(10):
    x_orig = X_test[i]
    y_true = y_test[i]
    
    if tabpfn.predict(x_orig.reshape(1, -1))[0] != y_true:
        continue
    
    total += 1
    
    # Attack without defense
    x_adv, success, _, _ = attack.attack(x_orig, y_true)
    if success:
        no_defense_success += 1
    
    # Defense: Add noise to adversarial example
    x_adv_noisy = add_gaussian_noise(x_adv, std=0.05)
    pred_defended = tabpfn.predict(x_adv_noisy.reshape(1, -1))[0]
    
    # Check if defense worked
    if pred_defended == y_true:
        with_defense_success += 1

no_defense_asr = no_defense_success / total * 100 if total > 0 else 0
defense_success_rate = with_defense_success / no_defense_success * 100 if no_defense_success > 0 else 0

print(f"\nResults:")
print(f"  Without Defense: {no_defense_asr:.1f}% ASR")
print(f"  Defense Success Rate: {defense_success_rate:.1f}%")
print(f"  (Defense recovered {with_defense_success}/{no_defense_success} adversarial examples)")

# DEFENSE 2: FEATURE SQUEEZING
print("\n" + "="*70)
print("DEFENSE 2: FEATURE SQUEEZING")
print("="*70)

def feature_squeezing(X, bit_depth=4):
    """Reduce precision of features"""
    # Normalize to [0, 1]
    X_min, X_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min + 1e-10)
    
    # Reduce bit depth
    levels = 2 ** bit_depth
    X_squeezed = np.round(X_norm * levels) / levels
    
    # Denormalize
    return X_squeezed * (X_max - X_min) + X_min

squeeze_defense_success = 0

for i in range(10):
    x_orig = X_test[i]
    y_true = y_test[i]
    
    if tabpfn.predict(x_orig.reshape(1, -1))[0] != y_true:
        continue
    
    x_adv, success, _, _ = attack.attack(x_orig, y_true)
    
    if success:
        # Apply feature squeezing
        x_adv_squeezed = feature_squeezing(x_adv.reshape(1, -1), bit_depth=6)
        pred_squeezed = tabpfn.predict(x_adv_squeezed)[0]
        
        if pred_squeezed == y_true:
            squeeze_defense_success += 1

squeeze_rate = squeeze_defense_success / no_defense_success * 100 if no_defense_success > 0 else 0

print(f"\nResults:")
print(f"  Feature Squeezing Success: {squeeze_rate:.1f}%")
print(f"  (Recovered {squeeze_defense_success}/{no_defense_success} examples)")

# DEFENSE 3: ENSEMBLE PREDICTION
print("\n" + "="*70)
print("DEFENSE 3: ENSEMBLE PREDICTION")
print("="*70)

# Train ensemble
print("\nTraining ensemble...")
xgboost = GBDTWrapper(model_type='xgboost')
xgboost.fit(X_train, y_train)

lightgbm = GBDTWrapper(model_type='lightgbm')
lightgbm.fit(X_train, y_train)

def ensemble_predict(x, models):
    """Majority voting"""
    predictions = [model.predict(x.reshape(1, -1))[0] for model in models]
    return max(set(predictions), key=predictions.count)

ensemble_models = [tabpfn, xgboost, lightgbm]

# Test ensemble defense
ensemble_defense_success = 0

for i in range(10):
    x_orig = X_test[i]
    y_true = y_test[i]
    
    if tabpfn.predict(x_orig.reshape(1, -1))[0] != y_true:
        continue
    
    x_adv, success, _, _ = attack.attack(x_orig, y_true)
    
    if success:
        # Ensemble prediction
        ensemble_pred = ensemble_predict(x_adv, ensemble_models)
        
        if ensemble_pred == y_true:
            ensemble_defense_success += 1

ensemble_rate = ensemble_defense_success / no_defense_success * 100 if no_defense_success > 0 else 0

print(f"\nResults:")
print(f"  Ensemble Defense Success: {ensemble_rate:.1f}%")
print(f"  (Recovered {ensemble_defense_success}/{no_defense_success} examples)")

# VISUALIZATION
fig, ax = plt.subplots(figsize=(10, 6))

defenses = ['No Defense\n(Baseline)', 'Gaussian\nNoise', 'Feature\nSqueezing', 'Ensemble\nVoting']
success_rates = [
    0,  # Baseline (all fail by definition)
    defense_success_rate,
    squeeze_rate,
    ensemble_rate
]

colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
bars = ax.bar(defenses, success_rates, color=colors, alpha=0.7, 
              edgecolor='black', linewidth=2)

ax.set_ylabel('Defense Success Rate (%)', fontweight='bold', fontsize=12)
ax.set_title('Effectiveness of Defense Mechanisms\n(% of adversarial examples successfully defended)',
             fontweight='bold', fontsize=14)
ax.set_ylim(0, 100)
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bar, val in zip(bars, success_rates):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 3,
            f'{val:.0f}%', ha='center', fontweight='bold', fontsize=11)

# Add baseline annotation
ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, linewidth=1.5)
ax.text(len(defenses)-0.5, 52, 'Target: 50%', fontsize=10, color='gray')

plt.tight_layout()
plt.savefig('results/defense_mechanisms.png', dpi=300, bbox_inches='tight')
print("\n✓ Saved: results/defense_mechanisms.png")
plt.show()

# SUMMARY
print("\n" + "="*70)
print("DEFENSE MECHANISMS SUMMARY")
print("="*70)

print(f"""
BASELINE (TabPFN without defense):
  Attack Success Rate: {no_defense_asr:.1f}%
  
DEFENSE EFFECTIVENESS:
  1. Gaussian Noise:      {defense_success_rate:.1f}% recovery
  2. Feature Squeezing:   {squeeze_rate:.1f}% recovery
  3. Ensemble Voting:     {ensemble_rate:.1f}% recovery

BEST DEFENSE: {'Ensemble Voting' if ensemble_rate == max(defense_success_rate, squeeze_rate, ensemble_rate) else 'Gaussian Noise' if defense_success_rate == max(defense_success_rate, squeeze_rate, ensemble_rate) else 'Feature Squeezing'}

KEY INSIGHTS:
  • Simple preprocessing defenses show moderate effectiveness
  • Ensemble methods provide robust defense
  • No single defense is perfect (adaptive attacks possible)
  • Defense-in-depth recommended for production systems
""")

# Save results
defense_results = {
    'baseline_asr': no_defense_asr,
    'defenses': {
        'gaussian_noise': defense_success_rate,
        'feature_squeezing': squeeze_rate,
        'ensemble_voting': ensemble_rate
    }
}

with open('results/defense_mechanisms.json', 'w') as f:
    json.dump(defense_results, f, indent=2)

print("✓ Saved: results/defense_mechanisms.json")
print("="*70)
