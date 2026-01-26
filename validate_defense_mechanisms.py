"""Statistical Validation of Defense Mechanisms"""
import numpy as np
from scipy import stats
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from src.models.tabpfn_wrapper import TabPFNWrapper
from src.attacks.boundary_attack import BoundaryAttack
import matplotlib.pyplot as plt

print("="*70)
print("DEFENSE MECHANISMS - STATISTICAL VALIDATION")
print("Testing with larger sample size + significance tests")
print("="*70)

# Load data
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
tabpfn = TabPFNWrapper(device='cpu')
tabpfn.fit(X_train, y_train)

def add_gaussian_noise(X, std=0.1):
    return X + np.random.randn(*X.shape) * std

# TEST WITH MORE SAMPLES
print("\nTesting Gaussian Noise defense on 20 samples...")
print("(This will take ~5 minutes)")

attack = BoundaryAttack(tabpfn, max_iterations=100, epsilon=0.5, verbose=False)

n_samples = 20
no_defense_results = []
with_defense_results = []

for i in range(n_samples):
    if i >= len(X_test):
        break
        
    x_orig = X_test[i]
    y_true = y_test[i]
    
    if tabpfn.predict(x_orig.reshape(1, -1))[0] != y_true:
        continue
    
    # Attack
    x_adv, success, _, _ = attack.attack(x_orig, y_true)
    
    # Without defense
    pred_no_def = tabpfn.predict(x_adv.reshape(1, -1))[0]
    no_defense_results.append(1 if pred_no_def != y_true else 0)  # 1 = attack success
    
    # With defense
    if success:
        x_adv_noisy = add_gaussian_noise(x_adv, std=0.05)
        pred_def = tabpfn.predict(x_adv_noisy.reshape(1, -1))[0]
        with_defense_results.append(1 if pred_def == y_true else 0)  # 1 = defense success
    
    if (i+1) % 5 == 0:
        print(f"  Tested {i+1}/{n_samples} samples...")

# Calculate statistics
no_defense_asr = np.mean(no_defense_results) * 100
defense_recovery = np.mean(with_defense_results) * 100 if with_defense_results else 0

print(f"\n{'='*70}")
print("RESULTS WITH LARGER SAMPLE SIZE")
print(f"{'='*70}")
print(f"Samples tested: {len(no_defense_results)}")
print(f"Without Defense ASR: {no_defense_asr:.1f}%")
print(f"Defense Recovery Rate: {defense_recovery:.1f}%")

# STATISTICAL SIGNIFICANCE TEST
print(f"\n{'='*70}")
print("STATISTICAL SIGNIFICANCE TEST")
print(f"{'='*70}")

# McNemar's test (paired test for binary outcomes)
if len(with_defense_results) > 5:
    # Create contingency table
    both_correct = sum(1 for i in range(len(with_defense_results)) 
                      if with_defense_results[i] == 1 and no_defense_results[i] == 0)
    both_wrong = sum(1 for i in range(len(with_defense_results)) 
                    if with_defense_results[i] == 0 and no_defense_results[i] == 1)
    
    # Simplified McNemar test
    if both_correct + both_wrong > 0:
        statistic = (abs(both_correct - both_wrong) - 1)**2 / (both_correct + both_wrong)
        p_value = 1 - stats.chi2.cdf(statistic, 1)
        
        print(f"McNemar's Test:")
        print(f"  Defense helps in: {both_correct} cases")
        print(f"  Defense hurts in: {both_wrong} cases")
        print(f"  Chi-square statistic: {statistic:.4f}")
        print(f"  p-value: {p_value:.4f}")
        print(f"  Result: {'SIGNIFICANT' if p_value < 0.05 else 'NOT SIGNIFICANT'} (α=0.05)")
        
        if p_value < 0.05:
            print(f"\n✓ Defense effectiveness is STATISTICALLY SIGNIFICANT!")
        else:
            print(f"\n⚠ Defense effectiveness is NOT statistically significant")

# CONFIDENCE INTERVAL
print(f"\n{'='*70}")
print("95% CONFIDENCE INTERVAL")
print(f"{'='*70}")

if len(with_defense_results) > 0:
    mean = defense_recovery / 100
    n = len(with_defense_results)
    se = np.sqrt(mean * (1 - mean) / n)
    ci_low = max(0, mean - 1.96 * se) * 100
    ci_high = min(1, mean + 1.96 * se) * 100
    
    print(f"Defense Recovery Rate: {defense_recovery:.1f}%")
    print(f"95% CI: [{ci_low:.1f}%, {ci_high:.1f}%]")
    
    if ci_low > 50:
        print(f"\n✓ We can be 95% confident that defense works >50% of the time!")

# VISUALIZATION
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Success rates
ax1.bar(['Without\nDefense', 'With\nDefense'], 
        [100-no_defense_asr, defense_recovery],
        color=['#e74c3c', '#2ecc71'], alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('Correct Predictions (%)', fontweight='bold', fontsize=12)
ax1.set_title('Defense Effectiveness\n(Higher = Better)', fontweight='bold', fontsize=14)
ax1.set_ylim(0, 100)
ax1.grid(axis='y', alpha=0.3)

for i, v in enumerate([100-no_defense_asr, defense_recovery]):
    ax1.text(i, v+3, f'{v:.0f}%', ha='center', fontweight='bold', fontsize=12)

# Plot 2: Sample distribution
ax2.hist([no_defense_results, with_defense_results], 
         bins=2, label=['No Defense (fail=1)', 'With Defense (success=1)'],
         color=['#e74c3c', '#2ecc71'], alpha=0.6, edgecolor='black')
ax2.set_xlabel('Outcome', fontweight='bold', fontsize=12)
ax2.set_ylabel('Count', fontweight='bold', fontsize=12)
ax2.set_title('Distribution of Results', fontweight='bold', fontsize=14)
ax2.legend()
ax2.grid(alpha=0.3)

plt.suptitle(f'Statistical Validation (n={len(no_defense_results)} samples)',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/defense_validation.png', dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: results/defense_validation.png")
plt.show()

print(f"\n{'='*70}")
print("CONCLUSION")
print(f"{'='*70}")
print(f"""
Based on {len(no_defense_results)} test samples:
  • Defense recovery rate: {defense_recovery:.1f}%
  • 95% CI: [{ci_low:.1f}%, {ci_high:.1f}%]
  • Statistical significance: {'YES (p < 0.05)' if p_value < 0.05 else 'NEEDS MORE SAMPLES'}

RECOMMENDATION:
  {'✓ Gaussian noise defense is VALIDATED and effective!' if defense_recovery > 70 and p_value < 0.05 else '⚠ Need more samples for conclusive validation'}
""")

print(f"{'='*70}")
