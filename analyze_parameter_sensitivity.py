"""Analyze parameter sensitivity results"""
import json
import numpy as np

print("="*80)
print("PARAMETER SENSITIVITY ANALYSIS - KEY FINDINGS")
print("="*80)

# Load results
with open('results/parameter_sensitivity.json', 'r') as f:
    results = json.load(f)

print("\n" + "="*80)
print("1. BOUNDARY ATTACK - max_iterations")
print("="*80)

print("\nASR by max_iterations:")
print(f"{'max_iter':<12} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12}")
print("-"*50)
for key in ['max_iter_50', 'max_iter_100', 'max_iter_200', 'max_iter_500']:
    data = results['boundary'][key]
    print(f"{key.split('_')[-1]:<12} {data['TabPFN']['asr']*100:<12.1f} {data['XGBoost']['asr']*100:<12.1f} {data['LightGBM']['asr']*100:<12.1f}")

print("\n→ FINDING: max_iterations has minimal effect on ASR")
print("→ XGBoost consistently 100% regardless of iterations")

print("\n" + "="*80)
print("2. BOUNDARY ATTACK - epsilon")
print("="*80)

print("\nASR by epsilon:")
print(f"{'epsilon':<12} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12}")
print("-"*50)
for key in ['epsilon_0.1', 'epsilon_0.3', 'epsilon_0.5', 'epsilon_1.0']:
    data = results['boundary'][key]
    print(f"{key.split('_')[-1]:<12} {data['TabPFN']['asr']*100:<12.1f} {data['XGBoost']['asr']*100:<12.1f} {data['LightGBM']['asr']*100:<12.1f}")

print("\n→ FINDING: epsilon has minimal effect on ASR")
print("→ Boundary attack is ROBUST to hyperparameter changes")

print("\n" + "="*80)
print("3. NES ATTACK - learning_rate (CRITICAL!)")
print("="*80)

print("\nASR by learning_rate:")
print(f"{'lr':<12} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12}")
print("-"*50)
for key in ['lr_0.1', 'lr_0.3', 'lr_0.5', 'lr_1.0']:
    data = results['nes'][key]
    print(f"{key.split('_')[-1]:<12} {data['TabPFN']['asr']*100:<12.1f} {data['XGBoost']['asr']*100:<12.1f} {data['LightGBM']['asr']*100:<12.1f}")

print("\n→ FINDING: Higher learning_rate → Higher ASR")
print("→ XGBoost: 0% at lr=0.1-0.3, 44% at lr=1.0 (huge jump!)")
print("→ Default lr=0.3 may be suboptimal!")

print("\n" + "="*80)
print("4. NES ATTACK - sigma (MOST CRITICAL!)")
print("="*80)

print("\nASR by sigma:")
print(f"{'sigma':<12} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12}")
print("-"*50)
for key in ['sigma_0.1', 'sigma_0.3', 'sigma_0.5', 'sigma_1.0']:
    data = results['nes'][key]
    print(f"{key.split('_')[-1]:<12} {data['TabPFN']['asr']*100:<12.1f} {data['XGBoost']['asr']*100:<12.1f} {data['LightGBM']['asr']*100:<12.1f}")

print("\n→ FINDING: sigma is CRITICAL for NES!")
print("→ sigma=0.1-0.3: XGBoost 0% ASR")
print("→ sigma=0.5: XGBoost 100% ASR")
print("→ sigma=1.0: All models 90-100% ASR")
print("→ Default sigma=0.3 was TOO LOW!")

print("\n" + "="*80)
print("5. NES ATTACK - n_samples")
print("="*80)

print("\nASR by n_samples:")
print(f"{'n_samples':<12} {'TabPFN':<12} {'XGBoost':<12} {'LightGBM':<12}")
print("-"*50)
for key in ['n_samples_10', 'n_samples_30', 'n_samples_50', 'n_samples_100']:
    data = results['nes'][key]
    print(f"{key.split('_')[-1]:<12} {data['TabPFN']['asr']*100:<12.1f} {data['XGBoost']['asr']*100:<12.1f} {data['LightGBM']['asr']*100:<12.1f}")

print("\n→ FINDING: n_samples has minimal effect")
print("→ Population size doesn't improve gradient estimation much")

print("\n" + "="*80)
print("THESIS IMPLICATIONS")
print("="*80)
print("""
KEY DISCOVERY:
- NES default parameters (lr=0.3, sigma=0.3) were SUBOPTIMAL
- With sigma=1.0, NES achieves 90-100% ASR (comparable to Boundary!)
- This partially explains why NES seemed "weaker"

RECOMMENDATION:
- For tabular data, use NES with sigma >= 0.5
- Boundary attack is more robust to hyperparameter choices
- Future work should tune NES specifically for tabular domains

LIMITATION ACKNOWLEDGED:
- Our NES results may underestimate true NES effectiveness
- With optimal parameters, NES could match Boundary
""")

# Summary table
print("\n" + "="*80)
print("OPTIMAL PARAMETERS")
print("="*80)
print("""
BOUNDARY ATTACK:
  max_iterations: 200 (diminishing returns after)
  epsilon: 0.5 (default is fine)

NES ATTACK:
  learning_rate: 0.5-1.0 (higher than default 0.3)
  sigma: 0.5-1.0 (MUCH higher than default 0.3!)
  n_samples: 30 (default is fine)
""")
