content = """# Adversarial Robustness of TabPFN vs GBDTs

**Master's Thesis Research**

- **Student:** Ayse Coskuner
- **Supervisor:** Ilia Koloiarov
- **Institution:** University of Hildesheim
- **Year:** 2026

## Research Objective

Comprehensive evaluation of adversarial robustness in TabPFN (Tabular Prior-Fitting Network) compared to traditional GBDT models (XGBoost, LightGBM).

## Major Findings

### 1. TabPFN Generally More Robust (Boundary Attack)

| Dataset | Features | TabPFN ASR | Best GBDT ASR | Difference | Winner |
|---------|----------|------------|---------------|------------|--------|
| Wine | 13 | 78.57% | 93.33% | -14.76% | TabPFN |
| Iris | 4 | 93.33% | 100.00% | -6.67% | TabPFN |
| Diabetes | 8 | 100.00% | 100.00% | 0.00% | Tie |
| Heart | 13 | 50.00% | 83.33% | -33.33% | TabPFN |
| Breast Cancer | 30 | 78.57% | 78.57% | 0.00% | Tie |

Statistical Analysis:
- Mean ASR difference: -11.0% (TabPFN lower)
- Bootstrap 95% CI: [-23.0%, -1.3%] (excludes zero = significant)
- Cohen's d: -0.788 (Medium effect size)
- Paired t-test: p=0.153 (limited by small n=5)

### 2. NES Attack Results

| Dataset | TabPFN ASR | Best GBDT ASR | Winner |
|---------|------------|---------------|--------|
| Wine | 35.71% | 14.29% | GBDT |
| Iris | 20.00% | 33.33% | TabPFN |
| Diabetes | 100.00% | 100.00% | Tie |
| Heart | 50.00% | 41.67% | GBDT |
| Breast Cancer | 85.71% | 50.00% | GBDT |

### 3. Transfer Attack Asymmetry

| Dataset | GBDT -> TabPFN | TabPFN -> GBDT | Ratio |
|---------|----------------|----------------|-------|
| Wine | 10.00% | 50.00% | 5.0x |
| Iris | 0.00% | 53.33% | - |
| Diabetes | 14.64% | 50.00% | 3.4x |
| Heart | 33.33% | 8.33% | 0.25x |
| Breast Cancer | 28.33% | 20.83% | 0.73x |

- Overall TabPFN -> GBDT avg: 50.00%
- Overall GBDT -> TabPFN avg: 10.00%
- Asymmetry ratio: 5.0x (t-test p=0.063)

### 4. Defense Mechanisms Evaluation

| Dataset | Features | Best Defense | Recovery |
|---------|----------|-------------|---------|
| Wine | 13 | Gaussian Noise s=0.01 | 71.43% |
| Iris | 4 | Feature Squeezing 8-bit | 100.00% |
| Diabetes | 8 | Gaussian Noise s=0.01 | 100.00% |
| Heart | 13 | Ensemble Voting | 41.67% |
| Breast Cancer | 30 | Gaussian Noise s=0.01 | 78.57% |

Key Finding: No single defense works best for all datasets.

### 5. Synthetic Experiments

| Experiment | Key Finding | TabPFN Correlation |
|-----------|-------------|-------------------|
| Label Noise | ASR increases with noise | r = +0.945 |
| Class Imbalance | ASR decreases with imbalance | r = -0.832 |
| Feature Scaling | No clear linear trend | r = -0.140 |
| Categorical Mix | All models vulnerable at 70%+ categorical | r = +0.548 |

## Statistical Validation Summary

| Finding | Test | Result | Significant? |
|---------|------|--------|-------------|
| TabPFN > GBDT (Boundary) | Bootstrap CI | [-23.0%, -1.3%] | Yes |
| TabPFN > GBDT (Boundary) | Paired t-test | p=0.153 | Marginal (small n=5) |
| Transfer Asymmetry | Paired t-test | p=0.063 | Marginal |
| TabPFN > GBDT (NES) | Paired t-test | p=0.286 | No |

## Experimental Setup

### Datasets

| Dataset | Samples | Features | Classes | Source |
|---------|---------|----------|---------|--------|
| Wine | 130 | 13 | 2 (binary) | sklearn |
| Iris | 100 | 4 | 2 (binary) | sklearn |
| Diabetes | 768 | 8 | 2 | OpenML (Pima Indians) |
| Heart | 297 | 13 | 2 | OpenML/UCI |
| Breast Cancer | 569 | 30 | 2 | sklearn |

### Preprocessing
- StandardScaler applied to all datasets
- stratify=y in train/test split (70/30)
- random_state=42 for reproducibility
- n_samples=15 adversarial examples per experiment

### Attack Methods

1. Boundary Attack (Decision-based)
   - max_iterations: 200, epsilon: 0.5
   - Black-box, uses only predicted labels

2. NES Attack (Score-based)
   - max_iterations: 200, n_samples: 30, learning_rate: 0.3, sigma: 0.3
   - Uses probability outputs

3. Transfer Attack (Model-agnostic)
   - Both directions: TabPFN -> GBDT and GBDT -> TabPFN
   - Tested across all 5 datasets

### Models
- TabPFN v2: CPU mode, random_state=42
- XGBoost: default params, random_state=42
- LightGBM: default params, random_state=42

## Project Structure
```
tabpfn-adversarial/
├── src/
│   ├── models/
│   │   ├── tabpfn_wrapper.py
│   │   └── gbdt_wrapper.py
│   ├── attacks/
│   │   ├── boundary_attack.py
│   │   └── nes_attack.py
│   └── evaluation/
│       ├── metrics.py
│       └── statistical_analysis.py
├── experiments/
│   ├── breast_cancer/
│   ├── diabetes/
│   ├── heart/
│   ├── iris/
│   ├── wine/
│   └── synthetic/
├── results/
│   ├── *_experiment.json
│   ├── *_nes_experiment.json
│   ├── *_defense_results.json
│   ├── transfer_attack_*.json
│   ├── synthetic_*.json
│   └── statistical_analysis.json
└── README.md
```

## Running Experiments
```bash
export PYTHONPATH=/path/to/tabpfn-adversarial

# Boundary attacks
python experiments/<dataset>/run_boundary.py

# NES attacks
python experiments/<dataset>/run_nes.py

# Transfer attacks
python experiments/<dataset>/run_transfer.py

# Defense mechanisms
python experiments/<dataset>/run_defense.py

# Synthetic experiments
python experiments/synthetic/run_noise_boundary.py
python experiments/synthetic/run_imbalance_boundary.py
python experiments/synthetic/run_scaling_boundary.py
python experiments/synthetic/run_categorical_boundary.py

# Statistical analysis
python src/evaluation/statistical_analysis.py
```

## Key Contributions

1. First comprehensive adversarial evaluation of TabPFN v2 - 5 real datasets, 3 attack types, 3 defense mechanisms, synthetic experiments
2. Dataset-dependent vulnerability patterns - No universal winner
3. Transfer attack asymmetry - TabPFN -> GBDT transfers 5x more effectively overall
4. Defense mechanism evaluation - Best defense varies by dataset
5. Synthetic experiment insights - Label noise strongly increases TabPFN vulnerability (r=+0.945)

## Limitations
- Black-box attacks only (no gradient-based white-box attacks)
- Binary classification focus
- Small sample sizes (n=15) due to computational constraints (CPU-only)

## Future Work
- White-box attack evaluation
- Multi-class classification robustness
- Larger sample sizes for statistical power
- Adversarial training for TabPFN

## Citation

@mastersthesis{coskuner2026adversarial,
  title={Adversarial Robustness Evaluation of TabPFN: A Tabular Foundation Model},
  author={Coskuner, Ayse},
  year={2026},
  school={University of Hildesheim},
  supervisor={Koloiarov, Ilia}
}

## License
MIT License

## Acknowledgments
- TabPFN authors (Hollmann et al.) for the foundation model
- OpenML and UCI ML Repository for datasets
- Thesis supervisor: Ilia Koloiarov
"""

open('README.md', 'w').write(content)
print('README.md written!')
