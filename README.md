# Adversarial Robustness of TabPFN vs GBDTs

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
| Wine | 13 | 93.33% | 100.00% (XGBoost) | -6.67% | TabPFN |
| Iris | 4 | 86.67% | 100.00% | -13.33% | TabPFN |
| Diabetes | 8 | 100.00% | 100.00% | 0.00% | Tie |
| Heart | 13 | 66.67% | 100.00% (XGBoost) | -33.33% | TabPFN |
| Breast Cancer | 30 | 80.00% | 86.67% | -6.67% | TabPFN |

Statistical Analysis:
- Mean ASR difference: -9.33% (TabPFN lower)
- Bootstrap 95% CI: [-18.7%, -1.3%] (excludes zero = significant)
- Cohen's d: -0.837 (Large effect size)
- Paired t-test: p=0.135 (limited by small n=5)

### 2. NES Attack Results

| Dataset | TabPFN ASR | Best GBDT ASR | Winner |
|---------|------------|---------------|--------|
| Wine | 73.33% | 73.33% (LightGBM) | Tie |
| Iris | 20.00% | 53.33% | TabPFN |
| Diabetes | 100.00% | 100.00% | Tie |
| Heart | 86.67% | 93.33% (LightGBM) | GBDT |
| Breast Cancer | 86.67% | 73.33% (XGBoost) | GBDT |

Statistical Analysis:
- Mean ASR difference: +5.33% (TabPFN slightly higher)
- Paired t-test: p=0.675 (not significant)

### 3. Transfer Attack Asymmetry

| Dataset | GBDT -> TabPFN | TabPFN -> GBDT | Ratio |
|---------|----------------|----------------|-------|
| Wine | 17.14% | 53.57% | 3.12x |
| Iris | 3.33% | 53.33% | 16.00x |
| Diabetes | 10.00% | 71.43% | 7.14x |
| Heart | 34.52% | 55.56% | 1.61x |
| Breast Cancer | 11.54% | 15.38% | 1.33x |

- Overall TabPFN -> GBDT avg: 49.85%
- Overall GBDT -> TabPFN avg: 15.31%
- Asymmetry ratio: 3.26x (t-test p=0.0004, highly significant)

### 4. Defense Mechanisms Evaluation

| Dataset | Features | Best Defense | Adv ASR After Defense | Recovery |
|---------|----------|--------------|-----------------------|---------|
| Wine | 13 | Gaussian Noise σ=0.01 | 0.00% | 93.33% |
| Iris | 4 | Ensemble Voting | 60.00% | 40.00% |
| Diabetes | 8 | Feature Squeezing 4-bit | 33.33% | 60.00% |
| Heart | 13 | Gaussian Noise σ=0.01 | 26.67% | 40.00% |
| Breast Cancer | 30 | Gaussian Noise σ=0.01 | 7.14% | 78.57% |

Key Finding: No single defense works best for all datasets.

### 5. Synthetic Experiments

| Experiment | Key Finding | TabPFN Correlation |
|-----------|-------------|-------------------|
| Label Noise | ASR increases with noise | r = +0.969 |
| Class Imbalance | ASR decreases with imbalance | r = -0.544 |
| Feature Scaling | ASR strongly decreases with more features | r = -0.965 |
| Categorical Mix | ASR increases with categorical ratio | r = +0.886 |

## Statistical Validation Summary

| Finding | Test | Result | Significant? |
|---------|------|--------|-------------|
| TabPFN > GBDT (Boundary) | Bootstrap CI | [-18.7%, -1.3%] | Yes |
| TabPFN > GBDT (Boundary) | Paired t-test | p=0.135 | Marginal (small n=5) |
| Transfer Asymmetry | Paired t-test | p=0.0004 | Yes (highly significant) |
| TabPFN > GBDT (NES) | Paired t-test | p=0.675 | No |

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
- **Common attack indices**: attack samples are drawn only from examples correctly classified by ALL models, ensuring fair comparison

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
│       ├── plot_boundary_results.py
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
│   ├── figures/
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
3. Transfer attack asymmetry - TabPFN -> GBDT transfers 3.26x more effectively on average (p=0.0004, highly significant)
4. Defense mechanism evaluation - Best defense varies by dataset
5. Synthetic experiment insights - Label noise and high categorical ratios increase vulnerability; more features strongly improve robustness (r=-0.965)

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
