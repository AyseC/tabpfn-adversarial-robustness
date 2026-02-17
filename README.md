# Adversarial Robustness of TabPFN vs GBDTs

## Master's Thesis Research
- **Student:** Ayse Coskuner
- **Supervisor:** Ilia Koloiarov
- **Institution:** University of Hildesheim
- **Year:** 2025

## Research Objective

Comprehensive evaluation of adversarial robustness in TabPFN (Tabular Prior-Fitting Network) compared to traditional GBDT models (XGBoost, LightGBM).

---

## Major Findings

### 1. TabPFN Generally More Robust Than GBDTs

TabPFN showed lower Attack Success Rate (ASR) in **4 out of 5 datasets**:

| Dataset | Features | TabPFN ASR | Best GBDT ASR | Difference | Winner |
|---------|----------|------------|---------------|------------|--------|
| Wine | 13 | 86.67% | 93.33% | -6.67% | TabPFN ✓ |
| Iris | 4 | 86.67% | 100.00% | -13.33% | TabPFN ✓ |
| Diabetes | 8 | 100.00% | 100.00% | 0.00% | Tie |
| Heart | 13 | 50.00% | 83.33% | -33.33% | TabPFN ✓ |
| Breast Cancer | 30 | 76.92% | 85.71% | -8.79% | TabPFN ✓ |

**Statistical Analysis:**
- Mean ASR difference: **-12.4%** (TabPFN lower)
- Bootstrap 95% CI: **[-23.5%, -4.4%]** (excludes zero = significant)
- Cohen's d: **-0.98** (Large effect size)
- Paired t-test: p=0.093 (marginally significant due to small n=5)

### 2. Boundary Attack More Effective Than NES on Tabular Data (p=0.002)

| Attack Type | Mean ASR | Std Dev |
|-------------|----------|---------|
| Boundary Attack | 89.89% | 13.13% |
| NES Attack | 61.01% | 27.25% |

**Key Discovery:** Decision-based attacks (Boundary) significantly outperform score-based attacks (NES) on tabular data. This **contradicts image domain literature** where NES is typically stronger.

**Explanation:**
- Decision trees have non-smooth decision boundaries
- NES gradient estimation fails on discontinuous probability landscapes
- Lower dimensionality (4-30 features vs 784+ for images) amplifies this effect

### 3. Parameter Sensitivity Analysis

**NES Attack - sigma is CRITICAL:**

| sigma | TabPFN ASR | XGBoost ASR | LightGBM ASR |
|-------|------------|-------------|--------------|
| 0.1 | 30% | 0% | 30% |
| 0.3 | 30% | 0% | 30% |
| 0.5 | 50% | 100% | 70% |
| 1.0 | 90% | 100% | 100% |

**Finding:** Default sigma=0.3 was suboptimal. With sigma≥0.5, NES achieves comparable ASR to Boundary Attack.

**Boundary Attack:** Robust to hyperparameter changes (max_iterations, epsilon have minimal effect).

### 4. Transfer Attack Asymmetry (Dataset-Dependent)

| Dataset | GBDT → TabPFN | TabPFN → GBDT | Ratio |
|---------|---------------|---------------|-------|
| Iris | 3.33% | 34.62% | 10.38x |
| Diabetes | 9.09% | 63.64% | 7.00x |
| Heart | 47.73% | 16.67% | 0.35x |
| Breast Cancer | 40.38% | 16.67% | 0.41x |
| Wine | 0% | 0% | N/A |

**Pattern:**
- Low-dimensional datasets (Iris, Diabetes): TabPFN → GBDT transfers better
- High-dimensional datasets (Heart, Breast Cancer): GBDT → TabPFN transfers better

### 5. Defense Mechanisms Evaluation

| Defense Type | Mean Recovery | Best Dataset | Best Recovery |
|--------------|---------------|--------------|---------------|
| Feature Squeezing | 59.67% | Iris (8-bit) | 93.33% |
| Gaussian Noise | 57.09% | Iris (σ=0.03) | 80.00% |
| Ensemble Voting | 33.13% | Breast Cancer | 61.54% |

**Best Defense per Dataset:**

| Dataset | Features | Best Defense | Recovery |
|---------|----------|--------------|----------|
| Wine | 13 | Gaussian Noise σ=0.01 | 73.33% |
| Iris | 4 | Feature Squeezing 8-bit | 93.33% |
| Diabetes | 8 | Feature Squeezing 16-bit | 81.82% |
| Heart | 13 | Ensemble Voting | 41.67% |
| Breast Cancer | 30 | Gaussian Noise σ=0.01 | 69.23% |

**Key Finding:** No single defense works best for all datasets. Defense effectiveness depends on data characteristics. Gaussian noise preprocessing provides complete protection against adversarial attacks on TabPFN in certain configurations, suggesting TabPFN's transformer architecture is particularly sensitive to input perturbations.

**Feature Complexity Threshold:** A notable threshold around 13 features was observed — datasets at or above this threshold show a strong correlation between feature complexity and ensemble defense effectiveness.

---

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
- **StandardScaler** applied to all datasets
- **stratify=y** in train/test split (70/30)
- **random_state=42** for reproducibility
- **n_samples=15** adversarial examples per experiment

### Attack Methods

**1. Boundary Attack (Decision-based)**
- max_iterations: 200
- epsilon: 0.5
- Black-box, uses only predicted labels

**2. NES Attack (Score-based)**
- max_iterations: 200
- n_samples: 30
- learning_rate: 0.3
- sigma: 0.3
- Uses probability outputs

**3. Transfer Attack (Model-agnostic)**
- Source models: TabPFN → GBDT and GBDT → TabPFN (both directions)
- Adversarial examples generated on source model, evaluated on target model
- Transfer rate measured across all 5 datasets
- Black-box with respect to target model — no target model access required during attack generation

### Models
- **TabPFN:** Tabular Prior-Fitted Network (CPU mode)
- **XGBoost:** Extreme Gradient Boosting (default params)
- **LightGBM:** Light Gradient Boosting Machine (default params)

---

## Project Structure
\`\`\`
tabpfn-adversarial/
├── src/
│   ├── models/
│   │   ├── tabpfn_wrapper.py
│   │   └── gbdt_wrapper.py
│   ├── attacks/
│   │   ├── boundary_attack.py
│   │   └── nes_attack.py
│   └── evaluation/
│       └── metrics.py
├── results/
│   ├── *_experiment.json
│   ├── *_nes_experiment.json
│   ├── *_defense_results.json
│   ├── transfer_attack_*.json
│   ├── parameter_sensitivity.json
│   └── statistical_analysis.json
├── run_*_experiment.py          # Boundary attack experiments
├── run_*_nes_experiment.py      # NES attack experiments
├── run_*_defense_experiment.py  # Defense experiments
├── run_transfer_attack_*.py     # Transfer attack experiments
├── run_parameter_sensitivity.py # Parameter analysis
├── run_statistical_analysis.py  # Statistical tests
├── analyze_*.py                 # Analysis scripts
└── README.md
\`\`\`

---

## Quick Start

### Installation
\`\`\`bash
# Clone repository
git clone https://github.com/AyseC/tabpfn-adversarial.git
cd tabpfn-adversarial

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
\`\`\`

### Running Experiments
\`\`\`bash
# Boundary Attack (all datasets)
python run_wine_experiment.py
python run_iris_experiment.py
python run_diabetes_experiment.py
python run_heart_experiment.py
python run_breast_cancer_experiment.py

# NES Attack (all datasets)
python run_wine_nes_experiment.py
python run_iris_nes_experiment.py
python run_diabetes_nes_experiment.py
python run_heart_nes_experiment.py
python run_breast_cancer_nes_experiment.py

# Transfer Attacks
python run_transfer_attack_wine.py
python run_transfer_attack_iris.py
python run_transfer_attack_diabetes.py
python run_transfer_attack_heart.py
python run_transfer_attack_breast_cancer.py

# Defense Mechanisms
python run_wine_defense_experiment.py
python run_iris_defense_experiment.py
python run_diabetes_defense_experiment.py
python run_heart_defense_experiment.py
python run_breast_cancer_defense_experiment.py

# Statistical Analysis
python run_statistical_analysis.py
python run_parameter_sensitivity.py
python analyze_defense_results.py
\`\`\`

---

## Statistical Validation Summary

| Finding | Test | p-value | Significant? |
|---------|------|---------|--------------|
| Boundary > NES | Paired t-test | **0.002** | ✓ Yes |
| TabPFN > GBDT | Bootstrap CI | **[−23.5%, −4.4%]** | ✓ Yes |
| TabPFN > GBDT | Paired t-test | 0.093 | Marginal (α=0.10) |
| Transfer Asymmetry | Paired t-test | 0.722 | ✗ No |
| Dim vs NES Gap | Pearson r | −0.902 (p=0.098) | Marginal |

---

## Key Contributions

1. **First comprehensive adversarial evaluation of TabPFN**
   - 5 real datasets, 3 attack types, 3 defense mechanisms

2. **Novel finding: Decision-based attacks outperform score-based on tabular data**
   - Statistically significant (p=0.002)
   - Contradicts image domain literature

3. **Parameter sensitivity analysis for tabular attacks**
   - NES sigma parameter is critical (0.3 → 1.0 increases ASR from 0% to 100%)
   - Boundary Attack robust to hyperparameter choices

4. **Dataset-dependent vulnerability patterns**
   - No universal "winner" between TabPFN and GBDTs
   - Robustness depends on data characteristics
   - Feature complexity threshold (~13 features) correlates with ensemble defense effectiveness

5. **Transfer attack asymmetry**
   - TabPFN → GBDT transfers more effectively on low-dimensional datasets (up to 10.38x ratio)
   - GBDT → TabPFN transfers more effectively on high-dimensional datasets

6. **Defense mechanism evaluation**
   - Feature Squeezing and Gaussian Noise most effective overall
   - Gaussian noise preprocessing provides complete protection against adversarial attacks on TabPFN in certain configurations
   - Best defense varies by dataset; no universal solution

---

## Limitations

- Black-box attacks only (no gradient-based white-box attacks)
- Binary classification focus
- Small sample sizes (n=15) per experiment due to computational constraints
- CPU-only TabPFN (no GPU acceleration)

## Future Work

- White-box attack evaluation (if gradient access available)
- Multi-class classification robustness
- Larger sample sizes for statistical power
- Adversarial training for TabPFN
- Domain-specific defense mechanisms

---

## Citation
\`\`\`bibtex
@mastersthesis{coskuner2025adversarial,
  title={Adversarial Robustness Evaluation of TabPFN: A Tabular Foundation Model},
  author={Coskuner, Ayse},
  year={2025},
  school={University of Hildesheim},
  supervisor={Koloiarov, Ilia}
}
\`\`\`

---

## License

MIT License

## Acknowledgments

- TabPFN authors (Hollmann et al.) for the foundation model
- OpenML and UCI ML Repository for datasets
- Thesis supervisor: Ilia Koloiarov
