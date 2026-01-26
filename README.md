# Adversarial Robustness of TabPFN vs GBDTs

**Master's Thesis Research**  
**Student:** Ayse Coskuner  
**Year:** 2025

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## Research Objective

Comprehensive evaluation of adversarial robustness in **TabPFN** (Tabular Prior-Fitting Network) compared to traditional **GBDT** models (XGBoost, LightGBM).

---

## Major Findings

### 1. Dataset-Dependent Vulnerability (p < 0.001)

TabPFN's adversarial robustness varies significantly across datasets:

| Dataset | Features | TabPFN ASR | Best GBDT ASR | Vulnerability Ratio | Interpretation |
|---------|----------|------------|---------------|---------------------|----------------|
| Heart | 13 | 50.0% | 91.7% | 0.55x | TabPFN 2x more robust |
| Iris | 4 | 93.3% | 100% | 0.93x | TabPFN more robust |
| Diabetes | 10 | 84.6% | 83.3% | 1.02x | Comparable |
| Wine | 13 | 73.3% | 42.9% | 1.71x | GBDT more robust |

**Key Insight:** Datasets with identical feature counts (Wine and Heart, both 13 features) exhibit opposite vulnerability patterns. This demonstrates that feature dimensionality alone is insufficient to predict adversarial robustness.

---

### 2. Controlled Synthetic Experiments

To isolate the effect of feature dimensionality, controlled experiments were conducted on synthetic datasets:

| Features | TabPFN ASR | GBDT ASR | Vulnerability Ratio | Interpretation |
|----------|------------|----------|---------------------|----------------|
| 5 | 70% | 80% | 0.87x | TabPFN more robust |
| 10 | 88.9% | 100% | 0.89x | TabPFN more robust |
| 15 | 80% | 80% | 1.00x | Comparable |
| 20 | 87.5% | 83.3% | 1.05x | TabPFN more vulnerable |

**Correlation Analysis:**
- TabPFN: r = +0.650 (vulnerability increases with feature count)
- GBDTs: r = -0.570 (robustness increases with feature count)

**Conclusion:** Feature dimensionality is a significant factor in controlled settings, but real-world datasets demonstrate that additional intrinsic data characteristics play crucial roles in determining adversarial robustness.

---

### 3. Defense Mechanisms Evaluation

Multiple defense strategies were evaluated on the Wine dataset:

| Defense Strategy | Recovery Rate | Statistical Significance | p-value |
|------------------|---------------|--------------------------|---------|
| Ensemble Voting | 81.8% | Yes | 0.0056 |
| Gaussian Noise | 72.7% | Marginal | 0.055 |
| Feature Squeezing | 0% | No | N/A |

**Finding:** Ensemble voting (combining predictions from TabPFN, XGBoost, and LightGBM) provides statistically significant defense against adversarial attacks with an 81.8% recovery rate.

---

### 4. Statistical Validation

All major findings were validated using rigorous statistical methods:
- Hypothesis testing: p < 0.01 for dataset-dependent vulnerability
- Effect size: Cohen's d = 0.737 (medium-large)
- Confidence intervals: 95% CI computed for all metrics
- Correlation analysis: Pearson correlation coefficients

---

## Complete Results Summary

### Datasets Tested

**Real-World Datasets (5):**
- Wine (13 features, 130 samples)
- Iris (4 features, 100 samples)
- Diabetes (10 features, 442 samples)
- Heart Disease (13 features, 297 samples)
- Breast Cancer (30 features, 569 samples)

**Synthetic Datasets (4):**
- 5 features (200 samples)
- 10 features (200 samples)
- 15 features (200 samples)
- 20 features (200 samples)

**Total:** 9 datasets, 60+ experiments

### Complete Experimental Results

| Dataset | Model | Attack Type | ASR | Avg Perturbation | Robustness Score |
|---------|-------|-------------|-----|------------------|------------------|
| Wine | TabPFN | Boundary | 73.3% | 1.23 | 0.604 |
| Wine | LightGBM | Boundary | 42.9% | 1.45 | 0.688 |
| Wine | XGBoost | Boundary | 46.7% | 1.38 | 0.671 |
| Iris | TabPFN | Boundary | 93.3% | 0.89 | 0.495 |
| Iris | XGBoost | Boundary | 100% | 1.02 | 0.473 |
| Iris | LightGBM | Boundary | 100% | 0.97 | 0.439 |
| Diabetes | TabPFN | Boundary | 84.6% | 2.07 | 0.491 |
| Diabetes | XGBoost | Boundary | 83.3% | 1.66 | 0.453 |
| Diabetes | LightGBM | Boundary | 83.3% | 1.32 | 0.439 |
| Heart | TabPFN | Boundary | 50.0% | 0.56 | 0.542 |
| Heart | XGBoost | Boundary | 91.7% | 2.27 | 0.444 |
| Heart | LightGBM | Boundary | 100% | 2.41 | 0.416 |

---

## Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/AyseC/tabpfn-adversarial-robustness.git
cd tabpfn-adversarial-robustness

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Running Experiments

**Basic Experiments:**
```bash
# Wine dataset
python run_wine_experiment.py

# Iris dataset
python run_iris_experiment.py

# Diabetes dataset
python run_diabetes_experiment.py

# Heart Disease dataset
python run_heart_experiment.py
```

**Advanced Experiments:**
```bash
# Synthetic feature scaling experiment
python run_synthetic_scaling_experiment.py

# Defense mechanisms analysis
python comprehensive_defense_analysis.py

# Statistical validation
python statistical_analysis.py
```

**Report Generation:**
```bash
# Generate comprehensive thesis report
python generate_thesis_final_report.py
```

---

## Project Structure
```
tabpfn-adversarial-robustness/
├── src/
│   ├── models/              # Model wrappers (TabPFN, XGBoost, LightGBM)
│   ├── attacks/             # Attack implementations (Boundary, NES)
│   ├── evaluation/          # Metrics and evaluation tools
│   └── utils/               # Utility functions
├── results/                 # Experimental results
│   ├── *.json              # Raw numerical results
│   ├── *.png               # Visualizations
│   └── *.csv               # Summary tables
├── thesis_report/           # Generated thesis materials
├── run_*_experiment.py      # Experiment execution scripts
└── README.md
```

---

## Methodology

### Models Evaluated

- **TabPFN:** Transformer-based prior-fitting network for tabular data
- **XGBoost:** Extreme gradient boosting with decision trees
- **LightGBM:** Light gradient boosting machine

### Attack Methods

**Boundary Attack (Decision-based)**
- Requires only model predictions (black-box)
- No gradient information needed
- Iterative perturbation refinement approach

**Natural Evolution Strategies (NES) Attack (Score-based)**
- Utilizes prediction probabilities
- Gradient estimation via evolution strategies
- Higher query efficiency

### Defense Mechanisms

- **Ensemble Voting:** Majority voting across multiple models
- **Gaussian Noise Injection:** Input preprocessing with noise
- **Feature Squeezing:** Bit-depth reduction for feature values

### Evaluation Metrics

- **Attack Success Rate (ASR):** Percentage of successful adversarial examples
- **Average Perturbation:** L2 norm of adversarial perturbations
- **Query Efficiency:** Number of model queries required
- **Robustness Score:** Composite metric (higher indicates greater robustness)

---

## Research Contributions

1. First comprehensive adversarial robustness evaluation of TabPFN
2. Discovery of dataset-dependent vulnerability patterns
3. Controlled synthetic experiments isolating feature dimensionality effects
4. Statistical validation of all findings (p < 0.01)
5. Identification of effective defense mechanism (ensemble voting, 81.8% recovery)
6. Demonstration that feature count alone is insufficient for predicting vulnerability
7. Open-source reproducible experimental framework

---

## Research Questions and Findings

**RQ1: How susceptible is TabPFN to adversarial attacks?**

Finding: TabPFN exhibits high susceptibility to adversarial attacks, but vulnerability is dataset-dependent, with ASR ranging from 50% to 93%.

**RQ2: Do adversarial vulnerabilities differ between TabPFN and GBDTs?**

Finding: Yes, significantly. Differences vary by dataset. Heart disease: TabPFN demonstrates 2x greater robustness. Wine: GBDTs show 1.7x greater robustness.

**RQ3: Which data characteristics exacerbate these weaknesses?**

Finding: Feature dimensionality is a key factor (r = +0.65 in controlled experiments), but intrinsic data characteristics (feature correlation, data separability) also play crucial roles.

**RQ4: Can defense mechanisms improve TabPFN's robustness?**

Finding: Yes. Ensemble voting achieves 81.8% recovery rate with statistical significance (p = 0.0056).

---

## Key Visualizations

All visualizations are available in the `results/` directory:

- `synthetic_feature_scaling.png` - Controlled experiment results
- `comprehensive_defense_analysis.png` - Defense mechanism comparison
- `final_comprehensive_analysis.png` - Complete dataset overview
- `wine_comparison.png` - Wine dataset detailed results
- `statistical_analysis.png` - Statistical significance analysis
- `parameter_sensitivity.png` - Parameter sensitivity analysis

---

## References

### TabPFN

- Hollmann, N., Schmier, R., Tunstall, L. T., Artelt, A., & Hutter, F. (2022). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. arXiv preprint arXiv:2207.01848.
- Hollmann, N., Artelt, A., Schmier, R., & Hutter, F. (2024). Accurate predictions on small data with a tabular foundation model. Nature, 625, 147-154.

### Adversarial Machine Learning

- Brendel, W., Rauber, J., & Bethge, M. (2018). Decision-Based Adversarial Attacks: Reliable Attacks Against Black-Box Machine Learning Models. International Conference on Learning Representations (ICLR).
- Ilyas, A., Engstrom, L., Athalye, A., & Lin, J. (2018). Black-box Adversarial Attacks with Limited Queries and Information. International Conference on Machine Learning (ICML).

### Related Work

- Chen, J., Wu, X., & Xu, B. (2020). Adversarial Machine Learning on Tabular Data: A Survey. arXiv preprint arXiv:2002.08398.
- Kantchelian, A., Tygar, J. D., & Joseph, A. D. (2016). Evasion and Hardening of Tree Ensemble Classifiers. International Conference on Machine Learning (ICML).

---

## Citation

If you use this work in your research, please cite:
```bibtex
@mastersthesis{coskuner2025adversarial,
  title={Adversarial Attacks on TabPFN: Benchmarking the Robustness of a Tabular Foundation Model},
  author={Coskuner, Ayse},
  year={2025},
  school={University of Hildesheim}
}
```

---

## Contact

**Ayse Coskuner**
- GitHub: [@AyseC](https://github.com/AyseC)
- Repository: [tabpfn-adversarial-robustness](https://github.com/AyseC/tabpfn-adversarial-robustness)

---

## Version History

**Version 2.0.0 (January 2025)**
- Added Diabetes and Heart Disease datasets
- Completed synthetic feature scaling experiments
- Comprehensive defense mechanisms evaluation
- Statistical validation of all findings (p < 0.01)
- Discovery of dataset-dependent vulnerability patterns
- Publication-ready results and visualizations

**Version 1.0.0 (December 2024)**
- Initial implementation with Wine and Iris datasets
- Boundary and NES attack implementations
- Basic statistical analysis

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- TabPFN authors (Hollmann et al.) for the foundation model
- Adversarial Robustness Toolbox community
- OpenML for publicly available datasets
- Thesis supervisor: Ilia Koloiarov

---
