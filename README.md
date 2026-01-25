# Adversarial Robustness of TabPFN vs GBDTs

**Master's Thesis Research**  
**Student:** Ayse Coskuner  
**Year:** 2025

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 🎯 Research Objective

Comprehensive evaluation of adversarial robustness in **TabPFN** (Tabular Prior-Fitting Network) compared to traditional **GBDT** models (XGBoost, LightGBM).

## 🔥 Key Findings

### 1. **Dataset-Dependent Robustness** (p < 0.001)
- **Wine Dataset (13 features):** TabPFN **1.71x MORE vulnerable** than GBDTs
- **Iris Dataset (4 features):** TabPFN **0.93x LESS vulnerable** (more robust!)
- **Pattern:** Feature complexity correlates with TabPFN vulnerability

### 2. **Statistical Validation**
- All major findings statistically significant (p < 0.01)
- Cohen's d = 0.737 (medium-large effect size)
- Results suitable for publication ✅

### 3. **Attack Comparison**
- **Boundary Attack:** More effective (73% vs 67% ASR on TabPFN)
- **NES Attack:** Similar vulnerability patterns
- Consistent across attack types → Fundamental vulnerability

### 4. **Parameter Sensitivity**
- TabPFN vulnerability is **parameter-independent**
- XGBoost shows parameter sensitivity
- TabPFN's weakness is fundamental, not attack-specific

---

## 📊 Results Summary

| Dataset | Model | Attack | ASR | Robustness Score |
|---------|-------|--------|-----|------------------|
| Wine | TabPFN | Boundary | 73.3% | 0.604 |
| Wine | LightGBM | Boundary | 42.9% | 0.688 |
| Wine | XGBoost | Boundary | 46.7% | 0.671 |
| Iris | TabPFN | Boundary | 93.3% | 0.495 |
| Iris | XGBoost | Boundary | 100% | 0.473 |
| Iris | LightGBM | Boundary | 100% | 0.439 |

**Conclusion:** TabPFN's adversarial robustness is dataset-dependent, not universally inferior!

---

## 🚀 Quick Start

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

### Run Experiments
```bash
# Wine dataset with Boundary Attack
python run_wine_experiment.py

# Iris dataset with NES Attack
python run_iris_nes_experiment.py

# Generate comprehensive report
python generate_final_report.py

# Statistical analysis
python statistical_analysis.py

# Parameter sensitivity
python parameter_sensitivity_analysis.py
```

### Interactive Analysis
```bash
jupyter notebook notebooks/analysis.ipynb
```

---

## 📁 Project Structure
```
tabpfn-adversarial-robustness/
├── src/
│   ├── models/              # Model wrappers (TabPFN, XGBoost, LightGBM)
│   ├── attacks/             # Attack implementations (Boundary, NES)
│   ├── evaluation/          # Metrics and evaluation tools
│   └── utils/               # Utilities
├── experiments/             # Experiment scripts
├── notebooks/               # Jupyter notebooks
├── results/                 # Experimental results
│   ├── *.json              # Raw results
│   ├── *.png               # Visualizations
│   └── *.csv               # Summary tables
├── config/                  # Configuration files
└── README.md
```

---

## 🔬 Methodology

### Models
- **TabPFN:** Transformer-based tabular foundation model
- **XGBoost:** Gradient boosting decision trees
- **LightGBM:** Light gradient boosting machine

### Attacks
1. **Boundary Attack** (Decision-based)
   - Only requires model predictions
   - No gradient information needed
   - Iterative perturbation refinement

2. **NES Attack** (Score-based)
   - Uses prediction probabilities
   - Gradient estimation via evolution strategies
   - More query-efficient

### Datasets
- **Wine:** 130 samples, 13 features, binary classification
- **Iris:** 100 samples, 4 features, binary classification

### Metrics
- **Attack Success Rate (ASR):** % of successful adversarial examples
- **Avg Perturbation:** L2 norm of adversarial perturbations
- **Query Efficiency:** Number of model queries needed
- **Robustness Score:** Combined metric (higher = more robust)

---

## 📈 Main Contributions

1. **First comprehensive adversarial robustness evaluation of TabPFN**
2. **Discovery of dataset-dependent vulnerability pattern**
3. **Statistical validation of all findings (p < 0.01)**
4. **Multi-attack evaluation (Boundary + NES)**
5. **Parameter sensitivity analysis**
6. **Open-source reproducible framework**

---

## 📊 Visualizations

All visualizations available in `results/` directory:

- `wine_comparison.png` - Wine dataset model comparison
- `comprehensive_analysis.png` - All datasets overview
- `attack_comparison.png` - Boundary vs NES comparison
- `statistical_analysis.png` - Statistical significance plots
- `parameter_sensitivity.png` - Parameter analysis
- `final_comprehensive_analysis.png` - Ultimate summary

---

## 📚 References

### TabPFN
- Hollmann et al. (2022). TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second. [arXiv:2207.01848](https://arxiv.org/abs/2207.01848)
- Hollmann et al. (2024). Accurate predictions on small data with a tabular foundation model. *Nature*, 625, 147-154.

### Adversarial Attacks
- Brendel et al. (2018). Decision-Based Adversarial Attacks. [arXiv:1712.04248](https://arxiv.org/abs/1712.04248)
- Ilyas et al. (2018). Black-box Adversarial Attacks with Limited Queries. *ICML 2018*.

### Related Work
- Chen et al. (2020). Adversarial Machine Learning on Tabular Data: A Survey.
- Kantchelian et al. (2016). Evasion and Hardening of Tree Ensemble Classifiers. *ICML 2016*.

---

## 🎓 Citation

If you use this code or findings in your research, please cite:
```bibtex
@mastersthesis{coskuner2025adversarial,
  title={Adversarial Robustness of TabPFN: Benchmarking Foundation Models vs GBDTs},
  author={Coskuner, Ayse},
  year={2025},
  school={Your University}
}
```

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- TabPFN authors (Hollmann et al.)
- Adversarial Robustness Toolbox community
- OpenML for datasets

---

## 📧 Contact

**Ayse Coskuner**
- GitHub: [@AyseC](https://github.com/AyseC)
- Repository: [tabpfn-adversarial-robustness](https://github.com/AyseC/tabpfn-adversarial-robustness)

---

## 🔄 Updates

**Latest Version:** v1.0.0 (2025)
- Complete implementation of Boundary and NES attacks
- Statistical significance testing
- Parameter sensitivity analysis
- Comprehensive Jupyter notebook
- Publication-ready results

---

**⭐ Star this repository if you found it helpful!**
