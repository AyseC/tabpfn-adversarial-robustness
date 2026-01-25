# Adversarial Robustness of TabPFN

Master's thesis research on adversarial attacks against TabPFN compared to traditional GBDT models.

## Main Finding

**TabPFN is 1.7x more vulnerable to adversarial attacks than GBDTs (XGBoost, LightGBM)**

## Results (Wine Dataset)

| Model | Attack Success Rate | Robustness Score |
|-------|-------------------|------------------|
| LightGBM | 42.9% | 0.688 |
| XGBoost | 46.7% | 0.671 |
| TabPFN | 73.3% | 0.604 |

## Quick Start
```bash
# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run experiment
python run_wine_experiment.py
```

## 📁 Project Structure

- `src/` - Source code (models, attacks, evaluation)
- `experiments/` - Experiment runners
- `results/` - Experimental results
- `config/` - Configuration files

## Methods

- **Models**: TabPFN, XGBoost, LightGBM
- **Attack**: Boundary Attack (decision-based)
- **Metrics**: Attack Success Rate, Perturbation, Query Efficiency

## References

- Hollmann et al. (2024) - TabPFN: Tabular Prior-Fitting Network
- Brendel et al. (2018) - Decision-Based Adversarial Attacks

---

**Student**: Ayse Coskuner  
**Institution**: Hildesheim University
**Year**: 2025
