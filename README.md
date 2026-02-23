# Adversarial Robustness of TabPFN vs GBDTs

**Student:** Ayse Coskuner | **Supervisor:** Ilia Koloiarov | **Institution:** University of Hildesheim | **Year:** 2026


## Boundary Attack Results

| Dataset | Features | TabPFN ASR | XGBoost ASR | LightGBM ASR | Winner |
|---------|----------|------------|-------------|--------------|--------|
| Wine | 13 | 78.6% | 100.0% | 93.3% | TabPFN |
| Iris | 4 | 93.3% | 100.0% | 100.0% | TabPFN |
| Diabetes | 8 | 100.0% | 100.0% | 100.0% | Tie |
| Heart | 13 | 50.0% | 83.3% | 91.7% | TabPFN |
| Breast Cancer | 30 | 78.6% | 78.6% | 85.7% | Tie |

## NES Attack Results

| Dataset | TabPFN ASR | XGBoost ASR | LightGBM ASR | Winner |
|---------|------------|-------------|--------------|--------|
| Wine | 35.7% | 14.3% | 40.0% | GBDT |
| Iris | 20.0% | 40.0% | 33.3% | TabPFN |
| Diabetes | 100.0% | 100.0% | 100.0% | Tie |
| Heart | 50.0% | 58.3% | 41.7% | GBDT |
| Breast Cancer | 85.7% | 71.4% | 50.0% | GBDT |

## Experimental Setup

- **Models:** TabPFN v2 (random_state=42), XGBoost, LightGBM
- **Datasets:** Wine, Iris, Diabetes, Heart, Breast Cancer
- **Preprocessing:** StandardScaler, stratify=y, test_size=0.3, random_state=42
- **Sample size:** n=15 per experiment
- **Attacks:** Boundary Attack (max_iterations=200, epsilon=0.5), NES Attack (max_iterations=200, sigma=0.3), Transfer Attack
- **Defenses:** Gaussian Noise, Feature Squeezing, Ensemble Voting

## Project Structure

```
tabpfn-adversarial/
├── src/
│   ├── models/         # TabPFN and GBDT wrappers
│   ├── attacks/        # Boundary and NES attack implementations
│   └── evaluation/     # Metrics and statistical analysis
├── experiments/
│   ├── breast_cancer/
│   ├── diabetes/
│   ├── heart/
│   ├── iris/
│   ├── wine/
│   └── synthetic/
└── results/            # JSON results for all experiments
```

## Running Experiments

```bash
export PYTHONPATH=/path/to/tabpfn-adversarial
python experiments/<dataset>/run_boundary.py
python experiments/<dataset>/run_nes.py
python experiments/<dataset>/run_transfer.py
python experiments/<dataset>/run_defense.py
python src/evaluation/statistical_analysis.py
```