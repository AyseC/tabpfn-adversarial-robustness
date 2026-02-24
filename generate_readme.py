import json

def load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}

boundary = {ds: load(f'results/{ds}_experiment.json') for ds in ['wine','iris','diabetes','heart','breast_cancer']}
nes = {ds: load(f'results/{ds}_nes_experiment.json') for ds in ['wine','iris','diabetes','heart','breast_cancer']}
feat = {'wine':13,'iris':4,'diabetes':8,'heart':13,'breast_cancer':30}

lines = []
lines.append("# Adversarial Robustness of TabPFN vs GBDTs\n")
lines.append("**Student:** Ayse Coskuner | **Supervisor:** Ilia Koloiarov | **Institution:** University of Hildesheim | **Year:** 2026\n")
lines.append("\n## Boundary Attack Results\n")
lines.append("| Dataset | Features | TabPFN ASR | XGBoost ASR | LightGBM ASR | Winner |")
lines.append("|---------|----------|------------|-------------|--------------|--------|")

for ds, f in feat.items():
    d = boundary[ds]
    if d:
        t = d.get('TabPFN',{}).get('attack_success_rate',0)*100
        x = d.get('XGBoost',{}).get('attack_success_rate',0)*100
        l = d.get('LightGBM',{}).get('attack_success_rate',0)*100
        best = min(x,l)
        w = 'TabPFN' if t < best else ('Tie' if t == best else 'GBDT')
        lines.append(f"| {ds.replace('_',' ').title()} | {f} | {t:.1f}% | {x:.1f}% | {l:.1f}% | {w} |")

lines.append("\n## NES Attack Results\n")
lines.append("| Dataset | TabPFN ASR | XGBoost ASR | LightGBM ASR | Winner |")
lines.append("|---------|------------|-------------|--------------|--------|")

for ds in feat:
    d = nes[ds]
    if d:
        t = d.get('TabPFN',{}).get('attack_success_rate',0)*100
        x = d.get('XGBoost',{}).get('attack_success_rate',0)*100
        l = d.get('LightGBM',{}).get('attack_success_rate',0)*100
        best = min(x,l)
        w = 'TabPFN' if t < best else ('Tie' if t == best else 'GBDT')
        lines.append(f"| {ds.replace('_',' ').title()} | {t:.1f}% | {x:.1f}% | {l:.1f}% | {w} |")

lines.append("\n## Experimental Setup\n")
lines.append("- **Models:** TabPFN v2 (random_state=42), XGBoost, LightGBM")
lines.append("- **Datasets:** Wine, Iris, Diabetes, Heart, Breast Cancer")
lines.append("- **Preprocessing:** StandardScaler, stratify=y, test_size=0.3, random_state=42")
lines.append("- **Sample size:** n=15 per experiment")
lines.append("- **Attacks:** Boundary Attack (max_iterations=200, epsilon=0.5), NES Attack (max_iterations=200, sigma=0.3), Transfer Attack")
lines.append("- **Defenses:** Gaussian Noise, Feature Squeezing, Ensemble Voting")
lines.append("\n## Project Structure\n")
lines.append("```")
lines.append("tabpfn-adversarial/")
lines.append("├── src/")
lines.append("│   ├── models/         # TabPFN and GBDT wrappers")
lines.append("│   ├── attacks/        # Boundary and NES attack implementations")
lines.append("│   └── evaluation/     # Metrics and statistical analysis")
lines.append("├── experiments/")
lines.append("│   ├── breast_cancer/")
lines.append("│   ├── diabetes/")
lines.append("│   ├── heart/")
lines.append("│   ├── iris/")
lines.append("│   ├── wine/")
lines.append("│   └── synthetic/")
lines.append("└── results/            # JSON results for all experiments")
lines.append("```")
lines.append("\n## Running Experiments\n")
lines.append("```bash")
lines.append("export PYTHONPATH=/path/to/tabpfn-adversarial")
lines.append("python experiments/<dataset>/run_boundary.py")
lines.append("python experiments/<dataset>/run_nes.py")
lines.append("python experiments/<dataset>/run_transfer.py")
lines.append("python experiments/<dataset>/run_defense.py")
lines.append("python src/evaluation/statistical_analysis.py")
lines.append("```")

with open('README.md', 'w') as f:
    f.write('\n'.join(lines))
print('README.md updated!')
