"""Add StandardScaler to all experiment files"""
import re

files_to_fix = [
    'run_wine_experiment.py',
    'run_iris_experiment.py', 
    'run_breast_cancer_experiment.py',
    'run_breast_cancer_nes_experiment.py',
    'run_diabetes_nes_experiment.py',
    'run_heart_nes_experiment.py',
    'run_iris_nes_experiment.py',
    'run_wine_nes_experiment.py',
]

for filepath in files_to_fix:
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Skip if already has StandardScaler
        if 'StandardScaler' in content:
            print(f"✓ {filepath} - already has StandardScaler")
            continue
        
        # Add import
        if 'from sklearn.preprocessing import StandardScaler' not in content:
            content = content.replace(
                'from sklearn.model_selection import train_test_split',
                'from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler'
            )
        
        # Add scaler before train_test_split
        # Find the line with train_test_split and add scaler before it
        pattern = r'(X_train, X_test, y_train, y_test = train_test_split\()'
        replacement = r'# Standardize features\nscaler = StandardScaler()\nX = scaler.fit_transform(X)\n\n\1'
        content = re.sub(pattern, replacement, content, count=1)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        print(f"✓ {filepath} - updated")
        
    except Exception as e:
        print(f"✗ {filepath} - error: {e}")

print("\nDone! Now run all experiments again.")
