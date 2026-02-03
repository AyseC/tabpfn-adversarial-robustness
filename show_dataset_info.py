"""Show dataset information for all datasets"""
import numpy as np
from sklearn.datasets import load_wine, load_iris, load_breast_cancer, fetch_openml
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("DATASET INFORMATION - ALL DATASETS")
print("="*80)

datasets_info = []

# Wine
print("\n[1/5] Wine Dataset...")
data = load_wine()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]
datasets_info.append({
    'name': 'Wine',
    'samples': len(X),
    'features': X.shape[1],
    'class_0': sum(y == 0),
    'class_1': sum(y == 1),
    'balance': f"{sum(y == 0) / len(y) * 100:.1f}% / {sum(y == 1) / len(y) * 100:.1f}%"
})

# Iris
print("[2/5] Iris Dataset...")
data = load_iris()
X, y = data.data, data.target
mask = y < 2
X, y = X[mask], y[mask]
datasets_info.append({
    'name': 'Iris',
    'samples': len(X),
    'features': X.shape[1],
    'class_0': sum(y == 0),
    'class_1': sum(y == 1),
    'balance': f"{sum(y == 0) / len(y) * 100:.1f}% / {sum(y == 1) / len(y) * 100:.1f}%"
})

# Diabetes (Pima Indians)
print("[3/5] Diabetes Dataset...")
try:
    diabetes = fetch_openml('diabetes', version=1, as_frame=False, parser='auto')
    X, y = diabetes.data, diabetes.target
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)
    else:
        y = y.astype(int)
except:
    from sklearn.datasets import load_diabetes as load_diabetes_reg
    diabetes_reg = load_diabetes_reg()
    X = diabetes_reg.data
    y = (diabetes_reg.target > np.median(diabetes_reg.target)).astype(int)

datasets_info.append({
    'name': 'Diabetes',
    'samples': len(X),
    'features': X.shape[1],
    'class_0': sum(y == 0),
    'class_1': sum(y == 1),
    'balance': f"{sum(y == 0) / len(y) * 100:.1f}% / {sum(y == 1) / len(y) * 100:.1f}%"
})

# Heart
print("[4/5] Heart Dataset...")
try:
    heart = fetch_openml('heart-statlog', version=1, as_frame=False, parser='auto')
    X, y = heart.data, heart.target.astype(int)
    if len(np.unique(y)) > 2:
        y = (y > 0).astype(int)
except:
    import pandas as pd
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
               'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
    df = pd.read_csv(url, names=columns, na_values='?')
    df = df.dropna()
    X = df.drop('target', axis=1).values
    y = (df['target'].values > 0).astype(int)

datasets_info.append({
    'name': 'Heart',
    'samples': len(X),
    'features': X.shape[1],
    'class_0': sum(y == 0),
    'class_1': sum(y == 1),
    'balance': f"{sum(y == 0) / len(y) * 100:.1f}% / {sum(y == 1) / len(y) * 100:.1f}%"
})

# Breast Cancer
print("[5/5] Breast Cancer Dataset...")
data = load_breast_cancer()
X, y = data.data, data.target
datasets_info.append({
    'name': 'Breast Cancer',
    'samples': len(X),
    'features': X.shape[1],
    'class_0': sum(y == 0),
    'class_1': sum(y == 1),
    'balance': f"{sum(y == 0) / len(y) * 100:.1f}% / {sum(y == 1) / len(y) * 100:.1f}%"
})

# Print table
print("\n" + "="*80)
print("SUMMARY TABLE")
print("="*80)
print(f"\n{'Dataset':<15} {'Samples':<10} {'Features':<10} {'Class 0':<10} {'Class 1':<10} {'Balance':<15}")
print("-"*70)

for d in datasets_info:
    print(f"{d['name']:<15} {d['samples']:<10} {d['features']:<10} {d['class_0']:<10} {d['class_1']:<10} {d['balance']:<15}")

print("-"*70)

# Totals
total_samples = sum(d['samples'] for d in datasets_info)
print(f"\n{'TOTAL':<15} {total_samples:<10}")

print("\n" + "="*80)
