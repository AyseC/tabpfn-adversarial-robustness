import numpy as np
import torch
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion


class TabPFNWrapper:
    """Wrapper for TabPFN v2"""
    
    def __init__(self, device="cpu", random_state=42, version="v2"):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = "cpu"
        
        self.device = device
        self.random_state = random_state
        self.version = version
        self.model = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit TabPFN model"""
        if X.shape[0] > 10000:
            print(f"Warning: TabPFN v2 works best with <10000 samples. Got {X.shape[0]}")
        if X.shape[1] > 500:
            print(f"Warning: TabPFN v2 works best with <500 features. Got {X.shape[1]}")
        
        # TabPFN v2
        self.model = TabPFNClassifier.create_default_for_version(ModelVersion.V2)
        self.model.random_state = self.random_state
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict labels"""
        if not self.is_fitted:
            raise ValueError("Model not fitted!")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model not fitted!")
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Compute accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)


if __name__ == "__main__":
    from sklearn.datasets import make_classification, load_iris
    
    print("Testing TabPFN v2...")
    
    # Binary test
    print("\n1. Binary classification:")
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    model = TabPFNWrapper(device="cpu")
    model.fit(X[:150], y[:150])
    acc = model.score(X[150:], y[150:])
    print(f"   ✓ Binary accuracy: {acc:.3f}")
    
    # Multi-class test
    print("\n2. Multi-class classification (Iris - 3 classes):")
    data = load_iris()
    X, y = data.data, data.target
    model = TabPFNWrapper(device="cpu")
    model.fit(X[:120], y[:120])
    acc = model.score(X[120:], y[120:])
    print(f"   ✓ Multi-class accuracy: {acc:.3f}")
    
    print("\n✓ TabPFN v2 works!")
