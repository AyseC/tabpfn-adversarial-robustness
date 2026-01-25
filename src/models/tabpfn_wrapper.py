import numpy as np
import torch
from tabpfn import TabPFNClassifier


class TabPFNWrapper:
    """Wrapper for TabPFN"""
    
    def __init__(self, device="cpu", random_state=42):
        if device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, using CPU")
            device = "cpu"
        
        self.device = device
        self.random_state = random_state
        self.model = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit TabPFN model"""
        if X.shape[0] > 1000:
            print(f"Warning: TabPFN works best with <1000 samples. Got {X.shape[0]}")
        if X.shape[1] > 100:
            print(f"Warning: TabPFN works best with <100 features. Got {X.shape[1]}")
        
        # TabPFN yeni versiyonu daha basit
        self.model = TabPFNClassifier(device=self.device)
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
    from sklearn.datasets import make_classification
    
    print("Testing TabPFN...")
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    
    model = TabPFNWrapper(device="cpu")
    model.fit(X[:150], y[:150])
    acc = model.score(X[150:], y[150:])
    print(f"✓ TabPFN works! Accuracy: {acc:.3f}")
