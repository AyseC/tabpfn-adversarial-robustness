import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class GBDTWrapper:
    """Wrapper for GBDT models"""
    
    def __init__(self, model_type="xgboost", random_state=42):
        self.model_type = model_type
        self.random_state = random_state
        
        if model_type == "xgboost":
            self.model = XGBClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=random_state
            )
        elif model_type == "lightgbm":
            self.model = LGBMClassifier(
                max_depth=6,
                learning_rate=0.1,
                n_estimators=100,
                random_state=random_state,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.is_fitted = False
    
    def fit(self, X, y):
        """Fit model"""
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
    
    X, y = make_classification(n_samples=200, n_features=10, n_classes=2, random_state=42)
    
    # Test XGBoost
    print("Testing XGBoost...")
    xgb = GBDTWrapper(model_type="xgboost")
    xgb.fit(X[:150], y[:150])
    acc = xgb.score(X[150:], y[150:])
    print(f"✓ XGBoost works! Accuracy: {acc:.3f}")
    
    # Test LightGBM
    print("\nTesting LightGBM...")
    lgbm = GBDTWrapper(model_type="lightgbm")
    lgbm.fit(X[:150], y[:150])
    acc = lgbm.score(X[150:], y[150:])
    print(f"✓ LightGBM works! Accuracy: {acc:.3f}")
