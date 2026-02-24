"""Natural Evolution Strategies (NES) Attack - Improved"""
import numpy as np
from tqdm import tqdm


class NESAttack:
    """
    NES Attack - Score-based attack using evolutionary strategies
    """
    
    def __init__(self, model, max_iterations=300, max_queries=10000, 
                 sigma=0.1, learning_rate=0.1, n_samples=100, verbose=True):
        self.model = model
        self.max_iterations = max_iterations
        self.max_queries = max_queries
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.n_samples = n_samples
        self.verbose = verbose
        self.query_count = 0
    
    def _get_proba(self, x):
        """Get prediction probabilities"""
        self.query_count += 1
        return self.model.predict_proba(x.reshape(1, -1))[0]
    
    def _get_loss(self, x, y_true):
        """Loss function: negative confidence in true class"""
        probs = self._get_proba(x)
        # We want to MINIMIZE confidence in true class
        return probs[y_true]
    
    def _estimate_gradient(self, x, y_true):
        """Estimate gradient using antithetic sampling"""
        gradient = np.zeros_like(x)
        
        # Antithetic sampling for variance reduction
        for _ in range(self.n_samples // 2):
            # Sample noise
            noise = np.random.randn(*x.shape)
            
            # Evaluate at +/- noise
            loss_plus = self._get_loss(x + self.sigma * noise, y_true)
            loss_minus = self._get_loss(x - self.sigma * noise, y_true)
            
            # Gradient contribution (we want to DECREASE confidence)
            gradient += (loss_plus - loss_minus) * noise
        
        # Normalize
        gradient = gradient / (self.n_samples * self.sigma)
        return gradient
    
    def attack(self, x_orig, y_true):
        """
        Execute NES attack
        
        Returns:
            x_adv, success, queries, perturbation
        """
        self.query_count = 0
        x_adv = x_orig.copy()
        best_x = x_orig.copy()
        best_conf = 1.0
        current_lr = self.learning_rate
        
        iterator = range(self.max_iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc="NES Attack", leave=False)
        
        for i in iterator:
            if self.query_count >= self.max_queries:
                break
            
            # Estimate gradient
            grad = self._estimate_gradient(x_adv, y_true)
            
            # Update (gradient DESCENT to minimize confidence in true class)
            x_adv = x_adv - current_lr * grad
            
            # Check current state
            probs = self._get_proba(x_adv)
            pred_label = np.argmax(probs)
            conf_true_class = probs[y_true]
            
            # Track best (lowest confidence in true class)
            if conf_true_class < best_conf:
                best_conf = conf_true_class
                best_x = x_adv.copy()
            
            # Early stop if successful
            if pred_label != y_true:
                x_adv = best_x
                break
            
            # Adaptive learning rate
            if i % 50 == 0 and i > 0:
                current_lr *= 0.9
        
        # Final check
        final_probs = self._get_proba(best_x)
        success = np.argmax(final_probs) != y_true
        perturbation = np.linalg.norm(best_x - x_orig)
        
        return best_x, success, self.query_count, perturbation


if __name__ == "__main__":
    print("NES Attack module loaded successfully")
