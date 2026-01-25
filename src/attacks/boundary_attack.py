"""Boundary Attack implementation"""
import numpy as np
from tqdm import tqdm


class BoundaryAttack:
    """
    Boundary Attack - Decision-based adversarial attack
    Requires only model predictions (no gradients needed)
    """
    
    def __init__(self, model, max_iterations=100, epsilon=0.3, verbose=True):
        self.model = model
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.verbose = verbose
        self.query_count = 0
    
    def _predict_label(self, x):
        """Get predicted label"""
        self.query_count += 1
        return self.model.predict(x.reshape(1, -1))[0]
    
    def attack(self, x_orig, y_true):
        """
        Execute boundary attack on a single sample
        
        Args:
            x_orig: Original sample
            y_true: True label
            
        Returns:
            x_adv: Adversarial example
            success: Whether attack succeeded
            queries: Number of queries used
            perturbation: L2 norm of perturbation
        """
        self.query_count = 0
        
        # Start with random adversarial sample
        x_adv = self._initialize_adversarial(x_orig, y_true)
        
        if x_adv is None:
            perturbation = 0.0
            return x_orig, False, self.query_count, perturbation
        
        # Iteratively move toward original while staying adversarial
        step_size = 0.1
        
        iterator = range(self.max_iterations)
        if self.verbose:
            iterator = tqdm(iterator, desc="Attack", leave=False)
        
        for i in iterator:
            # Random perturbation
            noise = np.random.randn(*x_orig.shape)
            noise = noise / (np.linalg.norm(noise) + 1e-10)
            
            # Perturb current adversarial example
            x_new = x_adv + step_size * noise
            
            # Check if still adversarial
            if self._predict_label(x_new) != y_true:
                x_adv = x_new
                
                # Try to move closer to original
                direction = x_orig - x_adv
                direction = direction / (np.linalg.norm(direction) + 1e-10)
                
                x_closer = x_adv + step_size * direction
                
                if self._predict_label(x_closer) != y_true:
                    x_adv = x_closer
            else:
                # Reduce step size if perturbation failed
                step_size *= 0.9
            
            # Stop if close enough to original
            if np.linalg.norm(x_adv - x_orig) < self.epsilon:
                break
        
        # Final check
        success = self._predict_label(x_adv) != y_true
        perturbation = np.linalg.norm(x_adv - x_orig)
        
        return x_adv, success, self.query_count, perturbation
    
    def _initialize_adversarial(self, x_orig, y_true, max_tries=100):
        """Find initial adversarial example"""
        for _ in range(max_tries):
            # Random sample
            x_random = x_orig + np.random.randn(*x_orig.shape)
            
            if self._predict_label(x_random) != y_true:
                return x_random
        
        return None
