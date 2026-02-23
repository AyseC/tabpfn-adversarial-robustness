"""Evaluation metrics for adversarial robustness"""
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class AttackResult:
    """Single attack result"""
    original_label: int
    predicted_label: int
    adversarial_label: int
    success: bool
    perturbation: float
    queries: int
    original_sample: np.ndarray
    adversarial_sample: np.ndarray


class RobustnessMetrics:
    """Compute robustness metrics"""
    
    @staticmethod
    def clean_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Standard accuracy"""
        return np.mean(y_true == y_pred)
    
    @staticmethod
    def attack_success_rate(results: List[AttackResult]) -> float:
        """Attack success rate (ASR)"""
        if len(results) == 0:
            return 0.0
        successes = sum(1 for r in results if r.success)
        return successes / len(results)
    
    @staticmethod
    def average_perturbation(results: List[AttackResult]) -> float:
        """Average L2 perturbation for successful attacks"""
        successful = [r for r in results if r.success]
        if len(successful) == 0:
            return 0.0
        return np.mean([r.perturbation for r in successful])
    
    @staticmethod
    def average_queries(results: List[AttackResult]) -> float:
        """Average queries for successful attacks"""
        successful = [r for r in results if r.success]
        if len(successful) == 0:
            return 0.0
        return np.mean([r.queries for r in successful])
    
    @staticmethod
    def adversarial_accuracy(clean_acc: float, asr: float) -> float:
        """
        Adversarial Accuracy - Literature standard metric
        
        Measures the proportion of samples that remain correctly 
        classified after adversarial attack.
        
        Args:
            clean_acc: Clean accuracy (before attack)
            asr: Attack success rate
        
        Returns:
            Adversarial accuracy = clean_acc * (1 - asr)
        """
        return 1.0 - asr
    
    @staticmethod
    def robustness_score(clean_acc: float, asr: float, avg_pert: float) -> float:
        """
        Composite Robustness Score (higher = more robust)
        
        Combines three factors:
        - Clean accuracy (40%): Baseline model performance
        - Attack resistance (40%): 1 - ASR
        - Perturbation magnitude (20%): Normalized avg perturbation
        
        Args:
            clean_acc: Clean accuracy
            asr: Attack success rate
            avg_pert: Average L2 perturbation for successful attacks
        
        Returns:
            Weighted robustness score in [0, 1]
        """
        # Normalize perturbation (assume max reasonable is 5.0)
        norm_pert = min(avg_pert / 5.0, 1.0)
        
        # Score: high accuracy + low ASR + high perturbation needed = robust
        score = 0.4 * clean_acc + 0.4 * (1 - asr) + 0.2 * norm_pert
        return score
    
    @staticmethod
    def compute_all(results: List[AttackResult], y_true: np.ndarray,
                   y_pred: np.ndarray) -> Dict[str, float]:
        """Compute all metrics"""
        metrics = {
            'clean_accuracy': RobustnessMetrics.clean_accuracy(y_true, y_pred),
            'attack_success_rate': RobustnessMetrics.attack_success_rate(results),
            'avg_perturbation': RobustnessMetrics.average_perturbation(results),
            'avg_queries': RobustnessMetrics.average_queries(results),
        }
        
        # Primary metric: Adversarial Accuracy (literature standard)
        metrics['adversarial_accuracy'] = RobustnessMetrics.adversarial_accuracy(
            metrics['clean_accuracy'],
            metrics['attack_success_rate']
        )
        
        # Secondary metric: Composite Robustness Score
        metrics['robustness_score'] = RobustnessMetrics.robustness_score(
            metrics['clean_accuracy'],
            metrics['attack_success_rate'],
            metrics['avg_perturbation']
        )
        
        return metrics


if __name__ == "__main__":
    # Test
    print("Testing metrics...")
    
    # Mock data
    results = [
        AttackResult(0, 0, 1, True, 2.5, 300, np.zeros(10), np.ones(10)),
        AttackResult(1, 1, 0, True, 2.0, 250, np.zeros(10), np.ones(10)),
        AttackResult(0, 0, 0, False, 0.0, 100, np.zeros(10), np.ones(10)),
    ]
    
    y_true = np.array([0, 1, 0])
    y_pred = np.array([0, 1, 0])
    
    metrics = RobustnessMetrics.compute_all(results, y_true, y_pred)
    
    print("\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    
    print("\n✓ Metrics module working!")
