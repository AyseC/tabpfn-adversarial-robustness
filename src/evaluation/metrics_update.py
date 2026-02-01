# metrics.py dosyasına eklenecek değişiklikler

# robustness_score fonksiyonundan sonra ekle:

@staticmethod
def adversarial_accuracy(clean_acc: float, asr: float) -> float:
    """
    Adversarial Accuracy - Literature standard metric
    
    Args:
        clean_acc: Clean accuracy
        asr: Attack success rate
    
    Returns:
        Adversarial accuracy score
    """
    return clean_acc * (1 - asr)

# compute_all fonksiyonuna ekle:
# metrics['adversarial_accuracy'] = RobustnessMetrics.adversarial_accuracy(
#     metrics['clean_accuracy'],
#     metrics['attack_success_rate']
# )
