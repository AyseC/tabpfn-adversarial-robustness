from dataclasses import dataclass, field
from typing import List, Dict, Optional
import os


@dataclass
class DataConfig:
    """Data configuration"""
    datasets: List[str] = field(default_factory=lambda: ["wine", "breast_cancer"])
    test_size: float = 0.2
    random_state: int = 42
    normalize: bool = True
    data_dir: str = "./data/offline"


@dataclass
class ModelConfig:
    """Model configuration"""
    device: str = "cpu"
    random_state: int = 42


@dataclass
class AttackConfig:
    """Attack configuration"""
    attack_types: List[str] = field(default_factory=lambda: ["boundary"])
    epsilon: float = 0.3
    max_iterations: int = 500
    max_queries: int = 5000


@dataclass
class ExperimentConfig:
    """Main experiment configuration"""
    exp_name: str = "first_experiment"
    output_dir: str = "./results"
    seed: int = 42
    device: str = "cpu"
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    attack: AttackConfig = field(default_factory=AttackConfig)
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)


if __name__ == "__main__":
    config = ExperimentConfig()
    print(f"Config created: {config.exp_name}")
    print(f"Datasets: {config.data.datasets}")
