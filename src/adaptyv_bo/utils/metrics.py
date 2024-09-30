from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict

class BaseMetric(ABC):
    @abstractmethod
    def compute(self, fitness_values: List[float], iteration: int) -> float:
        pass

    @abstractmethod
    def aggregate(self, values: List[float]) -> float:
        pass

class MaxFitness(BaseMetric):
    def compute(self, fitness_values: List[float], iteration: int) -> float:
        return max(fitness_values) if fitness_values else 0.0

    def aggregate(self, values: List[float]) -> float:
        return max(values) if values else 0.0

class AverageFitness(BaseMetric):
    def compute(self, fitness_values: List[float], iteration: int) -> float:
        return np.mean(fitness_values) if fitness_values else 0.0

    def aggregate(self, values: List[float]) -> float:
        return np.mean(values) if values else 0.0

class StandardDeviationFitness(BaseMetric):
    def compute(self, fitness_values: List[float], iteration: int) -> float:
        return np.std(fitness_values) if len(fitness_values) > 1 else 0.0

    def aggregate(self, values: List[float]) -> float:
        return np.mean(values) if values else 0.0

class Diversity(BaseMetric):
    def compute(self, fitness_values: List[float], iteration: int) -> float:
        return len(set(fitness_values)) / len(fitness_values) if fitness_values else 0.0

    def aggregate(self, values: List[float]) -> float:
        return np.mean(values) if values else 0.0

class Coverage(BaseMetric):
    def __init__(self, total_space_size: int):
        self.total_space_size = total_space_size

    def compute(self, fitness_values: List[float], iteration: int) -> float:
        return len(set(fitness_values)) / self.total_space_size if self.total_space_size > 0 else 0.0

    def aggregate(self, values: List[float]) -> float:
        return max(values) if values else 0.0

class ExpectedShortfall(BaseMetric):
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compute(self, fitness_values: List[float], iteration: int) -> float:
        if not fitness_values:
            return 0.0
        sorted_values = sorted(fitness_values)
        cutoff = max(1, int(len(sorted_values) * self.alpha))
        return np.mean(sorted_values[:cutoff])

    def aggregate(self, values: List[float]) -> float:
        return np.mean(values) if values else 0.0

class ConditionalValueAtRisk(BaseMetric):
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    def compute(self, fitness_values: List[float], iteration: int) -> float:
        if not fitness_values:
            return 0.0
        sorted_values = sorted(fitness_values, reverse=True)
        cutoff = max(1, int(len(sorted_values) * self.alpha))
        return np.mean(sorted_values[:cutoff])

    def aggregate(self, values: List[float]) -> float:
        return np.mean(values) if values else 0.0

class MetricsTracker:
    def __init__(self, metrics: Dict[str, BaseMetric]):
        self.metrics = metrics
        self.results = {name: [] for name in metrics.keys()}

    def update(self, fitness_values: List[float], iteration: int):
        for name, metric in self.metrics.items():
            value = metric.compute(fitness_values, iteration)
            self.results[name].append(value)

    def get_final_metrics(self) -> Dict[str, float]:
        return {name: metric.aggregate(self.results[name]) for name, metric in self.metrics.items()}
