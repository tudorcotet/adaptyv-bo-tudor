from abc import ABC, abstractmethod
from typing import List
from config.optimization import OptimizationConfig

class BaseGenerator(ABC):
    def __init__(self, config: OptimizationConfig):
        self.config = config

    @abstractmethod
    def generate(self, n_candidates: int) -> List[str]:
        pass

    @abstractmethod
    def generate_all(self) -> List[str]:
        pass

    @abstractmethod
    def update_sequences(self, new_sequences: List[str]) -> None:
        pass