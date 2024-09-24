from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseTracker(ABC):
    @abstractmethod
    def start_run(self, run_name: str):
        pass

    @abstractmethod
    def end_run(self):
        pass

    @abstractmethod
    def log_params(self, params: Dict[str, Any]):
        pass

    @abstractmethod
    def log_metric(self, key: str, value: float, step: int = None):
        pass

    @abstractmethod
    def log_artifact(self, local_path: str):
        pass

    @abstractmethod
    def log_figure(self, figure, filename: str):
        pass

    @abstractmethod
    def __enter__(self):
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    @abstractmethod
    def log_model_summary(self, model_name: str, model_summary: str):
        pass

    @abstractmethod
    def log_model_graph(self, model_name: str, model_graph: str):
        pass

    @abstractmethod
    def log_model_weights(self, model_name: str, model_weights: Dict[str, Any]):
        pass

