from abc import ABC, abstractmethod
import torch
from typing import Any
from config.optimization import SurrogateConfig

class BaseSurrogate(ABC):
    """
    Abstract base class for surrogate models in optimization.

    This class defines the interface for surrogate models used in optimization processes.
    Concrete implementations should inherit from this class and implement the abstract methods.

    Attributes:
        config (OptimizationConfig): Configuration for the optimization process.
        device (torch.device): The device (CPU or GPU) on which the model will run.
    """

    def __init__(self, config: SurrogateConfig):
        """
        Initialize the BaseSurrogate.

        Args:
            config (OptimizationConfig): Configuration for the optimization process.
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() and config.use_gpu else "cpu")

    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        """
        Fit the surrogate model to the given data.

        Args:
            X: Input features.
            y: Target values.
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """
        Make predictions using the surrogate model.

        Args:
            X: Input features to make predictions for.

        Returns:
            Predictions for the given input.
        """
        pass

    @abstractmethod
    def batch_predict(self, X: Any, batch_size: int = 1000) -> Any:
        """
        Make batch predictions using the surrogate model.

        Args:
            X: Input features to make predictions for.
            batch_size: Size of batches to use for prediction. Defaults to 1000.

        Returns:
            Batch predictions for the given input.
        """
        pass
