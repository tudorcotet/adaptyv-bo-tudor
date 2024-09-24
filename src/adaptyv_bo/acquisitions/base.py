from abc import ABC, abstractmethod
from typing import Any, List

import numpy as np
import torch

from config.optimization import AcquisitionConfig


class BaseAcquisition(ABC):
    """
    Abstract base class for acquisition functions in Bayesian optimization.

    This class defines the interface for acquisition functions used to select
    candidates for evaluation in the optimization process.

    Attributes:
        config (OptimizationConfig): Configuration object containing optimization parameters.
    """

    def __init__(self, config: AcquisitionConfig):
        """
        Initialize the BaseAcquisition object.

        Args:
            config (AcquisitionConfig): Configuration object containing optimization parameters.
        """
        self.config = config

    @abstractmethod
    def acquire(self, surrogate: Any, encoded_candidates: List[np.ndarray]) -> torch.Tensor:
        """
        Acquire candidates for evaluation based on the surrogate model.

        Args:
            surrogate (Any): The surrogate model used for predictions.
            encoded_candidates (List[np.ndarray]): List of encoded candidate sequences.

        Returns:
            torch.Tensor: Acquisition scores for each candidate.
        """
        pass
