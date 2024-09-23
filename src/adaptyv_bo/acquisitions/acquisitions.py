import torch
import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from config.optimization import OptimizationConfig
from acquisitions.base import BaseAcquisition

class UCBAcquisition(BaseAcquisition):
    """
    Upper Confidence Bound (UCB) acquisition function.

    This class implements the UCB acquisition strategy, which balances
    exploration and exploitation by selecting candidates with high predicted
    mean and high uncertainty.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize the UCB acquisition function.

        Args:
            config (OptimizationConfig): Configuration object containing
                optimization parameters, including the beta value for UCB.
        """
        super().__init__(config)
        self.beta = config.beta

    def acquire(self, mu: np.ndarray, sigma: np.ndarray, best_f: float) -> np.ndarray:
        """
        Compute the UCB scores for candidate points.

        Args:
            mu (np.ndarray): Mean predictions for the candidates.
            sigma (np.ndarray): Standard deviation of predictions for the candidates.
            best_f (float): The current best observed value (not used in UCB).

        Returns:
            np.ndarray: UCB scores for each candidate point.
        """
        return mu + self.beta * np.sqrt(sigma)

class ExpectedImprovementAcquisition(BaseAcquisition):
    """
    Expected Improvement (EI) acquisition function.

    This class implements the EI acquisition strategy, which selects candidates
    that are expected to improve upon the current best observed value.
    """

    def acquire(self, mu: np.ndarray, sigma: np.ndarray, best_f: float) -> np.ndarray:
        """
        Compute the Expected Improvement scores for candidate points.

        Args:
            mu (np.ndarray): Mean predictions for the candidates.
            sigma (np.ndarray): Standard deviation of predictions for the candidates.
            best_f (float): The current best observed value.

        Returns:
            np.ndarray: EI scores for each candidate point.
        """
        with np.errstate(divide='ignore'):
            imp = mu - best_f
            Z = imp / sigma
            ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            ei[sigma == 0.0] = 0.0

        return ei

class ProbabilityOfImprovementAcquisition(BaseAcquisition):
    """
    Probability of Improvement (PI) acquisition function.

    This class implements the PI acquisition strategy, which selects candidates
    based on their probability of improving upon the current best observed value.
    """

    def __init__(self, config: OptimizationConfig):
        """
        Initialize the PI acquisition function.

        Args:
            config (OptimizationConfig): Configuration object containing
                optimization parameters, including the xi value for PI.
        """
        super().__init__(config)
        self.xi = config.xi

    def acquire(self, mu: np.ndarray, sigma: np.ndarray, best_f: float) -> np.ndarray:
        """
        Compute the Probability of Improvement scores for candidate points.

        Args:
            mu (np.ndarray): Mean predictions for the candidates.
            sigma (np.ndarray): Standard deviation of predictions for the candidates.
            best_f (float): The current best observed value.

        Returns:
            np.ndarray: PI scores for each candidate point.
        """
        with np.errstate(divide='ignore'):
            z = (mu - best_f - self.xi) / sigma
            pi = norm.cdf(z)

        return pi
