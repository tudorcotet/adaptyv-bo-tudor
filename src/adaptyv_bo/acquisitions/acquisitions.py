import torch
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

    def acquire(self, mu, sigma, best_f):
        """
        Compute the UCB scores for candidate points.

        Args:
            mu (torch.Tensor): Mean predictions for the candidates.
            sigma (torch.Tensor): Standard deviation of predictions for the candidates.
            best_f (float): The current best observed value (not used in UCB).

        Returns:
            torch.Tensor: UCB scores for each candidate point.
        """
        ucb_scores = mu + self.beta * torch.sqrt(sigma)
        return ucb_scores

class ExpectedImprovementAcquisition(BaseAcquisition):
    """
    Expected Improvement (EI) acquisition function.

    This class implements the EI acquisition strategy, which selects candidates
    that are expected to improve upon the current best observed value.
    """

    def acquire(self, mu, sigma, best_f):
        """
        Acquire the best candidate using the EI strategy.

        Args:
            mu: Mean predictions for the candidates.
            sigma: Standard deviation of predictions for the candidates.
            best_f: The current best observed value.

        Returns:
            A list containing the index of the best candidate according to the EI score.
        """
        z = (mu - best_f) / torch.sqrt(sigma)
        ei = (mu - best_f) * torch.normal.cdf(z) + torch.sqrt(sigma) * torch.normal.pdf(z)
        best_idx = torch.argmax(ei)
        return [best_idx]
