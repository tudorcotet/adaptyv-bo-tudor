import torch
import gpytorch
import numpy as np
from typing import Tuple
from config.optimization import OptimizationConfig
from surrogates.base import BaseSurrogate

class CustomGPModel(gpytorch.models.ExactGP):
    """
    Custom Gaussian Process model using GPyTorch.

    This model uses a constant mean and a scaled RBF kernel.
    """
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, likelihood: gpytorch.likelihoods.Likelihood):
        super(CustomGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x: torch.Tensor) -> gpytorch.distributions.MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class GPSurrogate(BaseSurrogate):
    """
    Gaussian Process surrogate model.

    This class implements a Gaussian Process surrogate model using GPyTorch.
    It provides methods for fitting the model to data and making predictions.
    """
    def __init__(self, config: OptimizationConfig):
        super().__init__(config)
        self.model = None
        self.likelihood = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the Gaussian Process model to the given data.

        Args:
            X (np.ndarray): Input features.
            y (np.ndarray): Target values.
        """
        X_train = torch.FloatTensor(X).to(self.device)
        y_train = torch.FloatTensor(y).to(self.device)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = CustomGPModel(X_train, y_train, self.likelihood).to(self.device)
        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(self.config.n_training_iter):
            optimizer.zero_grad()
            output = self.model(X_train)
            loss = -mll(output, y_train)
            loss.backward()
            optimizer.step()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using the fitted Gaussian Process model.

        Args:
            X (np.ndarray): Input features to predict on.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted means and variances.
        """
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            X = torch.FloatTensor(X).to(self.device)
            observed_pred = self.likelihood(self.model(X))
        return observed_pred.mean.cpu().numpy(), observed_pred.variance.cpu().numpy()

    def batch_predict(self, X: np.ndarray, batch_size: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make batch predictions using the fitted Gaussian Process model.

        Args:
            X (np.ndarray): Input features to predict on.
            batch_size (int): Size of each batch for prediction.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted means and variances.
        """
        self.model.eval()
        self.likelihood.eval()
        means, variances = [], []
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(0, len(X), batch_size):
                batch = torch.FloatTensor(X[i:i+batch_size]).to(self.device)
                observed_pred = self.likelihood(self.model(batch))
                means.append(observed_pred.mean.cpu().numpy())
                variances.append(observed_pred.variance.cpu().numpy())
        return np.concatenate(means), np.concatenate(variances)
