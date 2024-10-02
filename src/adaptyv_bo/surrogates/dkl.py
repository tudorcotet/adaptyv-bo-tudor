import torch
import torch.nn as nn
import gpytorch
import numpy as np
from typing import Tuple
from config.optimization import SurrogateConfig
from surrogates.base import BaseSurrogate

class DeepKernelGP(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, input_dim, hidden_dim):
        super(DeepKernelGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        mean_x = self.mean_module(features)
        covar_x = self.covar_module(features)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class DeepKernelGPSurrogate(BaseSurrogate):
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.model = None
        self.likelihood = None

    def _initialize_model(self, input_dim):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = DeepKernelGP(None, None, self.likelihood, input_dim, self.hidden_dim)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        train_x = torch.FloatTensor(X)
        train_y = torch.FloatTensor(y)

        # Initialize the model with the correct input dimension
        if self.model is None:
            self._initialize_model(X.shape[1])

        # Set the train data
        self.model.set_train_data(train_x, train_y, strict=False)

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam([
            {'params': self.model.feature_extractor.parameters()},
            {'params': self.model.covar_module.parameters()},
            {'params': self.model.mean_module.parameters()},
            {'params': self.model.likelihood.parameters()},
        ], lr=self.config.learning_rate)

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        for _ in range(self.config.n_epochs):
            optimizer.zero_grad()
            output = self.model(train_x)
            loss = -mll(output, train_y)
            loss.backward()
            optimizer.step()

        return loss.item(), loss.item()

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        test_x = torch.FloatTensor(X)
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(test_x))

        mean = observed_pred.mean.numpy()
        std = observed_pred.stddev.numpy()
        return mean, std
    
    def batch_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = self.config.batch_size
        num_samples = X.shape[0]
        means = []
        stds = []
        
        for i in range(0, num_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_mean, batch_std = self.predict(batch_X)
            means.append(batch_mean)