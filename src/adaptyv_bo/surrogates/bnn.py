import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from config.optimization import SurrogateConfig
from surrogates.base import BaseSurrogate
from surrogates.loss import BaseLoss, bt_loss

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_mu, prior_sigma):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(prior_mu, prior_sigma))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-3, 0.1))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(prior_mu, prior_sigma))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).normal_(-3, 0.1))
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma

    def forward(self, x):
        weight = self.weight_mu + torch.log1p(torch.exp(self.weight_rho)) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + torch.log1p(torch.exp(self.bias_rho)) * torch.randn_like(self.bias_mu)
        return nn.functional.linear(x, weight, bias)

    def kl_loss(self):
        kl = self._kl_loss(self.weight_mu, self.weight_rho) + self._kl_loss(self.bias_mu, self.bias_rho)
        return kl

    def _kl_loss(self, mu, rho):
        sigma = torch.log1p(torch.exp(rho))
        return 0.5 * torch.sum(
            ((mu - self.prior_mu) ** 2) / (self.prior_sigma ** 2)
            + (sigma ** 2) / (self.prior_sigma ** 2)
            - 2 * torch.log(sigma / self.prior_sigma)
            - 1
        )

class BNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers, prior_mu, prior_sigma):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(BayesianLinear(input_dim, hidden_dim, prior_mu, prior_sigma))
        for _ in range(n_layers - 2):
            self.layers.append(BayesianLinear(hidden_dim, hidden_dim, prior_mu, prior_sigma))
        self.layers.append(BayesianLinear(hidden_dim, output_dim, prior_mu, prior_sigma))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = nn.functional.relu(layer(x))
        return self.layers[-1](x)

    def kl_loss(self):
        return sum(layer.kl_loss() for layer in self.layers)

class BNNSurrogate(BaseSurrogate):
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.config = config
        self.device = torch.device(config.device)
        self.loss_fn = BaseLoss(config)
        self.model = None  

    def _initialize_model(self, input_dim):
        self.model = BNN(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            n_layers=self.config.n_layers,
            prior_mu=self.config.prior_mu,
            prior_sigma=self.config.prior_sigma
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        # Initialize the model with the correct input dimension
        if self.model is None:
            self._initialize_model(X.shape[1])
        
        for epoch in range(self.config.n_epochs):
            self.optimizer.zero_grad()
            output = self.model(X)
            if self.config.loss_fn == "bt":
                loss = bt_loss(output.squeeze(), y, beta=self.config.bt_beta, noise=self.config.bt_noise)
            else:
                loss = self.loss_fn(output.squeeze(), y)
            kl_loss = self.model.kl_loss()
            total_loss = loss + kl_loss / len(X)
            total_loss.backward()
            self.optimizer.step()
        
        final_loss = total_loss.item()
        return final_loss, final_loss

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()  # Set the model to evaluation mode
        with torch.no_grad():
            predictions = []
            for _ in range(self.config.num_monte_carlo):
                output = self.model(X)
                predictions.append(output.cpu().numpy())
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0).squeeze()
        std = np.std(predictions, axis=0).squeeze()
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
            stds.append(batch_std)
        
        mean = np.concatenate(means, axis=0)
        std = np.concatenate(stds, axis=0)
        return mean, std