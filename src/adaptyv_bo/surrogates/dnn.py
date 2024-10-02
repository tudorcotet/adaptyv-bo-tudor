import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple
from config.optimization import SurrogateConfig
from surrogates.base import BaseSurrogate
from surrogates.loss import BaseLoss, bt_loss

class DropoutLinear(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout_rate: float):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.dropout(x)
        return self.linear2(x)

class DropoutLinearSurrogate(BaseSurrogate):
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.config = config
        self.device = torch.device(config.device)
        self.loss_fn = BaseLoss(config)
        self.model = None  # We'll initialize the model in the fit method

    def _initialize_model(self, input_dim):
        self.model = DropoutLinear(
            input_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            dropout_rate=self.config.dropout_rate
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

            loss.backward()
            self.optimizer.step()
        
        final_loss = loss.item()
        return final_loss, final_loss

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = torch.FloatTensor(X).to(self.device)
        self.model.train()  # Enable dropout during prediction
        with torch.no_grad():
            predictions = []
            for _ in range(self.config.mc_samples):
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