import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple
from config.optimization import SurrogateConfig
from surrogates.base import BaseSurrogate
from torch.utils.data import DataLoader, TensorDataset

class CNN1DDropout(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_rate: float = 0.5):
        super(CNN1DDropout, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension
        x = F.relu(self.conv1(x))
        x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNN1DDropoutSurrogate(BaseSurrogate):
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.model = CNN1DDropout(self.input_dim, self.output_dim, config.dropout_rate).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.learning_rate)
        self.mc_samples = config.mc_samples
        self.batch_size = config.batch_size

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        X = torch.FloatTensor(X).to(self.device)
        y = torch.FloatTensor(y).to(self.device)
        
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        losses = []
        for epoch in range(self.config.n_epochs):
            epoch_losses = []
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)
                loss = F.mse_loss(output.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(np.mean(epoch_losses))
        
        return np.mean(losses), losses[-1]

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict_batch(X[np.newaxis, :])[0]

    def predict_batch(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = torch.FloatTensor(X).to(self.device)
        self.model.train()  # Enable dropout during prediction
        
        dataloader = DataLoader(X, batch_size=self.batch_size)
        
        with torch.no_grad():
            predictions = []
            for _ in range(self.mc_samples):
                batch_predictions = []
                for batch in dataloader:
                    batch_predictions.append(self.model(batch).cpu().numpy())
                predictions.append(np.concatenate(batch_predictions, axis=0))
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std
