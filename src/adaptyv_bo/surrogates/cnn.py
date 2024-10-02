import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
from config.optimization import SurrogateConfig
from surrogates.base import BaseSurrogate
from surrogates.loss import BaseLoss, bt_loss
from torch.utils.data import Dataset, DataLoader

class SequenceDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        if self.y is not None:
            y = torch.FloatTensor([self.y[idx]])
            return X, y
        return X

class MinimalCNN1D(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, vocab_size: int, seq_len: Optional[int], dropout_rate: float = 0.1):
        super(MinimalCNN1D, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.seq_len = seq_len

        self.embedding = nn.Embedding(vocab_size, 8)
        self.conv = nn.Conv1d(8, 16, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(16, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.long()
            x = self.embedding(x).transpose(1, 2)
        elif x.dim() != 3:
            raise ValueError("Input must be 2D (one-hot) or 3D (pre-computed features)")
        
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        x = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        x = self.fc(x)
        return x

class CNN1DDropoutSurrogate(BaseSurrogate):
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.config = config
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.vocab_size = config.vocab_size
        self.seq_len = config.seq_len
        self.mc_samples = config.mc_samples
        self.batch_size = min(config.batch_size, 128)
        self.loss_fn = BaseLoss(config)
        self.model = None

    def _initialize_model(self, input_dim):
        self.model = MinimalCNN1D(
            input_dim=input_dim,
            output_dim=self.output_dim,
            vocab_size=self.vocab_size,
            seq_len=self.seq_len,
            dropout_rate=self.config.dropout_rate
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        if self.model is None:
            input_dim = X.shape[1] if X.ndim == 2 else X.shape[1]
            self.seq_len = X.shape[2] if X.ndim == 3 else None
            self._initialize_model(input_dim)
        
        dataset = SequenceDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.config.n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                output = self.model(batch_X)

                if self.config.loss_fn == "bt":
                 loss = bt_loss(output.squeeze(), y, beta=self.config.bt_beta, noise=self.config.bt_noise)
                else:
                 loss = self.loss_fn(output.squeeze(), y)                
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            
        return avg_loss, avg_loss

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dataset = SequenceDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.mc_samples):
                batch_predictions = []
                for batch_X in dataloader:
                    output = self.model(batch_X)
                    batch_predictions.append(output.numpy())
                predictions.append(np.concatenate(batch_predictions, axis=0))
        
        predictions = np.array(predictions)
        mean = np.mean(predictions, axis=0).squeeze()
        std = np.std(predictions, axis=0).squeeze()
        return mean, std

    def batch_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict(X)  # Use the same method for both single and batch predictions