import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
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

class MCDropoutNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = F.relu(self.dropout(self.fc1(x)))
        x = F.relu(self.dropout(self.fc2(x)))
        return self.fc3(x)

class DeepEnsembleSurrogate(BaseSurrogate):
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.config = config
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.n_estimators = config.n_estimators
        self.batch_size = min(config.batch_size, 128)
        self.loss_fn = BaseLoss(config)
        self.models: List[nn.Module] = []
        self.optimizers: List[torch.optim.Optimizer] = []

    def _initialize_models(self, input_dim):
        self.models = [MCDropoutNet(input_dim, self.hidden_dim, self.output_dim, self.config.dropout_rate) 
                       for _ in range(self.n_estimators)]
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=self.config.learning_rate) 
                           for model in self.models]

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        # Initialize the models with the correct input dimension
        if not self.models:
            input_dim = X.shape[1]
            self._initialize_models(input_dim)
        
        dataset = SequenceDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for model in self.models:
            model.train()
        
        for epoch in range(self.config.n_epochs):
            total_loss = 0
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten the input
                for model, optimizer in zip(self.models, self.optimizers):
                    optimizer.zero_grad()
                    output = model(batch_X)
                    
                    if self.config.loss_fn == "bt":
                        loss = bt_loss(output.squeeze(), y, beta=self.config.bt_beta, noise=self.config.bt_noise)
                    else:
                        loss = self.loss_fn(output.squeeze(), y)        

                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
            
            avg_loss = total_loss / (len(dataloader) * self.n_estimators)
            
        return avg_loss, avg_loss

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dataset = SequenceDataset(X)
        dataloader = DataLoader(dataset, batch_size=self.batch_size)
        
        for model in self.models:
            model.eval()
        
        predictions = []
        for model in self.models:
            batch_predictions = []
            for batch_X in dataloader:
                batch_X = batch_X.view(batch_X.size(0), -1)  # Flatten the input
                with torch.no_grad():
                    output = model(batch_X)
                batch_predictions.append(output.numpy())
            predictions.append(np.concatenate(batch_predictions, axis=0))
        
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