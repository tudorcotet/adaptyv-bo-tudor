import numpy as np
from typing import Tuple
from sklearn.ensemble import RandomForestRegressor
from config.optimization import SurrogateConfig
from surrogates.base import BaseSurrogate

class RandomForestSurrogate(BaseSurrogate):
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.model = RandomForestRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            random_state=config.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        self.model.fit(X, y)
        train_score = self.model.score(X, y)
        return train_score, train_score  # Return the same score for consistency with other surrogates

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        predictions = np.array([tree.predict(X) for tree in self.model.estimators_])
        mean = np.mean(predictions, axis=0)
        std = np.std(predictions, axis=0)
        return mean, std

    #Add better batching
    def batch_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.predict(X)
