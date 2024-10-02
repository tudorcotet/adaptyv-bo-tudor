import xgboost as xgb
import numpy as np
from typing import Tuple
from config.optimization import SurrogateConfig
from surrogates.base import BaseSurrogate

class XGBoostSurrogate(BaseSurrogate):
    def __init__(self, config: SurrogateConfig):
        super().__init__(config)
        self.model = xgb.XGBRegressor(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth,
            learning_rate=config.learning_rate,
            subsample=config.subsample,
            colsample_bytree=0.8,
            eta=config.eta,
            gamma=config.gamma,
            min_child_weight=config.min_child_weight,
            random_state=config.random_state
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        self.model.fit(X, y)
        train_score = self.model.score(X, y)
        return train_score, train_score

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.model.predict(X)
        # XGBoost doesn't provide uncertainty estimates by default, so we'll use the variance of tree predictions
        predictions = np.array([tree.predict(xgb.DMatrix(X)) for tree in self.model.get_booster()])
        std = np.std(predictions, axis=0)
        return mean, std
    
    def batch_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        mean = self.model.predict(X)
        # XGBoost doesn't provide uncertainty estimates by default, so we'll use the variance of tree predictions
        predictions = np.array([tree.predict(xgb.DMatrix(X)) for tree in self.model.get_booster()])
        std = np.std(predictions, axis=0)
        return mean, std    