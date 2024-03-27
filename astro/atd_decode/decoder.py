from .config import ModelConfig
import numpy as np
from sklearn.model_selection import cross_validate


class DecoderCV:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config

    def cross_validate(
        self, X: np.ndarray, y: np.ndarray, groups: np.ndarray | None = None
    ) -> dict:
        cv_results = cross_validate(
            estimator=self.model_config.model,
            X=X,
            y=y,
            groups=groups,
            cv=self.model_config.cv,
            scoring=self.model_config.scoring,
            n_jobs=self.model_config.n_jobs,
            return_estimator=self.model_config.return_estimator,
            return_train_score=self.model_config.return_train_score,
        )
        return cv_results
