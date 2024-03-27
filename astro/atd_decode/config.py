from astro.load import GroupSessionData, DataConfig
from astro.preprocess.config import PreprocessConfig
from sklearn.pipeline import Pipeline
from sklearn.model_selection import BaseCrossValidator
from dataclasses import dataclass
from typing import Callable
import pandas as pd
import numpy as np


@dataclass
class ATDPrepConfig:
    data_getter: Callable[[GroupSessionData], pd.DataFrame]
    target_getter: Callable[[GroupSessionData], np.ndarray]
    time_subsetter: Callable[[pd.DataFrame], pd.DataFrame]
    group_getter: Callable[[GroupSessionData], np.ndarray] | None = None


@dataclass
class ModelConfig:
    model: Pipeline
    cv: BaseCrossValidator
    scoring: str | None = None
    n_jobs: int = 1
    return_estimator: bool = True
    return_train_score: bool = False


@dataclass
class DecodeConfig:
    data_config: DataConfig
    preprocess_config: PreprocessConfig
    model_config: ModelConfig
