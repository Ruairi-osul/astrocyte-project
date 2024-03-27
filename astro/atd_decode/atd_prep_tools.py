import pandas as pd
import numpy as np
from typing import Callable, Any
from abc import ABC, abstractmethod


class ATDTimeSubsetter:
    def __init__(
        self,
        pre_masker: Callable[[np.ndarray], np.ndarray],
        post_masker: Callable[[np.ndarray], np.ndarray],
        time_col: str = "aligned_time",
    ):
        self.time_col = time_col
        self.pre_masker = pre_masker
        self.post_masker = post_masker

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        pre_mask = self.pre_masker(df_traces[self.time_col].values)
        post_mask = self.post_masker(df_traces[self.time_col].values)

        mask = np.logical_or(pre_mask, post_mask)

        return df_traces.loc[mask, :]


class BaseATDGetter(ABC):
    def __init__(
        self, time_col: str = "aligned_time", event_idx_col: str | None = "event_idx"
    ):
        self.time_col = time_col
        self.event_idx_col = event_idx_col

        self._meta_cols = (
            [self.time_col]
            if self.event_idx_col is None
            else [self.time_col, self.event_idx_col]
        )

    @abstractmethod
    def __call__(self, df_traces: pd.DataFrame) -> Any:
        ...


class ATDDataGetter(BaseATDGetter):
    def __call__(self, df_traces: pd.DataFrame) -> np.ndarray:
        df_neurons = df_traces.drop(columns=self._meta_cols)
        self.neuron_mapping_ = df_neurons.columns
        return df_neurons.values


class ATDTargetGetter(BaseATDGetter):
    def __init__(
        self,
        pre_masker: Callable[[np.ndarray], np.ndarray],
        post_masker: Callable[[np.ndarray], np.ndarray],
        time_col: str = "aligned_time",
        event_idx_col: str | None = "event_idx",
    ):
        super().__init__(time_col=time_col, event_idx_col=event_idx_col)
        self.pre_masker = pre_masker
        self.post_masker = post_masker


class ATDEventGetter(BaseATDGetter):
    def __call__(self, df_traces: pd.DataFrame) -> np.ndarray:
        return df_traces[self.event_idx_col].values


class ATDTargeGetter(BaseATDGetter):
    def __init__(
        self,
        pre_masker: Callable[[np.ndarray], np.ndarray],
        post_masker: Callable[[np.ndarray], np.ndarray],
        time_col: str = "aligned_time",
        event_idx_col: str | None = "event_idx",
    ):
        super().__init__(time_col=time_col, event_idx_col=event_idx_col)
        self.pre_masker = pre_masker
        self.post_masker = post_masker

    def __call__(self, df_traces: pd.DataFrame) -> np.ndarray:
        is_pre = self.pre_masker(df_traces[self.time_col].values)
        is_post = self.post_masker(df_traces[self.time_col].values)
        target = np.select([is_pre, is_post], [0, 1], default=np.nan)

        return target


def latency_mask_factory(
    t_min: float, t_max: float
) -> Callable[[np.ndarray], np.ndarray]:
    def latency_mask(arr: np.ndarray) -> np.ndarray:
        return np.logical_and(arr >= t_min, arr < t_max)

    return latency_mask
