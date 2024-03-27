from .config import ATDPrepConfig
import pandas as pd
import numpy as np


class ATDPrepper:
    def __init__(self, adt_config: ATDPrepConfig):
        self.adt_config = adt_config

    def subset_time(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        if self.adt_config.time_subsetter is not None:
            df_traces = self.adt_config.time_subsetter(df_traces)
        return df_traces

    def get_target(self, df_traces: pd.DataFrame) -> np.ndarray:
        return self.adt_config.target_getter(df_traces)

    def get_data(self, df_traces: pd.DataFrame) -> np.ndarray:
        return self.adt_config.data_getter(df_traces)

    def get_group(self, df_traces: pd.DataFrame) -> np.ndarray | None:
        if self.adt_config.group_getter is not None:
            return self.adt_config.group_getter(df_traces)
        return None

    def __call__(
        self, df_traces: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, (np.ndarray | None)]:
        df_traces = self.subset_time(df_traces)
        data = self.get_data(df_traces)
        target = self.get_target(df_traces)
        group = self.get_group(df_traces)
        return data, target, group
