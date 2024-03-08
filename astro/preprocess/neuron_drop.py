import pandas as pd
import numpy as np


class NeuronDropper:
    def __init__(
        self,
        neuron_to_drop: np.ndarray,
        copy: bool = True,
        to_str: bool = True,
    ):
        self.neuron_to_drop = neuron_to_drop
        if to_str:
            self.neuron_to_drop = self.neuron_to_drop.astype(str)

    def drop_wide(self, df_wide: pd.DataFrame) -> pd.DataFrame:
        return df_wide.drop(columns=self.neuron_to_drop)

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        if self.copy:
            df_traces = df_traces.copy()
        df_traces = self.drop_wide(df_traces)
        return df_traces
