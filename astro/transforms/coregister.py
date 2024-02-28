import pandas as pd
from typing import List


def coregister_wide(df_list: List[pd.DataFrame], not_neuron_cols: List[str] = None):
    if not_neuron_cols is None:
        not_neuron_cols = ["time"]

    neuron_sets = [set(df.columns) - set(not_neuron_cols) for df in df_list]
    common_neurons = set.intersection(*neuron_sets)
    all_cols = not_neuron_cols + list(common_neurons)
    df_list = [df[all_cols] for df in df_list]
    return df_list


def coregister_long(df_list: List[pd.DataFrame], neuron_col: str = "cell_id"):
    neuron_sets = [set(df[neuron_col]) for df in df_list]
    neuron_intersection = set.intersection(*neuron_sets)
    df_list = [df[df[neuron_col].isin(neuron_intersection)] for df in df_list]
    return df_list


def concat_wide(df_list: List[pd.DataFrame]):
    # reorder cols to be the same for all dfs
    cols = df_list[0].columns
    df_list = [df[cols] for df in df_list]
    df = pd.concat(df_list)
    return df


class CoRegistrar:
    def __init__(self, not_neuron_cols: List[str] = None, neuron_col: str = "cell_id"):
        self.not_neuron_cols = not_neuron_cols
        self.neuron_col = neuron_col

    def coregister_wide(self, df_list: List[pd.DataFrame]):
        return coregister_wide(df_list, self.not_neuron_cols)

    def coregister_long(self, df_list: List[pd.DataFrame]):
        return coregister_long(df_list, self.neuron_col)

    def concat_wide(self, df_list: List[pd.DataFrame]):
        return concat_wide(df_list)
