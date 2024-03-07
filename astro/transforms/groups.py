import pandas as pd
from typing import Optional, Dict, List
import numpy as np


def mice_by_group(
    df_mice: pd.DataFrame,
    df_mice_group_col: str = "group",
    df_mice_mouse_col: str = "mouse_name",
    permute_mice: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Get a dictionary with groups as keys and mice in each group as values.

    Args:
        df_mice (pd.DataFrame): Dataframe of mice metadata.
        df_mice_group_col (str, optional): Name of the column with group information. Defaults to "group".
        df_mice_mouse_col (str, optional): Name of the column with mouse names. Defaults to "mouse_name".
        permute_mice (bool, optional): Whether to permute mice. Defaults to False.

    Returns:
        Dict[str, np.ndarray]: Dictionary with groups as keys and mice in each group as values.
    """
    if permute_mice:
        df_mice[df_mice_group_col] = np.random.permutation(
            df_mice[df_mice_group_col].values
        )
    mice_by_group = (
        df_mice.groupby(df_mice_group_col)[df_mice_mouse_col].unique().to_dict()
    )
    return mice_by_group


def neurons_by_group(
    df_neurons: pd.DataFrame,
    df_mice: pd.DataFrame,
    df_mice_group_col: str = "group",
    df_mice_mouse_col: str = "mouse_name",
    df_neurons_mouse_col: str = "mouse_name",
    df_neurons_neuron_col: str = "cell_id",
    map_int_to_str: bool = True,
    permute_neurons: bool = False,
    equalize_neuron_numbers: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Get a dictionary with groups as keys and neurons in each group as values.

    Args:
        df_neurons (pd.DataFrame): Dataframe of neuron metadata.
        df_mice (pd.DataFrame): Dataframe of mice metadata.
        df_mice_group_col (str, optional): Name of the column with group information. Defaults to "group".
        df_mice_mouse_col (str, optional): Name of the column with mouse names. Defaults to "mouse_name".
        df_neurons_mouse_col (str, optional): Name of the column with mouse names in neuron dataframe. Defaults to "mouse_name".
        df_neurons_neuron_col (str, optional): Name of the column with neuron names. Defaults to "cell_id".
        map_int_to_str (bool, optional): Whether to map neuron names to strings. Defaults to True.
        permute_neurons (bool, optional): Whether to permute neuron groups. Defaults to False.
        equalize_neuron_numbers (bool, optional): Whether to equalize neuron numbers. Defaults to False.

    Returns:
        Dict[str, np.ndarray]: Dictionary with groups as keys and neurons in each group as values.
    """
    mice_dict = mice_by_group(
        df_mice=df_mice,
        df_mice_group_col=df_mice_group_col,
        df_mice_mouse_col=df_mice_mouse_col,
    )
    neurons_by_group = {}
    for group, mice in mice_dict.items():
        df_group = df_neurons[df_neurons[df_neurons_mouse_col].isin(mice)]
        neurons_by_group[group] = df_group[df_neurons_neuron_col].unique()

    if equalize_neuron_numbers:
        min_neurons = min([len(neurons) for neurons in neurons_by_group.values()])
        for group in neurons_by_group:
            np.random.shuffle(neurons_by_group[group])
            neurons_by_group[group] = neurons_by_group[group][:min_neurons]

    if permute_neurons:
        neurons = np.concatenate(list(neurons_by_group.values()))
        np.random.shuffle(neurons)
        for group in neurons_by_group:
            neurons_by_group[group] = neurons[: len(neurons_by_group[group])]
    if map_int_to_str:
        neurons_by_group = {
            group: [str(n) for n in neurons]
            for group, neurons in neurons_by_group.items()
        }
    return neurons_by_group


def neurons_by_mouse(
    df_neurons: pd.DataFrame,
    df_mice: pd.DataFrame,
    df_mice_mouse_col: str = "mouse_name",
    df_neurons_mouse_col: str = "mouse_name",
    df_neurons_neuron_col: str = "cell_id",
    map_int_to_str: bool = True,
    equalize_neuron_numbers: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Get a dictionary with mouse names as keys and neurons in each mouse as values.

    Args:
        df_neurons (pd.DataFrame): Dataframe of neuron metadata.
        df_mice (pd.DataFrame): Dataframe of mice metadata.
        df_mice_mouse_col (str, optional): Name of the column with mouse names. Defaults to "mouse_name".
        df_neurons_mouse_col (str, optional): Name of the column with mouse names in neuron dataframe. Defaults to "mouse_name".
        df_neurons_neuron_col (str, optional): Name of the column with neuron names. Defaults to "cell_id".
        map_int_to_str (bool, optional): Whether to map neuron names to strings. Defaults to True.
        equalize_neuron_numbers (bool, optional): Whether to equalize neuron numbers. Defaults to False.

    Returns:
        Dict[str, np.ndarray]: Dictionary with mouse names as keys and neurons in each mouse as values.
    """
    by_mouse = {}
    for mouse in df_mice[df_mice_mouse_col].unique():
        mouse_neurons = df_neurons[df_neurons[df_neurons_mouse_col] == mouse][
            df_neurons_neuron_col
        ].unique()
        if map_int_to_str:
            mouse_neurons = mouse_neurons.astype(str)
        by_mouse[mouse] = np.asarray(mouse_neurons)
    if equalize_neuron_numbers:
        min_neurons = min([len(neurons) for neurons in by_mouse.values()])
        for mouse in by_mouse:
            np.random.shuffle(by_mouse[mouse])
            by_mouse[mouse] = by_mouse[mouse][:min_neurons]

    if map_int_to_str:
        by_mouse = {
            mouse: [str(n) for n in neurons] for mouse, neurons in by_mouse.items()
        }

    return by_mouse


def traces_by_group(
    df_traces: pd.DataFrame,
    df_neurons: pd.DataFrame,
    df_mice: pd.DataFrame,
    df_mice_group_col: str = "group",
    df_mice_mouse_col: str = "mouse_name",
    df_neurons_mouse_col: str = "mouse_name",
    df_traces_time_col: Optional[str] = "time",
    map_int_to_str: bool = True,
    permute_neurons: bool = False,
    equalize_neuron_numbers: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Get a dictionary with groups as keys and traces in each group as values.

    Args:
        df_traces (pd.DataFrame): Dataframe of traces.
        df_neurons (pd.DataFrame): Dataframe of neuron metadata.
        df_mice (pd.DataFrame): Dataframe of mice metadata.
        df_mice_group_col (str, optional): Name of the column with group information. Defaults to "group".
        df_mice_mouse_col (str, optional): Name of the column with mouse names. Defaults to "mouse_name".
        df_neurons_mouse_col (str, optional): Name of the column with mouse names in neuron dataframe. Defaults to "mouse_name".
        df_traces_time_col (Optional[str], optional): Name of the column with time information in traces dataframe. Defaults to "time".
        map_int_to_str (bool, optional): Whether to map neuron names to strings. Defaults to True.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with groups as keys and traces in each group as values.
    """
    neurons_dict = neurons_by_group(
        df_neurons=df_neurons,
        df_mice=df_mice,
        df_mice_group_col=df_mice_group_col,
        df_mice_mouse_col=df_mice_mouse_col,
        df_neurons_mouse_col=df_neurons_mouse_col,
        map_int_to_str=map_int_to_str,
        permute_neurons=permute_neurons,
        equalize_neuron_numbers=equalize_neuron_numbers,
    )
    traces_by_group = {}
    for group, neurons in neurons_dict.items():
        if df_traces_time_col is None:
            cols = [c for c in df_traces.columns if c in neurons]
        else:
            cols = [df_traces_time_col] + [c for c in df_traces.columns if c in neurons]

        traces_by_group[group] = df_traces[cols]
    return traces_by_group


def traces_by_group_long(
    df_traces_long: pd.DataFrame,
    df_neurons: pd.DataFrame,
    df_mice: pd.DataFrame,
    df_mice_group_col: str = "group",
    df_mice_mouse_col: str = "mouse_name",
    df_neurons_mouse_col: str = "mouse_name",
    df_neurons_neuron_col: str = "cell_id",
    df_traces_long_neuron_col: str = "cell_id",
    permute_neurons: bool = False,
    equalize_neuron_numbers: bool = False,
) -> pd.DataFrame:
    """
    Add group information to a long-format dataframe of traces.

    Args:
        df_traces_long (pd.DataFrame): Long-format dataframe of traces.
        df_neurons (pd.DataFrame): Dataframe of neuron metadata.
        df_mice (pd.DataFrame): Dataframe of mice metadata.
        df_mice_group_col (str, optional): Name of the column with group information. Defaults to "group".
        df_mice_mouse_col (str, optional): Name of the column with mouse names. Defaults to "mouse_name".
        df_neurons_mouse_col (str, optional): Name of the column with mouse names in neuron dataframe. Defaults to "mouse_name".
        df_neurons_neuron_col (str, optional): Name of the column with neuron names. Defaults to "cell_id".
        df_traces_long_neuron_col (str, optional): Name of the column with neuron names in long-format dataframe. Defaults to "cell_id".
        permute_neurons (bool, optional): Whether to permute neuron groups. Defaults to False.
        equalize_neuron_numbers (bool, optional): Whether to equalize neuron numbers. Defaults to False.

    Returns:
        pd.DataFrame: Long-format dataframe of traces with group information.
    """
    if permute_neurons:
        df_neurons[df_mice_group_col] = np.random.permutation(
            df_neurons[df_mice_group_col].values
        )
    if equalize_neuron_numbers:
        min_neurons = min(
            df_neurons.groupby([df_mice_group_col, df_mice_mouse_col])[
                df_neurons_neuron_col
            ].count()
        )
        df_neurons = df_neurons.groupby([df_mice_group_col, df_mice_mouse_col]).apply(
            lambda x: x.sample(min_neurons)
        )
    df_neuron_groups = pd.merge(
        df_neurons,
        df_mice[[df_mice_mouse_col, df_mice_group_col]].drop_duplicates(),
        right_on=df_mice_mouse_col,
        left_on=df_neurons_mouse_col,
    )
    df_traces_long = pd.merge(
        df_traces_long,
        df_neuron_groups[[df_neurons_neuron_col, df_mice_group_col]].drop_duplicates(),
        left_on=df_traces_long_neuron_col,
        right_on=df_neurons_neuron_col,
    )
    return df_traces_long


def traces_by_mouse(
    df_traces: pd.DataFrame,
    df_neurons: pd.DataFrame,
    df_neurons_mouse_col: str = "mouse_name",
    df_neurons_neuron_col: str = "cell_id",
    df_traces_time_col: Optional[str] = "time",
    map_int_to_str: bool = True,
    permute_neurons: bool = False,
) -> Dict[str, pd.DataFrame]:
    """
    Get a dictionary with mouse names as keys and traces in each mouse as values.

    Args:
        df_traces (pd.DataFrame): Dataframe of traces.
        df_neurons (pd.DataFrame): Dataframe of neuron metadata.
        df_neurons_mouse_col (str, optional): Name of the column with mouse names in neuron dataframe. Defaults to "mouse_name".
        df_neurons_neuron_col (str, optional): Name of the column with neuron names. Defaults to "cell_id".
        df_traces_time_col (Optional[str], optional): Name of the column with time information in traces dataframe
        map_int_to_str (bool, optional): Whether to map neuron names to strings. Defaults to True.
        permute_neurons (bool, optional): Whether to permute neuron groups. Defaults to False.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary with mouse names as keys and traces in each mouse as values.
    """
    traces_by_mouse = {}

    for mouse in df_neurons[df_neurons_mouse_col].unique():
        mouse_neurons = df_neurons[df_neurons[df_neurons_mouse_col] == mouse][
            df_neurons_neuron_col
        ].unique()
        if map_int_to_str:
            mouse_neurons = [str(n) for n in mouse_neurons]

        cols = [c for c in df_traces.columns if c in mouse_neurons]
        if df_traces_time_col is not None:
            cols = [df_traces_time_col] + cols

        traces_by_mouse[mouse] = df_traces[cols]

    if permute_neurons:
        actual_keys = list(traces_by_mouse.keys())
        random_keys = np.random.permutation(list(traces_by_mouse.keys()))
        traces_by_mouse = {
            random_key: traces_by_mouse[actual_key]
            for actual_key, random_key in zip(actual_keys, random_keys)
        }

    return traces_by_mouse


class GroupSplitter:
    """
    Class to split data by groups and mouse names.

    Args:
        df_mice (Optional[pd.DataFrame], optional): Dataframe of mice metadata. Defaults to None.
        df_neurons (Optional[pd.DataFrame], optional): Dataframe of neuron metadata. Defaults to None.
        df_mice_mouse_col (str, optional): Name of the column with mouse names. Defaults to "mouse_name".
        df_mice_group_col (str, optional): Name of the column with group information. Defaults to "group".
        df_neurons_mouse_col (str, optional): Name of the column with mouse names in neuron dataframe. Defaults to "mouse_name".
        df_neurons_neuron_col (str, optional): Name of the column with neuron names. Defaults to "cell_id".
        df_traces_time_col (str, optional): Name of the column with time information in traces dataframe. Defaults to "time".
        excluded_groups (Optional[List[str]], optional): List of groups to exclude. Defaults to None.
        map_int_to_str (bool, optional): Whether to map neuron names to strings. Defaults to True.
    """

    def __init__(
        self,
        df_mice: Optional[pd.DataFrame] = None,
        df_neurons: Optional[pd.DataFrame] = None,
        df_mice_mouse_col: str = "mouse_name",
        df_mice_group_col: str = "group",
        df_neurons_mouse_col: str = "mouse_name",
        df_neurons_neuron_col: str = "cell_id",
        df_traces_time_col: str = "time",
        excluded_groups: Optional[List[str]] = None,
        map_int_to_str: bool = True,
        permute_neurons: bool = False,
        permute_mice: bool = False,
        equalize_neuron_numbers: bool = False,
    ):
        self._df_mice = df_mice
        self._df_neurons = df_neurons

        self.df_mice_mouse_col = df_mice_mouse_col
        self.df_mice_group_col = df_mice_group_col
        self.df_neurons_mouse_col = df_neurons_mouse_col
        self.df_neurons_neuron_col = df_neurons_neuron_col
        self.df_traces_time_col = df_traces_time_col

        self.map_int_to_str = map_int_to_str
        self.permute_neurons = permute_neurons
        self.permute_mice = permute_mice
        self.equalize_neuron_numbers = equalize_neuron_numbers

        self._mouse_group_mapping: dict[str, np.ndarray] | None = None
        self._neuron_group_mapping: dict[str, np.ndarray] | None = None
        self._neuron_mouse_mapping: dict[str, np.ndarray] | None = None

        if excluded_groups is None:
            excluded_groups = []
        self.excluded_groups = excluded_groups

    def set_df_mice(
        self,
        df_mice: pd.DataFrame,
        mouse_col: str | None = None,
        group_col: str | None = None,
    ) -> None:
        self._df_mice = df_mice
        if mouse_col is not None:
            self.df_mice_mouse_col = mouse_col
        if group_col is not None:
            self.df_mice_group_col = group_col

    def set_df_neurons(
        self,
        df_neurons: pd.DataFrame,
        mouse_col: str | None = None,
        neuron_col: str | None = None,
    ) -> None:
        self._df_neurons = df_neurons
        if mouse_col is not None:
            self.df_neurons_mouse_col = mouse_col
        if neuron_col is not None:
            self.df_neurons_neuron_col = neuron_col

    @property
    def df_mice(self) -> pd.DataFrame:
        if self._df_mice is None:
            raise ValueError("df_mice not set")
        return self._df_mice[
            ~self._df_mice[self.df_mice_group_col].isin(self.excluded_groups)
        ].copy()

    @property
    def df_neurons(self) -> pd.DataFrame:
        if self._df_neurons is None:
            raise ValueError("df_neurons not set")
        return self._df_neurons.loc[
            lambda x: x[self.df_neurons_mouse_col].isin(self.mice)
        ].copy()

    @property
    def groups(self) -> np.ndarray:
        """
        Get unique group names.

        Returns:
            np.ndarray: Unique group names.
        """
        return self.df_mice[self.df_mice_group_col].unique()

    @property
    def mice(self) -> np.ndarray:
        """
        Get unique mouse names.

        Returns:
            np.ndarray: Unique mouse names.
        """
        return self.df_mice[self.df_mice_mouse_col].unique()

    @property
    def neurons(self) -> np.ndarray:
        return self.df_neurons[self.df_neurons_neuron_col].unique()

    @property
    def mice_by_group(
        self,
    ) -> dict[str, np.ndarray]:
        """
        Get a dictionary with groups as keys and mice in each group as values.

        Returns:
            dict[str, np.ndarray]: Dictionary with groups as keys and mice in each group as values.
        """
        if self._mouse_group_mapping is None:
            self._mouse_group_mapping = mice_by_group(
                df_mice=self.df_mice,
                df_mice_group_col=self.df_mice_group_col,
                df_mice_mouse_col=self.df_mice_mouse_col,
                permute_mice=self.permute_mice,
            )
        return self._mouse_group_mapping

    @property
    def neurons_by_group(self) -> dict[str, np.ndarray]:
        """
        Get a dictionary with groups as keys and neurons in each group as values.

        Returns:
            dict[str, np.ndarray]: Dictionary with groups as keys and neurons in each group as values.
        """
        if self._neuron_group_mapping is None:
            self._neuron_group_mapping = neurons_by_group(
                df_neurons=self.df_neurons,
                df_mice=self.df_mice,
                df_mice_group_col=self.df_mice_group_col,
                df_mice_mouse_col=self.df_mice_mouse_col,
                df_neurons_mouse_col=self.df_neurons_mouse_col,
                df_neurons_neuron_col=self.df_neurons_neuron_col,
                map_int_to_str=self.map_int_to_str,
                permute_neurons=self.permute_neurons,
                equalize_neuron_numbers=self.equalize_neuron_numbers,
            )

        return self._neuron_group_mapping

    @property
    def neurons_by_mouse(
        self,
    ) -> dict[str, np.ndarray]:
        """
        Get a dictionary with mouse names as keys and neurons in each mouse as values.

        Returns:
            dict[str, np.ndarray]: Dictionary with mouse names as keys and neurons in each mouse as values.
        """
        if self._neuron_mouse_mapping is None:
            self._neuron_mouse_mapping = neurons_by_mouse(
                df_neurons=self.df_neurons,
                df_mice=self.df_mice,
                df_mice_mouse_col=self.df_mice_mouse_col,
                df_neurons_mouse_col=self.df_neurons_mouse_col,
                df_neurons_neuron_col=self.df_neurons_neuron_col,
                map_int_to_str=self.map_int_to_str,
                equalize_neuron_numbers=self.equalize_neuron_numbers,
            )
        return self._neuron_mouse_mapping

    def refresh_mappings(self):
        self._mouse_group_mapping = None
        self._neuron_group_mapping = None
        self._neuron_mouse_mapping = None

    def traces_by_group(
        self,
        df_traces: pd.DataFrame,
        df_traces_time_col: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Get a dictionary with groups as keys and traces in each group as values.

        Args:
            df_traces (pd.DataFrame): Dataframe of traces.
            df_traces_time_col (Optional[str], optional): Name of the column with time information in traces dataframe. Defaults to None.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with groups as keys and traces in each group as values.
        """

        if df_traces_time_col is None:
            df_traces_time_col = self.df_traces_time_col

        trace_dict = {}
        for group, neurons in self.neurons_by_group.items():
            cols = [c for c in df_traces.columns if c in neurons]
            if df_traces_time_col is not None:
                cols = [df_traces_time_col] + cols
            trace_dict[group] = df_traces[cols]
        return trace_dict

    def traces_by_mouse(
        self,
        df_traces: pd.DataFrame,
        df_traces_time_col: Optional[str] = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Get a dictionary with mouse names as keys and traces in each mouse as values.

        Args:
            df_traces (pd.DataFrame): Dataframe of traces.
            df_traces_time_col (Optional[str], optional): Name of the column with time information in traces dataframe. Defaults to None.

        Returns:
            Dict[str, pd.DataFrame]: Dictionary with mouse names as keys and traces in each mouse as values.
        """
        if df_traces_time_col is None:
            df_traces_time_col = self.df_traces_time_col

        trace_dict = {}
        for mouse, neurons in self.neurons_by_mouse.items():
            cols = [c for c in df_traces.columns if c in neurons]
            if df_traces_time_col is not None:
                cols = [df_traces_time_col] + cols
            trace_dict[mouse] = df_traces[cols]
        return trace_dict

    def traces_by_group_long(
        self,
        df_traces_long: pd.DataFrame,
        df_traces_long_neuron_col: str | None = None,
    ) -> pd.DataFrame:
        """
        Add group information to a long-format dataframe of traces.

        Args:
            df_traces_long (pd.DataFrame): Long-format dataframe of traces.
            df_traces_long_time_col (str, optional): Name of the column with time information in long-format traces dataframe

        Returns:
            pd.DataFrame: Long-format dataframe of traces with group information.
        """

        if df_traces_long_neuron_col is None:
            df_traces_long_neuron_col = self.df_neurons_neuron_col

        neuron_group_mapper = self.neurons_by_group
        neurons_by_group_inv = {
            neuron: group
            for group, neurons in neuron_group_mapper.items()
            for neuron in neurons
        }

        df_traces_long[self.df_mice_group_col] = df_traces_long[
            df_traces_long_neuron_col
        ].map(neurons_by_group_inv)

        return df_traces_long
