import pandas as pd
from typing import Optional, Dict, List
import numpy as np


def mice_by_group(
    df_mice: pd.DataFrame,
    df_mice_group_col: str = "group",
    df_mice_mouse_col: str = "mouse_name",
) -> Dict[str, np.ndarray]:
    """
    Get a dictionary with groups as keys and mice in each group as values.

    Args:
        df_mice (pd.DataFrame): Dataframe of mice metadata.
        df_mice_group_col (str, optional): Name of the column with group information. Defaults to "group".
        df_mice_mouse_col (str, optional): Name of the column with mouse names. Defaults to "mouse_name".

    Returns:
        Dict[str, np.ndarray]: Dictionary with groups as keys and mice in each group as values.
    """
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

    Returns:
        pd.DataFrame: Long-format dataframe of traces with group information.
    """
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
) -> Dict[str, pd.DataFrame]:
    """
    Get a dictionary with mouse names as keys and traces in each mouse as values.
    
    Args:
        df_traces (pd.DataFrame): Dataframe of traces.
        df_neurons (pd.DataFrame): Dataframe of neuron metadata.
        df_neurons_mouse_col (str, optional): Name of the column with mouse names in neuron dataframe. Defaults to "mouse_name".
        df_neurons_neuron_col (str, optional): Name of the column with neuron names. Defaults to "cell_id".
        df_traces_time_col (Optional[str], optional): Name of the column with time information in traces dataframe

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
        df_traces_long_neuron_col (str, optional): Name of the column with neuron names in long-format dataframe. Defaults to "cell_id".
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
        df_traces_long_neuron_col: str = "cell_id",
        excluded_groups: Optional[List[str]] = None,
        map_int_to_str: bool = True,
    ):
        self.df_mice = df_mice
        self.df_neurons = df_neurons
        self.df_mice_mouse_col = df_mice_mouse_col
        self.df_mice_group_col = df_mice_group_col
        self.df_neurons_mouse_col = df_neurons_mouse_col
        self.df_neurons_neuron_col = df_neurons_neuron_col
        self.df_traces_time_col = df_traces_time_col
        self.df_traces_long_neuron_col = df_traces_long_neuron_col
        self.map_int_to_str = map_int_to_str

        if excluded_groups is None:
            excluded_groups = []
        self.excluded_groups = excluded_groups

    def _exclude_groups(self, df_mice: pd.DataFrame) -> pd.DataFrame:
        return df_mice[
            ~df_mice[self.df_mice_group_col].isin(self.excluded_groups)
        ].copy()

    @property
    def groups(self) -> np.ndarray:
        """
        Get unique group names.

        Returns:
            np.ndarray: Unique group names.
        """
        if self.df_mice is None:
            raise ValueError("df_mice must be provided")
        df_mice = self.df_mice
        df_mice = self._exclude_groups(df_mice)
        return df_mice[self.df_mice_group_col].unique()

    @property
    def mice(self):
        """
        Get unique mouse names.
        
        Returns:
            np.ndarray: Unique mouse names.
        """
        if self.df_mice is None:
            raise ValueError("df_mice must be provided")
        df_mice = self.df_mice
        df_mice = self._exclude_groups(df_mice)
        return df_mice[self.df_mice_mouse_col].unique()

    def mice_by_group(
        self, df_mice: Optional[pd.DataFrame] = None
    ) -> Dict[str, np.ndarray]:
        """
        Get a dictionary with groups as keys and mice in each group as values.
        
        Args:
            df_mice (Optional[pd.DataFrame], optional): Dataframe of mice metadata. Defaults to None.
        """
        if df_mice is None:
            assert self.df_mice is not None, "df_mice must be provided"
            df_mice = self.df_mice

        df_mice = self._exclude_groups(df_mice)

        return mice_by_group(
            df_mice=df_mice,
            df_mice_group_col=self.df_mice_group_col,
            df_mice_mouse_col=self.df_mice_mouse_col,
        )

    def neurons_by_group(
        self,
        df_neurons: Optional[pd.DataFrame] = None,
        df_mice: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get a dictionary with groups as keys and neurons in each group as values.
        
        Args:
            df_neurons (Optional[pd.DataFrame], optional): Dataframe of neuron metadata. Defaults to None.
            df_mice (Optional[pd.DataFrame], optional): Dataframe of mice metadata. Defaults to None.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with groups as keys and neurons in each group as values.
        """
        if df_neurons is None:
            assert self.df_neurons is not None, "df_neurons must be provided"
            df_neurons = self.df_neurons

        if df_mice is None:
            assert self.df_mice is not None, "df_mice must be provided"
            df_mice = self.df_mice

        df_mice = self._exclude_groups(df_mice)

        return neurons_by_group(
            df_neurons=df_neurons,
            df_mice=df_mice,
            df_mice_group_col=self.df_mice_group_col,
            df_mice_mouse_col=self.df_mice_mouse_col,
            df_neurons_mouse_col=self.df_neurons_mouse_col,
            df_neurons_neuron_col=self.df_neurons_neuron_col,
            map_int_to_str=self.map_int_to_str,
        )

    def neurons_by_mouse(
        self,
        df_neurons: Optional[pd.DataFrame] = None,
        df_mice: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Get a dictionary with mouse names as keys and neurons in each mouse as values.
        
        Args:
            df_neurons (Optional[pd.DataFrame], optional): Dataframe of neuron metadata. Defaults to None.
            df_mice (Optional[pd.DataFrame], optional): Dataframe of mice metadata. Defaults to None.
            
        Returns:
            Dict[str, np.ndarray]: Dictionary with mouse names as keys and neurons in each mouse as values.
        """
        if df_neurons is None:
            assert self.df_neurons is not None, "df_neurons must be provided"
            df_neurons = self.df_neurons

        if df_mice is None:
            assert self.df_mice is not None, "df_mice must be provided"
            df_mice = self.df_mice
        df_mice = self._exclude_groups(df_mice)

        return neurons_by_mouse(
            df_neurons=df_neurons,
            df_mice=df_mice,
            df_mice_mouse_col=self.df_mice_mouse_col,
            df_neurons_mouse_col=self.df_neurons_mouse_col,
            df_neurons_neuron_col=self.df_neurons_neuron_col,
            map_int_to_str=self.map_int_to_str,
        )

    def traces_by_group(
        self,
        df_traces: pd.DataFrame,
        df_neurons: Optional[pd.DataFrame] = None,
        df_mice: Optional[pd.DataFrame] = None,
        df_traces_time_col: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Get a dictionary with groups as keys and traces in each group as values.
        
        Args:
            df_traces (pd.DataFrame): Dataframe of traces.
            df_neurons (Optional[pd.DataFrame], optional): Dataframe of neuron metadata. Defaults to None.
            df_mice (Optional[pd.DataFrame], optional): Dataframe of mice metadata. Defaults to None.
            df_traces_time_col (Optional[str], optional): Name of the column with time information in traces dataframe. Defaults to None.
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with groups as keys and traces in each group as values.
        """
        if df_neurons is None:
            assert self.df_neurons is not None, "df_neurons must be provided"
            df_neurons = self.df_neurons

        if df_mice is None:
            assert self.df_mice is not None, "df_mice must be provided"
            df_mice = self.df_mice

        if df_traces_time_col is None:
            df_traces_time_col = self.df_traces_time_col
        df_mice = self._exclude_groups(df_mice)

        return traces_by_group(
            df_traces=df_traces,
            df_neurons=df_neurons,
            df_mice=df_mice,
            df_mice_group_col=self.df_mice_group_col,
            df_mice_mouse_col=self.df_mice_mouse_col,
            df_neurons_mouse_col=self.df_neurons_mouse_col,
            df_traces_time_col=df_traces_time_col,
            map_int_to_str=self.map_int_to_str,
        )

    def traces_by_group_long(
        self,
        df_traces_long: pd.DataFrame,
        df_neurons: Optional[pd.DataFrame] = None,
        df_mice: Optional[pd.DataFrame] = None,
        df_traces_long_time_col: Optional[str] = None,
    ) -> pd.DataFrame:
        if df_neurons is None:
            assert self.df_neurons is not None, "df_neurons must be provided"
            df_neurons = self.df_neurons

        if df_mice is None:
            assert self.df_mice is not None, "df_mice must be provided"
            df_mice = self.df_mice

        if df_traces_long_time_col is None:
            df_traces_long_time_col = self.df_traces_time_col

        df_mice = self._exclude_groups(df_mice)

        return traces_by_group_long(
            df_traces_long=df_traces_long,
            df_neurons=df_neurons,
            df_mice=df_mice,
            df_mice_group_col=self.df_mice_group_col,
            df_mice_mouse_col=self.df_mice_mouse_col,
            df_neurons_mouse_col=self.df_neurons_mouse_col,
            df_neurons_neuron_col=self.df_neurons_neuron_col,
            df_traces_long_neuron_col=self.df_traces_long_neuron_col,
        )
