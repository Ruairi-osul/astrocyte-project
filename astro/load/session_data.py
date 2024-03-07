from .loader import Loader
import pandas as pd
from astro.transforms import GroupSplitter


class SessionData:
    """
    Object containing data for a single session.

    Methods and Attributes:
        df_traces: traces for all neurons in session
        df_block_starts: block start times for all blocks in session
        df_mice: mice in session
        df_cell_props: cell properties for all neurons in session
        df_neurons: neurons in session
    """

    def __init__(
        self,
        loader: Loader,
        session_name: str | None = None,
        group_splitter: GroupSplitter | None = None,
    ) -> None:
        """
        Args:
            loader (Loader): loader containing loading and preprocessing configuration.
            session_name (str, optional): Name of the session. Must be specified if loader does not have a session_name specified. Defaults to None.
        """
        if session_name is not None:
            loader.session_name = session_name
        else:
            self._check_loader_session(loader)
        self.loader = loader
        self.group_splitter = group_splitter

    def _check_loader_session(self, loader: Loader) -> None:
        if loader.session_name is None:
            raise ValueError("loader must not have a session_name specified")

    @property
    def df_traces(self) -> pd.DataFrame:
        """
        Traces for all neurons in session in wide format. Columns = ['time'] + neuron_ids.

        Returns:
            pd.DataFrame: traces for all neurons in session
        """
        df_traces = self.loader.load_traces().copy()
        if self.group_splitter is not None:
            neurons_in_session = self.group_splitter.neurons.astype(str)
            time_col = self.group_splitter.df_traces_time_col
            cols = [time_col] + [
                c for c in df_traces.columns if c in neurons_in_session
            ]
            df_traces = df_traces.loc[:, cols]
        return df_traces

    def df_block_starts(
        self, block_group: str = None, block_name: str | None = None
    ) -> pd.DataFrame:
        """
        A dataframe containing block start times for all blocks in the session.

        Args:
            block_group (str, optional): Group name for the block. Defaults to None.
            block_name (str, optional): Name of the block. Defaults to None.

        Returns:
            pd.DataFrame: block start times for all blocks in session
        """
        df_block_starts = self.loader.load_blockstarts(
            block_group=block_group, block_name=block_name
        ).copy()

        if self.group_splitter is not None:
            mice_in_session = self.group_splitter.mice
            mouse_col = self.group_splitter.df_mice_mouse_col
            df_block_starts = df_block_starts.loc[
                lambda x: x[mouse_col].isin(mice_in_session)
            ]
        return df_block_starts

    @property
    def df_mice(self) -> pd.DataFrame:
        """
        Get dataframe of mice in the experiment. Columns are ['mouse_name', 'group'].

        Returns:
            pd.DataFrame: mice in session
        """
        df_mice = self.loader.load_mice().copy()
        if self.group_splitter is not None:
            mice_in_session = self.group_splitter.mice
            mouse_col = self.group_splitter.df_mice_mouse_col
            df_mice = df_mice.loc[lambda x: x[mouse_col].isin(mice_in_session)]
        return df_mice

    @property
    def df_cell_props(self) -> pd.DataFrame:
        """
        Get dataframe of cell properties for all neurons in the session. Columns include ["cell_id", "mouse_name", "centroid_x" "centroid_y"]

        Returns:
            pd.DataFrame: cell properties for all neurons in session
        """

        df_cell_props = self.loader.load_cell_props().copy()
        if self.group_splitter is not None:
            neurons_in_session = self.group_splitter.neurons.astype(str)
            neurons_col = self.group_splitter.df_neurons_neuron_col
            df_cell_props = df_cell_props.loc[
                lambda x: x[neurons_col].astype(str).isin(neurons_in_session)
            ]
        return df_cell_props

    @property
    def df_neurons(self) -> pd.DataFrame:
        """
        Get a dataframe of the neurons in the session. Columns include ['cell_id', 'mouse_name']

        Returns:
            pd.DataFrame: neurons in session
        """
        df_neurons_all = self.loader.load_neurons()
        neurons_in_session = self.df_cell_props["cell_id"].unique()
        return df_neurons_all[df_neurons_all["cell_id"].isin(neurons_in_session)].copy()

    @property
    def traces_by_group(self) -> dict[str, pd.DataFrame]:
        """
        Get a dict of traces for all groups in a session.

        Returns:
            dict[str, pd.DataFrame]: traces for all groups in session
        """
        if self.group_splitter is None:
            raise ValueError("group_splitter is required for traces_by_group")
        return self.group_splitter.traces_by_group(df_traces=self.df_traces)

    def df_block_starts_by_group(
        self,
        block_group: str | None = None,
        block_name: str | None = None,
        mouse_col: str = "mouse_name",
    ) -> dict[str, pd.DataFrame]:
        """
        Get block start time dataframe for all groups in a session.

        Args:
            block_group (str, optional): Name of the block group. Defaults to None.
            block_name (str, optional): Name of the block. Defaults to None.

        Returns:
            dict[str, pd.DataFrame]: block start times for all groups in session
        """
        if self.group_splitter is None:
            raise ValueError("group_splitter is required for df_block_starts_by_group")
        df_block_starts = self.df_block_starts(
            block_group=block_group, block_name=block_name
        )
        mice_by_group = self.group_splitter.mice_by_group
        return {
            group: df_block_starts.loc[lambda x: x[mouse_col].isin(mice)]
            for group, mice in mice_by_group.items()
        }


class GroupSessionData:
    """Represents data from a single group in a single session.

    Attributes:
        group: group name
        loader: loader for session
        group_splitter: group splitter for session
        block_starts_mouse_col: column name in block_starts for mouse name
        session_data: session data
        df_traces: traces for all neurons in group
        df_block_starts: block start times for all blocks in group
        df_mice: mice in group
        df_cell_props: cell properties for all neurons in group
        df_neurons: neurons in group
    """

    def __init__(
        self,
        loader: Loader,
        group: str,
        group_splitter: GroupSplitter,
        session_name: str | None = None,
        block_starts_mouse_col: str = "mouse_name",
    ) -> None:
        """Initializes GroupSessionData.

        Args:
            loader: loader for session, must not have a session_name specified
            group: group name
            group_splitter: group splitter for session
            session_name: session name
            block_starts_mouse_col: column name in block_starts for mouse name
        """
        self.group = group
        self.session_name = session_name
        self.loader = loader
        self.group_splitter = group_splitter
        self.block_starts_mouse_col = block_starts_mouse_col

        self.session_data = SessionData(loader=loader, session_name=session_name)

    @property
    def df_traces(self) -> pd.DataFrame:
        """
        Traces for all neurons in group in wide format. Columns = ['time'] + neuron_ids.

        Returns:
            pd.DataFrame: traces for all neurons in group
        """
        df = self.session_data.df_traces
        trace_dict = self.group_splitter.traces_by_group(df_traces=df)
        return trace_dict[self.group]

    def df_block_starts(
        self, block_name: str | None = None, block_group: str | None = None
    ) -> pd.DataFrame:
        """
        Get block start time dataframe for the group in a particular session.

        Args:
            block_name (str, optional): Name of a specific block. Defaults to None.
            block_group (str, optional): Name for the set of blocks. Defaults to None.

        Returns:
            pd.DataFrame: block start times for all blocks in group
        """
        mice = self.group_splitter.mice_by_group[self.group]
        df_block_starts = self.session_data.df_block_starts(
            block_group=block_group, block_name=block_name
        ).loc[lambda x: x[self.block_starts_mouse_col].isin(mice)]
        df_block_starts = df_block_starts.reset_index(drop=True).copy()
        return df_block_starts

    @property
    def df_mice(self) -> pd.DataFrame:
        """
        Get dataframe of mice in the group.

        Returns:
            pd.DataFrame: mice in group
        """
        mice = self.group_splitter.mice_by_group[self.group]
        mouse_col = self.group_splitter.df_mice_mouse_col
        df_mice = self.session_data.df_mice.loc[lambda x: x[mouse_col].isin(mice)]
        df_mice = df_mice.reset_index(drop=True).copy()
        return df_mice

    @property
    def df_cell_props(self) -> pd.DataFrame:
        """
        Get dataframe of cell properties for all neurons in the group.

        Returns:
            pd.DataFrame: cell properties for all neurons in group
        """
        neurons = self.group_splitter.neurons_by_group[self.group]
        neurons_col = self.group_splitter.df_neurons_neuron_col
        df_cell_props = self.session_data.df_cell_props.loc[
            lambda x: x[neurons_col].astype(str).isin(neurons)
        ]
        df_cell_props = df_cell_props.reset_index(drop=True).copy()
        return df_cell_props

    @property
    def df_neurons(self) -> pd.DataFrame:
        """
        Get a dataframe of the neurons in the group in long format.

        Returns:
            pd.DataFrame: neurons in group
        """
        neurons = self.group_splitter.neurons_by_group[self.group]
        neurons_col = self.group_splitter.df_neurons_neuron_col
        df_neurons = self.session_data.df_neurons.loc[
            lambda x: x[neurons_col].astype(str).isin(neurons)
        ]
        df_neurons = df_neurons.reset_index(drop=True).copy()
        return df_neurons

    @classmethod
    def from_session_data(
        cls,
        group: str,
        session_data: SessionData,
        group_splitter: GroupSplitter | None = None,
        block_starts_mouse_col: str = "mouse_name",
    ) -> "GroupSessionData":
        """Creates GroupSessionData from SessionData.

        Args:
            group: group name
            session_data: session data
            group_splitter: group splitter for session
            block_starts_mouse_col: column name in block_starts for mouse name

        Returns:
            GroupSessionData
        """
        if group_splitter is None:
            group_splitter = GroupSplitter(
                df_mice=session_data.df_mice,
                df_neurons=session_data.df_neurons,
            )

        loader = session_data.loader
        return cls(
            group=group,
            loader=loader,
            group_splitter=group_splitter,
            block_starts_mouse_col=block_starts_mouse_col,
        )
