from pathlib import Path
import pandas as pd
from typing import Optional
from astro.preprocess import Preprocessor
from cc_blocks.block_collections import GroupedSessionBlocks
from .load_fn import (
    load_traces,
    load_mice,
    load_cell_props,
    load_blocks,
    load_blockstarts,
    load_master_cellset,
)


class Loader:
    """
    Loader for astro data.

    Can be preconfigured to load from a specific session and/or to store loading parameters.

    Methods:
        load_traces: Load traces for a session.
        load_mice: Load mice for a session.
        load_cell_props: Load cell properties for a session.
        load_neurons: Load neurons for a session.
        load_blocks: Load blocks for a session.
        load_blockstarts: Load block start times for a session.
        set_preprocessor: Set the preprocessor to use for loading data.
    """

    def __init__(
        self,
        data_dir: Path,
        session_name: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
        block_group: Optional[str] = None,
        block_name: Optional[str] = None,
        mouse_col_name: str = "mouse_name",
    ):
        """

        Args:
            data_dir (Path): Path to the data directory.
            session_name (Optional[str], optional): Name of the session. Defaults to None.
            preprocessor (Optional[Preprocessor], optional): Optional atro.preprocess.Preprocessor to apply to traces and events post loading. Defaults to None.
            block_group (Optional[str], optional): Name of the block group (set of similar blocks) to load for events. Defaults to None.
            block_name (Optional[str], optional): Name of the a specific block to load for events. Defaults to None.
            mouse_col_name (str, optional): Name of the column in long-format dataframes that contains the mouse name. Defaults to "mouse_name".
        """
        self.data_dir = data_dir
        self.session_name = session_name
        self.preprocessor = preprocessor
        self.block_group = block_group
        self.block_name = block_name
        self.mouse_col_name = mouse_col_name

    def _check_session_name(self, session_name: Optional[str]) -> str:
        session_name = self.session_name if session_name is None else session_name
        assert (
            session_name is not None
        ), "session_name must be specified either during initialization or during call"
        return session_name

    def load_traces(
        self, session_name: Optional[str] = None, preprocess: bool = True
    ) -> pd.DataFrame:
        """
        Load traces for a session in wide format. Columns = ['time'] + neuron_ids.

        Args:
            session_name (Optional[str], optional): Name of the session. Defaults to value at initialization.
            preprocess (bool, optional): Whether to preprocess the traces using the astro.preprocess.Preprocessor preprocessor attribute. Defaults to True.

        Returns:
            pd.DataFrame: Traces for the session in wide format.
        """
        session_name = self._check_session_name(session_name)

        df_traces = load_traces(self.data_dir, session_name)
        if preprocess and self.preprocessor is not None:
            df_traces = self.preprocessor.preprocess_traces(df_traces)
        return df_traces

    def load_mice(self) -> pd.DataFrame:
        """
        Load mice dataframe for the entire dataset. Columns are ['mouse_name', 'group'].

        Returns:
            pd.DataFrame: Dataframe of mouse metadata.
        """
        return load_mice(self.data_dir)

    def load_cell_props(self, session_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load cell properties dataframe for a session.

        Args:
            session_name (Optional[str], optional): Name of the session. Defaults to value at initialization.

        Returns:
            pd.DataFrame: Dataframe of cell properties.
        """
        session_name = self._check_session_name(session_name)
        return load_cell_props(self.data_dir, session_name)

    def load_neurons(
        self,
    ) -> pd.DataFrame:
        """
        Load neurons dataframe for the entire dataset. Columns include ['cell_id', 'mouse_name',]

        Returns:
            pd.DataFrame: Dataframe of neuron metadata.
        """
        return load_master_cellset(self.data_dir)

    def load_blocks(
        self,
        session_name: Optional[str] = None,
    ) -> GroupedSessionBlocks:
        """
        Load the cc_blocks.GroupedSessionBlocks object for a session.

        Args:
            session_name (Optional[str], optional): Name of the session. Defaults to value at initialization.

        Returns:
            GroupedSessionBlocks: GroupedSessionBlocks object for the session.
        """
        session_name = self._check_session_name(session_name)
        return load_blocks(self.data_dir, session_name)

    def load_blockstarts(
        self,
        block_name: Optional[str] = None,
        block_group: Optional[str] = None,
        session_name: Optional[str] = None,
        preprocess: bool = True,
    ) -> pd.DataFrame:
        """
        Load block start times dataframe for a session.

        Args:
            block_name (Optional[str], optional): Name of the block. Defaults to value at initialization.
            block_group (Optional[str], optional): Name of the block group. Defaults to value at initialization.
            session_name (Optional[str], optional): Name of the session. Defaults to value at initialization.
            preprocess (bool, optional): Whether to preprocess the block start times using the astro.preprocess.Preprocessor preprocessor attribute. Defaults to True.

        Returns:
            pd.DataFrame: Dataframe of block start times.
        """
        session_name = self._check_session_name(session_name)

        if block_group is None:
            block_group = self.block_group
        if block_name is None:
            block_name = self.block_name

        df_block_starts = load_blockstarts(
            self.data_dir,
            session_name=session_name,
            block_name=block_name,
            block_group=block_group,
            mouse_col_name=self.mouse_col_name,
        )
        if preprocess and self.preprocessor is not None:
            df_block_starts = self.preprocessor.process_events(df_block_starts)
        return df_block_starts

    def set_preprocessor(self, preprocessor: Preprocessor) -> None:
        """
        Update the preprocessor attribute.

        Args:
            preprocessor (Preprocessor): Preprocessor to use for loading data.
        """
        self.preprocessor = preprocessor
