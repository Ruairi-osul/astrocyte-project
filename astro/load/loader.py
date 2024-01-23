from pathlib import Path
import pandas as pd
from typing import Optional
import pickle
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
from functools import cached_property


class Loader:
    def __init__(
        self,
        data_dir: Path,
        session_name: Optional[str] = None,
        preprocessor: Optional[Preprocessor] = None,
        block_group: Optional[str] = None,
        block_name: Optional[str] = None,
        mouse_col_name: str = "mouse_name",
    ):
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
        session_name = self._check_session_name(session_name)

        df_traces = load_traces(self.data_dir, session_name)
        if preprocess and self.preprocessor is not None:
            df_traces = self.preprocessor.preprocess_traces(df_traces)
        return df_traces

    def load_mice(self) -> pd.DataFrame:
        return load_mice(self.data_dir)

    def load_cell_props(self, session_name: Optional[str] = None) -> pd.DataFrame:
        session_name = self._check_session_name(session_name)
        return load_cell_props(self.data_dir, session_name)

    def load_neurons(
        self,
    ) -> pd.DataFrame:
        return load_master_cellset(self.data_dir)

    def load_blocks(
        self,
        session_name: Optional[str] = None,
    ) -> GroupedSessionBlocks:
        session_name = self._check_session_name(session_name)
        return load_blocks(self.data_dir, session_name)

    def load_blockstarts(
        self,
        block_name: Optional[str] = None,
        block_group: Optional[str] = None,
        session_name: Optional[str] = None,
        preprocess: bool = True,
    ) -> pd.DataFrame:
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
        self.preprocessor = preprocessor

    def load_master_cellset(self) -> pd.DataFrame:
        return load_master_cellset(self.data_dir)
