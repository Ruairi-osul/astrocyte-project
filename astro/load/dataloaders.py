from .config import DataConfig
from astro.load import GroupSessionData
import pandas as pd
from abc import ABC, abstractmethod


class DataLoader(ABC):
    def __init__(self, data_config: DataConfig):
        self.data_config = data_config

    @abstractmethod
    def get_traces(self) -> pd.DataFrame:
        ...

    @abstractmethod
    def get_events(self) -> pd.DataFrame:
        ...


class GroupDataloader(DataLoader):
    """
    A class for loading data from a group session. Useful for dynamic configuration changes.

    - Stores configuration (astro.load.config.DataConfig)
    - Generates astro.load.GroupSessionData instance on the fly (allows for dynamic configuration changes)
    - Exposes methods for getting traces and events
    """

    def __init__(self, data_config: DataConfig):
        if data_config.group_splitter is None:
            raise ValueError("GroupSplitter is required for GroupDataloader")
        if data_config.group is None:
            raise ValueError("Group is required for GroupDataloader")

        super().__init__(data_config)

    @property
    def group_session_data(self) -> GroupSessionData:
        return GroupSessionData(
            loader=self.data_config.loader,
            session_name=self.data_config.session_name,
            group=self.data_config.group,
            group_splitter=self.data_config.group_splitter,
        )

    def get_traces(self) -> pd.DataFrame:
        return self.group_session_data.df_traces

    def get_events(self) -> pd.DataFrame:
        if self.data_config.block_group is None:
            raise ValueError("block_group is required for GroupDataloader")
        return self.group_session_data.df_block_starts(
            block_group=self.data_config.block_group
        )

    def __call__(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        df_traces = self.get_traces()
        df_events = self.get_events()
        return df_traces, df_events
