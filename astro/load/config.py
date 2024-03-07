from astro.transforms import GroupSplitter
from .loader import Loader
from dataclasses import dataclass


@dataclass
class DataConfig:
    """
    Container for loading configuration.

    Args:
        loader (Loader): astro.load.Loader instance
        session_name (str): Name of the session
        block_group (str, optional): Name of the block group. Defaults to None.
        group (str, optional): Name of the group. Defaults to None.
        group_splitter (GroupSplitter, optional): astro.transforms.GroupSplitter instance. Defaults to None.
    """

    loader: Loader
    session_name: str
    block_group: str | None = None
    group: str | None = None
    group_splitter: GroupSplitter | None = None
