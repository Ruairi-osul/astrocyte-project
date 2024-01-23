from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional
import pickle
from astro.preprocess import Preprocessor
from cc_blocks.block_collections import GroupedSessionBlocks


SESSION_SUBDIRS = {
    "hab-cage": "00-hab-cage",
    "hab-tone": "01-hab-tone",
    "cond": "02-cond",
    "ret": "03-ret",
    "ext": "04-ext",
    "diff-ret": "05-diff-ret",
    "late-ret": "06-late-ret",
    "renewal": "07-renewal",
}

DATA_FILENAMES = {
    "master_cellset": "master_cellset.parquet",
    "mice": "mice.csv",
    "cell_props": "cell_props.parquet",
    "traces": "traces.parquet",
    "blocks": "session_blocks.pkl",
}


def _verify_data_dir(data_dir: Path) -> None:
    """Verify that data_dir contains the expected subdirectories."""
    assert data_dir.exists(), f"Data directory {data_dir} does not exist."
    assert data_dir.is_dir(), f"Data directory {data_dir} is not a directory."


def _verify_session_name(session_name: str) -> None:
    """Verify that session_name is one of the expected values."""
    assert (
        session_name in SESSION_SUBDIRS.keys()
    ), f"session_name must be one of {SESSION_SUBDIRS.keys()}"


def load_master_cellset(data_dir: Path) -> pd.DataFrame:
    """Load master_cellset.parquet file from data_dir."""
    _verify_data_dir(data_dir)
    return pd.read_parquet(data_dir / DATA_FILENAMES["master_cellset"])


def load_mice(data_dir: Path) -> pd.DataFrame:
    """Load mice.csv file from data_dir."""
    _verify_data_dir(data_dir)
    return pd.read_csv(data_dir / DATA_FILENAMES["mice"])


def load_cell_props(data_dir: Path, session_name: str) -> pd.DataFrame:
    """Load cell_props.parquet file from data_dir/session_name."""
    _verify_data_dir(data_dir)
    _verify_session_name(session_name)
    return pd.read_parquet(
        data_dir / SESSION_SUBDIRS[session_name] / DATA_FILENAMES["cell_props"]
    )


def load_traces(data_dir: Path, session_name: str) -> pd.DataFrame:
    """Load traces.parquet file from data_dir/session_name."""
    _verify_data_dir(data_dir)
    _verify_session_name(session_name)
    return pd.read_parquet(
        data_dir / SESSION_SUBDIRS[session_name] / DATA_FILENAMES["traces"]
    )


def load_blocks(data_dir: Path, session_name: str) -> GroupedSessionBlocks:
    """Load session_blocks.pkl file from data_dir/session_name."""
    _verify_data_dir(data_dir)
    _verify_session_name(session_name)
    with open(
        data_dir / SESSION_SUBDIRS[session_name] / DATA_FILENAMES["blocks"], "rb"
    ) as f:
        blocks = pickle.load(f)
    return blocks


def load_block_timeseries(
    data_dir: Path, session_name: str, sampling_rate: float
) -> pd.DataFrame:
    """Load blocks and get blocks timeseries dataframe from data_dir/session_name"""

    _verify_data_dir(data_dir)
    _verify_session_name(session_name)
    session_blocks = load_blocks(data_dir, session_name)
    block_timeseries = session_blocks.get_binary_block_time_series()
    return block_timeseries


def load_blockstarts(
    data_dir: Path,
    session_name: str,
    block_name: Optional[str] = None,
    block_group: Optional[str] = None,
    mouse_col_name: str = "mouse_name",
) -> pd.DataFrame:
    """Load blocks and get blockstarts datadrame from data_dir/session_name"""
    _verify_data_dir(data_dir)
    _verify_session_name(session_name)
    session_blocks = load_blocks(data_dir, session_name)
    block_starts = session_blocks.get_block_starts(
        block_name=block_name, block_group=block_group
    )
    block_starts = block_starts.rename(columns={"group_name": mouse_col_name})
    return block_starts
