from astro.preprocess import TracePreprocessor
from trace_minder.align import GroupedAligner
import numpy as np
import pandas as pd
from typing import Callable, Optional, Tuple, Union


ElementWiseMasker = Callable[[np.ndarray], np.ndarray]


def _is_negative(arr: np.ndarray) -> np.ndarray:
    return arr < 0


def _is_positive(arr: np.ndarray) -> np.ndarray:
    return arr > 0


def latency_mask_factory(t_min: float, t_max: float) -> ElementWiseMasker:
    def latency_mask(arr: np.ndarray) -> np.ndarray:
        return np.logical_and(arr >= t_min, arr < t_max)

    return latency_mask


class ATDecodePreprocessor:
    _in_block_int = 1
    _out_of_block_int = -1

    def __init__(
        self,
        aligner: GroupedAligner,
        latency_in_block: Optional[ElementWiseMasker] = None,
        latency_out_of_block: Optional[ElementWiseMasker] = None,
        aligned_preprocessor: Optional[TracePreprocessor] = None,
        in_block_string: str = "in_block",
        out_of_block_string: str = "out_of_block",
    ):
        self.aligner = aligner
        self.latency_in_block = latency_in_block
        self.latency_out_of_block = latency_out_of_block
        self.aligned_preprocessor = aligned_preprocessor
        self.in_block_string = in_block_string
        self.out_of_block_string = out_of_block_string

    def _gen_block_ts(
        self,
        latency_arr: Union[np.ndarray, pd.Series],
        latency_in_block: Callable[
            [np.ndarray],
            np.ndarray,
        ],
        latency_out_of_block: ElementWiseMasker,
    ) -> np.ndarray:
        """Takes a numpy array of latencies and returns a numpy array of block_ts

        The block_ts array is the same shape as the latency array, and contains
        the following values:
        - ATDecodePreprocessor._in_block_int: if latency_in_block is True
        - ATDecodePreprocessor._out_of_block_int: if latency_out_of_block is True
        - np.nan: if neither latency_in_block nor latency_out_of_block is True

        Raises a ValueError if latency_in_block and latency_out_of_block overlap

        Args:
            latency_arr (Union[np.ndarray, pd.Series]): Array of latencies
            latency_in_block (ElementWiseMasker): Callable that takes a numpy array of latencies and returns a numpy array of booleans
            latency_out_of_block (ElementWiseMasker): Callable that takes a numpy array of latencies and returns a numpy array of booleans

        Returns:
            np.ndarray: Array of block_ts
        """
        in_block = latency_in_block(latency_arr)
        out_of_block = latency_out_of_block(latency_arr)

        if np.any(np.logical_and(in_block, out_of_block)):
            raise ValueError("latency_in_block and latency_out_of_block overlap")

        block_ts = np.empty(latency_arr.shape)
        block_ts = np.select(
            condlist=[in_block, out_of_block],
            choicelist=[
                ATDecodePreprocessor._in_block_int,
                ATDecodePreprocessor._out_of_block_int,
            ],
            default=np.nan,
        )
        return block_ts

    def _filter_window(
        self, df_aligned: pd.DataFrame, block_ts: np.ndarray
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Filters df_aligned and block_ts to only include rows where block_ts is not np.nan

        block_ts is np.nan when the latency is not in a block or out of a block

        Args:
            df_aligned (pd.DataFrame): Aligned dataframe
            block_ts (np.ndarray): Array of block_ts

        Returns:
            Tuple[pd.DataFrame, np.ndarray]: Filtered df_aligned and block_ts
        """
        block_ts_not_nan = ~np.isnan(block_ts)
        return (
            df_aligned.iloc[block_ts_not_nan].reset_index(drop=True),
            block_ts[block_ts_not_nan],
        )

    def _map_block_ts_int_to_str(self, block_ts: pd.Series) -> pd.Series:
        """Maps block_ts from int to str

        Args:
            block_ts (pd.Series): block_ts series with int values as defined in ATDecodePreprocessor class attributes

        Returns:
            pd.Series: block_ts series with str values as defined in instance attributes
        """
        mapper = {
            ATDecodePreprocessor._in_block_int: self.in_block_string,
            ATDecodePreprocessor._out_of_block_int: self.out_of_block_string,
        }
        return block_ts.map(mapper)

    def _extract_temporal_cols(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extracts temporal columns from df

        Args:
            df (pd.DataFrame): Dataframe with temporal columns

        Returns:
            pd.DataFrame: Dataframe with only temporal columns
            pd.DataFrame: Input Dataframe without temporal columns
        """
        temporal_cols = [
            self.aligner.created_aligned_time_col,
            self.aligner.created_event_index_col,
            self.aligner.time_col,
        ]
        extracted_cols = [df.pop(col) for col in temporal_cols if col in df.columns]
        temporal_df = pd.concat(extracted_cols, axis=1)

        return temporal_df, df

    def __call__(
        self,
        df_traces: pd.DataFrame,
        block_starts: pd.DataFrame,
        latency_in_block: Optional[ElementWiseMasker] = None,
        latency_out_of_block: Optional[ElementWiseMasker] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
        if latency_in_block is None:
            if self.latency_in_block is None:
                raise ValueError("latency_in_block must be provided")
            latency_in_block = self.latency_in_block

        if latency_out_of_block is None:
            if self.latency_out_of_block is None:
                raise ValueError("latency_out_of_block must be provided")
            latency_out_of_block = self.latency_out_of_block

        # get aligned traces
        df_aligned = self.aligner.align(df_traces, block_starts)
        if self.aligned_preprocessor is not None:
            df_aligned = self.aligned_preprocessor(df_aligned)

        df_aligned = df_aligned.sort_values(self.aligner.time_col)

        # get block time series (with np.nan for out of range latencies and ints for in/out of block)
        block_ts = self._gen_block_ts(
            df_aligned[self.aligner.created_aligned_time_col].values,
            latency_in_block,
            latency_out_of_block,
        )

        # use block_ts to filter df_aligned and block_ts
        df_aligned, block_ts = self._filter_window(df_aligned, block_ts)

        # convert the block_ts array to a series of strings with the same index as df_aligned
        block_ts = pd.Series(block_ts, index=df_aligned.index)
        block_ts = self._map_block_ts_int_to_str(block_ts)

        # extract temporal columns from df_aligned
        temporal_df, df_aligned = self._extract_temporal_cols(df_aligned)

        return temporal_df, df_aligned, block_ts


def all_time_decode_pp_fac(
    window_1: Tuple[int, int],
    window_2: Tuple[int, int],
    aligner_fac: Callable[[], GroupedAligner],
) -> ATDecodePreprocessor:
    min_window_checker = latency_mask_factory(t_min=window_1[0], t_max=window_1[1])
    max_window_checker = latency_mask_factory(t_min=window_2[0], t_max=window_2[1])

    preprocessor = ATDecodePreprocessor(
        aligner=aligner_fac(),
        latency_out_of_block=min_window_checker,
        latency_in_block=max_window_checker,
    )
    return preprocessor
