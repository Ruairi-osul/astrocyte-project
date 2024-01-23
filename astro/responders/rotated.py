from typing import Optional
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from trace_minder.preprocess import TracePreprocessor
from trace_minder.align import GroupedAligner
from trace_minder.trace_aggregation import PrePostAggregator
from trace_minder.responders.rotated_responder import AUCDiffResponders
from astro.preprocess import Preprocessor
from astro.load import Loader


class RespondersSaver:
    def __init__(
        self,
        root_data_dir: Path,
        responders_fn: bool = "responders.csv",
        reps_fn: Optional[str] = "responders_reps.parquet",
        aligned_fn: Optional[str] = "aligned_traces.parquet",
        compression: Optional[str] = "snappy",
    ):
        self.root_data_dir = root_data_dir
        self.responders_fn = responders_fn
        self.reps_fn = reps_fn
        self.aligned_fn = aligned_fn
        self.compression = compression
        self.save_dir_ = None

    def _get_save_dir(self, name: Optional[str] = None):
        match name:
            case None:
                self.save_dir_ = self.root_data_dir
            case _:
                self.save_dir_ = self.root_data_dir / name

        self.save_dir_.mkdir(exist_ok=True, parents=True)
        return self.save_dir_

    def set_fn_suffix(self, suffix: str):
        self.responders_fn = Path(self.responders_fn).stem + suffix + ".csv"
        if self.reps_fn is not None:
            self.reps_fn = Path(self.reps_fn).stem + suffix + ".parquet"
        if self.aligned_fn is not None:
            self.aligned_fn = Path(self.aligned_fn).stem + suffix + ".parquet"

    def save(
        self,
        df_responders: pd.DataFrame,
        name: Optional[str] = None,
        df_aligned: Optional[pd.DataFrame] = None,
        df_reps: Optional[pd.DataFrame] = None,
    ):
        save_dir = self._get_save_dir(name)
        df_responders.to_csv(save_dir / self.responders_fn, index=False)

        if df_aligned is not None and self.aligned_fn is not None:
            df_aligned.to_parquet(
                save_dir / self.aligned_fn, index=False, compression=self.compression
            )
        if df_reps is not None and self.reps_fn is not None:
            df_reps.to_parquet(
                save_dir / self.reps_fn, index=False, compression=self.compression
            )


@dataclass
class RespondersConfig:
    loader_preprocessor: Preprocessor
    aligner: GroupedAligner
    average_trace_preprocessor: TracePreprocessor
    aggregator: PrePostAggregator
    n_boot: int = 50


def run_responders(
    name: str,
    loader: Loader,
    responders_config: RespondersConfig,
    saver: Optional[RespondersSaver] = None,
):
    store_reps = True if saver is not None else False

    df_traces = loader.load_traces()
    df_events = loader.load_blockstarts()

    responders_calculator = AUCDiffResponders(
        aligner=responders_config.aligner,
        average_trace_preprocessor=responders_config.average_trace_preprocessor,
        aggregator=responders_config.aggregator,
        n_boot=responders_config.n_boot,
        _store_reps=store_reps,
    )
    df_responders = responders_calculator.get_responders(df_traces, df_events)

    if saver is not None:
        saver.save(df_responders, name=name, df_aligned=df_responders)
