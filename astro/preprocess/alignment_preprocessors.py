import pandas as pd
from .config import PreprocessConfig
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    def __init__(self, preprocess_config: PreprocessConfig):
        self.preprocess_config = preprocess_config

    @abstractmethod
    def preprocess_raw_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def preprocess_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        ...

    @abstractmethod
    def align_traces(
        self, df_traces: pd.DataFrame, df_events: pd.DataFrame
    ) -> pd.DataFrame:
        ...

    @abstractmethod
    def preprocess_aligned_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        ...


class AlignmentPreprocessor(BasePreprocessor):
    def preprocess_raw_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_config.trace_preprocessor_load is not None:
            df_traces = self.preprocess_config.trace_preprocessor_load(df_traces)
        if self.preprocess_config.trace_rotator is not None:
            df_traces = self.preprocess_config.trace_rotator(df_traces)
        if self.preprocess_config.trace_sampler is not None:
            df_traces = self.preprocess_config.trace_sampler(df_traces)
        return df_traces

    def preprocess_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_config.event_preprocessor is not None:
            df_events = self.preprocess_config.event_preprocessor(df_events)
        return df_events

    def align_traces(
        self, df_traces: pd.DataFrame, df_events: pd.DataFrame
    ) -> pd.DataFrame:
        return self.preprocess_config.aligner.align(df_traces, df_events)

    def preprocess_aligned_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_config.trace_preprocessor_post_alignment is not None:
            df_traces = self.preprocess_config.trace_preprocessor_post_alignment(
                df_traces
            )
        return df_traces

    def __call__(
        self, df_traces: pd.DataFrame, df_events: pd.DataFrame
    ) -> pd.DataFrame:
        df_traces = self.preprocess_raw_traces(df_traces)
        df_events = self.preprocess_events(df_events)
        df_traces = self.align_traces(df_traces, df_events)
        df_traces = self.preprocess_aligned_traces(df_traces)
        return df_traces
