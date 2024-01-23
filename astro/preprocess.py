from typing import Optional, Callable, Union
import numpy as np
import pandas as pd
from trace_minder.preprocess import TracePreprocessor


class EventPreprocessor:
    def __init__(
        self,
        max_time: Optional[float] = None,
        min_time: Optional[float] = None,
        first_x_events: Optional[int] = None,
        last_x_events: Optional[int] = None,
        every_x_events: Optional[int] = None,
        x_set_of_y_events: Optional[int] = None,
    ):
        self.max_time = max_time
        self.min_time = min_time
        self.first_x_events = first_x_events
        self.last_x_events = last_x_events
        self.every_x_events = every_x_events
        self.x_set_of_y_events = x_set_of_y_events

    def subset_first_x_events(self, events: np.ndarray) -> np.ndarray:
        return events[: self.first_x_events]

    def subset_last_x_events(self, events: np.ndarray) -> np.ndarray:
        assert self.last_x_events is not None
        return events[-self.last_x_events :]

    def subset_every_x_events(self, events: np.ndarray) -> np.ndarray:
        return events[:: self.every_x_events]

    def subset_x_set_of_y_events(self, events: np.ndarray) -> np.ndarray:
        return events[: self.x_set_of_y_events]

    def subset_max_time(self, events: np.ndarray) -> np.ndarray:
        return events[events <= self.max_time]

    def subset_min_time(self, events: np.ndarray) -> np.ndarray:
        return events[events >= self.min_time]

    def __call__(self, events: np.ndarray) -> np.ndarray:
        if self.first_x_events is not None:
            events = self.subset_first_x_events(events)
        if self.last_x_events is not None:
            events = self.subset_last_x_events(events)
        if self.every_x_events is not None:
            events = self.subset_every_x_events(events)
        if self.x_set_of_y_events is not None:
            events = self.subset_x_set_of_y_events(events)
        if self.max_time is not None:
            events = self.subset_max_time(events)
        if self.min_time is not None:
            events = self.subset_min_time(events)
        return events


class GroupedEventPreprocessor:
    def __init__(
        self,
        df_events_group_col: str = "mouse_name",
        df_events_time_col: str = "start_time",
        df_events_event_time_col: str = "start_time",
        max_time: Optional[float] = None,
        min_time: Optional[float] = None,
        first_x_events: Optional[int] = None,
        last_x_events: Optional[int] = None,
        every_x_events: Optional[int] = None,
        x_set_of_y_events: Optional[int] = None,
    ):
        self.df_events_mouse_col = df_events_group_col
        self.df_events_time_col = df_events_time_col
        self.df_events_event_time_col = df_events_event_time_col
        self.event_preprocessor = EventPreprocessor(
            max_time=max_time,
            min_time=min_time,
            first_x_events=first_x_events,
            last_x_events=last_x_events,
            every_x_events=every_x_events,
            x_set_of_y_events=x_set_of_y_events,
        )
        self.max_time = max_time
        self.min_time = min_time
        self.first_x_events = first_x_events
        self.last_x_events = last_x_events
        self.every_x_events = every_x_events
        self.x_set_of_y_events = x_set_of_y_events

    def _filter_events(self, df, func):
        events = func(df[self.df_events_time_col].values)
        df = df.loc[df[self.df_events_time_col].isin(events)]
        return df

    def _apply_event_preprocessor(
        self, df_events: pd.DataFrame, func: Callable[[np.ndarray], np.ndarray]
    ) -> pd.DataFrame:
        df_events = df_events.groupby(self.df_events_mouse_col).apply(
            lambda df: self._filter_events(df, func)
        )
        df_events = df_events.reset_index(drop=True)
        return df_events

    def subset_first_x_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_first_x_events
        )

    def subset_last_x_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_last_x_events
        )

    def subset_every_x_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_every_x_events
        )

    def subset_x_set_of_y_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_x_set_of_y_events
        )

    def subset_max_time(self, df_events: pd.DataFrame) -> pd.DataFrame:
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_max_time
        )

    def subset_min_time(self, df_events: pd.DataFrame) -> pd.DataFrame:
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_min_time
        )

    def __call__(self, df_events: pd.DataFrame) -> pd.DataFrame:
        if self.first_x_events is not None:
            df_events = self.subset_first_x_events(df_events)
        if self.last_x_events is not None:
            df_events = self.subset_last_x_events(df_events)
        if self.every_x_events is not None:
            df_events = self.subset_every_x_events(df_events)
        if self.x_set_of_y_events is not None:
            df_events = self.subset_x_set_of_y_events(df_events)
        if self.max_time is not None:
            df_events = self.subset_max_time(df_events)
        if self.min_time is not None:
            df_events = self.subset_min_time(df_events)
        return df_events


class Preprocessor:
    def __init__(
        self,
        trace_preprocessor: Optional[TracePreprocessor] = None,
        event_preprocessor: Optional[EventPreprocessor] = None,
        grouped_event_preprocessor: Optional[GroupedEventPreprocessor] = None,
    ):
        self.trace_preprocessor = trace_preprocessor
        self.event_preprocessor = event_preprocessor
        self.grouped_event_preprocessor = grouped_event_preprocessor

    def preprocess_traces(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.trace_preprocessor is not None:
            df = self.trace_preprocessor(df)
            return df
        else:
            raise ValueError("No trace preprocessor provided.")

    def process_events(
        self, events: Union[np.ndarray, pd.DataFrame]
    ) -> Union[np.ndarray, pd.DataFrame]:
        if self.event_preprocessor is not None:
            events = self.event_preprocessor(events)
            return events
        elif self.grouped_event_preprocessor is not None:
            events = self.grouped_event_preprocessor(events)
            return events
        else:
            raise ValueError("No event preprocessor provided.")
