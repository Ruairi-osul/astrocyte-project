from typing import Callable
import numpy as np
import pandas as pd
from trace_minder.preprocess import TracePreprocessor


class EventPreprocessor:
    """
    Preprocessor for event timings expressed as a float-valued array.

    Args:
        max_time (Optional[float], optional): Maximum time to include. Defaults to None.
        min_time (Optional[float], optional): Minimum time to include. Defaults to None.
        first_x_events (Optional[int], optional): Number of first events to include. Defaults to None.
        last_x_events (Optional[int], optional): Number of last events to include. Defaults to None.
        every_x_events (Optional[int], optional): Include every xth event. Defaults to None.
        x_set_of_y_events (Optional[int], optional): Include the first x events of y. Defaults to None.
    """

    def __init__(
        self,
        max_time: float | None = None,
        min_time: float | None = None,
        first_x_events: int | None = None,
        last_x_events: int | None = None,
        every_x_events: int | None = None,
        x_set_of_y_events: int | None = None,
    ):
        self.max_time = max_time
        self.min_time = min_time
        self.first_x_events = first_x_events
        self.last_x_events = last_x_events
        self.every_x_events = every_x_events
        self.x_set_of_y_events = x_set_of_y_events

    def subset_first_x_events(self, events: np.ndarray) -> np.ndarray:
        """
        Subset the first x events from an array of events.

        Args:
            events (np.ndarray): Array of events

        Returns:
            np.ndarray: Subset of the first x events
        """
        return events[: self.first_x_events]

    def subset_last_x_events(self, events: np.ndarray) -> np.ndarray:
        """
        Subset the last x events from an array of events.

        Args:
            events (np.ndarray): Array of events

        Returns:
            np.ndarray: Subset of the last x events
        """
        assert self.last_x_events is not None
        return events[-self.last_x_events :]

    def subset_every_x_events(self, events: np.ndarray) -> np.ndarray:
        """
        Subset every xth event from an array of events.

        Args:
            events (np.ndarray): Array of events

        Returns:
            np.ndarray: Subset of every xth event
        """
        return events[:: self.every_x_events]

    def subset_x_set_of_y_events(self, events: np.ndarray) -> np.ndarray:
        """
        Subset the first x events of y from an array of events.

        Args:
            events (np.ndarray): Array of events

        Returns:
            np.ndarray: Subset of the first x events of y
        """
        return events[: self.x_set_of_y_events]

    def subset_max_time(self, events: np.ndarray) -> np.ndarray:
        """
        Subset events that occur before a maximum time.

        Args:
            events (np.ndarray): Array of events

        Returns:
            np.ndarray: Subset of events that occur before a maximum time
        """
        return events[events <= self.max_time]

    def subset_min_time(self, events: np.ndarray) -> np.ndarray:
        """
        Subset events that occur after a minimum time.

        Args:
            events (np.ndarray): Array of events

        Returns:
            np.ndarray: Subset of events that occur after a minimum time
        """
        return events[events >= self.min_time]

    def __call__(self, events: np.ndarray) -> np.ndarray:
        """
        Call the event preprocessor on an array of events.

        Uses the set of event preprocessing sets defined during initialization.

        Args:
            events (np.ndarray): Array of events

        Returns:
            np.ndarray: Preprocessed array of events
        """
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
    """
    Preprocessor for grouped event timings expressed as a dataframe with event timing and group identifier columns.

    Args:
        df_events_group_col (str, optional): Name of the column in the dataframe that contains the group identifier. Defaults to "mouse_name".
        df_events_time_col (str, optional): Name of the column in the dataframe that contains the event timing. Defaults to "start_time".
        df_events_event_time_col (str, optional): Name of the column in the dataframe that contains the event timing. Defaults to "start_time".
        max_time (Optional[float], optional): Maximum time to include. Defaults to None.
        min_time (Optional[float], optional): Minimum time to include. Defaults to None.
        first_x_events (Optional[int], optional): Number of first events to include. Defaults to None.
        last_x_events (Optional[int], optional): Number of last events to include. Defaults to None.
        every_x_events (Optional[int], optional): Include every xth event. Defaults to None.
        x_set_of_y_events (Optional[int], optional): Include the first x events of y. Defaults to None.
    """

    def __init__(
        self,
        df_events_group_col: str = "mouse_name",
        df_events_time_col: str = "start_time",
        df_events_event_time_col: str = "start_time",
        max_time: float | None = None,
        min_time: float | None = None,
        first_x_events: int | None = None,
        last_x_events: int | None = None,
        every_x_events: int | None = None,
        x_set_of_y_events: int | None = None,
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

    def _filter_events(
        self, df: pd.DataFrame, func: Callable[[np.ndarray], np.ndarray]
    ) -> pd.DataFrame:
        events = func(df[self.df_events_time_col].values)
        df = df.loc[df[self.df_events_time_col].isin(events)]
        return df

    def _apply_event_preprocessor(
        self, df_events: pd.DataFrame, func: Callable[[np.ndarray], np.ndarray]
    ) -> pd.DataFrame:
        """
        Apply a subsetting function to each group of events

        Args:
            df_events (pd.DataFrame): DataFrame of event timings
            func (Callable[[np.ndarray], np.ndarray]): Function to apply to each group of events

        Returns:
            pd.DataFrame: DataFrame of event timings after applying the function
        """
        df_events = df_events.groupby(self.df_events_mouse_col).apply(
            lambda df: self._filter_events(df, func)
        )
        df_events = df_events.reset_index(drop=True)
        return df_events

    def subset_first_x_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Subset the first x events from each group of events

        Args:
            df_events (pd.DataFrame): DataFrame of event timings

        Returns:
            pd.DataFrame: DataFrame of event timings after subsetting the first x events
        """
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_first_x_events
        )

    def subset_last_x_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Subset the last x events from each group of events

        Args:
            df_events (pd.DataFrame): DataFrame of event timings

        Returns:
            pd.DataFrame: DataFrame of event timings after subsetting the last x events
        """
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_last_x_events
        )

    def subset_every_x_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Subset every xth event from each group of events

        Args:
            df_events (pd.DataFrame): DataFrame of event timings

        Returns:
            pd.DataFrame: DataFrame of event timings after subsetting every xth event
        """
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_every_x_events
        )

    def subset_x_set_of_y_events(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Subset the first x events of y from each group of events

        Args:
            df_events (pd.DataFrame): DataFrame of event timings

        Returns:
            pd.DataFrame: DataFrame of event timings after subsetting the first x events of y
        """
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_x_set_of_y_events
        )

    def subset_max_time(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Subset events that occur before a maximum time from each group of events

        Args:
            df_events (pd.DataFrame): DataFrame of event timings

        Returns:
            pd.DataFrame: DataFrame of event timings after subsetting events that occur before a maximum time
        """
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_max_time
        )

    def subset_min_time(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Subset events that occur after a minimum time from each group of events

        Args:
            df_events (pd.DataFrame): DataFrame of event timings
        """
        return self._apply_event_preprocessor(
            df_events, self.event_preprocessor.subset_min_time
        )

    def __call__(self, df_events: pd.DataFrame) -> pd.DataFrame:
        """
        Call the grouped event preprocessor on a dataframe of event timings

        Uses the set of event preprocessing sets defined during initialization.

        Args:
            df_events (pd.DataFrame): DataFrame of event timings

        Returns:
            pd.DataFrame: Preprocessed dataframe of event timings
        """
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
    """
    Preprocessor for traces and events.

    Container for trace and event preprocessors.

    Args:
        trace_preprocessor (TracePreprocessor, optional): Preprocessor for traces. Defaults
        to None.
        event_preprocessor (EventPreprocessor, optional): Preprocessor for events. Defaults
        to None.
        grouped_event_preprocessor (GroupedEventPreprocessor, optional): Preprocessor for
        grouped events. Defaults to None."""

    def __init__(
        self,
        trace_preprocessor: TracePreprocessor | None = None,
        event_preprocessor: EventPreprocessor | None = None,
        grouped_event_preprocessor: GroupedEventPreprocessor | None = None,
    ):
        self.trace_preprocessor = trace_preprocessor
        self.event_preprocessor = event_preprocessor
        self.grouped_event_preprocessor = grouped_event_preprocessor

    def preprocess_traces(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess traces using the trace preprocessor.

        Args:
            df (pd.DataFrame): DataFrame of traces in wide format.

        Returns:
            pd.DataFrame: Preprocessed DataFrame of traces in wide format.
        """
        if self.trace_preprocessor is not None:
            df = self.trace_preprocessor(df)
            return df
        else:
            raise ValueError("No trace preprocessor provided.")

    def process_events(
        self, events: np.ndarray | pd.DataFrame
    ) -> np.ndarray | pd.DataFrame:
        """
        Process events using the event preprocessor or grouped event preprocessor.

        Args:
            events (np.ndarray | pd.DataFrame): Array or DataFrame of events.

        Returns:
            np.ndarray | pd.DataFrame: Processed array or DataFrame of events.
        """
        if self.event_preprocessor is not None:
            events = self.event_preprocessor(events)
            return events
        elif self.grouped_event_preprocessor is not None:
            events = self.grouped_event_preprocessor(events)
            return events
        else:
            raise ValueError("No event preprocessor provided.")
