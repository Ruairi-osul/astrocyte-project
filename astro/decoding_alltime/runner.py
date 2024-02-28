# from .config import RunConfig, RunConfigBlock, RunResults
from .preprocess import ATDecodePreprocessor
from .model_fit import ModelFitter
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelEncoder
from dataclasses import dataclass
import numpy as np
from typing import Callable


class ATDecodeRunner:
    def run(self, *args, **kwargs):
        raise NotImplementedError


@dataclass
class RunResults:
    """
    The results of a single run of all time decoding.

    Args:
        df_temporal (pd.DataFrame): A dataframe of time values after preprocessing
        df_aligned (pd.DataFrame): A dataframe of aligned predictors after preprocessing
        target_ts (pd.Series): A series of target values after preprocessing
        groups (pd.Series | np.ndarray): If applicable, a series of group values after preprocessing
        clf (BaseEstimator): A fitted classifier
        le (LabelEncoder): A fitted label encoder
        score (float): The score of the classifier
        desc (str): A description of the run
    """

    df_temporal: pd.DataFrame = None
    df_aligned: pd.DataFrame = None
    target_ts: pd.Series = None
    groups: pd.Series | np.ndarray = None
    clf: BaseEstimator = None
    le: LabelEncoder = None
    score: float = None
    desc: str = None


def all_time_decode_block(
    df_traces: pd.DataFrame,
    df_event_starts: pd.DataFrame,
    preprocessor: ATDecodePreprocessor,
    model_fitter: ModelFitter,
    model: BaseEstimator,
    group_getter: Callable[
        [pd.DataFrame, pd.DataFrame, pd.DataFrame], pd.Series | np.ndarray
    ]
    | None = None,
) -> RunResults:
    """
    Run the ATDecode model on the given data.
    """
    # preprocess data
    df_temporal, df_aligned, target_ts = preprocessor(
        df_traces=df_traces, block_starts=df_event_starts
    )
    match group_getter:
        case None:
            groups = None
        case _:
            groups = group_getter(df_temporal, df_aligned, target_ts)

    le = LabelEncoder()
    y = le.fit_transform(target_ts)
    # print(groups)
    # fit model
    model = model_fitter.fit_model(
        df_predictors=df_aligned, target=y, model=model, groups=groups
    )

    score = model_fitter.eval_model(
        df_predictors=df_aligned, target=y, model=model, groups=groups
    )

    # construct results
    results = RunResults(
        df_temporal=df_temporal,
        df_aligned=df_aligned,
        target_ts=target_ts,
        groups=groups,
        clf=model,
        le=le,
        score=score,
    )

    return results
