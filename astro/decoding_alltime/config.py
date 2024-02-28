from .preprocess import ATDecodePreprocessor
from .model_fit import ModelFitter
from .runner import all_time_decode_block, RunResults
from astro.preprocess import Preprocessor
from astro.load import Loader, GroupSessionData
from astro.transforms import GroupSplitter
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generator
from trace_minder.surrogates import SurrogateGenerator, SurrogateTemplate
import pandas as pd
import numpy as np

# ADD PARAMS FOR GROUP GETTER


@dataclass
class ConfigTemplate:
    """
    A template for creating a configuration that holds all data and metadata
    needed for all time decoding.

    Args:
        data_dir: The directory containing the raw data.
        session_name: The name of the session.
        group: The group to analyze.
        loader_preprocessor: The preprocessor to use when loading data.
        block_group: The block group to analyze.
        atd_preprocessor: The ATDecodePreprocessor to use. This takes aligned traces and events and returns (temporal_df, df_preds, df_targets)
        model_factory: A callable that returns a model to use for decoding.
        model_fitter: A ModelFitter to use for fitting the model. This class has methods for fitting and evaluating the model.
        group_getter: If using group cross validation, a callable that returns a group array, else None.
        surrogate_factory: If generating surrogates, a callable that returns either a SurrogateGenerator or a SurrogateTemplate, else None.

    """

    data_dir: Path
    session_name: str
    group: str
    loader_preprocessor: Preprocessor
    block_group: str
    atd_preprocessor: ATDecodePreprocessor
    group_splitter: GroupSplitter
    model_factory: Callable[[], Pipeline]
    model_fitter: ModelFitter
    group_getter: Callable[
        [pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray
    ] | None = None
    surrogate_factory: Callable[
        [], SurrogateGenerator | SurrogateTemplate
    ] | None = None


@dataclass
class ATDConfig:
    """
    A configuration that holds all data and metadata needed for all time decoding.

    Args:
        session_name: The name of the session being analyzed.
        group: The group being analyzed.
        block_group: The block group being analyzed.
        df_traces: The raw traces for neurons of the group on the session.
        df_block_starts: The block start times for the group on the session.
        atd_preprocessor: The ATDecodePreprocessor used. This takes aligned traces and events and returns (temporal_df, df_preds, df_targets)
        model: The model used for decoding.
        model_fitter: The ModelFitterFrac used for fitting the model. This class has methods for fitting and evaluating the model.
        group_getter: If using group cross validation, a callable that returns a group array, else None.
        surrogate_generater: If generating surrogates, either a SurrogateGenerator or a SurrogateTemplate, else None.
    """

    session_name: str
    group: str
    block_group: str
    df_traces: pd.DataFrame
    df_block_starts: pd.DataFrame
    atd_preprocessor: ATDecodePreprocessor
    model: Pipeline
    model_fitter: ModelFitter
    group_getter: Callable[
        [pd.DataFrame, pd.DataFrame, pd.DataFrame], np.ndarray
    ] | None = None
    surrogate_generator: SurrogateTemplate | SurrogateGenerator | None = None


def generate_configuration(
    atd_config_templates: list[ConfigTemplate],
) -> Generator[ATDConfig, None, None]:
    """
    A generator that yields ATDConfig objects for each ConfigTemplate in the list.

    Args:
        atd_config_templates: A list of ConfigTemplates to use to generate ATDConfig objects.

    Yields:
        ATDConfig objects.
    """
    for config_template in atd_config_templates:
        yield create_configuration(config_template)


def create_configuration(config_template: ConfigTemplate) -> ATDConfig:
    """
    Create an ATDConfig object from a ConfigTemplate.

    Args:
        config_template: The ConfigTemplate to use to create the ATDConfig.

    Returns:
        An ATDConfig object.
    """
    loader = Loader(
        data_dir=config_template.data_dir,
        preprocessor=config_template.loader_preprocessor,
        session_name=config_template.session_name,
    )

    group_session_data = GroupSessionData(
        group=config_template.group,
        loader=loader,
        group_splitter=config_template.group_splitter,
    )

    df_traces = group_session_data.df_traces
    df_block_starts = group_session_data.df_block_starts(
        block_group=config_template.block_group
    )
    match config_template.surrogate_factory:
        case None:
            surrogate_generator = None
        case _:
            surrogate_generator = config_template.surrogate_factory()

    return ATDConfig(
        session_name=config_template.session_name,
        group=config_template.group,
        block_group=config_template.block_group,
        df_traces=df_traces,
        df_block_starts=df_block_starts,
        atd_preprocessor=config_template.atd_preprocessor,
        surrogate_generator=surrogate_generator,
        model=config_template.model_factory(),
        group_getter=config_template.group_getter,
        model_fitter=config_template.model_fitter,
    )


def process_atd_config(atd_config: ATDConfig, desc: str | None = None) -> RunResults:
    match atd_config.surrogate_generator:
        case None:
            df_traces = atd_config.df_traces
        case _:
            df_traces = atd_config.surrogate_generator(df_traces=atd_config.df_traces)
    res = all_time_decode_block(
        df_traces=df_traces,
        df_event_starts=atd_config.df_block_starts,
        preprocessor=atd_config.atd_preprocessor,
        model=atd_config.model,
        group_getter=atd_config.group_getter,
        model_fitter=atd_config.model_fitter,
    )
    if desc is None:
        res.desc = (
            f"{atd_config.session_name}__{atd_config.group}__{atd_config.block_group}"
        )
    return res
