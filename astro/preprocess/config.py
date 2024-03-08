from dataclasses import dataclass
from trace_minder.align import GroupedAligner
from trace_minder.preprocess import TracePreprocessor
from trace_minder.surrogates import Rotater, TraceSampler
from .raw_preprocessors import GroupedEventPreprocessor
from .neuron_drop import NeuronDropper


@dataclass
class PreprocessConfig:
    """
    Container for preprocessing configuration.

    Args:
        aligner (GroupedAligner): trace_minder.align.GroupedAligner instance
        trace_preprocessor_load (TracePreprocessor, optional): trace_minder.preprocess.TracePreprocessor instance. Defaults to None.
        event_preprocessor (GroupedEventPreprocessor, optional): trace_minder.preprocess.GroupedEventPreprocessor instance. Defaults to None.
        neuron_dropper (NeuronDropper, optional): astro.preprocess.neuron_drop.NeuronDropper instance. Defaults to None.
        trace_rotator (Rotater, optional): trace_minder.surrogates.Rotater instance. Defaults to None.
        trace_sampler (TraceSampler, optional): trace_minder.surrogates.TraceSampler instance. Defaults to None.
        trace_preprocessor_post_alignment (TracePreprocessor, optional): trace_minder.preprocess.TracePreprocessor instance. Defaults to None.
    """

    aligner: GroupedAligner
    trace_preprocessor_load: TracePreprocessor | None = None
    event_preprocessor: GroupedEventPreprocessor | None = None
    neuron_dropper: NeuronDropper | None = None
    trace_rotator: Rotater | None = None
    trace_sampler: TraceSampler | None = None
    trace_preprocessor_post_alignment: TracePreprocessor | None = None
