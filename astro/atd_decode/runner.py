from astro.load import GroupDataloader
from astro.preprocess.alignment_preprocessors import AlignmentPreprocessor
from astro.atd_decode.decoder import DecoderCV
from astro.atd_decode.atd_prepper import ATDPrepper


class ATDDecodeRunner:
    def __init__(
        self,
        data_loader: GroupDataloader,
        preprocessor: AlignmentPreprocessor,
        atd_prepper: ATDPrepper,
        decoder: DecoderCV,
    ):
        self.data_loader = data_loader
        self.preprocessor = preprocessor
        self.atd_prepper = atd_prepper
        self.decoder = decoder
        self.results_: dict | None = None

    def run(self) -> dict:
        df_traces, df_events = self.data_loader()
        df_aligned = self.preprocessor(df_traces, df_events)
        data, target, groups = self.atd_prepper(df_aligned)
        results = self.decoder.cross_validate(X=data, y=target, groups=groups)
        self.results_ = results
        return results
