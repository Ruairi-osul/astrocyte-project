from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import os
from dotenv import load_dotenv


@dataclass
class Config:
    data_dir: Optional[Path] = None
    derived_data_dir: Optional[Path] = None
    fig_dir: Optional[Path] = None

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> "Config":
        if env_file is not None:
            load_dotenv(dotenv_path=env_file)
        data_dir = Path(os.environ["DATA_DIR"])
        derived_data_dir = Path(os.environ["DERIVED_DATA_DIR"])
        fig_dir = Path(os.environ["FIG_DIR"])
        return cls(
            data_dir=data_dir, derived_data_dir=derived_data_dir, fig_dir=fig_dir
        )
