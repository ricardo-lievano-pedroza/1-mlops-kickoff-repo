import logging
from pathlib import Path

import pandas as pd

from src.utils import load_csv

logger = logging.getLogger(__name__)


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path to the raw CSV file.

    Outputs:
    - df_raw: Raw DataFrame loaded from disk.

    Why this contract matters for reliable ML delivery:
    - “Same inputs, same outputs” is the foundation of reproducible ML pipelines.
    """
    logger.info(f"Loading raw data from {raw_data_path}")

    if not raw_data_path.exists():
        raise FileNotFoundError(
            f"Ingestion Error: The raw data file was not found at {raw_data_path}"
            f"Ensure raw dataset is placed in the 'data/raw/' directory"
        )

    if not raw_data_path.is_file():
        raise ValueError(
            f"Ingestion Error: {raw_data_path} is a directory, not a file"
            "Check path in the config.yaml file"
        )

    df_raw = load_csv(raw_data_path)

    if df_raw is None or df_raw.empty:
        raise ValueError(
            "Loaded dataframe is empty.\n"
            f"File path: {raw_data_path}\n"
            "Fix:\n"
            "Check the file contents"
        )

    logger.info(f"Loaded dataframe shape: {df_raw.shape}")

    return df_raw
