import logging
from typing import Optional
import pandas as pd

"""
Educational Goal:
- Why this module exists in an MLOps system: Separate dataset cleaning from modeling to reduce hidden notebook state and rework.
- Responsibility (separation of concerns): Apply deterministic cleaning rules and produce a cleaned DataFrame for downstream steps.
- Pipeline contract (inputs and outputs): Input is raw df + target column name; output is cleaned df (same rows/cols unless student changes).
"""
# --------------------------------------------------------
# START STUDENT CODE
# --------------------------------------------------------

logger = logging.getLogger(__name__)


def clean_dataframe(
        df_raw: pd.DataFrame,
        target_column: Optional[str] = None
        ) -> pd.DataFrame:
    """
    Data cleeaner module usefull for both training and inference: 

    Training:
    - target_column is required
    - Standarization of column names
    - Drop duplicates
    - Drop rows with missing target values

    Inference:
    - Standarization of column names
    - Drop duplicates
    - target_column not requiered

    Inputs:
    - df_raw: Raw pandas DataFrame
    - target_column: Name of the target column (optional)
    
    Outputs:
    - df_clean: Cleaned pandas DataFrame
    """
    logger.info("Cleaning dataframe")

    if df_raw is None:
        raise ValueError(
            "df_raw is empty. Check src/load_data.py and path for raw data in config.yaml"
        )
    if not isinstance(df_raw, pd.DataFrame):
        raise TypeError(
            f"df_rawmust be a pandas DataFrame. df_raw type: {type(df_raw)}"
        )

    df_clean = df_raw.copy()

    initial_rows = len(df_clean)

    # Standarization of the column headers
    df_clean.columns = (
        df_clean.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
    )

    categorical_columns = df_clean.select_dtypes(include=["object", "category"]).columns.tolist()
    for c in categorical_columns:
        df_clean[c] = df_clean[c].str.strip().str.lower().str.replace(" ", "_")

    df_clean.drop_duplicates(inplace=True)

    df_clean.columns = [c.lower() for c in df_raw.columns]

    if target_column is not None:
        if target_column not in df_clean.columns:
            raise ValueError(
                f"FATAL: target_column '{target_column}' missing after cleaning. "
                "Check SETTINGS[target_column] and CSV headers."
                )

        try:
            df_clean[target_column] = df_clean[target_column].astype(float)  # actually assign it back
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Target variable is not numeric, got type={df_clean[target_column].dtype}"
            ) from e

        # Drop records where there are missing values in the target column.
        df_clean.dropna(subset=[target_column], inplace= True)

    df_clean.reset_index(drop=True, inplace=True)

    dropped_rows = initial_rows - len(df_clean)

    if dropped_rows > 0:
        logger.info(f'Dropped {dropped_rows} rows')

    logger.info(f"Rows after cleaning {dropped_rows} rows")

    return df_clean

# --------------------------------------------------------
# END STUDENT CODE
# --------------------------------------------------------
