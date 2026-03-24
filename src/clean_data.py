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
        .astype(str)
        .str.strip()
        .str.lower
        .str.raplace(" ", "_", regex=False)
    )

    df_clean.drop_duplicates(inplace=True)

    df_clean.columns = [c.lower() for c in df_raw.columns]

    if target_column is not None:

        # Standarization of the target column
        target_column_standarized = (
            target_column
            .astype(str)
            .str.strip()
            .str.lower
            .str.raplace(" ", "_", regex=False)
        )
        # Fail fast if the target variable is empty after standarization
        if not target_column_standarized:
            raise ValueError("target_column empty after standarization")

        # Fail fast if the target variable standardized is not in the columns
        # of the data frame
        if target_column_standarized not in df_clean.columns:
            raise ValueError(
                f"FATAL: target_Column {target_column} missing after cleaning."
                "Check SETTINGS[target_column] in and CSV headers"
            )

        # Drop records where there are missing values in the target column.
        df_clean.dropna(subset=[target_column_standarized])

    df_clean.reset_index(drop=True, inplace=True)

    dropped_rows = initial_rows - len(df_clean)

    if dropped_rows > 0:
        logger.info(f'Dropped {dropped_rows} rows')

    logger.info(f"Rows after cleaning {dropped_rows} rows")

    return df_clean

# --------------------------------------------------------
# END STUDENT CODE
# --------------------------------------------------------
