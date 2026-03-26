import logging
from typing import Optional, List
import pandas as pd

logger = logging.getLogger(__name__)


def validate_dataframe(
        df: pd.DataFrame,
        required_columns: List,
        target_column: Optional[str] = None,
        numeric_non_negative_cols: Optional[List[str]] = None
        ) -> bool:

    """
    Inputs:
    - df: DataFrame to validate
    - required_columns: List of required column names
    - numeric_non_negative_cols: optional list of columns where the values must be positive

    Outputs:
    - bool: True if valid; raises ValueError for obvious issues

    Why this contract matters for reliable ML delivery:
    - Prevents downstream crashes and protects business timelines by catching schema breaks immediately.
    """
    logger.info("Validating dataframe")

    if df is None or len(df) == 0:
        raise ValueError(
            "Validation failed: DataFrame is empty. Check your data loading and src/clean_data.py module."
        )

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Validation failed: df must be a DataFrame, got type ={type(df)}"
        )

    if required_columns is None or len(required_columns) == 0:
        raise ValueError(
            "Validation failed: required columns is empty"
        )
    required_columns = [
        str(col)
        .lower()
        .strip()
        .replace(" ", "_")
        .replace(".", "_")
        for col in required_columns
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Validation failed: Missing required columns: {missing}"
            )

    if target_column is not None:
        if target_column.lower() not in df.columns:
            raise ValueError(
                f"Validation failed: target_column: {target_column} not found in df"
            )

        if df[target_column.lower()].isna().any():
            raise ValueError(
                "Validation failed: missing values found in the target column"
            )

    numeric_non_negative_cols = numeric_non_negative_cols or []
    for c in numeric_non_negative_cols:
        if c not in df.columns:
            raise ValueError(
                f"Validaiton failed: {c} not found in df"
            )

        if df[c].dtype not in [int, float]:
            raise TypeError(
                f"Validation failed: column {c} found but is not numeric"
            )

        if (df[c] < 0).any():
            raise ValueError(
                f"Validation failed: column {c} has negative values"
            )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # Put this on the config file
    NULL_RATE_THRESHOLD = 0.8
    RENT_UPPER_BOUND = 20000.0   # monthly rent - adjust to your market
    SQMT_UPPER_BOUND = 5000.0    # max plausible area (m^2)
    BEDROOMS_MAX = 10
    FLOORS_MAX = 100

    # Convenient short-hands
    missing_rates = df.isna().mean()

    # Null-rate checks
    high_nulls = [
        c for c, r in missing_rates.items() if r > NULL_RATE_THRESHOLD
        ]

    if high_nulls:
        raise ValueError(
            f"Validation failed: High null rate (> {NULL_RATE_THRESHOLD:.0%})"
            f"in columns: {high_nulls}")

    # 3) Numeric dtype & range checks
    numeric_checks = {
        "rent": {"min": 0.0, "max": RENT_UPPER_BOUND},
        "sq.mt": {"min": 0.01, "max": SQMT_UPPER_BOUND},
        "floors": {"min": 0, "max": FLOORS_MAX},
        "bedrooms": {"min": 0, "max": BEDROOMS_MAX},
    }

    for col, bounds in numeric_checks.items():
        if col in df.columns:
            # dtype/coercion check
            try:
                col_series = pd.to_numeric(df[col].dropna())
            except Exception:
                raise ValueError(f"Validation failed: Column '{
                    col}' must be numeric or coercible to numeric."
                    )
            # range checks (only applied to non-empty series)
            if len(col_series) > 0:
                if (col_series < bounds["min"]).any():
                    raise ValueError(
                        f"Validation failed: Column '{col}' "
                        f"has values < {bounds['min']} (invalid).")
                if (col_series > bounds["max"]).any():
                    raise ValueError(
                        f"Validation failed: Column '{col}' has "
                        f"values > {bounds['max']} (suspicious).")

    # 4) Binary column checks
    # TODO put this line in terms of the config file
    binary_cols = ["outer", "duplex", "semidetached", "cottage", "elevator",
                   "penthouse"]
    for col in binary_cols:
        if col in df.columns:
            # allowed set: {0,1,True,False,"0","1","Y","N"} - we convert
            # strings where sensible
            unique_vals = set(df[col].dropna().unique())
            # normalize some common representations to check membership
            normalized = set()
            for v in unique_vals:
                if isinstance(v, str):
                    vv = v.strip().lower()
                    if vv in {"y", "yes", "true", "1"}:
                        normalized.add(1)
                    elif vv in {"n", "no", "false", "0"}:
                        normalized.add(0)
                    else:
                        normalized.add(vv)
                elif isinstance(v, (int, float, bool)):
                    normalized.add(int(v))
                else:
                    normalized.add(v)
            # Accept if all normalized values are subset of {0,1}
            if not normalized.issubset({0, 1}):
                raise ValueError(f"Validation failed: Binary-like column "
                                 f"'{col}' contains unexpected values:"
                                 f"{unique_vals}")

    # 5) Duplicate rows check (optional)
    if df.duplicated().any():
        raise ValueError(
                "Warning: Duplicate rows detected in dataframe. "
                "Consider deduping in clean step."
            )

    return True
