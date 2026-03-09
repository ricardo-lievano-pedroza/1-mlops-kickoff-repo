"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Stop the pipeline early when data quality or schema drift would break training or inference
- Responsibility (separation of concerns): Validate schema and simple column expectations, without transforming data
- Pipeline contract (inputs and outputs): DataFrame in, True out, raises ValueError on critical failures

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


# --------------------------------------------------------
# START STUDENT CODE
# --------------------------------------------------------
def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: pandas DataFrame to validate
    - required_columns: list of column names expected to exist
    Outputs:
    - is_valid: True if validation passes
    Why this contract matters for reliable ML delivery:
    - Early validation prevents wasted runs and reduces silent failures caused by broken or drifting schemas
    """
    print("[validate.validate_dataframe] Validating dataframe")

    if df.empty:
        raise ValueError("Validation failed: dataframe is empty")

    print("[validate] Normalizing column names")
    df.columns = df.columns.str.strip()

    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        missing_str = ", ".join(missing_required)
        raise ValueError(
            f"Validation failed: missing required columns: {missing_str}"
        )

    target_column = "Rent"

    if target_column not in df.columns:
        raise ValueError(
            "Validation failed: target column 'Rent' not found"
        )

    if df[target_column].isna().any():
        raise ValueError(
            "Validation failed: target column 'Rent' "
            "contains missing values"
        )

    total_records = len(df)

    print("[validate] Missing values summary per column")

    for col in df.columns:
        na_count = int(df[col].isna().sum())
        na_ratio = na_count / total_records
        summary = (
            f"[validate] {col} NA: {na_count}/{total_records} "
            f"({na_ratio:.2%})"
        )
        print(summary)

        if na_ratio > 0.5:
            raise ValueError(
                f"Validation failed: column '{col}' "
                f"has too many missing values ({na_ratio:.2%})"
            )

    start_numeric_col = "Rent"
    start_numeric_idx = df.columns.get_loc(start_numeric_col)
    numeric_cols = list(df.columns[start_numeric_idx:])

    def _to_numeric_temp(series: pd.Series) -> pd.Series:
        s = series.copy()

        s_str = (
            s.astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .astype("string")
        )

        s_str = s_str.str.replace(".", "", regex=False)
        s_str = s_str.str.replace(",", ".", regex=False)

        s_num = pd.to_numeric(s_str, errors="coerce")
        return s_num

    print("[validate] Checking numeric columns from Rent to last column")

    for col in numeric_cols:
        s_num = _to_numeric_temp(df[col])

        non_null_count = int(df[col].notna().sum())
        converted_non_null_count = int(s_num.notna().sum())

        if non_null_count > 0 and converted_non_null_count == 0:
            raise ValueError(
                f"Validation failed: column '{col}' must be numeric"
            )

    start_binary_col = "Outer"
    start_binary_idx = df.columns.get_loc(start_binary_col)
    binary_cols = list(df.columns[start_binary_idx:])

    print("[validate] Checking binary columns from Outer to last column")

    for col in binary_cols:
        s_num = _to_numeric_temp(df[col])
        s_non_null = s_num.dropna()

        if s_non_null.empty:
            raise ValueError(
                f"Validation failed: binary column '{col}' "
                "has only missing values"
            )

        unique_vals = set(s_non_null.unique().tolist())

        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Validation failed: column '{col}' must be binary 0/1"
            )

    return True
