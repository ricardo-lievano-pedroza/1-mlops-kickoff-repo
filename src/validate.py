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
    print("[validate.validate_dataframe] Validating dataframe")  # TODO: replace with logging later

    if df.empty:
        raise ValueError("Validation failed: dataframe is empty")
    print("[validate] Normalizing column names")  # TODO: replace with logging later
    df.columns = df.columns.str.strip()

    missing_required = [c for c in required_columns if c not in df.columns]
    if missing_required:
        raise ValueError(f"Validation failed: missing required columns: {missing_required}")
        
    target_column = "Rent"

    if target_column not in df.columns:
        raise ValueError("Validation failed: target column 'Rent' not found")

    if df[target_column].isna().any():
        raise ValueError("Validation failed: target column 'Rent' contains missing values")

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Validation rules depend on business rules and the dataset contract
    # Examples:
    # 1. Add column-level missingness thresholds
    # 2. Add range checks and consistency checks across columns
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")

    total_records = len(df)

   # 1) Missing values ratio per column
print("[validate] Missing values summary per column")  # TODO: replace with logging later

for col in df.columns:
    na_count = int(df[col].isna().sum())
    na_ratio = na_count / total_records
    print(f"[validate] {col} NA: {na_count}/{total_records} ({na_ratio:.2%})")

    # Fail if more than 50% missing
    if na_ratio > 0.5:
        raise ValueError(
            f"Validation failed: column '{col}' has too many missing values ({na_ratio:.2%})"
        )

    # 2) Numeric check from Rent to last column, without modifying df
    start_numeric_col = "Rent"
    if start_numeric_col not in df.columns:
        raise ValueError("Validation failed: column 'Rent' not found")

    start_numeric_idx = df.columns.get_loc(start_numeric_col)
    numeric_cols = list(df.columns[start_numeric_idx:])

    def _to_numeric_temp(series: pd.Series) -> pd.Series:
        print(f"[validate] Converting to numeric for check: {series.name}")  # TODO: replace with logging later
        s = series.copy()

        s_str = (
            s.astype(str)
            .str.strip()
            .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
            .astype("string")
        )

        # Support thousands dot and decimal comma formatting
        s_str = s_str.str.replace(".", "", regex=False)
        s_str = s_str.str.replace(",", ".", regex=False)

        s_num = pd.to_numeric(s_str, errors="coerce")
        return s_num

    print("[validate] Checking numeric columns from Rent to last column")  # TODO: replace with logging later
    for col in numeric_cols:
        s_num = _to_numeric_temp(df[col])

        non_null_count = int(df[col].notna().sum())
        converted_non_null_count = int(s_num.notna().sum())

        print(
            f"[validate] {col} non-null: {non_null_count}, numeric after parse: {converted_non_null_count}"
        )

        if non_null_count > 0 and converted_non_null_count == 0:
            raise ValueError(
                f"Validation failed: column '{col}' must be numeric. No values could be parsed as numbers"
            )

    # 3) Binary check from Outer to last column, values must be 0 or 1 (after numeric parsing)
    start_binary_col = "Outer"
    if start_binary_col not in df.columns:
        raise ValueError("Validation failed: column 'Outer' not found")

    start_binary_idx = df.columns.get_loc(start_binary_col)
    binary_cols = list(df.columns[start_binary_idx:])

    print("[validate] Checking binary columns from Outer to last column")  # TODO: replace with logging later
    for col in binary_cols:
        s_num = _to_numeric_temp(df[col])
        s_non_null = s_num.dropna()

        if s_non_null.empty:
            raise ValueError(f"Validation failed: binary column '{col}' has only missing values")

        unique_vals = set(s_non_null.unique().tolist())
        print(f"[validate] {col} unique values (numeric-parsed): {sorted(list(unique_vals))}")

        if not unique_vals.issubset({0, 1}):
            raise ValueError(
                f"Validation failed: column '{col}' must be binary 0/1. Found {sorted(list(unique_vals))}"
            )
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return True
