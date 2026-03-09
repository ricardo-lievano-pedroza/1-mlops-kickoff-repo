"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

"""
Module: Data Cleaning
---------------------
Role: Preprocessing, missing value imputation, and feature engineering.
Input: pandas.DataFrame (Raw).
Output: pandas.DataFrame (Processed/Clean).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Separate dataset cleaning from modeling to reduce hidden notebook state and rework.
- Responsibility (separation of concerns): Apply deterministic cleaning rules and produce a cleaned DataFrame for downstream steps.
- Pipeline contract (inputs and outputs): Input is raw df + target column name; output is cleaned df (same rows/cols unless student changes).

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""
# --------------------------------------------------------
# START STUDENT CODE
# --------------------------------------------------------
import pandas as pd



# target_column is the dependent variable that we want to predict. It is used in cleaning to ensure we don't drop rows with missing target values, which would affect model training.
def clean_dataframe(df_raw: pd.DataFrame, target_column: str, required_columns: list ) -> pd.DataFrame:
    """
    Inputs:
    - df_raw: Raw pandas DataFrame
    - target_column: Name of the target column
    Outputs:
    - df_clean: Cleaned pandas DataFrame
    Why this contract matters for reliable ML delivery:
    - Cleaning changes data semantics; isolating it makes changes reviewable, testable, and less risky.
    """
    print("[clean_data.clean_dataframe] Cleaning raw dataframe (baseline = identity transform)")
    if target_column not in df_raw.columns:
        raise ValueError(f"Target column '{target_column}' not found in dataframe.")
    df_clean = df_raw.copy()

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the baseline
    # Why: Cleaning rules depend on data quirks (missingness, outliers, business rules, leakage rules).
    # Examples:
    # 1. Drop rows with missing target values
    # 2. Normalize text categories, trim whitespace, standardize casing
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #remove non_required columns
    non_required_columns = [col for col in df_clean.columns if col not in required_columns]
    df_clean.drop(columns=non_required_columns, inplace=True)

    #Drop duplicates
    df_clean.drop_duplicates(inplace=True)

    #drop rows with missing target values
    before_rows = len(df_clean)
    df_clean = df_clean[df_clean[target_column].notna()].copy()
    after_rows = len(df_clean)
    print(f"[clean_data.clean_dataframe] Dropped {before_rows - after_rows} rows with missing target")
    #reset index after dropping rows
    df_clean.reset_index(inplace=True)


    #force target column to be numeric (if not already)
    df_clean[target_column] = pd.to_numeric(df_clean[target_column], errors='raise')


    #detect numeric columns
    numeric_columns = df_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col != target_column]

    #normalize categorical columns
    categorical_columns = df_clean.select_dtypes(include=["object","string", "category"]).columns.tolist()
    for col in categorical_columns:
        df_clean[col] = (df_clean[col].astype(str).str.strip().str.lower())
        print(f"[clean_data.clean_dataframe] Normalized categorical column {col}")

    print("Warning: Student has not implemented this section yet")


    return df_clean

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
    # Minimal baseline sanity: ensure target exists if referenced
