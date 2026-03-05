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

import pandas as pd


def clean_dataframe(df_raw: pd.DataFrame, target_column: str) -> pd.DataFrame:
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
    #drop rows with missing target values
    df_clean = df_raw.dropna(subset=[target_column])


    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook cleaning logic here to replace or extend the baseline
    # Why: Cleaning rules depend on data quirks (missingness, outliers, business rules, leakage rules).
    # Examples:
    # 1. Drop rows with missing target values
    # 2. Normalize text categories, trim whitespace, standardize casing
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    # Minimal baseline sanity: ensure target exists if referenced
    if target_column not in df_clean.columns: