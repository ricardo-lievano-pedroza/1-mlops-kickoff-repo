import pandas as pd
import pytest
from src.validate import validate_dataframe

def test_validate_success_parsing_and_binary():
    # Columns include a trailing space to exercise column name normalization
    df = pd.DataFrame(
        {
            "Id ": [1, 2, 3],
            "Rent": ["1.234,56", "2.000", "3000"],  # mixed thousand dot and plain
            "Value1": ["10", "20", "30"],  # numeric column after Rent
            "Outer": ["1", "0", "1"],  # binary as strings
            "Flag1": [0, 1, 0],  # binary as ints
        }
    )
    required = ["Id", "Rent", "Value1", "Outer", "Flag1"]
    assert validate_dataframe(df, required) is True


def test_empty_dataframe_raises():
    df = pd.DataFrame(columns=["Rent", "Outer"])
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Rent", "Outer"])


def test_missing_required_column_raises():
    df = pd.DataFrame({"Rent": [100], "Outer": [1]})
    # required includes a column that's not present
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Rent", "Outer", "MissingCol"])


def test_missing_target_column_raises():
    df = pd.DataFrame({"SomeOther": [1, 2], "Outer": [0, 1]})
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["SomeOther", "Outer"])


def test_target_contains_missing_values_raises():
    df = pd.DataFrame({"Rent": [100, None], "Outer": [1, 0]})
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Rent", "Outer"])


def test_excessive_missing_values_in_column_raises():
    # Create a column with 75% missing values -> should fail (>50%)
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "Rent": [100, 200, 300, 400],
            "Value1": [None, None, 5, None],  # 3/4 missing = 75%
            "Outer": [1, 0, 1, 0],
        }
    )
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Id", "Rent", "Value1", "Outer"])


def test_numeric_parsing_failure_raises():
    # Column between Rent and last contains non-parsable strings -> should raise
    df = pd.DataFrame(
        {
            "Id": [1, 2],
            "Rent": ["1000", "2000"],
            "BadNum": ["abc", "def"],  # non-null but cannot be parsed as numeric
            "Outer": [1, 0],
        }
    )
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Id", "Rent", "BadNum", "Outer"])


def test_binary_column_all_missing_raises():
    # Binary column 'Outer' exists but only has missing values -> binary check should raise
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "Rent": [100, 200, 300],
            "Outer": [None, None, None],  # all missing
        }
    )
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Id", "Rent", "Outer"])


def test_binary_column_invalid_values_raises():
    # Binary column contains a value outside {0,1} after numeric parsing -> raise
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3],
            "Rent": [100, 200, 300],
            "Outer": ["0", "1", "2"],  # '2' is invalid for binary
        }
    )
    with pytest.raises(ValueError):
        validate_dataframe(df, required_columns=["Id", "Rent", "Outer"])