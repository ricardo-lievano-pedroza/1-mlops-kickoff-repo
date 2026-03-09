import pandas as pd
import pytest

from src.validate import validate_dataframe


def test_validate_dataframe_returns_true_on_valid_data():
    df = pd.DataFrame(
        {
            "rent": [1000.0, 1500.0, 1200.0],
            "outer": [0, 1, 0],
        }
    )
    assert validate_dataframe(df, required_columns=["rent", "outer"]) is True


def test_validate_dataframe_raises_on_empty_df():
    df_empty = pd.DataFrame()
    with pytest.raises(ValueError, match="DataFrame is empty"):
        validate_dataframe(df_empty, required_columns=["rent"])


def test_validate_dataframe_raises_on_missing_required_columns():
    df = pd.DataFrame({"rent": [1000.0]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_dataframe(df, required_columns=["rent", "outer"])


def test_validate_dataframe_raises_on_high_null_rate():
    # 4 rows: 4/4 nulls => 100% null rate > 0.8 threshold
    df = pd.DataFrame(
        {
            "rent": [1000.0, 1200.0, 1100.0, 1300.0],
            "outer": [None, None, None, None],
        }
    )
    with pytest.raises(ValueError, match="High null rate"):
        validate_dataframe(df, required_columns=["rent", "outer"])


def test_validate_dataframe_raises_on_rent_out_of_range_low():
    df = pd.DataFrame(
        {
            "rent": [-1.0, 1000.0],
            "outer": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="values < 0.0"):
        validate_dataframe(df, required_columns=["rent", "outer"])


def test_validate_dataframe_raises_on_rent_out_of_range_high():
    df = pd.DataFrame(
        {
            "rent": [25000.0, 1000.0],  # default max is 20000
            "outer": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="values > 20000.0"):
        validate_dataframe(df, required_columns=["rent", "outer"])


def test_validate_dataframe_raises_on_invalid_binary_values():
    df = pd.DataFrame(
        {
            "rent": [1000.0, 1200.0, 1100.0],
            "outer": [0, 2, 1],  # 2 is invalid
        }
    )
    with pytest.raises(ValueError, match="Binary-like column"):
        validate_dataframe(df, required_columns=["rent", "outer"])


def test_validate_dataframe_accepts_common_string_binary_values():
    df = pd.DataFrame(
        {
            "rent": [1000.0, 1200.0, 1100.0, 1300.0],
            "outer": ["Y", "n", "1", "0"],
        }
    )
    assert validate_dataframe(df, required_columns=["rent", "outer"]) is True