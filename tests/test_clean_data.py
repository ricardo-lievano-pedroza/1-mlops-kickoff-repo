import pandas as pd
import pytest

from src.clean_data import clean_dataframe


def test_drops_missing_target():
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": [10, None, 30]
    })

    df_clean = clean_dataframe(
        df_raw=df,
        target_column="target",
        required_columns=[
            "feature",
            "target"]
        )

    # One row should be dropped
    assert len(df_clean) == 2
    assert df_clean["target"].isna().sum() == 0


def test_target_is_numeric():
    df = pd.DataFrame({
        "feature": [1, 2, 3],
        "target": ["10", "20", "30"]
    })

    df_clean = clean_dataframe(
        df_raw=df,
        target_column="target",
        required_columns=["feature", "target"]
    )

    assert pd.api.types.is_numeric_dtype(df_clean["target"])


def test_categorical_normalization():
    df = pd.DataFrame({
        "city": [" New York ", "LONDON", "paris"],
        "target": [1, 2, 3]
    })

    df_clean = clean_dataframe(
        df_raw=df,
        target_column="target",
        required_columns=["city", "target"]
    )

    assert df_clean["city"].iloc[0] == "new york"
    assert df_clean["city"].iloc[1] == "london"
    assert df_clean["city"].iloc[2] == "paris"


def test_raises_if_target_missing():
    df = pd.DataFrame({
        "feature": [1, 2, 3]
    })

    with pytest.raises(ValueError):
        clean_dataframe(
            df_raw=df,
            target_column="target",
            required_columns=["feature"]
        )


def test_drops_duplicates():
    df = pd.DataFrame({"city": ["a", "a"], "target": [1, 1]})
    df_clean = clean_dataframe(
        df_raw=df,
        target_column="target",
        required_columns=["city", "target"]
    )
    assert len(df_clean) == 1
