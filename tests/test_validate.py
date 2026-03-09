def test_validate():
    import pandas as pd
    import pytest
    from src import validate  # change if your module path differs

    schema = {
        "age": {"dtype": "integer", "nullable": False, "min": 0, "max": 120},
        "income": {"dtype": "numeric", "nullable": True, "min": 0},
        "country": {"dtype": "string", "nullable": False, "allowed_values": ["ES", "FR", "DE"]},
        "id": {"dtype": "integer", "nullable": False, "unique": True},
    }

    # --- valid case ---
    df_ok = pd.DataFrame(
        {
            "age": pd.Series([25, 40, 18], dtype="int64"),
            "income": pd.Series([1000.0, 2000.0, 0.0], dtype="float64"),
            "country": pd.Series(["ES", "FR", "DE"], dtype="object"),
            "id": pd.Series([1, 2, 3], dtype="int64"),
        }
    )
    assert validate(df_ok, schema) is True

    # --- invalid: missing column ---
    with pytest.raises(ValueError, match="Missing required columns"):
        validate(df_ok.drop(columns=["income"]), schema)

    # --- invalid: extra column (when allow_extra_columns=False) ---
    df_extra = df_ok.copy()
    df_extra["extra"] = 1
    with pytest.raises(ValueError, match="Unexpected columns present"):
        validate(df_extra, schema, allow_extra_columns=False)

    # --- invalid: null in non-nullable ---
    df_null = df_ok.copy()
    df_null.loc[0, "age"] = None
    with pytest.raises(ValueError, match="non-nullable"):
        validate(df_null, schema)

    # --- invalid: range violation ---
    df_range = df_ok.copy()
    df_range.loc[1, "age"] = 200
    with pytest.raises(ValueError, match=r"values > 120"):
        validate(df_range, schema)

    # --- invalid: allowed values violation ---
    df_allowed = df_ok.copy()
    df_allowed.loc[2, "country"] = "UK"
    with pytest.raises(ValueError, match=r"invalid values"):
        validate(df_allowed, schema)

    # --- invalid: unique violation ---
    df_dup = df_ok.copy()
    df_dup.loc[2, "id"] = 2
    with pytest.raises(ValueError, match=r"expected unique"):
        validate(df_dup, schema)