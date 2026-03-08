import numpy as np
import pandas as pd

from src.features import get_feature_preprocessor

def test_output_is_array():
    df = pd.DataFrame({
        "numerical_features": [1, 4, 9, 5],
        "categorical_features": ["D1", "D2", "D4", "D1"],
        "bin_features": [10000, 3000, 2000, 1500]
    })
    
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["bin_features"],
        categorical_onehot_cols=["categorical_features"],
        numeric_passthrough_cols=["numerical_features"],
        n_bins=3
    )
    
    Xt = preprocessor.fit_transform(df)
    assert isinstance(Xt, np.ndarray)


def test_handles_missing_numeric_values():
    df = pd.DataFrame({
        "numerical_features": [1, np.nan, 9, 5],
        "binning_features": [10000, 3000, np.nan, 1500]
    })
    
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["binning_features"],
        numeric_passthrough_cols=["numerical_features"],
        n_bins=3
    )
    Xt = preprocessor.fit_transform(df)
    
    Xt_np = Xt.toarray() if hasattr(Xt, "toarray") else np.asarray(Xt)
    print(preprocessor)
    assert not np.isnan(Xt_np).any()


def test_handles_missing_categorical_values():
    df = pd.DataFrame({
        "categorical_features": ["D1", None, "D4", "D1"]
    })
    
    preprocessor = get_feature_preprocessor(
        categorical_onehot_cols=["categorical_features"]
    )
    Xt = preprocessor.fit_transform(df)
    assert Xt.shape[0] == 4


def test_empty_lists_returns_valid_transformer():
    df = pd.DataFrame({
        "col1": [1, 2, 3],
        "col2": ["a", "b", "c"]
    })
    
    preprocessor = get_feature_preprocessor()
    Xt = preprocessor.fit_transform(df)
    assert Xt.shape[0] == 3


def test_output_shape_matches_expected_features():
    df = pd.DataFrame({
        "numerical_features": [1, 4, 9, 5],
        "categorical_features": ["D1", "D2", "D4", "D1"],
        "binning_features": [10000, 3000, 2000, 1500]
    })
    
    preprocessor = get_feature_preprocessor(
        quantile_bin_cols=["binning_features"],
        categorical_onehot_cols=["categorical_features"],
        numeric_passthrough_cols=["numerical_features"],
        n_bins=3
    )
    
    Xt = preprocessor.fit_transform(df)
    assert Xt.shape[0] == 4
    assert Xt.shape[1] > 0