"""
Educational Goal:
- Why this module exists in an MLOps system: Feature logic must be repeatable across training and inference to avoid training/serving skew.
- Responsibility (separation of concerns): Define a preprocessing “recipe” without fitting it (fit happens only on train split).
- Pipeline contract (inputs and outputs): Inputs are column name lists; output is an unfitted ColumnTransformer.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def get_feature_preprocessor(
    bin_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols: Optional[List[str]] numeric columns to quantile-bin
    - categorical_onehot_cols: Optional[List[str]] categorical columns to one-hot encode
    - numeric_passthrough_cols: Optional[List[str]] numeric columns to pass through unchanged
    - n_bins: number of quantile bins for KBinsDiscretizer
    Outputs:
    - ColumnTransformer preprocessing object (NOT fitted)
    Why this contract matters for reliable ML delivery:
    - A “recipe-only” preprocessor ensures transforms are fit only on training data inside the Pipeline.
    """
    print("[features.get_feature_preprocessor] Building ColumnTransformer feature recipe")  # TODO: replace with logging later
  
    bin_cols = bin_cols or []
    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []

    transformers = []

    if bin_cols:
        numeric_bin_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="mean")),
                (
                    "bin",
                    KBinsDiscretizer(
                        n_bins=n_bins,
                        encode="onehot-dense",
                        strategy="quantile"
                    )
                ),
            ]
        )
        transformers.append(
            ("quantile_bin", numeric_bin_pipeline, bin_cols)
        )

    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
                ("onehot", ohe),
            ]
        )
        transformers.append(("categorical_onehot", categorical_pipeline, categorical_cols))

    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
            ]
        )
        transformers.append(("numeric_scaler", numeric_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor