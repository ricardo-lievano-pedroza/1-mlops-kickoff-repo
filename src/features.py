"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""

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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
):
    """
    Inputs:
    - quantile_bin_cols: Numeric columns to discretize into quantile bins
    - categorical_onehot_cols: Categorical columns to one-hot encode
    - numeric_passthrough_cols: Numeric columns to pass through unchanged
    - n_bins: Number of quantile bins
    Outputs:
    - preprocessor: Unfitted sklearn ColumnTransformer
    Why this contract matters for reliable ML delivery:
    - Keeping this as an unfitted recipe prevents leakage and ensures identical transforms at train and serve time.
    """
    print("[features.get_feature_preprocessor] Building ColumnTransformer recipe (unfitted)")  # TODO: replace with logging later

    quantile_bin_cols = quantile_bin_cols or []
    categorical_onehot_cols = categorical_onehot_cols or []
    numeric_passthrough_cols = numeric_passthrough_cols or []

    # Robust OneHotEncoder creation across sklearn versions
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    quantile_pipe = Pipeline(
        steps=[
            ("kbins", KBinsDiscretizer(n_bins=n_bins, encode="onehot-dense", strategy="quantile")),
        ]
    )

    cat_pipe = Pipeline(
        steps=[
            ("onehot", ohe),
        ]
    )

    transformers = []
    if quantile_bin_cols:
        transformers.append(("quantile_bin", quantile_pipe, quantile_bin_cols))
    if categorical_onehot_cols:
        transformers.append(("categorical_onehot", cat_pipe, categorical_onehot_cols))
    if numeric_passthrough_cols:
        transformers.append(("numeric_passthrough", "passthrough", numeric_passthrough_cols))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Modify feature recipe (scaling, imputation, text, interactions) as needed
    # Why: Feature engineering depends on data modality and business goals
    # Examples:
    # 1. Add SimpleImputer for missing numeric values
    # 2. Add StandardScaler for linear models
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return preprocessor