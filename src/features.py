import logging
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def get_feature_preprocessor(
    bin_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
    numeric_cols: Optional[List[str]] = None,
    n_bins: int = 3,
) -> ColumnTransformer:
    """
    Inputs:
    - bin_cols: Optional[List[str]] numeric columns to quantile-bin
    - categorical_cols: Optional[List[str]] categorical columns to one-hot encode
    - numeric_cols: Optional[List[str]] numeric columns to pass through unchanged
    - n_bins: number of quantile bins for KBinsDiscretizer

    Outputs:
    - ColumnTransformer preprocessing object (NOT fitted)

    Why this contract matters for reliable ML delivery:
    - A “recipe-only” - preprocessor - fitting happens later on X_train only.
    """

    logger.info("Building ColumnTransfomer")

    if n_bins < 2 and bin_cols is not None:
        raise ValueError("FATAL: n_bins must be >= 2 for quantile binning")

    if not (bin_cols or categorical_cols or numeric_cols):
        raise ValueError(
            "Fatal: No feature columns configured for the preprocessor"
            )

    bin_cols = bin_cols or []
    categorical_cols = categorical_cols or []
    numeric_cols = numeric_cols or []

    transformers = []

    # Quantile features: Impute -> Quanitle bin
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

    # Cateogrical features: Impute, One hot encoding
    if categorical_cols:
        try:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

        categorical_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="constant",
                                         fill_value="__MISSING__")),
                ("onehot", ohe),
            ]
        )
        transformers.append(("categorical_onehot", categorical_pipeline,
                             categorical_cols))

    # Numerical features: Impute, Scale
    if numeric_cols:
        numeric_pipeline = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", RobustScaler()),
            ]
        )
        transformers.append(("numeric_scaler", numeric_pipeline, numeric_cols))

    preprocessor = ColumnTransformer(transformers=transformers,
                                     remainder="drop")
    return preprocessor
