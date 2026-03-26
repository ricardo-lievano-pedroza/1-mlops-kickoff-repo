import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def train_model(
        X_train: pd.DataFrame,
        y_train: pd.Series,
        preprocessor: ColumnTransformer,
        problem_type: str = "regression"
        ) -> Pipeline:
    """
    Inputs:
    - X_train: Training features (DataFrame)
    - y_train: Training labels (Series)
    - preprocessor: ColumnTransformer (not fitted)
    - problem_type: "regression" or "classification"

    Outputs:
    - model: Fitted scikit-learn Pipeline

    Why this contract matters for reliable ML delivery:
    - Training is repeatable and leakage-resistant because preprocessing is
    fit only within the Pipeline on training data.
    """
    logger.info(f"Training model pipeline for {problem_type}")

    if X_train is None or len(X_train) == 0:
        raise ValueError("Training failed: X_train is empty.")

    if y_train is None or len(y_train) == 0:
        raise ValueError("Training failed: y_train is empty.")

    if len(y_train) != len(X_train):
        raise ValueError(
            f"Training failed: X_trian and y_train are different sized"
            f" X_train: {len(X_train)}, y_train: {len(y_train)}."
        )

    if not isinstance(preprocessor, ColumnTransformer):
        raise TypeError(
            f"preprocessor must be of type ColumnTransformer, got type = {type(preprocessor)}"
        )

    if problem_type.lower() == "regression":
        estimator = LinearRegression()
    else:
        raise ValueError(f"Training failed: problem_type not supported: got '{problem_type}'")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    model.fit(X_train, y_train)

    logger.info("Model training completed")

    return model
