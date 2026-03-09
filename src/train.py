"""
Educational Goal:
- Why this module exists in an MLOps system: Training should be a repeatable function with a stable contract.
- Responsibility (separation of concerns): Fit a Pipeline(preprocess -> model) on training data only.
- Pipeline contract (inputs and outputs): Inputs are X_train/y_train + preprocessor + problem type; output is a fitted model.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str = "regression"):
    """
    Inputs:
    - X_train: Training features (DataFrame)
    - y_train: Training labels (Series)
    - preprocessor: ColumnTransformer (not fitted)
    - problem_type: "regression" or "classification"
    Outputs:
    - model: Fitted scikit-learn Pipeline
    Why this contract matters for reliable ML delivery:
    - Training is repeatable and leakage-resistant because preprocessing is fit only within the Pipeline on training data.
    """
    print(f"[train.train_model] Training model for problem_type={problem_type}")  # TODO: replace with logging later

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    if X_train is None or len(X_train) == 0:
        raise ValueError("Training failed: X_train is empty.")

    if y_train is None or len(y_train) == 0:
        raise ValueError("Training failed: y_train is empty.")

    if problem_type == "regression":
        estimator = LinearRegression()
    else:
        raise ValueError("Training failed: problem_type not supported")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )

    model.fit(X_train, y_train)

    return model
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------
