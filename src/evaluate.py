import logging


from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary.
"""
logger = logging.getLogger(__name__)


def evaluate_model(
    model,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    problem_type: str
) -> Dict[str, float]:
    """
    Evaluate a trained regression model on test data.

    Input:
    - model: Fitted model with predict() method
    - X_test: Evaluation feautures
    - Y_test: Evaluation target
    - problem_type : "regression" or "classification"

    Output:
    - metrics: Dictionary of metrics as python floats

    """
    logger.info("Starting evaluation")

    if X_test is None or len(X_test) == 0:
        raise ValueError(
            "FATAL: X_test is empty. Cannot evaluate the model"
        )

    if y_test is None or len(y_test) == 0:
        raise ValueError(
            "FATAL: y_test is empty. Cannot evaluate the model"
        )

    if len(X_test) != len(y_test):
        raise ValueError(
            f"FATAL: X_test rows:{len(X_test)}do not match y_test rows: {len(y_test)}"
        )

    if not hasattr(model, "predict"):
        raise TypeError(
            f"FATAL: model must have the method predict(), got type= {type(model)}"
        )

    if problem_type.lower() != "regression":
        raise ValueError(
            f"FATAL: Unsupported problem_type: {problem_type}: Use 'regression'"
        )

    y_pred = model.predict(X_test)

    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))
    metrics = {"mae": mae, "rmse": rmse, "r2": r2}

    logger.info("Metrics=%s", metrics)

    return metrics
