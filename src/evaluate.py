"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""
v"""
Educational Goal:
- Why this module exists in an MLOps system: Evaluation must be consistent, testable, and separated from training logic.
- Responsibility (separation of concerns): Compute a single metric on held-out data and return it for reporting/alerts.
- Pipeline contract (inputs and outputs): Inputs are fitted model Pipeline + test data + problem_type; output is a float metric.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """
    Inputs:
    - model: Fitted sklearn Pipeline
    - X_test: Test features DataFrame
    - y_test: Test target Series
    - problem_type: "regression" or "classification"
    Outputs:
    - metric_value: float (RMSE for regression, F1 weighted for classification)
    Why this contract matters for reliable ML delivery:
    - Standardized evaluation prevents metric drift and makes it easy to compare runs.
    """
    print(f"[evaluate.evaluate_model] Evaluating model")  # TODO: replace with logging later

    y_pred = model.predict(X_test)
    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Replace/extend metrics (MAE, ROC-AUC, calibration, business KPIs)
    # Why: What “good” means differs across products and stakeholders
    # Examples:
    # 1. Add MAE for regression
    # 2. Add per-segment evaluation
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    mae = mean_absolute_error(y_pred,y_test )
    mape = np.mean(100 * abs(y_test - y_pred) / y_test)
    metric_values = {
        'mae': mae,
        'mape': mape
    }
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return metric_values