"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""
"""
Educational Goal:
- Why this module exists in an MLOps system: Evaluation must be consistent, testable, and separated from training logic.
- Responsibility (separation of concerns): Compute a single metric on held-out data and return it for reporting/alerts.
- Pipeline contract (inputs and outputs): Inputs are fitted model Pipeline + test data + problem_type; output is a float metric.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, x_test: pd.DataFrame, y_test: pd.Series, report_dir: Optional[str | Path] = None,) -> Dict[str,float]:
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

    y_pred = model.predict(x_test)
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
    y_true= y_test.to_numpy()
    y_hat= np.asarray(y_pred)

    mae = float(mean_absolute_error(y_true, y_hat))
    rmse = float(mean_squared_error(y_true, y_hat, squared=False))
    r2= float(r2_score(y_true, y_hat))


    eps=1e-8
    denom= np.maximum(np.abs(y_true),eps) #avoid division by zero
    mape = float(np.mean(np.abs((y_true - y_hat) / denom)) * 100)

    metric_values = {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}
    if report_dir is not None:
        r_save_regression_plots(y_true=y_true, y_hat=y_hat, report_dir=Path(report_dir)) # saves plots to disk for later analysis


    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return metric_values

def _save_regression_plots(y_true: np.ndarray, y_hat: np.ndarray, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt  # local import so eval works without matplotlib if plots not requested

    # Predicted vs Actual
    plt.figure()
    plt.scatter(y_true, y_hat)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual")
    plt.savefig(report_dir / "pred_vs_actual.png", bbox_inches="tight")
    plt.close()

    # Residuals histogram
    residuals = y_true - y_hat
    plt.figure()
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("Residuals Distribution")
    plt.savefig(report_dir / "residuals_hist.png", bbox_inches="tight")
    plt.close()