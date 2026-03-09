"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    """
    Evaluate a trained regression model on test data.

    Input:
    - Trained model + test data

    Output:
    - Metrics dictionary
    - Plots saved to `reports/`
    """
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(model.predict(X_test)).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_test length {len(y_true)} != y_pred length {len(y_pred)}")

    # Metrics
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    metrics: Dict[str, float] = {"mae": mae, "rmse": rmse, "r2": r2}

    # Plot 1: Predicted vs Actual
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    mn = float(min(np.min(y_true), np.min(y_pred)))
    mx = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([mn, mx], [mn, mx])
    fig.tight_layout()
    fig.savefig(reports_dir / "pred_vs_actual.png", dpi=150)
    plt.close(fig)

    # Plot 2: Residuals vs Predicted
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0.0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(reports_dir / "residuals_vs_pred.png", dpi=150)
    plt.close(fig)

    return metrics