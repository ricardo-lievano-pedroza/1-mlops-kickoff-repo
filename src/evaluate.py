"""
Module: Evaluation
------------------
Role: Generate metrics and plots for model performance.
Input: Trained Model + Test Data.
Output: Metrics dictionary and plots saved to `reports/`.
"""


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass(frozen=True)
class EvalConfig:
    reports_dir: Union[str, Path] = "reports"
    save_prefix: str = "eval"
    save_predictions_csv: bool = True
    # If you have an ID column in X_test (DataFrame), store it in the predictions CSV
    id_column: Optional[str] = None


def _ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    mask = np.ones_like(arrays[0], dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return mask


def evaluate(
    model: Any,
    X_test: Union[pd.DataFrame, np.ndarray],
    y_test: Union[pd.Series, np.ndarray],
    config: Optional[EvalConfig] = None,
) -> Dict[str, Any]:
    """
    Evaluate a trained regression model on test data.

    Saves to reports_dir:
      - *_pred_vs_actual.png
      - *_residuals_vs_pred.png
      - *_residual_hist.png
      - *_abs_error_hist.png
      - *_metrics.json
      - *_predictions.csv (optional)

    Returns:
      metrics dict
    """
    cfg = config or EvalConfig()
    reports_dir = _ensure_dir(cfg.reports_dir)

    y_true = np.asarray(y_test).reshape(-1)
    y_pred = np.asarray(model.predict(X_test)).reshape(-1)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(f"y_test length {len(y_true)} != y_pred length {len(y_pred)}")

    # Remove non-finite values (protects metrics + plots)
    mask = _finite_mask(y_true, y_pred)
    dropped = int((~mask).sum())
    y_true_f = y_true[mask]
    y_pred_f = y_pred[mask]

    residuals = y_true_f - y_pred_f
    abs_err = np.abs(residuals)

    # Metrics
    mae = float(mean_absolute_error(y_true_f, y_pred_f))
    rmse = float(np.sqrt(mean_squared_error(y_true_f, y_pred_f)))
    r2 = float(r2_score(y_true_f, y_pred_f))

    metrics: Dict[str, Any] = {
        "problem_type": "regression",
        "n_samples": int(len(y_true)),
        "n_used": int(len(y_true_f)),
        "n_dropped_nonfinite": dropped,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "abs_error_p50": float(np.percentile(abs_err, 50)),
        "abs_error_p90": float(np.percentile(abs_err, 90)),
        "abs_error_p95": float(np.percentile(abs_err, 95)),
    }

    # --- Plot: Predicted vs Actual ---
    fig, ax = plt.subplots()
    ax.scatter(y_true_f, y_pred_f, alpha=0.5)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")

    mn = float(min(np.min(y_true_f), np.min(y_pred_f)))
    mx = float(max(np.max(y_true_f), np.max(y_pred_f)))
    ax.plot([mn, mx], [mn, mx])

    pred_path = reports_dir / f"{cfg.save_prefix}_pred_vs_actual.png"
    fig.tight_layout()
    fig.savefig(pred_path, dpi=150)
    plt.close(fig)
    metrics["pred_vs_actual_path"] = str(pred_path)

    # --- Plot: Residuals vs Predicted ---
    fig, ax = plt.subplots()
    ax.scatter(y_pred_f, residuals, alpha=0.5)
    ax.axhline(0.0)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs Predicted")

    res_scatter_path = reports_dir / f"{cfg.save_prefix}_residuals_vs_pred.png"
    fig.tight_layout()
    fig.savefig(res_scatter_path, dpi=150)
    plt.close(fig)
    metrics["residuals_vs_pred_path"] = str(res_scatter_path)

    # --- Plot: Residual Histogram ---
    fig, ax = plt.subplots()
    ax.hist(residuals, bins=50)
    ax.set_xlabel("Residual")
    ax.set_ylabel("Count")
    ax.set_title("Residual Distribution")

    res_hist_path = reports_dir / f"{cfg.save_prefix}_residual_hist.png"
    fig.tight_layout()
    fig.savefig(res_hist_path, dpi=150)
    plt.close(fig)
    metrics["residual_hist_path"] = str(res_hist_path)

    # --- Plot: Absolute Error Histogram ---
    fig, ax = plt.subplots()
    ax.hist(abs_err, bins=50)
    ax.set_xlabel("|Residual|")
    ax.set_ylabel("Count")
    ax.set_title("Absolute Error Distribution")

    abs_hist_path = reports_dir / f"{cfg.save_prefix}_abs_error_hist.png"
    fig.tight_layout()
    fig.savefig(abs_hist_path, dpi=150)
    plt.close(fig)
    metrics["abs_error_hist_path"] = str(abs_hist_path)

    # --- Save metrics JSON ---
    metrics_path = reports_dir / f"{cfg.save_prefix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    metrics["metrics_path"] = str(metrics_path)

    # --- Save predictions CSV (optional) ---
    if cfg.save_predictions_csv:
        pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
        pred_df["residual"] = pred_df["y_true"] - pred_df["y_pred"]
        pred_df["abs_error"] = pred_df["residual"].abs()

        if isinstance(X_test, pd.DataFrame) and cfg.id_column and cfg.id_column in X_test.columns:
            pred_df.insert(0, cfg.id_column, X_test[cfg.id_column].values)

        preds_path = reports_dir / f"{cfg.save_prefix}_predictions.csv"
        pred_df.to_csv(preds_path, index=False)
        metrics["predictions_path"] = str(preds_path)

    return metrics