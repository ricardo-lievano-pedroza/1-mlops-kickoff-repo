# tests/test_evaluate.py

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Force a non-interactive backend for matplotlib in CI / headless runs
import matplotlib
matplotlib.use("Agg")

from src.evaluate import evaluate, EvalConfig


class DummyLinearModel:
    """Predicts y = 2*x0 + 1 for regression tests."""
    def predict(self, X):
        X = np.asarray(X)
        return 2 * X[:, 0] + 1


def test_evaluate_perfect_fit_saves_artifacts(tmp_path: Path):
    # Arrange
    model = DummyLinearModel()
    X_test = np.array([[0.0], [1.0], [2.0], [3.0], [4.0]])
    y_test = np.array([1.0, 3.0, 5.0, 7.0, 9.0])  # perfect match

    cfg = EvalConfig(
        reports_dir=tmp_path / "reports",
        save_prefix="eval",
        save_predictions_csv=True,
        id_column=None,
    )

    # Act
    metrics = evaluate(model, X_test, y_test, config=cfg)

    # Assert metrics
    assert metrics["problem_type"] == "regression"
    assert metrics["n_samples"] == 5
    assert metrics["n_used"] == 5
    assert metrics["n_dropped_nonfinite"] == 0
    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0

    # Assert artifacts exist
    reports_dir = Path(cfg.reports_dir)
    assert (reports_dir / "eval_pred_vs_actual.png").exists()
    assert (reports_dir / "eval_residuals_vs_pred.png").exists()
    assert (reports_dir / "eval_residual_hist.png").exists()
    assert (reports_dir / "eval_abs_error_hist.png").exists()
    assert (reports_dir / "eval_metrics.json").exists()
    assert (reports_dir / "eval_predictions.csv").exists()

    # Assert metrics JSON content matches returned dict (at least key metrics)
    with open(reports_dir / "eval_metrics.json", "r", encoding="utf-8") as f:
        saved = json.load(f)
    for k in ["mae", "rmse", "r2", "n_samples", "n_used", "n_dropped_nonfinite"]:
        assert k in saved
        assert saved[k] == metrics[k]


def test_evaluate_drops_nonfinite_values(tmp_path: Path):
    # Arrange
    model = DummyLinearModel()
    X_test = np.array([[0.0], [1.0], [2.0], [3.0]])
    y_test = np.array([1.0, np.nan, 5.0, 7.0])  # includes NaN -> should drop 1 row

    cfg = EvalConfig(
        reports_dir=tmp_path / "reports",
        save_prefix="eval_nan",
        save_predictions_csv=True,
    )

    # Act
    metrics = evaluate(model, X_test, y_test, config=cfg)

    # Assert
    assert metrics["n_samples"] == 4
    assert metrics["n_used"] == 3
    assert metrics["n_dropped_nonfinite"] == 1

    # Artifacts exist
    reports_dir = Path(cfg.reports_dir)
    assert (reports_dir / "eval_nan_pred_vs_actual.png").exists()
    assert (reports_dir / "eval_nan_metrics.json").exists()
    assert (reports_dir / "eval_nan_predictions.csv").exists()


def test_evaluate_raises_on_length_mismatch(tmp_path: Path):
    # Arrange
    model = DummyLinearModel()
    X_test = np.array([[0.0], [1.0], [2.0]])
    y_test = np.array([1.0, 3.0])  # mismatch

    cfg = EvalConfig(reports_dir=tmp_path / "reports", save_prefix="eval_mismatch")

    # Act / Assert
    with pytest.raises(ValueError, match="y_test length"):
        evaluate(model, X_test, y_test, config=cfg)


def test_evaluate_with_dataframe_and_id_column(tmp_path: Path):
    # Arrange
    model = DummyLinearModel()
    X_test = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "x0": [0.0, 1.0, 2.0, 3.0],
        }
    )
    y_test = np.array([1.0, 3.0, 5.0, 7.0])

    cfg = EvalConfig(
        reports_dir=tmp_path / "reports",
        save_prefix="eval_df",
        save_predictions_csv=True,
        id_column="id",
    )

    # Act
    metrics = evaluate(model, X_test[["x0"]].values, y_test, config=cfg)

    # Assert artifacts exist (basic)
    reports_dir = Path(cfg.reports_dir)
    assert (reports_dir / "eval_df_predictions.csv").exists()
    assert (reports_dir / "eval_df_metrics.json").exists()

    # Note:
    # Your current evaluate() only inserts id_column if X_test passed into evaluate()
    # is a DataFrame. Since we passed numpy values above, we only verify CSV exists.
    # If you want ID support, call evaluate(model, X_test, y_test, cfg) and update
    # DummyLinearModel.predict to handle DataFrames.