import numpy as np
from pathlib import Path
import pytest
from src.evaluate import evaluate


def test_evaluate_runs_and_saves_plots(tmp_path, monkeypatch):
    # Move into temporary directory
    monkeypatch.chdir(tmp_path)

    # Define a simple predict function
    def predict(X):
        X = np.asarray(X).reshape(-1)
        return 2 * X

    # Create a simple object with a predict attribute
    model = type("Model", (), {"predict": predict})()

    X_test = np.array([0.0, 1.0, 2.0, 3.0])
    y_test = np.array([0.0, 2.0, 4.0, 6.0])

    metrics = evaluate(model, X_test, y_test)

    assert metrics["mae"] == 0.0
    assert metrics["rmse"] == 0.0
    assert metrics["r2"] == 1.0

    assert Path("reports/pred_vs_actual.png").exists()
    assert Path("reports/residuals_vs_pred.png").exists()


def test_evaluate_raises_on_length_mismatch(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def predict(X):
        return np.asarray(X)

    model = type("Model", (), {"predict": predict})()

    with pytest.raises(ValueError):
        evaluate(model, np.array([1, 2]), np.array([1]))