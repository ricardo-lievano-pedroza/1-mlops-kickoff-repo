def test_evaluate(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    import pytest

    from src.evaluate import evaluate_model

    # run in isolated folder so reports/ is created under tmp_path
    monkeypatch.chdir(tmp_path)

    class DummyModel:
        def predict(self, X):
            return np.array([1.0, 2.0, 3.0])

    X_test = pd.DataFrame({"x": [0, 1, 2]})
    y_test = pd.Series([1.0, 2.0, 3.0])

    metrics = evaluate_model(
        DummyModel(),
        X_test,
        y_test,
        problem_type="regression"
        )

    # metrics exist and are numeric
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)
