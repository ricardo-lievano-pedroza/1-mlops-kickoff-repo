def test_evaluate(tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    import pytest

    from evaluation import evaluate  # change if your module path differs

    # run in isolated folder so reports/ is created under tmp_path
    monkeypatch.chdir(tmp_path)

    class DummyModel:
        def predict(self, X):
            return np.array([1.0, 2.0, 3.0])

    X_test = pd.DataFrame({"x": [0, 1, 2]})
    y_test = pd.Series([1.0, 2.0, 3.0])

    metrics = evaluate(DummyModel(), X_test, y_test)

    # metrics exist and are numeric
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    assert metrics["mae"] == pytest.approx(0.0)
    assert metrics["rmse"] == pytest.approx(0.0)
    assert metrics["r2"] == pytest.approx(1.0)

    # plots were saved
    assert (Path("reports") / "pred_vs_actual.png").exists()
    assert (Path("reports") / "residuals_vs_pred.png").exists()