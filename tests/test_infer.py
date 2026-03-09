import numpy as np
import pandas as pd
import pytest

from src.infer import run_inference


# ---- Simple stub models (no sklearn dependency needed) ----

class PredictModel:
    def __init__(self, preds):
        self._preds = np.asarray(preds)

    def predict(self, X):
        # Return as-is; shape could be 1D or (n,1) depending on the test
        return self._preds


class ProbaModel:
    def __init__(self, proba):
        self._proba = np.asarray(proba)

    def predict(self, X):
        # Not used when use_proba=True, but included for completeness
        return np.zeros(len(X))

    def predict_proba(self, X):
        return self._proba


# ---- Fixtures ----

@pytest.fixture
def X_infer():
    return pd.DataFrame(
        {"f1": [1, 2, 3], "f2": [10, 20, 30]},
        index=pd.Index(["a", "b", "c"], name="id")
    )


# ---- Tests ----

def test_returns_prediction_only_dataframe(X_infer):
    model = PredictModel([0.1, 0.2, 0.3])
    df_pred = run_inference(model, X_infer)

    assert isinstance(df_pred, pd.DataFrame)
    assert list(df_pred.columns) == ["prediction"]
    assert len(df_pred) == len(X_infer)


def test_preserves_index_exactly(X_infer):
    model = PredictModel([1, 2, 3])
    df_pred = run_inference(model, X_infer)

    assert df_pred.index.equals(X_infer.index)


def test_empty_input_returns_empty_prediction_df():
    X_empty = pd.DataFrame({"f1": []}, index=pd.Index([], name="id"))
    model = PredictModel([])

    df_pred = run_inference(model, X_empty)

    assert list(df_pred.columns) == ["prediction"]
    assert df_pred.index.equals(X_empty.index)
    assert df_pred.empty


def test_raises_if_model_has_no_predict(X_infer):
    class BadModel:
        pass

    with pytest.raises(TypeError):
        run_inference(BadModel(), X_infer)


def test_raises_if_X_infer_not_dataframe():
    model = PredictModel([1, 2, 3])
    with pytest.raises(TypeError):
        run_inference(model, X_infer=[1, 2, 3])  # not a DataFrame


def test_use_proba_returns_selected_class_column(X_infer):
    # 3 samples, 2 classes -> choose class index 1 by default
    proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.4, 0.6],
    ])
    model = ProbaModel(proba)

    df_pred = run_inference(model, X_infer, use_proba=True)

    assert np.allclose(df_pred["prediction"].to_numpy(), proba[:, 1])


def test_use_proba_raises_if_predict_proba_missing(X_infer):
    model = PredictModel([1, 2, 3])  # has predict but no predict_proba
    with pytest.raises(TypeError):
        run_inference(model, X_infer, use_proba=True)


def test_threshold_binarizes_and_validates_bounds(X_infer):
    proba = np.array([
        [0.9, 0.1],
        [0.2, 0.8],
        [0.4, 0.6],
    ])
    model = ProbaModel(proba)

    df_pred = run_inference(model, X_infer, use_proba=True, threshold=0.7)
    assert df_pred["prediction"].tolist() == [0, 1, 0]

    with pytest.raises(ValueError):
        run_inference(model, X_infer, use_proba=True, threshold=1.5)


def test_clipping_applies_correctly(X_infer):
    model = PredictModel([-5.0, 0.5, 10.0])
    df_pred = run_inference(model, X_infer, clip_min=0.0, clip_max=1.0)

    assert df_pred["prediction"].tolist() == [0.0, 0.5, 1.0]


def test_inverse_transform_applies_and_errors_surface(X_infer):
    model = PredictModel([0.0, 1.0, 2.0])

    df_pred = run_inference(model, X_infer, inverse_transform_fn=lambda a: a + 10)
    assert df_pred["prediction"].tolist() == [10.0, 11.0, 12.0]

    def bad_fn(a):
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        run_inference(model, X_infer, inverse_transform_fn=bad_fn)