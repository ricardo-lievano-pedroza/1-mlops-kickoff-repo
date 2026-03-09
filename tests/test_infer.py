import numpy as np
import pandas as pd

from src.infer import run_inference


class DummyModel:
    def _init_(self, preds):
        self._preds = np.asarray(preds)

    def predict(self, X):
        return self._preds


def test_returns_prediction_only_dataframe():
    X = pd.DataFrame({"f1": [1, 2, 3]}, index=[10, 11, 12])
    model = DummyModel([0.1, 0.2, 0.3])

    df_pred = run_inference(model, X)

    assert isinstance(df_pred, pd.DataFrame)
    assert list(df_pred.columns) == ["prediction"]
    assert df_pred.shape == (3, 1)


def test_preserves_index_exactly():
    X = pd.DataFrame({"f1": [1, 2, 3]},
                     index=pd.Index(["a", "b", "c"], name="id"))
    model = DummyModel([1, 2, 3])

    df_pred = run_inference(model, X)

    assert df_pred.index.equals(X.index)


def test_empty_input_returns_empty_prediction_df():
    X_empty = pd.DataFrame({"f1": []}, index=pd.Index([], name="id"))
    model = DummyModel([])

    df_pred = run_inference(model, X_empty)

    assert isinstance(df_pred, pd.DataFrame)
    assert list(df_pred.columns) == ["prediction"]
    assert df_pred.index.equals(X_empty.index)
    assert df_pred.shape == (0, 1)