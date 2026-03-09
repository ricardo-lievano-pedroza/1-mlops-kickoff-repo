from pathlib import Path

import numpy as np
import pandas as pd

from src.clean_data import clean_dataframe
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.train import train_model
from src.utils import load_model, load_csv, save_csv, save_model
from src.validate import validate_dataframe


def test_end_to_end_pipeline_integration(tmp_path: Path):
    # 1) Arrange: tiny deterministic dataset (like your rent use case)
    df = pd.DataFrame(
        {
            # TODO_STUDENT: Replace these names to match your real config if
            # needed
            "sq_mt": [50.0, 70.0, np.nan, 40.0],
            "floors": [1, 2, 3, 1],
            "bedrooms": [1, 2, 3, 1],
            "outer": [1, 0, 1, 0],
            "duplex": [0, 0, 1, 0],
            "semidetached": [0, 1, 0, 0],
            "rent": [1200.0, 1800.0, 2500.0, 1000.0],
        }
    )

    target_column = "rent"

    # temp artifact paths (no touching real repo folders)
    raw_path = tmp_path / "raw.csv"
    clean_path = tmp_path / "clean.csv"
    model_path = tmp_path / "model.joblib"
    preds_path = tmp_path / "predictions.csv"

    # 2) Materialize raw
    save_csv(df, raw_path)
    df_raw = load_csv(raw_path)

    # 3) Validate required columns exist
    required_columns = [
        "sq_mt",
        "floors",
        "bedrooms",
        "outer",
        "duplex",
        "semidetached",
        target_column,
    ]

    # 4) Clean
    df_clean = clean_dataframe(df_raw, target_column=target_column,
                               required_columns=required_columns)
    save_csv(df_clean, clean_path)

    assert validate_dataframe(
        df_clean, required_columns=required_columns) is True

    # 5) Split (tiny and deterministic)
    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    X_train = X.iloc[:3].copy()
    y_train = y.iloc[:3].copy()
    X_test = X.iloc[3:].copy()

    # 6) Build preprocessor (numeric passthrough + optional categorical lists)
    preprocessor = get_feature_preprocessor(
        bin_cols=[],
        categorical_cols=[],
        # TODO_STUDENT: add categorical cols if you have them as strings
        numeric_cols=list(X.columns),
        n_bins=3,
    )

    # 7) Train
    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type="regression",
    )

    # 8) Save/load model artifact
    save_model(model, model_path)
    loaded = load_model(model_path)

    # 9) Inference
    df_pred = run_inference(loaded, X_infer=X_test)

    # 10) Assertions: output contract
    assert isinstance(df_pred, pd.DataFrame)
    assert list(df_pred.columns) == ["prediction"]
    assert df_pred.index.equals(X_test.index)
    assert len(df_pred) == len(X_test)

    # 11) Save predictions artifact
    save_csv(df_pred, preds_path)
    df_pred_reload = load_csv(preds_path)
    assert "prediction" in df_pred_reload.columns