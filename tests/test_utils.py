from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from src.utils import load_csv, load_model, save_csv, save_model

def test_save_and_load_csv_roundtrip(tmp_path: Path):
    df_in = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    csv_path = tmp_path / "nested" / "data.csv"
    
    save_csv(df_in, csv_path)
    df_out = load_csv(csv_path)

    assert csv_path.exists()
    pd.testing.assert_frame_equal(df_in, df_out)


def test_save_csv_creates_parent_directories(tmp_path: Path):
    df_in = pd.DataFrame({"col": [1]})
    csv_path = tmp_path / "a" / "b" / "c" / "file.csv"
    assert not csv_path.parent.exists()

    save_csv(df_in, csv_path)
    assert csv_path.parent.exists()
    assert csv_path.exists()


def test_load_csv_raises_if_missing_file(tmp_path: Path):
    missing = tmp_path / "does_not_exist.csv"

    with pytest.raises(FileNotFoundError):
        load_csv(missing)


def test_save_and_load_model_roundtrip(tmp_path: Path):
    model_in = Ridge()
    model_path = tmp_path / "models" / "model.joblib"

    save_model(model_in, model_path)
    model_out = load_model(model_path)

    assert model_path.exists()
    assert type(model_out) is type(model_in)


def test_save_model_creates_parent_directories(tmp_path: Path):
    model_in = Ridge()
    model_path = tmp_path / "x" / "y" / "model.joblib"
    assert not model_path.parent.exists()
    
    save_model(model_in, model_path)

    assert model_path.parent.exists()
    assert model_path.exists()