import pandas as pd
import pytest

from src import load_data


def test_creates_dummy_dataset_when_missing_and_example_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("IS_EXAMPLE_CONFIG", "true")
    raw_path = tmp_path / "raw" / "people-10000.csv"

    df = load_data.load_raw_data(raw_path)

    assert raw_path.exists()
    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert set(["num_feature", "cat_feature", "target"]).issubset(df.columns)


def test_raises_when_missing_and_not_example_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("IS_EXAMPLE_CONFIG", "false")
    raw_path = tmp_path / "raw" / "people-10000.csv"

    with pytest.raises(FileNotFoundError):
        load_data.load_raw_data(raw_path)


def test_loads_existing_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("IS_EXAMPLE_CONFIG", "false")
    raw_path = tmp_path / "raw" / "existing.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(
        "num_feature,cat_feature,target\n"
        "1.0,A,0.0\n"
        "2.0,B,1.0\n",
        encoding="utf-8",
    )

    df = load_data.load_raw_data(raw_path)

    assert df.shape == (2, 3)
    assert list(df.columns) == ["num_feature", "cat_feature", "target"]


def test_raises_on_empty_csv(tmp_path, monkeypatch):
    monkeypatch.setenv("IS_EXAMPLE_CONFIG", "false")
    raw_path = tmp_path / "raw" / "empty.csv"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    raw_path.write_text("num_feature,cat_feature,target\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_data.load_raw_data(raw_path)