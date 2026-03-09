from pathlib import Path
import pandas as pd
import joblib


def load_csv(filepath: Path) -> pd.DataFrame:
    print(f"[utils.load_csv] Loading CSV from: {filepath}")
    return pd.read_csv(filepath)


def save_csv(df: pd.DataFrame, filepath: Path) -> None:
    print(f"[utils.save_csv] Saving CSV to: {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=False)


def save_model(model, filepath: Path) -> None:
    print(f"[utils.save_model] Saving model to: {filepath}")
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath: Path):
    print(f"[utils.load_model] Loading model from: {filepath}")
    return joblib.load(filepath)