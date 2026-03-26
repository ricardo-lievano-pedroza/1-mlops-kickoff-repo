# 1) Imports
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import logging

import pandas as pd

from sklearn.model_selection import train_test_split

from src.validate import validate_dataframe
from src.clean_data import clean_dataframe
from src.evaluate import evaluate_model
from src.features import get_feature_preprocessor
from src.infer import run_inference
from src.load_data import load_raw_data
from src.train import train_model
from src.utils import load_csv, save_csv, save_model

from src.logger import configure_logging

logger = logging.getLogger(__name__)

configure_logging(
        log_level="INFO",
        log_file="logs/pipeline.log"
    )

# 2) CONFIGURATION (SETTINGS dictionary bridge)
# LOUD REMINDER:
# This SETTINGS dict is a temporary bridge until config.yml is introduced.
# You MUST map these fields to your real dataset columns and business
# requirements.
SETTINGS = {
    "is_example_config": False,
    "problem_type": "regression",
    "random_state": 42,
    "test_size": 0.25,
    "target_column": "rent",
    "paths": {
        "raw_data": "data/raw/Houses_for_rent.csv",
        "processed_data": "data/processed/clean.csv",
        "model": "models/model.joblib",
        "predictions": "reports/predictions.csv",
        "inference": "data/inference/Houses_for_rent_inference.csv"
    },
    "features": {
        # Pre-configured to match the dummy CSV created by src/load_data.py
        "quantile_bin": [],
        # keep empty by default; dummy numeric feature passes through
        "categorical_onehot": ["district"],
        "numeric_passthrough": [
            "sq.mt",
            "floor",
            "bedrooms",
            "outer",
            "duplex",
            "cottage",
            "elevator",
            "penthouse",
            "semidetached"
            ],
        "n_bins": 3,
    },
}


def main():

    """
    Inputs:
    - None (reads SETTINGS and files from repo-relative paths)
    Outputs:
    - None (side effects: writes clean.csv, model.joblib, predictions.csv;
    prints metric)
    Why this contract matters for reliable ML delivery:
    - A single entrypoint makes execution consistent across laptops, CI, and
    future schedulers (Airflow, Prefect, etc.).
    """
    logger.info("Stating pipeline")
    print("1") # TODO: replace with logging later

    # --------------------------------------------------------
    # Step 0: Ensure output directories exist (manual materialization only)
    # --------------------------------------------------------
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    Path("models").mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(parents=True, exist_ok=True)
    Path("data/inference").mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Step 0: Example-config check
    # --------------------------------------------------------
    if SETTINGS.get("is_example_config", False):
        print("LOUD NOTE: SETTINGS is using the example dummy configuration.")
        # TODO: replace with logging later
        print(
            "Update SETTINGS to match your real dataset before production use."
            )
        # TODO: replace with logging later

    # Resolve paths
    raw_path = Path(SETTINGS["paths"]["raw_data"])
    processed_path = Path(SETTINGS["paths"]["processed_data"])
    model_path = Path(SETTINGS["paths"]["model"])
    preds_path = Path(SETTINGS["paths"]["predictions"])
    inference_path = Path(SETTINGS["paths"]["inference"])

    target_column = SETTINGS["target_column"]
    problem_type = SETTINGS["problem_type"]

    # --------------------------------------------------------
    # Step 1: Load
    # --------------------------------------------------------
    logger.info("1) LOAD raw data")
    df_raw = load_raw_data(raw_path)

    # --------------------------------------------------------
    # Step 2: Clean
    # --------------------------------------------------------
    logger.info("2) CLEAN training data")

    feature_cfg = SETTINGS["features"]
    configured_feature_cols = (
        feature_cfg.get("quantile_bin", [])
        + feature_cfg.get("categorical_onehot", [])
        + feature_cfg.get("numeric_passthrough", [])
    )
    required_columns = list(
        dict.fromkeys(configured_feature_cols + [target_column])
    )
    df_clean = clean_dataframe(df_raw, target_column=target_column)

    # --------------------------------------------------------
    # Step 3: Save processed CSV (artifact requirement)
    # --------------------------------------------------------
    logger.info("3) SAVE processed data")

    save_csv(df_clean, processed_path)

    # --------------------------------------------------------
    # Step 4: Validate
    # --------------------------------------------------------
    logger.info("4) VALIDATE training data")

    validate_dataframe(
        df=df_clean,
        required_columns=required_columns,
        target_column=SETTINGS['target_column'],
        numeric_non_negative_cols=[c for c in SETTINGS['features']['numeric_passthrough'] if c != 'floor']
        )

    # --------------------------------------------------------
    # Step 5: Train/test split (BEFORE any feature fitting to prevent leakage)
    # --------------------------------------------------------
    logger.info("5) SPLIT train/val/test")

    X = df_clean.drop(columns=[target_column])
    y = df_clean[target_column]

    stratify = y if problem_type == "classification" else None

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS["test_size"],
            random_state=SETTINGS["random_state"],
            stratify=stratify,
        )
    except ValueError as e:
        logger.info(
            f"Stratified split failed ({e});"
            "falling back to non-stratified split."
            )

        X_train, X_val, X_test, y_train, y_val, y_test = train_test_split(
            X,
            y,
            test_size=SETTINGS['test_size'],
            random_state=SETTINGS['random_state'],
            stratify=(problem_type == "classification"),
        )
    logger.info("Split sizes | train=%s | test=%s", X_train.shape, X_test.shape)

    # --------------------------------------------------------
    # Step 5.1: Fail-fast feature checks (columns exist + quantile_bin cols are
    # numeric)
    # --------------------------------------------------------
    logger.info("Running fail-fast feature configuration checks")

    missing_cols = [
        c for c in configured_feature_cols if c not in X_train.columns
    ]
    if missing_cols:
        raise ValueError(
            "Feature config error: these configured feature columns are"
            f"missing from X_train: {missing_cols}. "
            "Update SETTINGS['features'] to match your dataset."
        )

    # Explicitly check that quantile_bin columns are numeric
    quantile_cols = feature_cfg.get("quantile_bin", [])
    for c in quantile_cols:
        if not pd.api.types.is_numeric_dtype(X_train[c]):
            raise ValueError(
                f"Feature config error: column '{c}' is in "
                "SETTINGS['features'] ['quantile_bin'] but is not numeric."
                "Move it to categorical_onehot or fix the dtype in cleaning."
            )

    # --------------------------------------------------------
    # Step 6: Build feature recipe (unfitted ColumnTransformer)
    # --------------------------------------------------------
    logger.info("6) BUILD feature recipe")

    preprocessor = get_feature_preprocessor(
        bin_cols=feature_cfg.get("quantile_bin", []),
        categorical_cols=feature_cfg.get("categorical_onehot", []),
        numeric_cols=feature_cfg.get("numeric_passthrough", []),
        n_bins=int(feature_cfg.get("n_bins", 3)),
    )

    # --------------------------------------------------------
    # Step 7: Train model (Pipeline fits preprocess+model on TRAIN only)
    # --------------------------------------------------------
    logger.info("7) TRAIN base model pipeline")

    model = train_model(
        X_train=X_train,
        y_train=y_train,
        preprocessor=preprocessor,
        problem_type=problem_type
    )

    # --------------------------------------------------------
    # Step 8: Evaluate
    # --------------------------------------------------------
    logger.info("8) EVALUATE on validation split")

    metric_value = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        problem_type=problem_type
    )

    if problem_type == "regression":
        logger.info(f"Held-out RMSE: {metric_value}")
    else:
        logger.info(f"[main.main] Held-out weighted F1: {metric_value}")

    # --------------------------------------------------------
    # Step 9 Save model (artifact requirement)
    # --------------------------------------------------------
    logger.info("9) SAVE model artifact")

    save_model(model, model_path)

    # --------------------------------------------------------
    # Step 10: Inference on example data + save predictions
    # (artifact requirement)
    # --------------------------------------------------------
    logger.info("10) INFER on new data file")

    df_infer = load_csv(inference_path)
    df_infer_clean = clean_dataframe(
        df_raw=df_infer,
        target_column=SETTINGS['target_column']
        )

    validate_dataframe(df=df_infer_clean,
                       required_columns=required_columns,
                       target_column=SETTINGS['target_column'],
                       numeric_non_negative_cols=SETTINGS['features']['numeric_passthrough']
                       )

    X_infer = df_infer_clean[required_columns]

    df_pred = run_inference(model=model, X_infer=X_infer)

    logger.debug("Inference preview\n%s", df_pred.head(10).to_string(index=False))

    save_csv(df_pred, preds_path)

    logger.info("Done")
    logger.info("Wrote processed data: %s", processed_path)
    logger.info("Wrote model artifact: %s", model_path)
    logger.info("Wrote predictions: %s", preds_path)
    print("2")

if __name__ == "__main__":
    main()
