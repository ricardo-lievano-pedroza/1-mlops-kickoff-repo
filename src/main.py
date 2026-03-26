# 1) Imports
from pathlib import Path
from typing import Any, Dict, List, Optional

import logging

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import wandb

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


# ------------------------------------------------------------
# Config validation functions
# ------------------------------------------------------------


def load_config(config_path: Path) -> Dict[str, Any]:
    """
    Load YAML configuration from disk

    Why this exists
    - Centralizing config loading prevents "config drift" where different modules parse YAML differently
    - Fail fast with clear messages when config.yaml is missing or malformed
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(
            "config.yaml must parse into a dictionary at the top level")

    return cfg


def require_section(cfg: Dict[str, Any], section: str) -> Dict[str, Any]:
    """
    Enforce a required top-level config section

    Why this exists
    - This produces an actionable error tied to config.yaml structure
    """
    value = cfg.get(section)
    if not isinstance(value, dict):
        raise ValueError(
            f"config.yaml must contain a top-level '{section}' mapping")
    return value


def require_str(section: Dict[str, Any], key: str) -> str:
    value = section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"config.yaml: '{key}' must be a non-empty string")
    return value.strip()


def require_float(section: Dict[str, Any], key: str) -> float:
    value = section.get(key)
    try:
        return float(value)
    except Exception as e:
        raise ValueError(
            f"config.yaml: '{key}' must be a number. Got '{value}'") from e


def require_int(section: Dict[str, Any], key: str) -> int:
    value = section.get(key)
    try:
        return int(value)
    except Exception as e:
        raise ValueError(
            f"config.yaml: '{key}' must be an integer. Got '{value}'") from e


def require_list(section: Dict[str, Any], key: str) -> List[str]:
    value = section.get(key)
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(
            f"config.yaml: '{key}' must be a list. Got type={type(value)}")
    out: List[str] = []
    for item in value:
        if isinstance(item, str) and item.strip():
            out.append(item.strip())
    return out


def normalize_problem_type(problem_type: Optional[str]) -> str:
    return (problem_type or "").strip().lower()


def resolve_repo_path(project_root: Path, relative_path: str) -> Path:
    """
    Resolve a config path relative to the repo root

    This makes the repo reproducible across machines because we never rely on the current working directory
    """
    if not isinstance(relative_path, str) or not relative_path.strip():
        raise ValueError("config.yaml: path values must be non-empty strings")
    return project_root / relative_path.strip()


def dedupe_preserve_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


# Weight and bias:

def _wandb_is_enabled(cfg: Dict[str, Any]) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return False
    return bool(wandb_cfg.get("enabled", False))


def _wandb_get_str(cfg: Dict[str, Any], key: str, default: str = "") -> str:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return str(value).strip() if value is not None else default


def _wandb_get_bool(cfg: Dict[str, Any], key: str, default: bool = False) -> bool:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    return bool(value)


def _wandb_get_int(cfg: Dict[str, Any], key: str, default: int = 0) -> int:
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return default
    value = wandb_cfg.get(key, default)
    try:
        return int(value)
    except Exception:
        return default


def _wandb_get_list(cfg: Dict[str, Any], key: str) -> List[str]:
    """Safely extract a list of strings, stripping whitespace and dropping empty values."""
    wandb_cfg = cfg.get("wandb")
    if not isinstance(wandb_cfg, dict):
        return []

    value = wandb_cfg.get(key, [])
    if not isinstance(value, list):
        return []

    out: List[str] = []
    for v in value:
        if v is None:
            continue
        s = str(v).strip()
        if s:
            out.append(s)
    return out


def main():
    project_root = Path(__file__).resolve().parents[1]
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

    # Load and validate config.yaml
    
    cfg = load_config(project_root / "config.yaml")

    paths_cfg = require_section(cfg, "paths")
    problem_cfg = require_section(cfg, "problem")
    split_cfg = require_section(cfg, "split")
    features_cfg = require_section(cfg, "features")
    logging_cfg = require_section(cfg, "logging")
    log_file_path = resolve_repo_path(
        project_root, require_str(paths_cfg, "log_file"))
    log_level = require_str(logging_cfg, "level")

    configure_logging(
        log_level=log_level,
        log_file=log_file_path,
    )

    # -----------------------------
    # Initialize W&B Experiment Tracking
    # -----------------------------
    wandb_run = None
    if _wandb_is_enabled(cfg):
        wandb_project = _wandb_get_str(cfg, "project")
        if not wandb_project:
            raise ValueError(
                "config.yaml: wandb.project must be a non-empty string when wandb.enabled is true"
            )

        wandb_name = _wandb_get_str(cfg, "name")
        wandb_job_type = _wandb_get_str(
            cfg, "job_type", default="factory-pipeline")
        wandb_group = _wandb_get_str(cfg, "group")
        wandb_notes = _wandb_get_str(cfg, "notes")
        wandb_tags = _wandb_get_list(cfg, "tags")

        wandb_run = wandb.init(
            project=wandb_project,
            name=wandb_name if wandb_name else None,
            job_type=wandb_job_type,
            group=wandb_group if wandb_group else None,
            notes=wandb_notes if wandb_notes else None,
            tags=wandb_tags if wandb_tags else None,
            config=cfg,
        )

        wandb_run.summary["entrypoint"] = "python -m src.main"
        wandb_run.summary["model_artifact_path"] = str(
            require_str(paths_cfg, "model_artifact"))

        logger.info(
            "Initialized W&B run | name=%s | project=%s | job_type=%s",
            wandb_run.name,
            wandb_project,
            wandb_job_type,
        )
    else:
        logger.info("W&B disabled, continuing without experiment tracking")

    try:
        logger.info("Stating pipeline")
        
        # Resolve paths
        raw_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "raw_data"))
        processed_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "processed_data"))
        model_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "model_artifact"))
        inference_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "inference_data"))
        preds_path = resolve_repo_path(
            project_root, require_str(paths_cfg, "predictions_artifact"))

        # Problem definition
        target_column = require_str(problem_cfg, "target_column")
        problem_type = require_str(problem_cfg, "problem_type")

        # Split paramters:
        test_size = require_float(split_cfg, "test_size")
        random_seed = require_int(split_cfg, "random_seed")

        # Features: 
        quantile_cols = require_list(features_cfg, "quantile_cols")
        categorical_cols = require_list(features_cfg, "categorical_onehot")
        numerical_cols = require_list(features_cfg, "numeric_passthrough")
        n_bins = require_int(features_cfg, "n_bins")

        configured_feature_cols = quantile_cols + categorical_cols + numerical_cols

        non_negative_cols = require_list(features_cfg, "non_negative")

        required_columns = list(
            dict.fromkeys(configured_feature_cols + [target_column])
        )

        # --------------------------------------------------------
        # Step 1: Load
        # --------------------------------------------------------
        logger.info("1) LOAD raw data")
        df_raw = load_raw_data(raw_path)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/raw_rows": int(df_raw.shape[0]),
                    "data/raw_cols": int(df_raw.shape[1]),
                }
            )

        # --------------------------------------------------------
        # Step 2: Clean
        # --------------------------------------------------------
        logger.info("2) CLEAN training data")

        df_clean = clean_dataframe(df_raw, target_column=target_column)

        if wandb_run is not None:
            wandb.log(
                {
                    "data/clean_rows": int(df_clean.shape[0]),
                    "data/clean_cols": int(df_clean.shape[1]),
                }
            )

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
            target_column=target_column,
            numeric_non_negative_cols=non_negative_cols
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
                test_size=test_size,
                random_state=random_seed,
                stratify=stratify,
            )
        except ValueError as e:
            logger.info(
                f"Stratified split failed ({e});"
                "falling back to non-stratified split."
                )

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                random_state=random_seed,
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
                "Update features in congif.yaml to match your dataset."
            )

        # Explicitly check that quantile_bin columns are numeric

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
            bin_cols=quantile_cols,
            categorical_cols=categorical_cols,
            numeric_cols=numerical_cols,
            n_bins=n_bins,
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
            logger.info(
            "Validation metrics | MAE=%.4f | RMSE=%.4f | R2=%.4f",
                metric_value["mae"],
                metric_value["rmse"],
                metric_value["r2"],
)
        else:
            logger.info(f" Held-out weighted F1: {metric_value}")

        if wandb_run is not None:
            wandb.log({f"metrics/val/{k}": float(v) for k, v in metric_value.items()})

        if wandb_run is not None and len(metric_value) > 1:
            comparison_df = pd.DataFrame.from_dict(
                {
                    "metric": list(metric_value.keys()),
                    "value": list(metric_value.values()),
                }
            )
            wandb.log(
                {"tables/metrics_comparison_val": wandb.Table(dataframe=comparison_df)})       

        # --------------------------------------------------------
        # Step 9 Save model (artifact requirement)
        # --------------------------------------------------------
        logger.info("9) SAVE model artifact")

        save_model(model, model_path)
        if wandb_run is not None:
            model_artifact_name = _wandb_get_str(
                cfg, "model_artifact_name", default="model")
            model_artifact = wandb.Artifact(
                name=model_artifact_name,
                type="model",
                description="Scikit-learn pipeline or calibrated model artifact",
            )
            model_artifact.add_file(str(model_path))
            wandb.log_artifact(model_artifact)

            if _wandb_get_bool(cfg, "log_processed_data", default=False):
                data_artifact = wandb.Artifact(
                    name=f"{model_artifact_name}-processed-data",
                    type="dataset",
                    description="Processed training dataset written by the factory pipeline",
                )
                data_artifact.add_file(str(processed_path))
                wandb.log_artifact(data_artifact)
        # --------------------------------------------------------
        # Step 10: Inference on example data + save predictions
        # (artifact requirement)
        # --------------------------------------------------------
        logger.info("10) INFER on new data file")

        df_infer = load_csv(inference_path)
        df_infer_clean = clean_dataframe(
            df_raw=df_infer
            )

        validate_dataframe(df=df_infer_clean,
                        required_columns=required_columns,
                        numeric_non_negative_cols=non_negative_cols)

        X_infer = df_infer_clean[configured_feature_cols]

        df_pred = run_inference(model=model, X_infer=X_infer)

        if wandb_run is not None and _wandb_get_bool(cfg, "log_predictions_table", default=False):
            n_rows = _wandb_get_int(cfg, "predictions_table_rows", default=200)
            sample_df = df_pred.head(n_rows)
            wandb.log(
                {"tables/predictions_preview": wandb.Table(dataframe=sample_df)})

        logger.debug("Inference preview\n%s", df_pred.head(10).to_string(index=False))

        save_csv(df_pred, preds_path)

        if wandb_run is not None and _wandb_get_bool(cfg, "log_predictions", default=False):
            model_artifact_name = _wandb_get_str(
                cfg, "model_artifact_name", default="model")
            pred_artifact = wandb.Artifact(
                name=f"{model_artifact_name}-predictions",
                type="predictions",
                description="Inference outputs written by the factory pipeline",
            )
            pred_artifact.add_file(str(preds_path))
            wandb.log_artifact(pred_artifact)

        logger.info("Done")
        logger.info("Wrote processed data: %s", processed_path)
        logger.info("Wrote model artifact: %s", model_path)
        logger.info("Wrote predictions: %s", preds_path)
    except Exception:
        logger.exception("Pipeline failed")
        if wandb_run is not None:
            wandb.finish(exit_code=1)
        raise

    finally:
        if wandb_run is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
