from contextlib import asynccontextmanager
import logging
import os
import time
import uuid
from threading import Lock
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
import wandb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from pydantic import BaseModel, ConfigDict

from src.clean_data import clean_dataframe
from src.infer import run_inference
from src.validate import validate_dataframe

# Keep this as the only coupling to main.py to keep the API decoupled
from src.main import load_config, require_section, require_str, resolve_repo_path

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

# Load environment variables from .env (e.g. for config paths or secrets)
load_dotenv()

# ---------------------------------------------------------
# Pydantic schemas
# IMPORTANT:
# - extra="forbid" rejects unknown JSON fields
# - keep this schema aligned with your current dataset
# ---------------------------------------------------------

class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    district: str
    sq_mt: float
    floor: int
    bedrooms: int
    outer: int
    duplex: int
    cottage: int
    elevator: int
    penthouse: int
    semidetached: int


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: List[HouseFeatures]


class PredictionItem(BaseModel):
    prediction: int


class PredictResponse(BaseModel):
    model_version: str
    predictions: List[PredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


# -------------------------------------------------------------------
# 2) Small Local Helpers
# -------------------------------------------------------------------
def _require_list(cfg: Dict[str, Any], key: str) -> List[Any]:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key}")
    value = cfg.get(key)
    if not isinstance(value, list):
        raise ValueError(
            f"Config key '{key}' must be a list, got {type(value).__name__}")
    return value


def _dedupe_preserve_order(items: List[Any]) -> List[Any]:
    seen = set()
    out: List[Any] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _configured_feature_columns(cfg: Dict[str, Any]) -> List[str]:
    features_cfg = require_section(cfg, "features")
    cols = _dedupe_preserve_order(
        list(_require_list(features_cfg, "quantile_cols"))
        + list(_require_list(features_cfg, "categorical_onehot"))
        + list(_require_list(features_cfg, "numeric_passthrough"))
    )
    return [str(c) for c in cols]


# -------------------------------------------------------------------
# 3) Lifespan: Load shared resources once at API startup
# -------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        project_root = Path(__file__).resolve().parents[1]
        app.state.global_config = load_config(project_root / "config.yaml")
        paths_cfg = require_section(app.state.global_config, "paths")

        model_source = os.getenv("MODEL_SOURCE", "local").lower()

        if model_source == "wandb":
            logger.info(
                "MODEL_SOURCE=wandb -> fetching model from W&B Registry")

            wandb_entity = os.getenv("WANDB_ENTITY")
            artifact_alias = os.getenv("WANDB_MODEL_ALIAS", "prod")

            wandb_cfg = app.state.global_config.get("wandb", {})
            wandb_project = wandb_cfg.get("project")
            artifact_name = wandb_cfg.get("model_artifact_name")

            if not wandb_entity or not wandb_project or not artifact_name:
                raise ValueError(
                    "Missing required W&B credentials or config settings")

            artifact_path = f"{wandb_entity}/{wandb_project}/{artifact_name}:{artifact_alias}"

            wandb.login(key=os.getenv("WANDB_API_KEY"), relogin=True)
            api = wandb.Api()
            artifact = api.artifact(artifact_path)
            artifact_dir = artifact.download()
            model_path = Path(artifact_dir) / "model.joblib"

            logger.info("Downloaded model from W&B: %s", artifact_path)

            if not model_path.exists():
                logger.error(
                    "Model file missing inside downloaded artifact at %s", model_path)
                app.state.model_pipeline = None
                app.state.model_version = "missing"
            else:
                app.state.model_pipeline = joblib.load(model_path)
                app.state.model_version = artifact_path
                logger.info(
                    "Startup complete, model loaded from W&B artifact %s", artifact_path)

        else:
            logger.info("MODEL_SOURCE=local -> using local model artifact")

            model_path = resolve_repo_path(
                project_root,
                require_str(paths_cfg, "model_artifact"),
            )

            if not model_path.exists():
                logger.error("Model file missing at %s", model_path)
                app.state.model_pipeline = None
                app.state.model_version = "missing"
            else:
                app.state.model_pipeline = joblib.load(model_path)
                app.state.model_version = model_path.name
                logger.info(
                    "Startup complete, model loaded from %s", model_path)

    except Exception as e:
        logger.exception("Startup failed: %s", str(e))
        app.state.global_config = {}
        app.state.model_pipeline = None
        app.state.model_version = "startup_error"

    yield
    logger.info("API shutdown complete")


app = FastAPI(title="Madrid rental prediction API",
              version="1.0.0", lifespan=lifespan)


# -------------------------------------------------------------------
# 4) Observability Architecture (Layer 1 & Layer 2)
# -------------------------------------------------------------------

# --- LAYER 1: System Monitoring Middleware ---
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()

    correlation_id = str(uuid.uuid4())
    request.state.correlation_id = correlation_id

    response = await call_next(request)

    latency = time.time() - start_time
    model_version = getattr(app.state, "model_version", "unloaded")

    logger.info(
        "correlation_id=%s path=%s method=%s status=%s latency_s=%.4f model_version=%s",
        correlation_id,
        request.url.path,
        request.method,
        response.status_code,
        latency,
        model_version,
    )

    response.headers["X-Correlation-ID"] = correlation_id
    return response


# --- LAYER 2: ML Monitoring Buffer ---
LOG_BUFFER = []
# Protects our buffer from race conditions during concurrent API requests
BUFFER_LOCK = Lock()
BATCH_SIZE = 1


def flush_logs_to_wandb(batch_data: list, project_name: str):
    """Ephemeral W&B run to securely log a batch of inference rows as a Table."""
    if not batch_data:
        return

    if os.getenv("WANDB_MODE", "").lower() == "disabled":
        logger.info("Skipping W&B flush because WANDB_MODE=disabled")
        return

    try:
        wandb_entity = os.getenv("WANDB_ENTITY")
        run = wandb.init(
            entity=wandb_entity if wandb_entity else None,
            project=project_name,
            job_type="inference-batch",
            reinit=True,
        )

        feature_keys = list(batch_data[0]["features"].keys())
        columns = [
            "correlation_id",
            "req_id",
            "timestamp",
            "path",
            "status_code",
            "model_version",
            "latency_s",
            "prediction",
            "probability",
        ] + feature_keys

        table = wandb.Table(columns=columns)
        for item in batch_data:
            row = [
                item["correlation_id"],
                item["req_id"],
                item["timestamp"],
                item["path"],
                item["status_code"],
                item["model_version"],
                item["latency_s"],
                item["prediction"],
                item["probability"],
            ] + [item["features"].get(k) for k in feature_keys]

            table.add_data(*row)

        run.log({"inference_logs": table})
        run.finish()
        logger.info("Flushed %s ML logs to W&B", len(batch_data))

    except Exception as e:
        logger.error("Failed to flush logs to W&B: %s", e)


# -------------------------------------------------------------------
# 5) Endpoints
# -------------------------------------------------------------------
@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "Use /health or /docs to test the API"}


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    model_loaded = getattr(app.state, "model_pipeline", None) is not None
    model_version = getattr(app.state, "model_version", "unloaded")

    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "model_not_loaded",
                "model_loaded": False,
                "model_version": model_version,
            },
        )

    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=model_version,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(
    req: PredictRequest,
    request: Request,
    background_tasks: BackgroundTasks,
) -> PredictResponse:
    model_pipeline = getattr(app.state, "model_pipeline", None)
    global_config = getattr(app.state, "global_config", {})
    model_version = getattr(app.state, "model_version", "unloaded")

    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Check startup logs, W&B credentials, artifact alias, and model availability.")

    try:
        # Start timing the inference specifically for our ML Logs
        inference_start_time = time.time()
        correlation_id = getattr(
            request.state, "correlation_id", "missing_correlation_id")
        records_dicts = [r.model_dump() for r in req.records]
        df_raw = pd.DataFrame(records_dicts)
        df_clean = clean_dataframe(df_raw, target_column=None)

        features_cfg = require_section(global_config, "features")
        configured_feature_cols = (
            list(_require_list(features_cfg, "quantile_cols"))
            + list(_require_list(features_cfg, "categorical_onehot"))
            + list(_require_list(features_cfg, "numeric_passthrough"))
)

        # validation_cfg = require_section(global_config, "validation")
        non_negative = _require_list(features_cfg, "non_negative")

        validate_dataframe(
            df=df_clean,
            required_columns=configured_feature_cols,
            target_column=None,
            numeric_non_negative_cols=non_negative,
        )

        # problem_cfg = require_section(global_config, "problem")
        #identifier_col = require_str(problem_cfg, "identifier_column")

        # ids = (
        #     df_clean[identifier_col].astype(str).tolist()
        #     if identifier_col in df_clean.columns
        #     else [str(i) for i in range(len(df_clean))]
        # )
        X_infer = df_clean[configured_feature_cols]

        # run_cfg = require_section(global_config, "run")
        # include_proba = bool(run_cfg.get(
        #     "include_proba_if_classification", True))

        df_pred = run_inference(model=model_pipeline, X_infer=X_infer)

        inference_latency = time.time() - inference_start_time
        current_time = time.time()
        req_id = str(uuid.uuid4())
        preds: List[PredictionItem] = []

        with BUFFER_LOCK:
            for i in range(len(records_dicts)):
                pred_val = int(df_pred.iloc[i]["prediction"])
                prob_val: Optional[float] = None

                if "proba" in df_pred.columns:
                    prob_val = float(df_pred.iloc[i]["proba"])

                preds.append(
                    PredictionItem(
                        prediction=pred_val)
                )

                LOG_BUFFER.append({
                    "correlation_id": correlation_id,
                    "timestamp": current_time,
                    "req_id": req_id,
                    "path": "/predict",
                    "status_code": 200,
                    "model_version": model_version,
                    "latency_s": inference_latency,
                    "prediction": pred_val,
                    "probability": prob_val,
                    "features": records_dicts[i],
                })

            if len(LOG_BUFFER) >= BATCH_SIZE:
                batch_copy = LOG_BUFFER.copy()
                LOG_BUFFER.clear()
                wandb_project = global_config.get("wandb", {}).get(
                    "project", "opioid-risk-classification")
                background_tasks.add_task(
                    flush_logs_to_wandb, batch_copy, wandb_project)

        return PredictResponse(
            model_version=model_version,
            predictions=preds,
        )

    except ValueError as e:
        logger.error("Validation error: %s", str(e))
        raise HTTPException(status_code=422, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction failed: %s", str(e))
        raise HTTPException(
            status_code=500, detail="Internal Server Error") from e