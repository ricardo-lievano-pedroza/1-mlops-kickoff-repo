from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.main import SETTINGS
from src.clean_data import clean_dataframe
from src.infer import run_inference
from src.validate import validate_dataframe


# ---------------------------------------------------------
# Serving helpers
# ---------------------------------------------------------
def _feature_config() -> dict:
    return SETTINGS["features"]


def _target_column() -> str:
    return SETTINGS["target_column"]


def _model_path() -> Path:
    return Path(SETTINGS["paths"]["model"])


def _problem_type() -> str:
    return SETTINGS["problem_type"]


def _feature_columns() -> list[str]:
    feature_cfg = _feature_config()
    cols = (
        feature_cfg.get("quantile_bin", [])
        + feature_cfg.get("categorical_onehot", [])
        + feature_cfg.get("numeric_passthrough", [])
    )
    return list(dict.fromkeys(cols))


def _required_columns_for_cleaning() -> list[str]:
    # In serving we may still call clean/validate with the same schema discipline,
    # but prediction input should ultimately use only feature columns.
    return list(dict.fromkeys(_feature_columns() + [_target_column()]))


# ---------------------------------------------------------
# Pydantic schemas
# IMPORTANT:
# - extra="forbid" rejects unknown JSON fields
# - keep this schema aligned with your current dataset
# ---------------------------------------------------------
class HouseFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    district: str = Field(..., description="District/category feature")
    sq_mt: float = Field(..., alias="sq.mt", description="Square meters")
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


class PredictResponse(BaseModel):
    status: Literal["ok"]
    problem_type: str
    n_predictions: int
    predictions: list[float]


class HealthResponse(BaseModel):
    status: Literal["ok", "error"]
    model_loaded: bool
    problem_type: str
    model_path: str


# ---------------------------------------------------------
# App lifespan
# Load heavy resources once at startup
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = _model_path()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. "
            "Run main.py first to train and save the model."
        )

    app.state.model = joblib.load(model_path)
    app.state.model_path = str(model_path)
    app.state.problem_type = _problem_type()

    yield


app = FastAPI(
    title="House Rent Prediction API",
    version="0.1.0",
    description=(
        "FastAPI wrapper around the trained sklearn pipeline. "
        "No new ML logic is introduced here."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------
@app.get("/", tags=["meta"])
def root() -> dict:
    return {
        "message": "House Rent Prediction API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    model_loaded = hasattr(app.state, "model") and app.state.model is not None

    return HealthResponse(
        status="ok" if model_loaded else "error",
        model_loaded=model_loaded,
        problem_type=app.state.problem_type,
        model_path=app.state.model_path,
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        # 1) JSON -> DataFrame
        df_infer = pd.DataFrame(
            [record.model_dump(by_alias=True) for record in payload.records]
        )

        # 2) Reuse the same cleaning/validation flow to avoid skew
        feature_columns = _feature_columns()
        required_columns_for_cleaning = _required_columns_for_cleaning()

        # If your clean_dataframe requires target_column to exist,
        # create a temporary placeholder just for schema compatibility.
        # This is a bridge until you split train-clean vs infer-clean config.
        if _target_column() not in df_infer.columns:
            df_infer[_target_column()] = 0

        df_infer_clean = clean_dataframe(
            df_raw=df_infer,
            target_column=_target_column(),
            required_columns=required_columns_for_cleaning,
        )

        validate_dataframe(
            df_infer_clean,
            required_columns=required_columns_for_cleaning,
        )

        # 3) Only pass FEATURES to inference, never the target
        X_infer = df_infer_clean[feature_columns]

        # 4) Run inference
        df_pred = run_inference(
            model=app.state.model,
            X_infer=X_infer,
        )

        # Make output robust to different run_inference return shapes
        if "prediction" in df_pred.columns:
            predictions = df_pred["prediction"].tolist()
        else:
            # fallback: use first column
            predictions = df_pred.iloc[:, 0].tolist()

        return PredictResponse(
            status="ok",
            problem_type=app.state.problem_type,
            n_predictions=len(predictions),
            predictions=[float(p) for p in predictions],
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except KeyError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Missing expected field/column: {str(e)}",
        ) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}") from e