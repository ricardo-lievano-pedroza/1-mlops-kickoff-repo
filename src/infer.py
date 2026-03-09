"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data.
Input: Trained Model + New Data.
Output: Predictions (Array or DataFrame).
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Make inference a first-class, testable step (training and inference often diverge in notebooks).
- Responsibility (separation of concerns): Run model.predict and return a clean predictions-only DataFrame.
- Pipeline contract (inputs and outputs): Input fitted Pipeline + feature DataFrame; output DataFrame with ONE column 'prediction' preserving index.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd
import numpy as np
from typing import Optional, Callable, Union


def run_inference(
    model,
    X_infer: pd.DataFrame,
    *,
    # --- Optional postprocessing knobs ---
    use_proba: bool = False,
    proba_class_index: int = 1,
    threshold: Optional[float] = None,          # if set, converts probabilities -> {0,1}
    clip_min: Optional[float] = None,           # clip regression (or proba) lower bound
    clip_max: Optional[float] = None,           # clip upper bound
    round_ndigits: Optional[int] = None,        # rounding (e.g., 2 decimals)
    inverse_transform_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Inputs:
    - model: Fitted scikit-learn Pipeline (must implement predict; optionally predict_proba)
    - X_infer: Feature DataFrame for inference

    Outputs:
    - df_pred: DataFrame with a single column named "prediction" preserving the input index

    Why this contract matters for reliable ML delivery:
    - A strict inference output contract prevents accidental schema drift and simplifies downstream consumers.
    """

    print("[infer.run_inference] Running inference and returning prediction-only DataFrame")  # TODO: replace with logging later

    # -----------------------
    # Basic validation
    # -----------------------
    if not hasattr(model, "predict"):
        raise TypeError("run_inference expected `model` to have a `.predict()` method (fitted sklearn estimator/pipeline).")

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError(f"run_inference expected `X_infer` to be a pandas DataFrame, got: {type(X_infer)}")

    if X_infer.shape[0] == 0:
        # Preserve schema + index even when empty
        return pd.DataFrame({"prediction": pd.Series(dtype=float)}, index=X_infer.index)

    # -----------------------
    # Prediction (predict or predict_proba)
    # -----------------------
    if use_proba:
        if not hasattr(model, "predict_proba"):
            raise TypeError("use_proba=True but model does not implement `.predict_proba()`.")

        proba = model.predict_proba(X_infer)

        # proba can be (n_samples, n_classes)
        proba = np.asarray(proba)
        if proba.ndim != 2:
            raise ValueError(f"predict_proba returned an unexpected shape: {proba.shape}")

        if proba_class_index < 0 or proba_class_index >= proba.shape[1]:
            raise ValueError(
                f"proba_class_index={proba_class_index} is out of bounds for predict_proba output with shape {proba.shape}"
            )

        preds = proba[:, proba_class_index]
    else:
        preds = model.predict(X_infer)

    # Normalize predictions to a clean 1D numpy array
    preds = np.asarray(preds)
    if preds.ndim == 2 and preds.shape[1] == 1:
        preds = preds[:, 0]
    elif preds.ndim != 1:
        raise ValueError(f"Model predictions have unexpected shape {preds.shape}. Expected 1D or (n,1).")

    # -----------------------
    # Postprocessing
    # -----------------------
    # Inverse transform (for targets that were transformed during training)
    if inverse_transform_fn is not None:
        try:
            preds = inverse_transform_fn(preds)
            preds = np.asarray(preds)
        except Exception as e:
            raise RuntimeError(f"inverse_transform_fn failed: {e}") from e

    # Clip (common for regression constraints)
    if clip_min is not None or clip_max is not None:
        preds = np.clip(preds, a_min=clip_min, a_max=clip_max)

    # Optional thresholding for classification probabilities
    if threshold is not None:
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"threshold must be between 0 and 1; got {threshold}")
        preds = (preds >= threshold).astype(int)

    # Optional rounding (useful for outputs shown to humans)
    if round_ndigits is not None:
        preds = np.round(preds, round_ndigits)

    # -----------------------
    # Output contract: ONE column 'prediction', preserve index
    # -----------------------
    df_pred = pd.DataFrame({"prediction": preds}, index=X_infer.index)

    # Hard guardrails: enforce contract strictly
    if list(df_pred.columns) != ["prediction"]:
        raise RuntimeError("Inference output contract violated: output must have exactly one column named 'prediction'.")
    if not df_pred.index.equals(X_infer.index):
        raise RuntimeError("Inference output contract violated: output index must match input index exactly.")

    return df_pred