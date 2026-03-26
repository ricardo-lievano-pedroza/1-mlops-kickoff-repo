"""
Educational Goal:
- ⁠Why this module exists in an MLOps system:
  Inference is where business value is delivered (predictions drive decisions).
  A dedicated inference function reduces deployment risk and ensures consistent output format.
- ⁠Responsibility (separation of concerns):
  Run model.predict on new data and return a clean predictions DataFrame.
- ⁠Pipeline contract (inputs and outputs):
  Inputs: model + X_infer DataFrame. Output: DataFrame with exactly one column "prediction".
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Executes prediction on unseen data.

    Inputs:
    - model: fitted sklearn Pipeline.
    - X_infer: DataFrame of features to predict on.

    Outputs:
    - predictions_df: DataFrame with a single column "prediction", preserving input index.

    Why this contract matters for reliable ML delivery:
    - Stable output schemas prevent downstream integration failures.
    - Retain previous index to joing back to original database records.
    """
    logger.info("Running Inference")

    if X_infer is None or len(X_infer) == 0:
        raise ValueError(
            "FATAL: X_infer is empty. Cannot run inference"
        ) 

    if not isinstance(X_infer, pd.DataFrame):
        raise TypeError(
            f"FATAL: X_infer must be a pandas DataFrame. got type = {type(X_infer)} "
        )

    if not hasattr(model, "predict"):
        raise TypeError(
            f"FATAL: model needs to have predict() method. got type= {type(model)}"
        )
    predictions_df = pd.DataFrame(index=X_infer.index)

    predictions_df['prediction'] = model.predict(X_infer)

    return predictions_df
