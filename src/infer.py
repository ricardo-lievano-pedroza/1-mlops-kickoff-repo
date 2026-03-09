"""
Educational Goal:
•⁠  ⁠Why this module exists in an MLOps system:
  Inference is where business value is delivered (predictions drive decisions).
  A dedicated inference function reduces deployment risk and ensures
  consistent output format.
•⁠  ⁠Responsibility (separation of concerns):
  Run model.predict on new data and return a clean predictions DataFrame.
•⁠  ⁠Pipeline contract (inputs and outputs):
  Inputs: model + X_infer DataFrame. Output: DataFrame with exactly one column
  "prediction".

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

import pandas as pd


def run_inference(model, X_infer: pd.DataFrame) -> pd.DataFrame:
    """
    Inputs:
    - model: fitted sklearn Pipeline.
    - X_infer: DataFrame of features to predict on.
    Outputs:
    - predictions_df: DataFrame with a single column "prediction", preserving
    input index.
    Why this contract matters for reliable ML delivery:
    - Stable output schemas prevent downstream integration failures (APIs,
    batch jobs, dashboards).
    """
    print("[infer.run_inference] Running inference")
    # TODO: replace with logging later

    if X_infer is None or len(X_infer) == 0:
        return pd.DataFrame({"prediction": []},
                            index=getattr(X_infer, "index", None))

    preds = model.predict(X_infer)
    predictions_df = pd.DataFrame({"prediction": preds}, index=X_infer.index)

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Add post-processing such as thresholding, rounding, or
    # business rules.
    # Why: Many real deployments require translating raw outputs into
    # decision-ready formats.
    # Examples:
    # 1. Clip regression predictions to non-negative values
    # 2. Convert probabilities into risk buckets
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to
    # proceed!")
    #
    # Placeholder (Remove this after implementing your code):
    print("Warning: Student has not implemented this section yet")
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return predictions_df