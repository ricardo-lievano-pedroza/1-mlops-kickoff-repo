"""
Module: Model Training
----------------------
Role: Bundle preprocessing and algorithms into a single Pipeline and fit on training data.
Input: pandas.DataFrame (Processed) + ColumnTransformer (Recipe).
Output: Serialized scikit-learn Pipeline in `models/`.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Training should be deterministic, repeatable, and isolated from evaluation/inference.
- Responsibility (separation of concerns): Build and fit a single Pipeline (preprocess + model) on training data only.
- Pipeline contract (inputs and outputs): Inputs are X_train, y_train, preprocessor, problem_type; output is a fitted Pipeline.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFECV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor, problem_type: str):
    """
    Inputs:
    - X_train: Training features DataFrame
    - y_train: Training target Series
    - preprocessor: Unfitted feature preprocessor (e.g., ColumnTransformer)
    - problem_type: "regression" or "classification"
    Outputs:
    - model: Fitted sklearn Pipeline (preprocess + estimator)
    Why this contract matters for reliable ML delivery:
    - A single Pipeline object prevents leakage and makes deployment simpler (one object to save/load/predict).
    """
    print(f"[train.train_model] Training model as problem_type='{problem_type}'")  # TODO: replace with logging later

    # Example: base estimator is ordinary least squares (scikit-learn)
    base_model = LinearRegression()

    # RFECV selector (wrapped as the model step)
    rfecv = RFECV(
        estimator=base_model,
        step=1,
        cv=RepeatedKFold(n_splits=5, n_repeats=3, random_state=42),
        scoring="neg_mean_squared_error",  # for regression
        n_jobs=-1,
    )

    # Build the pipeline with the preprocessor (ColumnTransformer) you already have
    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),  # from src.features.get_feature_preprocessor(...)
            ("model", rfecv),
        ]
    )


    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Swap estimator or tune hyperparameters
    # Why: Model choice depends on business constraints (interpretability, latency, accuracy)
    # Examples:
    # 1. Replace Ridge with RandomForestRegressor
    # 2. Add GridSearchCV (later session)
    #
    # Optional forcing function (leave commented)
    # raise NotImplementedError("Student: You must implement this logic to proceed!")
    #
    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    model.fit(X_train, y_train)
    return model