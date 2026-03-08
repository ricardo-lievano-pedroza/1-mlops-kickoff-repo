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

# TO DO Implement new logic 