"""
Educational Goal:
- Why this module exists in an MLOps system:
  Data loading is one of the highest-risk steps (wrong file, wrong schema,
  wrong environment).
  A dedicated loader gives you a single, testable place to control and audit
  data access.
- Responsibility (separation of concerns):
  Load raw data from disk. If not present, create a deterministic dummy
  dataset for scaffolding.
- Pipeline contract (inputs and outputs):
  Input: raw_data_path (Path). Output: raw DataFrame with expected columns.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

import os
from pathlib import Path

import pandas as pd

from src.utils import load_csv, save_csv


def load_raw_data(raw_data_path: Path) -> pd.DataFrame:
    """
    Inputs:
    - raw_data_path: Path to the raw CSV file.
    Outputs:
    - df_raw: Raw DataFrame loaded from disk.
    Why this contract matters for reliable ML delivery:
    - “Same inputs, same outputs” is the foundation of reproducible ML
      pipelines.
    """
    # TODO: replace with logging later
    print(f"[load_data.load_raw_data] Loading raw data from: {raw_data_path}")

    is_example_config = (
        os.getenv("IS_EXAMPLE_CONFIG", "true").lower() == "true"
    )

    if not raw_data_path.exists():
        if is_example_config:
            raw_data_path.parent.mkdir(parents=True, exist_ok=True)

            dummy = pd.DataFrame(
                {
                    "num_feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                    "cat_feature": ["A", "B", "A", "C", "B", "C"],
                    "target": [0.0, 1.2, 1.9, 3.1, 3.8, 5.2],
                }
            )

            # TODO: replace with logging later
            print(
                "\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                "LOUD WARNING (SCAFFOLDING ONLY):\n"
                f"- Raw data file was not found at: {raw_data_path}\n"
                "A tiny deterministic DUMMY dataset was created with:\n"
                'Columns:  ["num_feature", "cat_feature", "target"]\n'
                "ONLY to ensure the pipeline runs end-to-end immediately.\n"
                "MUST replace this dataset + update SETTINGS in src/main.py.\n"
                "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n"
                )

            save_csv(dummy, raw_data_path)
        else:
            raise FileNotFoundError(
                "[load_data.load_raw_data] Raw data file not found.\n"
                f"Expected at: {raw_data_path}\n"
                "Fix:\n"
                "1) Put your dataset at that path, OR\n"
                "2) Update SETTINGS['raw_data_path'] (and later config.yml) to"
                "the correct file.\n"
            )

    df_raw = load_csv(raw_data_path)

    if df_raw is None or df_raw.empty:
        raise ValueError(
            "[load_data.load_raw_data] Loaded dataframe is empty.\n"
            f"File path: {raw_data_path}\n"
            "Fix:\n"
            "Check the file contents (maybe header-only or wrong delimiter)\n"
            "Confirm your export/query actually produced rows.\n"
        )

    # TODO: replace with logging later
    print(
        "[load_data.load_raw_data] Loaded shape=%s, columns=%s"
        % (df_raw.shape, list(df_raw.columns))
    )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    # TODO_STUDENT: Paste your notebook logic here to replace or extend the
    # baseline
    # Why: Explain why this step varies per dataset or business context
    # Examples:
    # 1. ...
    # 2. ...
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

    return df_raw