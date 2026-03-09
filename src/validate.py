# """
# Module: Data Validation
# -----------------------
# Role: Check data quality (schema, types, ranges) before training.
# Input: pandas.DataFrame.
# Output: Boolean (True if valid) or raises Error.
# """

# from typing import Any, Dict, Iterable
# import pandas as pd
# import pandas.api.types as ptypes


# def validate_dataframe(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]],
# allow_extra_columns: bool = False) -> bool:
#     if not isinstance(df, pd.DataFrame):
#         raise TypeError("Input must be a pandas DataFrame.")

#     # ---- column presence ----
#     required = set(schema.keys())
#     cols = set(df.columns)

#     missing = required - cols
#     if missing:
#         raise ValueError(f"Missing required columns: {sorted(missing)}")

#     if not allow_extra_columns:
#         extra = cols - required
#         if extra:
#             raise ValueError(f"Unexpected columns present: {sorted(extra)}")

#     errors: list[str] = []

#     for col, rules in schema.items():
#         s = df[col]
#         null_mask = s.isna()
#         non_null = s[~null_mask]

#         # ---- nullable ----
#         nullable = bool(rules.get("nullable", True))
#         if not nullable and null_mask.any():
#             errors.append(f"[{col}] non-nullable but has {
#                int(null_mask.sum())} nulls")

#         # ---- dtype ----
#         expected = rules.get("dtype", None)
#         if expected is not None:
#             exp = str(expected)

#             dtype_err = None
#             if exp == "numeric" and not ptypes.is_numeric_dtype(s.dtype):
#                 dtype_err = f"expected numeric, found {s.dtype}"
#             elif exp == "integer" and not ptypes.is_integer_dtype(s.dtype):
#                 dtype_err = f"expected integer, found {s.dtype}"
#             elif exp == "float" and not ptypes.is_float_dtype(s.dtype):
#                 dtype_err = f"expected float, found {s.dtype}"
#             elif exp in ("bool", "boolean") and not ptypes.is_bool_dtype
# (s.dtype):
#                 dtype_err = f"expected boolean, found {s.dtype}"
#             elif exp == "datetime" and not ptypes.is_datetime64_any_dtype
# (s.dtype):
#                 dtype_err = f"expected datetime, found {s.dtype}"
#             elif exp == "string" and not (ptypes.is_string_dtype(s.dtype) or
# s.dtype == "object"):
#                 dtype_err = f"expected string/object, found {s.dtype}"
#             elif exp == "category" and not ptypes.is_categorical_dtype
# (s.dtype):
#                 dtype_err = f"expected category, found {s.dtype}"
#             else:
#                 # exact dtype fallback (only if exp is a valid dtype string)
#                 try:
#                     if not ptypes.is_dtype_equal(s.dtype, exp):
#                         dtype_err = f"expected dtype {exp}, found {s.dtype}"
#                 except Exception:
#                     dtype_err = None

#             if dtype_err:
#                 errors.append(f"[{col}] {dtype_err}")

#         # ---- min/max ranges (non-null only) ----
#         if "min" in rules:
#             mn = rules["min"]
#             bad = non_null < mn
#             if bad.any():
#                 sample = list(non_null[bad].dropna().unique()[:5])
#                 errors.append(f"[{col}] {int(bad.sum())} values < {mn};
# sample={sample}")

#         if "max" in rules:
#             mx = rules["max"]
#             bad = non_null > mx
#             if bad.any():
#                 sample = list(non_null[bad].dropna().unique()[:5])
#                 errors.append(f"[{col}] {int(bad.sum())} values > {mx};
# sample={sample}")

#         # ---- allowed values (non-null only) ----
#         if "allowed_values" in rules:
#             allowed: Iterable[Any] = rules["allowed_values"]
#             bad = ~non_null.isin(list(allowed))
#             if bad.any():
#                 sample = list(non_null[bad].dropna().unique()[:5])
#                 errors.append(f"[{col}] {int(bad.sum())} invalid values;
# sample={sample}")

#         # ---- unique (non-null only) ----
#         if bool(rules.get("unique", False)):
#             dup = non_null.duplicated()
#             if dup.any():
#                 sample = list(non_null[dup].dropna().unique()[:5])
#                 errors.append(f"[{col}] expected unique but has duplicates;
# sample={sample}")

#         # ---- max null fraction ----
#         if "max_null_frac" in rules:
#             frac = float(null_mask.mean())
#             if frac > float(rules["max_null_frac"]):
#                 errors.append(f"[{col}] null fraction {frac:.3f} exceeds
# {rules['max_null_frac']}")

#     if errors:
#         raise ValueError("Validation failed:\n- " + "\n- ".join(errors))

#     return True

# 6) src/validate.py
"""
Educational Goal:
- Why this module exists in an MLOps system: Validation catches data issues
early, preventing wasted training runs.
- Responsibility (separation of concerns): Fail fast on obvious problems
(empty data, missing columns).
- Pipeline contract (inputs and outputs): Input df + required column list;
output True if valid (or raise).

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from
config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: DataFrame to validate
    - required_columns: List of required column names
    Outputs:
    - bool: True if valid; raises ValueError for obvious issues
    Why this contract matters for reliable ML delivery:
    - Prevents downstream crashes and protects business timelines by catching
    schema breaks immediately.
    """
    print("[validate.validate_dataframe] Validating dataframe "
          "(fail fast checks)")
    # TODO: replace with logging later

    if df is None or len(df) == 0:
        raise ValueError("Validation failed: DataFrame is empty. Check your "
                         "data loading and filters.")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Validation failed: Missing required columns: {
            missing
            }"
            )

    # --------------------------------------------------------
    # START STUDENT CODE
    # --------------------------------------------------------
    NULL_RATE_THRESHOLD = 0.8
    RENT_UPPER_BOUND = 20000.0   # monthly rent - adjust to your market
    SQMT_UPPER_BOUND = 5000.0    # max plausible area (m^2)
    BEDROOMS_MAX = 10
    FLOORS_MAX = 100

    # Convenient short-hands
    missing_rates = df.isna().mean()

    # Null-rate checks
    high_nulls = [c for c, r in missing_rates.items() if r >
                  NULL_RATE_THRESHOLD]
    if high_nulls:
        raise ValueError(
            f"Validation failed: High null rate (> {NULL_RATE_THRESHOLD:.0%})"
            f"in columns: {high_nulls}")
    # 3) Numeric dtype & range checks
    numeric_checks = {
        "rent": {"min": 0.0, "max": RENT_UPPER_BOUND},
        "sq.mt": {"min": 0.01, "max": SQMT_UPPER_BOUND},
        "floors": {"min": 0, "max": FLOORS_MAX},
        "bedrooms": {"min": 0, "max": BEDROOMS_MAX},
    }

    for col, bounds in numeric_checks.items():
        if col in df.columns:
            # dtype/coercion check
            try:
                col_series = pd.to_numeric(df[col].dropna())
            except Exception:
                raise ValueError(f"Validation failed: Column '{
                    col}' must be numeric or coercible to numeric."
                    )
            # range checks (only applied to non-empty series)
            if len(col_series) > 0:
                if (col_series < bounds["min"]).any():
                    raise ValueError(
                        f"Validation failed: Column '{col}' "
                        f"has values < {bounds['min']} (invalid).")
                if (col_series > bounds["max"]).any():
                    raise ValueError(
                        f"Validation failed: Column '{col}' has "
                        f"values > {bounds['max']} (suspicious).")

    # 4) Binary column checks
    binary_cols = ["outer", "duplex", "semidetached", "cottage", "elevator",
                   "penthouse"]
    for col in binary_cols:
        if col in df.columns:
            # allowed set: {0,1,True,False,"0","1","Y","N"} - we convert
            # strings where sensible
            unique_vals = set(df[col].dropna().unique())
            # normalize some common representations to check membership
            normalized = set()
            for v in unique_vals:
                if isinstance(v, str):
                    vv = v.strip().lower()
                    if vv in {"y", "yes", "true", "1"}:
                        normalized.add(1)
                    elif vv in {"n", "no", "false", "0"}:
                        normalized.add(0)
                    else:
                        normalized.add(vv)
                elif isinstance(v, (int, float, bool)):
                    normalized.add(int(v))
                else:
                    normalized.add(v)
            # Accept if all normalized values are subset of {0,1}
            if not normalized.issubset({0, 1}):
                raise ValueError(f"Validation failed: Binary-like column "
                                 f"'{col}' contains unexpected values:"
                                 f"{unique_vals}")

        # 5) Duplicate rows check (optional)
    if df.duplicated().any():
        print("Warning: Duplicate rows detected in dataframe. "
              "Consider deduping in clean step.")
        # TODO: replace with logging later

    # --------------------------------------------------------
    # END STUDENT CODE
    # --------------------------------------------------------

    return True