"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

from typing import Any, Dict, Iterable
import pandas as pd
import pandas.api.types as ptypes


def validate(df: pd.DataFrame, schema: Dict[str, Dict[str, Any]], allow_extra_columns: bool = False) -> bool:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    # ---- column presence ----
    required = set(schema.keys())
    cols = set(df.columns)

    missing = required - cols
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if not allow_extra_columns:
        extra = cols - required
        if extra:
            raise ValueError(f"Unexpected columns present: {sorted(extra)}")

    errors: list[str] = []

    for col, rules in schema.items():
        s = df[col]
        null_mask = s.isna()
        non_null = s[~null_mask]

        # ---- nullable ----
        nullable = bool(rules.get("nullable", True))
        if not nullable and null_mask.any():
            errors.append(f"[{col}] non-nullable but has {int(null_mask.sum())} nulls")

        # ---- dtype ----
        expected = rules.get("dtype", None)
        if expected is not None:
            exp = str(expected)

            dtype_err = None
            if exp == "numeric" and not ptypes.is_numeric_dtype(s.dtype):
                dtype_err = f"expected numeric, found {s.dtype}"
            elif exp == "integer" and not ptypes.is_integer_dtype(s.dtype):
                dtype_err = f"expected integer, found {s.dtype}"
            elif exp == "float" and not ptypes.is_float_dtype(s.dtype):
                dtype_err = f"expected float, found {s.dtype}"
            elif exp in ("bool", "boolean") and not ptypes.is_bool_dtype(s.dtype):
                dtype_err = f"expected boolean, found {s.dtype}"
            elif exp == "datetime" and not ptypes.is_datetime64_any_dtype(s.dtype):
                dtype_err = f"expected datetime, found {s.dtype}"
            elif exp == "string" and not (ptypes.is_string_dtype(s.dtype) or s.dtype == "object"):
                dtype_err = f"expected string/object, found {s.dtype}"
            elif exp == "category" and not ptypes.is_categorical_dtype(s.dtype):
                dtype_err = f"expected category, found {s.dtype}"
            else:
                # exact dtype fallback (only if exp is a valid dtype string)
                try:
                    if not ptypes.is_dtype_equal(s.dtype, exp):
                        dtype_err = f"expected dtype {exp}, found {s.dtype}"
                except Exception:
                    dtype_err = None

            if dtype_err:
                errors.append(f"[{col}] {dtype_err}")

        # ---- min/max ranges (non-null only) ----
        if "min" in rules:
            mn = rules["min"]
            bad = non_null < mn
            if bad.any():
                sample = list(non_null[bad].dropna().unique()[:5])
                errors.append(f"[{col}] {int(bad.sum())} values < {mn}; sample={sample}")

        if "max" in rules:
            mx = rules["max"]
            bad = non_null > mx
            if bad.any():
                sample = list(non_null[bad].dropna().unique()[:5])
                errors.append(f"[{col}] {int(bad.sum())} values > {mx}; sample={sample}")

        # ---- allowed values (non-null only) ----
        if "allowed_values" in rules:
            allowed: Iterable[Any] = rules["allowed_values"]
            bad = ~non_null.isin(list(allowed))
            if bad.any():
                sample = list(non_null[bad].dropna().unique()[:5])
                errors.append(f"[{col}] {int(bad.sum())} invalid values; sample={sample}")

        # ---- unique (non-null only) ----
        if bool(rules.get("unique", False)):
            dup = non_null.duplicated()
            if dup.any():
                sample = list(non_null[dup].dropna().unique()[:5])
                errors.append(f"[{col}] expected unique but has duplicates; sample={sample}")

        # ---- max null fraction ----
        if "max_null_frac" in rules:
            frac = float(null_mask.mean())
            if frac > float(rules["max_null_frac"]):
                errors.append(f"[{col}] null fraction {frac:.3f} exceeds {rules['max_null_frac']}")

    if errors:
        raise ValueError("Validation failed:\n- " + "\n- ".join(errors))

    return True