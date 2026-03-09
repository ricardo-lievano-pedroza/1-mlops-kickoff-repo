"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Union
import pandas as pd
import pandas.api.types as ptypes


class DataValidationError(ValueError):
    pass


def _sample_values(s: pd.Series, mask: pd.Series, n: int = 5) -> list:
    # Return up to n distinct offending values (excluding NaNs)
    vals = s[mask].dropna().unique()
    return list(vals[:n])


def _check_dtype(series: pd.Series, expected: Any) -> Optional[str]:
    """
    expected can be:
      - pandas dtype string ('int64', 'Int64', 'float64', 'string', 'datetime64[ns]')
      - family token: 'numeric' | 'integer' | 'float' | 'bool' | 'datetime' | 'string' | 'category'
    """
    if expected is None:
        return None

    exp = str(expected)

    # type families (more robust than exact dtype equality)
    if exp == "numeric" and not ptypes.is_numeric_dtype(series.dtype):
        return f"expected numeric, found {series.dtype}"
    if exp == "integer" and not ptypes.is_integer_dtype(series.dtype):
        return f"expected integer, found {series.dtype}"
    if exp == "float" and not ptypes.is_float_dtype(series.dtype):
        return f"expected float, found {series.dtype}"
    if exp in ("bool", "boolean") and not ptypes.is_bool_dtype(series.dtype):
        return f"expected boolean, found {series.dtype}"
    if exp == "datetime" and not ptypes.is_datetime64_any_dtype(series.dtype):
        return f"expected datetime, found {series.dtype}"
    if exp == "string" and not (ptypes.is_string_dtype(series.dtype) or series.dtype == "object"):
        return f"expected string/object, found {series.dtype}"
    if exp == "category" and not ptypes.is_categorical_dtype(series.dtype):
        return f"expected category, found {series.dtype}"

    # exact dtype string fallback
    try:
        if not ptypes.is_dtype_equal(series.dtype, exp):
            return f"expected dtype {exp}, found {series.dtype}"
    except Exception:
        # If exp isn't a recognized dtype string, skip exact check.
        return None

    return None


def validate(
    df: pd.DataFrame,
    schema: Dict[str, Dict[str, Any]],
    allow_extra_columns: bool = False,
) -> bool:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    required = set(schema.keys())
    cols = set(df.columns)

    missing = required - cols
    if missing:
        raise DataValidationError(f"Missing required columns: {sorted(missing)}")

    if not allow_extra_columns:
        extra = cols - required
        if extra:
            raise DataValidationError(f"Unexpected columns present: {sorted(extra)}")

    errors: list[str] = []

    for col, rules in schema.items():
        s = df[col]
        null_mask = s.isna()
        non_null = s[~null_mask]

        # nullability
        nullable = bool(rules.get("nullable", True))
        if not nullable and null_mask.any():
            errors.append(f"[{col}] non-nullable but has {int(null_mask.sum())} nulls")

        # dtype
        dtype_err = _check_dtype(s, rules.get("dtype"))
        if dtype_err:
            errors.append(f"[{col}] {dtype_err}")

        # ranges (only on non-null)
        if "min" in rules:
            mn = rules["min"]
            bad = non_null < mn
            if bad.any():
                sample = _sample_values(non_null, bad)
                errors.append(
                    f"[{col}] {int(bad.sum())} values < {mn}; sample={sample}"
                )

        if "max" in rules:
            mx = rules["max"]
            bad = non_null > mx
            if bad.any():
                sample = _sample_values(non_null, bad)
                errors.append(
                    f"[{col}] {int(bad.sum())} values > {mx}; sample={sample}"
                )

        # allowed values (ignore nulls if nullable=True)
        if "allowed_values" in rules:
            allowed: Iterable[Any] = rules["allowed_values"]
            bad = ~non_null.isin(list(allowed))
            if bad.any():
                sample = _sample_values(non_null, bad)
                errors.append(
                    f"[{col}] {int(bad.sum())} invalid values; sample={sample}"
                )

        # uniqueness
        if rules.get("unique", False):
            dup = non_null.duplicated()
            if dup.any():
                sample = _sample_values(non_null, dup)
                errors.append(f"[{col}] expected unique but has duplicates; sample={sample}")

        # missingness threshold
        if "max_null_frac" in rules:
            frac = float(null_mask.mean())
            if frac > float(rules["max_null_frac"]):
                errors.append(f"[{col}] null fraction {frac:.3f} exceeds {rules['max_null_frac']}")

    if errors:
        raise DataValidationError("Validation failed:\n- " + "\n- ".join(errors))

    return True