"""
utils.py
--------
Shared utility functions for data loading, cleaning, preprocessing,
and evaluation reporting.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "study_hours",
    "sleep_hours",
    "attendance_pct",
    "previous_score",
    "assignments_done",
    "stress_level",
    "parental_education",
    "gender",
    "tutoring",
]
TARGET_COL = "exam_score"

NUMERIC_FEATURES = [
    "study_hours",
    "sleep_hours",
    "attendance_pct",
    "previous_score",
    "assignments_done",
    "stress_level",
    "tutoring",
]
CATEGORICAL_FEATURES = ["parental_education", "gender"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load CSV data and return a DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns from '%s'", *df.shape, path.name)
    return df


# ---------------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Perform data cleaning:
    - Report missing values
    - Impute numeric NaNs with median
    - Impute categorical NaNs with mode
    - Clip numeric features to valid ranges
    - Remove duplicate rows

    Parameters
    ----------
    df : pd.DataFrame  – raw dataframe

    Returns
    -------
    pd.DataFrame  – cleaned dataframe
    """
    df = df.copy()

    # --- Missing values ---
    missing = df.isnull().sum()
    if missing.any():
        logger.info("Missing values detected:\n%s", missing[missing > 0].to_string())

    # Impute numeric
    for col in NUMERIC_FEATURES:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("  Imputed '%s' with median=%.2f", col, median_val)

    # Impute categorical
    for col in CATEGORICAL_FEATURES:
        if col in df.columns and df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("  Imputed '%s' with mode='%s'", col, mode_val)

    # --- Range clipping ---
    clip_ranges: Dict[str, Tuple[float, float]] = {
        "study_hours":      (0, 24),
        "sleep_hours":      (0, 24),
        "attendance_pct":   (0, 100),
        "previous_score":   (0, 100),
        "assignments_done": (0, 10),
        "stress_level":     (1, 10),
        "exam_score":       (0, 100),
    }
    for col, (lo, hi) in clip_ranges.items():
        if col in df.columns:
            before = df[col].between(lo, hi).sum()
            df[col] = df[col].clip(lo, hi)
            after = df[col].between(lo, hi).sum()
            if before != after:
                logger.warning("Clipped %d values in '%s'", after - before, col)

    # --- Duplicates ---
    dupes = df.duplicated().sum()
    if dupes:
        df = df.drop_duplicates()
        logger.info("Removed %d duplicate rows", dupes)

    logger.info("Cleaning complete. Shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Feature / target split
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) split from a cleaned DataFrame."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return X, y


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
    cv_X: pd.DataFrame | None = None,
    cv_y: pd.Series | None = None,
) -> Dict[str, Any]:
    """
    Compute regression metrics and optionally cross-validation R² scores.

    Parameters
    ----------
    pipeline   : fitted sklearn Pipeline
    X_test     : test features
    y_test     : true targets
    model_name : label for logging
    cv_X / cv_y: if provided, 5-fold CV R² is also computed

    Returns
    -------
    dict with keys: mae, mse, rmse, r2, cv_r2_mean, cv_r2_std
    """
    y_pred = pipeline.predict(X_test)

    mae  = mean_absolute_error(y_test, y_pred)
    mse  = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_test, y_pred)

    metrics: Dict[str, Any] = dict(mae=mae, mse=mse, rmse=rmse, r2=r2)

    if cv_X is not None and cv_y is not None:
        cv_scores = cross_val_score(pipeline, cv_X, cv_y, cv=5, scoring="r2")
        metrics["cv_r2_mean"] = cv_scores.mean()
        metrics["cv_r2_std"]  = cv_scores.std()

    # Pretty-print
    bar = "─" * 45
    print(f"\n{bar}")
    print(f"  {model_name} — Test-Set Metrics")
    print(bar)
    print(f"  MAE  : {mae:>8.4f}")
    print(f"  MSE  : {mse:>8.4f}")
    print(f"  RMSE : {rmse:>8.4f}")
    print(f"  R²   : {r2:>8.4f}")
    if "cv_r2_mean" in metrics:
        print(f"  CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")
    print(bar)

    return metrics


# ---------------------------------------------------------------------------
# Input validation (used by CLI)
# ---------------------------------------------------------------------------

def validate_input(value_str: str, lo: float, hi: float, label: str) -> float:
    """Parse a float from a string and assert it falls within [lo, hi]."""
    try:
        val = float(value_str)
    except ValueError:
        raise ValueError(f"'{label}' must be a number, got '{value_str}'.")
    if not (lo <= val <= hi):
        raise ValueError(f"'{label}' must be between {lo} and {hi}, got {val}.")
    return val
