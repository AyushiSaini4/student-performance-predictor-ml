"""
model.py
--------
Defines, trains, evaluates, and persists ML pipelines for
student performance prediction.

Two models are compared:
  1. Linear Regression   – interpretable baseline
  2. Random Forest       – non-linear ensemble model

Each model is wrapped in an sklearn Pipeline that handles:
  - Column-level preprocessing (scaling + one-hot encoding)
  - Model fitting and prediction
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COL,
    evaluate_model,
    logger,
)

MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - StandardScales all numeric features
    - OneHotEncodes categorical features (drop first to avoid multicollinearity)
    """
    numeric_pipe = Pipeline([("scaler", StandardScaler())])
    categorical_pipe = Pipeline(
        [("ohe", OneHotEncoder(handle_unknown="ignore", drop="first", sparse_output=False))]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def _make_pipeline(estimator) -> Pipeline:
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ]
    )


MODELS: Dict[str, Pipeline] = {
    "Linear Regression": _make_pipeline(LinearRegression()),
    "Ridge Regression": _make_pipeline(Ridge(alpha=1.0)),
    "Random Forest": _make_pipeline(
        RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    ),
    "Gradient Boosting": _make_pipeline(
        GradientBoostingRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
    ),
}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    seed: int = 42,
) -> Tuple[Dict[str, Pipeline], Dict[str, Any], pd.DataFrame, pd.Series]:
    """
    Split data, fit every model, evaluate on hold-out test set,
    and return results + fitted pipelines.

    Parameters
    ----------
    X         : feature matrix
    y         : target vector
    test_size : fraction of data reserved for testing
    seed      : random seed for reproducibility

    Returns
    -------
    (fitted_pipelines, results_dict, X_test, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )
    logger.info(
        "Train/test split: %d / %d samples", len(X_train), len(X_test)
    )

    fitted: Dict[str, Pipeline] = {}
    results: Dict[str, Any] = {}
    predictions: Dict[str, np.ndarray] = {}

    for name, pipeline in MODELS.items():
        logger.info("Training '%s'…", name)
        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(
            pipeline, X_test, y_test,
            model_name=name,
            cv_X=X_train, cv_y=y_train,
        )
        fitted[name] = pipeline
        results[name] = metrics
        predictions[name] = pipeline.predict(X_test)

    return fitted, results, predictions, X_test, y_test


# ---------------------------------------------------------------------------
# Feature importance (tree models only)
# ---------------------------------------------------------------------------

def get_feature_importances(
    pipeline: Pipeline,
    model_name: str,
) -> Tuple[list[str], np.ndarray] | None:
    """
    Extract feature importances from a tree-based model pipeline.

    Returns (feature_names, importances) or None for linear models.
    """
    model = pipeline.named_steps["model"]
    if not hasattr(model, "feature_importances_"):
        return None

    preprocessor = pipeline.named_steps["preprocessor"]
    feature_names = list(preprocessor.get_feature_names_out())
    importances = model.feature_importances_
    return feature_names, importances


# ---------------------------------------------------------------------------
# Best-model selection
# ---------------------------------------------------------------------------

def select_best_model(
    results: Dict[str, Any],
    metric: str = "r2",
    higher_is_better: bool = True,
) -> str:
    """Return the name of the model with the best value for `metric`."""
    best = max(results, key=lambda k: results[k][metric]) if higher_is_better \
           else min(results, key=lambda k: results[k][metric])
    logger.info(
        "Best model by %s: '%s' (%.4f)", metric, best, results[best][metric]
    )
    return best


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(pipeline: Pipeline, name: str) -> Path:
    """Persist a fitted pipeline to disk using joblib."""
    safe_name = name.lower().replace(" ", "_")
    path = MODEL_DIR / f"{safe_name}.joblib"
    joblib.dump(pipeline, path)
    logger.info("Saved model → %s", path)
    return path


def load_model(name: str) -> Pipeline:
    """Load a previously saved pipeline from disk."""
    safe_name = name.lower().replace(" ", "_")
    path = MODEL_DIR / f"{safe_name}.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"No saved model found at '{path}'. Run train first."
        )
    pipeline = joblib.load(path)
    logger.info("Loaded model '%s' from %s", name, path)
    return pipeline


# ---------------------------------------------------------------------------
# Single-sample prediction helper
# ---------------------------------------------------------------------------

def predict_single(pipeline: Pipeline, sample: dict) -> float:
    """
    Predict the exam score for a single student represented as a dict.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    sample   : dict with keys matching FEATURE_COLS

    Returns
    -------
    float – predicted exam score (clipped to [0, 100])
    """
    df = pd.DataFrame([sample])
    score = pipeline.predict(df)[0]
    return float(np.clip(score, 0, 100))
