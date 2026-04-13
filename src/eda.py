"""
eda.py
------
Exploratory Data Analysis module.
Generates and saves visualisation plots to the plots/ directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from utils import NUMERIC_FEATURES, TARGET_COL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Style defaults
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.05)
PLOT_DIR = Path(__file__).parent.parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

FIG_DPI  = 150
SAVEKW   = dict(dpi=FIG_DPI, bbox_inches="tight")


def _save(fig: plt.Figure, name: str) -> None:
    path = PLOT_DIR / name
    fig.savefig(path, **SAVEKW)
    logger.info("Saved plot → %s", path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------

def plot_target_distribution(df: pd.DataFrame) -> None:
    """Histogram + KDE of the exam score distribution."""
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(df[TARGET_COL], kde=True, bins=30, color="steelblue", ax=ax)
    ax.axvline(df[TARGET_COL].mean(), color="firebrick", ls="--", lw=1.5, label=f"Mean = {df[TARGET_COL].mean():.1f}")
    ax.set_xlabel("Exam Score")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Exam Scores")
    ax.legend()
    _save(fig, "01_target_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """Heatmap of Pearson correlations among numeric features + target."""
    numeric_df = df[NUMERIC_FEATURES + [TARGET_COL]]
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, linewidths=0.5, ax=ax,
    )
    ax.set_title("Feature Correlation Heatmap")
    _save(fig, "02_correlation_heatmap.png")


def plot_scatter_vs_target(df: pd.DataFrame) -> None:
    """Scatter plots of each numeric feature vs exam score."""
    features = [f for f in NUMERIC_FEATURES if f != TARGET_COL]
    n = len(features)
    cols = 3
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    axes = axes.flatten()

    for i, feat in enumerate(features):
        ax = axes[i]
        ax.scatter(df[feat], df[TARGET_COL], alpha=0.35, s=15, color="steelblue")
        try:
            x_vals = df[feat].astype(float).values
            y_vals = df[TARGET_COL].astype(float).values
            m, b = np.polyfit(x_vals, y_vals, 1)
            xs = np.linspace(x_vals.min(), x_vals.max(), 200)
            ax.plot(xs, m * xs + b, color="firebrick", lw=1.5)
        except np.linalg.LinAlgError:
            pass  # skip trendline if SVD fails
        ax.set_xlabel(feat.replace("_", " ").title())
        ax.set_ylabel("Exam Score")
        ax.set_title(f"{feat.replace('_', ' ').title()} vs Score")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Numeric Features vs Exam Score", fontsize=14, y=1.01)
    fig.tight_layout()
    _save(fig, "03_scatter_vs_target.png")


def plot_categorical_analysis(df: pd.DataFrame) -> None:
    """Box plots of exam score by categorical variables."""
    cat_cols = ["parental_education", "gender", "tutoring"]
    fig, axes = plt.subplots(1, len(cat_cols), figsize=(14, 5))

    for ax, col in zip(axes, cat_cols):
        order = df.groupby(col)[TARGET_COL].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=col, y=TARGET_COL, order=order, ax=ax, palette="pastel")
        ax.set_xlabel(col.replace("_", " ").title())
        ax.set_ylabel("Exam Score")
        ax.set_title(f"Score by {col.replace('_', ' ').title()}")
        ax.tick_params(axis="x", rotation=20)

    fig.tight_layout()
    _save(fig, "04_categorical_boxplots.png")


def plot_pairplot(df: pd.DataFrame) -> None:
    """Pairplot for the most important numeric features."""
    key_features = ["study_hours", "attendance_pct", "previous_score", TARGET_COL]
    pair_df = df[key_features].copy()
    pair_df.columns = [c.replace("_", "\n") for c in pair_df.columns]

    g = sns.pairplot(pair_df, diag_kind="kde", plot_kws={"alpha": 0.3, "s": 15})
    g.figure.suptitle("Pairplot — Key Features", y=1.02, fontsize=13)
    _save(g.figure, "05_pairplot.png")


def plot_missing_values(df_raw: pd.DataFrame) -> None:
    """Bar chart showing missing value counts before cleaning."""
    missing = df_raw.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        logger.info("No missing values to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 4))
    missing.sort_values().plot.barh(ax=ax, color="salmon", edgecolor="white")
    ax.set_xlabel("Missing Count")
    ax.set_title("Missing Values per Column")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    _save(fig, "00_missing_values.png")


def plot_feature_importance(feature_names: list[str], importances: np.ndarray, model_name: str) -> None:
    """Horizontal bar chart of feature importances (tree models)."""
    idx = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in idx],
        importances[idx],
        color="steelblue",
        edgecolor="white",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importances — {model_name}")
    fig.tight_layout()
    _save(fig, "06_feature_importance.png")


def plot_model_comparison(results: dict) -> None:
    """Bar chart comparing MAE and R² across models."""
    models = list(results.keys())
    mae_vals = [results[m]["mae"] for m in models]
    r2_vals  = [results[m]["r2"]  for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(x, mae_vals, width, color=["steelblue", "darkorange", "seagreen"][:len(models)])
    ax1.set_xticks(x); ax1.set_xticklabels(models, rotation=15)
    ax1.set_ylabel("MAE"); ax1.set_title("Mean Absolute Error (lower ↓ is better)")

    ax2.bar(x, r2_vals, width, color=["steelblue", "darkorange", "seagreen"][:len(models)])
    ax2.set_xticks(x); ax2.set_xticklabels(models, rotation=15)
    ax2.set_ylabel("R² Score"); ax2.set_title("R² Score (higher ↑ is better)")
    ax2.set_ylim(0, 1.05)

    fig.suptitle("Model Comparison", fontsize=13)
    fig.tight_layout()
    _save(fig, "07_model_comparison.png")


def plot_residuals(y_test: pd.Series, predictions: dict) -> None:
    """Residual plots (actual vs predicted) for each model."""
    n = len(predictions)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, predictions.items()):
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.4, s=18, color="steelblue")
        ax.axhline(0, color="firebrick", ls="--", lw=1.5)
        ax.set_xlabel("Predicted Score")
        ax.set_ylabel("Residuals")
        ax.set_title(f"Residuals — {name}")

    fig.tight_layout()
    _save(fig, "08_residual_plots.png")


# ---------------------------------------------------------------------------
# Run all EDA
# ---------------------------------------------------------------------------

def run_eda(df_raw: pd.DataFrame, df_clean: pd.DataFrame) -> None:
    """Execute the full EDA pipeline."""
    logger.info("Running EDA…")
    plot_missing_values(df_raw)
    plot_target_distribution(df_clean)
    plot_correlation_heatmap(df_clean)
    plot_scatter_vs_target(df_clean)
    plot_categorical_analysis(df_clean)
    plot_pairplot(df_clean)
    logger.info("EDA complete. Plots saved to '%s'", PLOT_DIR)
