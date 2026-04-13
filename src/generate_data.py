"""
generate_data.py
----------------
Generates a realistic synthetic dataset for student performance prediction.
Run this once to create the dataset before training the model.
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_student_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic student performance dataset with realistic correlations.

    Parameters
    ----------
    n_samples : int
        Number of student records to generate.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        DataFrame containing student features and exam scores.
    """
    rng = np.random.default_rng(seed)

    # --- Core features ---
    study_hours     = rng.uniform(1, 10, n_samples)           # hrs/day
    sleep_hours     = rng.uniform(4, 10, n_samples)           # hrs/night
    attendance_pct  = rng.uniform(40, 100, n_samples)         # % of classes
    previous_score  = rng.uniform(30, 100, n_samples)         # prior exam score
    assignments_done = rng.integers(0, 11, n_samples)         # out of 10
    stress_level    = rng.integers(1, 11, n_samples)          # 1 (low) – 10 (high)

    # --- Categorical features ---
    parental_education = rng.choice(
        ["high_school", "bachelor", "master", "phd"],
        n_samples,
        p=[0.30, 0.40, 0.20, 0.10],
    )
    gender = rng.choice(["male", "female"], n_samples)
    tutoring = rng.choice([0, 1], n_samples, p=[0.60, 0.40])

    # --- Target: exam score (deterministic signal + noise) ---
    score = (
        study_hours     * 3.5
        + sleep_hours   * 1.2
        + attendance_pct * 0.25
        + previous_score * 0.30
        + assignments_done * 1.5
        - stress_level  * 0.8
        + tutoring      * 4.0
        + rng.normal(0, 4, n_samples)        # irreducible noise
    )
    # Scale score to a realistic exam range [0, 100]
    score = np.clip(score, 0, 100)

    df = pd.DataFrame(
        {
            "study_hours":        study_hours,
            "sleep_hours":        sleep_hours,
            "attendance_pct":     attendance_pct,
            "previous_score":     previous_score,
            "assignments_done":   assignments_done,
            "stress_level":       stress_level,
            "parental_education": parental_education,
            "gender":             gender,
            "tutoring":           tutoring,
            "exam_score":         score,
        }
    )

    # --- Introduce realistic missing values (~5 %) ---
    for col in ["sleep_hours", "stress_level", "parental_education"]:
        mask = rng.random(n_samples) < 0.05
        df.loc[mask, col] = np.nan

    return df


if __name__ == "__main__":
    output_path = Path(__file__).parent.parent / "data" / "student_data.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = generate_student_data()
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}  ({len(df)} rows, {df.shape[1]} columns)")
