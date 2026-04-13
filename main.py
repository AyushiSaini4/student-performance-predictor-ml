"""
main.py
-------
Entry point for the Student Performance Prediction project.

Usage
-----
  python main.py train         – generate data, run EDA, train & save models
  python main.py predict       – interactive CLI prediction (uses saved best model)
  python main.py train predict – do both in sequence
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure src/ is on the path when running from the project root
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging

from src.generate_data import generate_student_data
from src.utils import load_data, clean_data, split_features_target, logger
from src.eda import run_eda, plot_feature_importance, plot_model_comparison, plot_residuals
from src.model import (
    train_all_models,
    select_best_model,
    get_feature_importances,
    save_model,
    load_model,
    predict_single,
)

DATA_PATH  = Path(__file__).parent / "data" / "student_data.csv"
BEST_MODEL_FILE = Path(__file__).parent / "models" / "best_model_name.txt"


# ---------------------------------------------------------------------------
# Train pipeline
# ---------------------------------------------------------------------------

def run_training() -> None:
    """Generate data → clean → EDA → train models → save best model."""

    print("\n" + "═" * 55)
    print("  STUDENT PERFORMANCE PREDICTOR — TRAINING PIPELINE")
    print("═" * 55)

    # 1. Data generation
    if not DATA_PATH.exists():
        logger.info("No dataset found. Generating synthetic data…")
        df_raw = generate_student_data()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df_raw.to_csv(DATA_PATH, index=False)
    else:
        df_raw = load_data(DATA_PATH)

    # 2. Data cleaning
    df_clean = clean_data(df_raw)

    # 3. EDA
    run_eda(df_raw, df_clean)

    # 4. Feature / target split
    X, y = split_features_target(df_clean)

    # 5. Train & evaluate all models
    fitted_models, results, predictions, X_test, y_test = train_all_models(X, y)

    # 6. Post-training plots
    plot_model_comparison(results)
    plot_residuals(y_test, predictions)

    # 7. Feature importance (best tree model)
    for name in ["Gradient Boosting", "Random Forest"]:
        fi = get_feature_importances(fitted_models[name], name)
        if fi:
            plot_feature_importance(fi[0], fi[1], name)
            break

    # 8. Select and persist best model
    best_name = select_best_model(results, metric="r2")
    save_model(fitted_models[best_name], best_name)

    # Also save all models
    for name, pipeline in fitted_models.items():
        if name != best_name:
            save_model(pipeline, name)

    # Record best model name for predict mode
    BEST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    BEST_MODEL_FILE.write_text(best_name)

    print(f"\n✅  Training complete. Best model: '{best_name}'")
    print(f"    Plots saved to  : {Path('plots').resolve()}")
    print(f"    Models saved to : {Path('models').resolve()}\n")


# ---------------------------------------------------------------------------
# Interactive CLI prediction
# ---------------------------------------------------------------------------

def _prompt(prompt: str, lo: float, hi: float) -> float:
    from src.utils import validate_input
    while True:
        raw = input(f"  {prompt} [{lo}–{hi}]: ").strip()
        try:
            return validate_input(raw, lo, hi, prompt)
        except ValueError as e:
            print(f"  ⚠  {e}  — Please try again.")


def _prompt_choice(prompt: str, choices: list[str]) -> str:
    opts = " / ".join(choices)
    while True:
        raw = input(f"  {prompt} ({opts}): ").strip().lower()
        if raw in choices:
            return raw
        print(f"  ⚠  Invalid choice. Choose from: {opts}")


def run_prediction_cli() -> None:
    """Load the best saved model and interactively predict exam scores."""

    print("\n" + "═" * 55)
    print("  STUDENT PERFORMANCE PREDICTOR — PREDICT MODE")
    print("═" * 55)

    # Load best model
    if not BEST_MODEL_FILE.exists():
        print("❌  No trained model found. Run 'python main.py train' first.")
        sys.exit(1)

    best_name = BEST_MODEL_FILE.read_text().strip()
    pipeline  = load_model(best_name)
    print(f"\n  Using model: {best_name}\n")

    while True:
        print("─" * 40)
        print("  Enter student details (or Ctrl+C to quit):\n")

        sample = {
            "study_hours":      _prompt("Study hours per day",          1,   10),
            "sleep_hours":      _prompt("Sleep hours per night",        4,   10),
            "attendance_pct":   _prompt("Attendance percentage",       40,  100),
            "previous_score":   _prompt("Previous exam score",         30,  100),
            "assignments_done": int(_prompt("Assignments done (0-10)",  0,   10)),
            "stress_level":     int(_prompt("Stress level (1-10)",      1,   10)),
            "tutoring":         int(_prompt_choice("Tutoring?", ["0", "1"])),
            "parental_education": _prompt_choice(
                "Parental education",
                ["high_school", "bachelor", "master", "phd"],
            ),
            "gender": _prompt_choice("Gender", ["male", "female"]),
        }

        score = predict_single(pipeline, sample)

        print(f"\n  📊  Predicted Exam Score : {score:.1f} / 100")

        grade = (
            "A+" if score >= 90 else
            "A"  if score >= 80 else
            "B"  if score >= 70 else
            "C"  if score >= 60 else
            "D"  if score >= 50 else "F"
        )
        print(f"  🎓  Estimated Grade       : {grade}\n")

        again = input("  Predict for another student? (y/n): ").strip().lower()
        if again != "y":
            print("\n  Goodbye! 👋\n")
            break


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]

    if not args or args == ["--help"] or args == ["-h"]:
        print(__doc__)
        sys.exit(0)

    if "train" in args:
        run_training()

    if "predict" in args:
        run_prediction_cli()

    if not any(a in args for a in ("train", "predict")):
        print(f"Unknown command: {sys.argv[1:]}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
