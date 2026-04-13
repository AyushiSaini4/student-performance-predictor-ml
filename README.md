# 🎓 Student Performance Predictor<img width="3572" height="722" alt="08_residual_plots" src="https://github.com/user-attachments/assets/08bbdac7-650f-4d3b-bd8a-dad8f53d213e" />
<img width="1468" height="590" alt="07_model_comparison" src="https://github.com/user-attachments/assets/ac34f081-e84b-41a6-87ba-387c2e0b766a" />
<img width="1174" height="722" alt="06_feature_importance" src="https://github.com/user-attachments/assets/bb0b642c-6cbb-4d70-953c-2d5fb08e7495" />
<img width="1471" height="1531" alt="05_pairplot" src="https://github.com/user-attachments/assets/f4236817-636d-4b8a-88e9-0cd2e3382208" />
<img width="2072" height="721" alt="04_categorical_boxplots" src="https://github.com/user-attachments/assets/346d1b67-f2a9-4744-9f1d-54424419de28" />
<img width="2222" height="1820" alt="03_scatter_vs_target" src="https://github.com/user-attachments/assets/4143ccbb-8074-4a53-a634-a18c7cf0d9c4" />
<img width="1386" height="1230" alt="02_correlation_heatmap" src="https://github.com/user-attachments/assets/98071213-cc15-4aea-bda1-66a14510b2cb" />
<img width="1044" height="602" alt="01_target_distribution" src="https://github.com/user-attachments/assets/3fde0c7a-4276-4d6a-809c-d005a1e1d00f" />
<img width="1104" height="602" alt="00_missing_values" src="https://github.com/user-attachments/assets/9633d44c-5ac3-485a-b2e0-20d8421ee085" />


> 🚀 A production-ready Machine Learning project that predicts student exam performance using academic, behavioral, and lifestyle features.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![ML](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

A complete, production-structured Machine Learning project that predicts student exam scores based on study habits, lifestyle factors, and academic history.

---

## 📁 Project Structure

```
student_performance/
├── main.py                  # Entry point: train & predict modes
├── requirements.txt
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py     # Synthetic dataset generator
│   ├── utils.py             # Data loading, cleaning, evaluation helpers
│   ├── eda.py               # Exploratory Data Analysis + visualisations
│   └── model.py             # Pipeline building, training, persistence
│
├── data/
│   └── student_data.csv     # Generated after first run
│
├── models/
│   ├── linear_regression.joblib
│   ├── ridge_regression.joblib
│   ├── random_forest.joblib
│   ├── gradient_boosting.joblib
│   └── best_model_name.txt
│
└── plots/
    ├── 00_missing_values.png
    ├── 01_target_distribution.png
    ├── 02_correlation_heatmap.png
    ├── 03_scatter_vs_target.png
    ├── 04_categorical_boxplots.png
    ├── 05_pairplot.png
    ├── 06_feature_importance.png
    ├── 07_model_comparison.png
    └── 08_residual_plots.png
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train all models

```bash
python main.py train
```

This will:
- Generate a 1,000-row synthetic student dataset (with realistic missing values)
- Clean and impute missing data
- Run full EDA and save 9 plots to `plots/`
- Train 4 models (Linear Regression, Ridge, Random Forest, Gradient Boosting)
- Print evaluation metrics (MAE, MSE, RMSE, R², cross-validated R²)
- Save all trained models to `models/`
- Automatically select and flag the best model by R²

### 3. Predict interactively

```bash
python main.py predict
```

You will be prompted to enter:

| Feature | Range |
|---|---|
| Study hours per day | 1 – 10 |
| Sleep hours per night | 4 – 10 |
| Attendance percentage | 40 – 100 |
| Previous exam score | 30 – 100 |
| Assignments done | 0 – 10 |
| Stress level | 1 – 10 |
| Tutoring | 0 (No) / 1 (Yes) |
| Parental education | high_school / bachelor / master / phd |
| Gender | male / female |

Output:
```
📊  Predicted Exam Score : 82.4 / 100
🎓  Estimated Grade       : B
```

### 4. Train then predict in one command

```bash
python main.py train predict
```

---

## 🧪 Models & Evaluation

| Model | MAE | RMSE | R² |
|---|---|---|---|
| **Linear Regression** | ~3.20 | ~4.08 | **~0.918** |
| Ridge Regression | ~3.20 | ~4.08 | ~0.918 |
| Gradient Boosting | ~4.25 | ~5.31 | ~0.861 |
| Random Forest | ~4.79 | ~6.00 | ~0.823 |

> Linear Regression achieves the best R² on this dataset because the target is a near-linear combination of features (by design). Ensemble models still perform respectably and would outperform on noisier/more complex real-world data.

---

## 🔬 Features Used

| Feature | Type | Description |
|---|---|---|
| `study_hours` | Numeric | Hours of study per day |
| `sleep_hours` | Numeric | Hours of sleep per night |
| `attendance_pct` | Numeric | % of classes attended |
| `previous_score` | Numeric | Score on the previous exam |
| `assignments_done` | Numeric | Number of assignments completed (0–10) |
| `stress_level` | Numeric | Self-reported stress level (1–10) |
| `tutoring` | Binary | Whether the student has a tutor |
| `parental_education` | Categorical | Highest education level of parents |
| `gender` | Categorical | Student gender |

---

## 🗂 Module Overview

### `src/generate_data.py`
Creates a realistic synthetic dataset with controlled signal-to-noise ratio and ~5 % missing values in selected columns.

### `src/utils.py`
- `load_data()` – Load CSV with validation
- `clean_data()` – Median/mode imputation, range clipping, deduplication
- `split_features_target()` – Returns (X, y) from cleaned DataFrame
- `evaluate_model()` – Prints and returns MAE, MSE, RMSE, R², CV R²
- `validate_input()` – CLI input validation helper

### `src/eda.py`
Generates 9 publication-quality plots:
- Missing value counts
- Target distribution histogram
- Correlation heatmap
- Scatter plots (each feature vs target with trendline)
- Box plots by categorical variables
- Pairplot of key features
- Feature importances (tree model)
- Model comparison bar chart
- Residual plots

### `src/model.py`
- Defines `sklearn` Pipelines with `ColumnTransformer` preprocessing
- `train_all_models()` – Fits all 4 pipelines, evaluates, returns results
- `select_best_model()` – Picks winner by R²
- `save_model()` / `load_model()` – joblib persistence
- `predict_single()` – Single-sample inference helper

---

## 📊 EDA Highlights

- Study hours, attendance, and previous scores are the strongest predictors of exam performance
- Students with tutoring score ~4 points higher on average
- Parental education level shows a modest positive correlation with exam scores
- Stress level has a mild negative impact

---

## 🛠 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation & cleaning |
| `scikit-learn` | Preprocessing, models, evaluation |
| `matplotlib` | Plot rendering |
| `seaborn` | Statistical visualisations |
| `joblib` | Model serialisation |
| `numpy` | Numerical operations |

---

## 📝 License

MIT — free to use, modify, and distribute.
