# Boiler Efficiency Prediction and Sensor Anomaly Detection

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13%2B-red.svg)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()

An end-to-end Machine Learning project following the **23-Stage ML Model Lifecycle** framework to predict **Boiler Efficiency** and detect **Sensor Anomalies** in a coal-fired thermal power plant.

---

## Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Model Performance](#model-performance)
- [23-Stage ML Model Lifecycle](#23-stage-ml-model-lifecycle)
- [Key Figures](#key-figures)
- [Limitations](#limitations)
- [Recommendations](#recommendations)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Overview

This project uses **50,091 operational records** with **58 parameters** from a coal-fired thermal power plant to:

1. **Predict Boiler Efficiency** (`Boiler_Eff_`) - Regression
2. **Detect Sensor Anomalies** (`Anomaly_Label`) - Classification

### Dataset

| Property        | Value                                                        |
| --------------- | ------------------------------------------------------------ |
| Source          | [Kaggle - Power Plant Data](https://www.kaggle.com/datasets/pavanjitsubash/power-plant-data-steam-turbine-and-boiler-metrics) |
| Rows            | 50,091                                                       |
| Initial Columns | 55                                                           |
| Final Features  | 83                                                           |
| Time Resolution | 10 minutes                                                   |
| Missing Values  | 0                                                            |
| Anomaly Rate    | 1.51%                                                        |

---

## Key Results

| Target                         | Model                    | Result                    | Status          |
| ------------------------------ | ------------------------ | ------------------------- | --------------- |
| **Anomaly Detection**          | Isolation Forest (Tuned) | F1 = 0.44, ROC-AUC = 0.97 | Feasible        |
| **Boiler Efficiency**          | Regression               | R2 < 0                    | Not Predictable |
| **Boiler Efficiency (Binned)** | Random Forest            | F1 = 0.33                 | Not Predictable |

---

## Key Findings

### 1. Data Leakage Detected

Random Forest with Raw_Only features achieved **F1 = 1.0**, but this was **Data Leakage** (features derived from the target). After removing leakage, the realistic F1 was **0.44**.

**Leakage Features Detected:**

- `turbine_to_boiler_eff` (derived from `Boiler_Eff_`)
- `aph_effect_leak_ratio` (derived from `APH_Leakage_`)

### 2. Anomaly Detection is Feasible

With **ROC-AUC = 0.97**, the Isolation Forest model demonstrates excellent discriminative ability for anomaly detection.

### 3. Boiler Efficiency is Not Predictable

- **Regression**: R2 < 0
- **Binned Classification**: F1 = 0.33
- **Reason**: Weak correlation (Max = 0.28) and severe imbalance (63:1)

### 4. Isolation Forest is the Best Choice

- **Unsupervised**: No labels needed in production
- **Realistic**: F1 = 0.44 (within SOTA range)
- **Interpretable**: SHAP, Permutation Importance
- **Robust**: F1 Std = 0.036
- **Fair**: F1 Range = 0.126

---

## Project Structure

boiler-efficiency-anomaly-detection/
│
├── 01_problem_definition/          # Phase 01: Problem definition
│   ├── problem_statement.md
│   ├── business_understanding.md
│   ├── success_criteria.md
│   ├── hypothesis.md
│   └── assumptions_constraints.md
│
├── 02_data_understanding/          # Phase 02: Data understanding
├── 03_data_preparation/            # Phase 03: Data preparation
├── 04_modeling/                    # Phase 04-23: Modeling
├── 05_evaluation/                  # Phase 17-22: Evaluation
├── 06_deployment/                  # Phase 23: Deployment
│
├── data/
│   ├── raw/                        # Original dataset
│   │   └── industrial_dataset.csv
│   ├── processed/                  # Cleaned data
│   │   ├── industrial_dataset_phase02_final.csv
│   │   ├── industrial_dataset_phase03_final.csv
│   │   └── industrial_dataset_phase05_final.csv
│   └── splits/                     # Train/Val/Test splits
│       ├── train.csv
│       ├── validation.csv
│       ├── test.csv
│       └── feature_columns.csv
│
├── models/                         # Trained models
│   ├── isolation_forest_tuned.pkl
│   ├── isolation_forest_final.pkl
│   ├── anomaly_mlp_final_v3.keras
│   ├── random_forest_anomaly.pkl
│   └── best_hyperparameters.json
│
├── reports/
│   ├── figures/                    # 50+ figures
│   ├── resume_figures/             # 8 professional figures
│   │   ├── FIGURE_01_overview.png
│   │   ├── FIGURE_02_model_comparison.png
│   │   ├── FIGURE_03_final_performance.png
│   │   ├── FIGURE_04_cm_roc.png
│   │   ├── FIGURE_05_feature_importance.png
│   │   ├── FIGURE_06_fairness_robustness.png
│   │   ├── FIGURE_07_learning_curve.png
│   │   └── FIGURE_08_data_leakage.png
│   ├── tables/                     # 100+ CSV tables
│   ├── FINAL_REPORT.txt
│   └── phase*_log.txt              # Phase logs
│
├── deployment/                     # Deployment artifacts
│   ├── anomaly_model_v1.0.0.pkl
│   ├── model_hash.json
│   ├── model_approval.json
│   └── model_signoff.json
│
├── docs/                           # Documentation
│   ├── FINAL_REPORT.md
│   ├── EXECUTIVE_SUMMARY.json
│   ├── model_card.md
│   ├── model_documentation.md
│   ├── model_version.json
│   ├── model_governance.json
│   ├── model_compliance.json
│   └── model_justification.json
│
├── src/                            # Source code (37 files)
│   ├── phase01_problem_definition.py
│   ├── phase02_data_collection_preparation.py
│   ├── phase03_transformation_encoding.py
│   ├── phase04_eda.py
│   ├── phase05_feature_engineering.py
│   ├── phase06_data_splitting_pipeline.py
│   ├── phase07_assumption_checking.py
│   ├── phase08_model_selection.py
│   ├── phase09_architecture_design.py
│   ├── phase10_loss_objective.py
│   ├── phase11_optimizer_selection.py
│   ├── phase12_regularization.py
│   ├── phase13_hyperparameters.py
│   ├── phase14_training.py
│   ├── phase15_validation.py
│   ├── phase16_hyperparameter_refinement.py
│   ├── phase17_model_evaluation.py
│   ├── phase18_diagnostics.py
│   ├── phase19_hypothesis_testing.py
│   ├── phase20_interpretability.py
│   ├── phase21_error_fairness.py
│   ├── phase22_model_comparison.py
│   ├── phase23_finalization_deployment.py
│   ├── phase24_final_report.py
│   ├── phase25_final_figures.py
│   └── phase26_github_preparation.py
│
├── app/                            # Streamlit app
├── notebooks/                      # Jupyter notebooks
│
├── README.md                       # This file
├── LICENSE                         # MIT License
├── requirements.txt                # Dependencies
├── .gitignore                      # Git ignore rules
├── CONTRIBUTING.md                 # Contribution guidelines
└── CHANGELOG.md                    # Version history

---

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/boiler-efficiency-anomaly-detection.git
cd boiler-efficiency-anomaly-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Data

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/pavanjitsubash/power-plant-data-steam-turbine-and-boiler-metrics) and place it in:

```
data/raw/industrial_dataset.csv
```

### 4. Run the Pipeline

```bash
python src/phase01_problem_definition.py
python src/phase02_data_collection_preparation.py
python src/phase03_transformation_encoding.py
python src/phase04_eda.py
python src/phase05_feature_engineering.py
python src/phase06_data_splitting_pipeline.py
python src/phase07_assumption_checking.py
python src/phase08_model_selection.py
python src/phase09_architecture_design.py
python src/phase10_loss_objective.py
python src/phase11_optimizer_selection.py
python src/phase12_regularization.py
python src/phase13_hyperparameters.py
python src/phase14_training.py
python src/phase15_validation.py
python src/phase16_hyperparameter_refinement.py
python src/phase17_model_evaluation.py
python src/phase18_diagnostics.py
python src/phase19_hypothesis_testing.py
python src/phase20_interpretability.py
python src/phase21_error_fairness.py
python src/phase22_model_comparison.py
python src/phase23_finalization_deployment.py
python src/phase24_final_report.py
python src/phase25_final_figures.py
```

### 5. Load the Final Model

```python
import joblib
import numpy as np

# Load the model
model = joblib.load("models/isolation_forest_tuned.pkl")

# Prepare input (4 raw anomaly features)
# Features: APH_Leakage, CO, Dust, Reheater
X = np.array([[1.5, 250, 14, 2.5]])

# Predict
prediction = model.predict(X)
# 1 = Normal, -1 = Anomaly

if prediction[0] == -1:
    print("ANOMALY DETECTED")
else:
    print("Normal operation")
```

---

## Model Performance

### Final Model: Isolation Forest (Tuned)

| Metric | Value |
|--------|-------|
| **F1 Score** | 0.4437 |
| **ROC-AUC** | 0.9710 |
| **Precision** | 0.3679 |
| **Recall** | 0.5591 |
| **PR-AUC** | 0.3593 |
| **MCC** | 0.4421 |
| **Cohen's Kappa** | 0.4322 |
| **F1 Std (Seeds)** | 0.0356 |
| **F1 Range (Shifts)** | 0.1259 |

### Model Parameters

```python
{
    "n_estimators": 300,
    "max_samples": 0.9,
    "contamination": 0.02,
    "max_features": 0.5,
    "bootstrap": False,
    "random_state": 42
}
```

### Model Comparison

| Model | F1 | ROC-AUC | Leakage |
|-------|-----|---------|---------|
| **Isolation Forest (Tuned)** | **0.4437** | **0.9710** | No |
| Isolation Forest (Default) | 0.1708 | 0.9508 | No |
| Random Forest (Raw_Only) | 1.0000 | 1.0000 | **Yes** |
| AdaBoost (Leaky) | 1.0000 | 1.0000 | **Yes** |
| MLP | 0.0000 | 0.5199 | No |
| Dummy (All Normal) | 0.0000 | 0.5000 | No |

### Confusion Matrix (Test Set)

| | Predicted Normal | Predicted Anomaly |
|---|---|---|
| **Actual Normal** | TN = 7,265 | FP = 122 |
| **Actual Anomaly** | FN = 56 | TP = 71 |

---

## 23-Stage ML Model Lifecycle

| Phase | Title | Status |
|-------|-------|--------|
| 01 | Problem Definition | Complete |
| 02 | Data Collection & Preparation | Complete |
| 03 | Data Transformation & Encoding | Complete |
| 04 | Exploratory Data Analysis | Complete |
| 05 | Feature Engineering | Complete |
| 06 | Data Splitting & Pipeline | Complete |
| 07 | Assumption Checking | Complete |
| 08 | Model Selection | Complete |
| 09 | Architecture Design | Complete |
| 10 | Loss / Objective Function | Complete |
| 11 | Optimizer Selection | Complete |
| 12 | Regularization | Complete |
| 13 | Hyperparameters | Complete |
| 14 | Training | Complete |
| 15 | Validation | Complete |
| 16 | Hyperparameter Tuning & Refinement | Complete |
| 17 | Model Evaluation | Complete |
| 18 | Diagnostics & Assumption Validation | Complete |
| 19 | Hypothesis Testing | Complete |
| 20 | Interpretability & Explainability | Complete |
| 21 | Error & Fairness Analysis | Complete |
| 22 | Model Comparison & Final Selection | Complete |
| 23 | Finalization & Deployment | Complete |

---

## Key Figures

### Project Overview

![Project Overview](FIGURE_01_overview.png)

### Model Comparison

![Model Comparison](FIGURE_02_model_comparison.png)

### Final Model Performance

![Final Performance](FIGURE_03_final_performance.png)

### Confusion Matrix and ROC

![Confusion Matrix](FIGURE_04_cm_roc.png)

### Feature Importance

![Feature Importance](FIGURE_05_feature_importance.png)

### Fairness and Robustness

![Fairness](FIGURE_06_fairness_robustness.png)

### Learning Curve

![Learning Curve](FIGURE_07_learning_curve.png)

### Data Leakage Analysis

![Data Leakage](FIGURE_08_data_leakage.png)

---

## Limitations

| ID | Category | Limitation | Impact |
|----|----------|------------|--------|
| LIM-01 | Performance | F1 = 0.44 (not perfect) | HIGH |
| LIM-02 | Recall | 44% of anomalies missed | HIGH |
| LIM-03 | Precision | 63% false alarms | MEDIUM |
| LIM-04 | Bias | F1 varies by month (Std = 0.088) | MEDIUM |
| LIM-05 | Fairness | F1 varies by shift (Range = 0.126) | LOW |
| LIM-06 | Robustness | F1 Std = 0.036 | LOW |
| LIM-07 | Data Leakage | RF F1 = 1.0 is artificial | HIGH |
| LIM-08 | Method | Unsupervised (no labels) | MEDIUM |
| LIM-09 | Features | Only 4 raw features | MEDIUM |
| LIM-10 | Data | 63:1 class imbalance | HIGH |

---

## Recommendations

### For Deployment

1. **Use with Human Oversight** - F1 = 0.44 is not perfect
2. **Retrain Every 6 Months** - or when F1 < 0.35
3. **Monitor Weekly** - F1, Precision, Recall
4. **Add New Features** - to improve Precision
5. **Review Data License** - Kaggle License unspecified

### For Future Work

1. **Redefine Anomaly** using statistical methods (Z-Score, IQR)
2. **Add More Sensor Features** - temperature, pressure, flow
3. **Use Ensemble Methods** - combine Isolation Forest with Autoencoder
4. **Collect More Data** - to improve generalization
5. **Domain-Specific Features** - based on 17 years of experience

---

## Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.9+ |
| **ML** | scikit-learn, TensorFlow/Keras |
| **Interpretability** | SHAP, LIME |
| **Data** | pandas, numpy |
| **Visualization** | matplotlib, seaborn, plotly |
| **Statistics** | scipy, statsmodels |

---

## Author

**Amin Mosallanejad**

- Senior Utility Engineer (17 years of experience)
- M.Sc. in Data Science and Business Analytics
- M.Sc. in Chemical Engineering

**Contact:**

- Email: mosallanejadamin1400@gmail.com
- LinkedIn: [amin-mosallanejad](https://linkedin.com/in/amin-mosallanejad)
- GitHub: [AminMosallanejad](https://github.com/AminMosallanejad)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Note**: The dataset is from Kaggle and its license is unspecified. This project is for **educational/research purposes only**.

---

## Acknowledgments

- [Kaggle](https://www.kaggle.com/) for the dataset
- [scikit-learn](https://scikit-learn.org/) for Isolation Forest
- [SHAP](https://github.com/slundberg/shap) for interpretability
- [LIME](https://github.com/marcotcr/lime) for local interpretability

---

**Status**: Complete - All 23 phases done

**Last Updated**: 2026-09-16

