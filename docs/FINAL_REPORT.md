# Final Report: Boiler Efficiency Prediction and Sensor Anomaly Detection

**Version**: v1.0.0  
**Date**: 2026-09-16  
**Author**: Amin Mosallanejad  
**Framework**: ML Model Lifecycle - 23 Main Stages  

---

## Executive Summary

This project followed the **23-Stage ML Model Lifecycle** framework to build 
a machine learning system for **Boiler Efficiency Prediction** and 
**Sensor Anomaly Detection** using 50,091 operational records from a coal-fired 
thermal power plant.

### Key Findings

- **Boiler_Eff_ is NOT predictable** (R² < 0 for Regression, F1 = 0.33 for Binned Classification)
- **Anomaly Detection IS feasible** (F1 = 0.44, ROC-AUC = 0.97 with Isolation Forest)
- **Data Leakage inflates results** (Random Forest F1 = 1.0 is artificial)
- **Isolation Forest is the realistic choice** (Unsupervised, Interpretable, Robust)

---

## Final Model

| Metric | Value |
|--------|-------|
| Model | Isolation Forest (Tuned) |
| Version | v1.0.0 |
| Type | Unsupervised Anomaly Detection |
| F1 (Test) | 0.4437 |
| ROC-AUC (Test) | 0.9710 |
| Precision (Test) | 0.3679 |
| Recall (Test) | 0.5591 |
| PR-AUC (Test) | 0.3593 |
| MCC | 0.4421 |
| Cohen's Kappa | 0.4322 |
| Leakage | No |
| Interpretability | 9/10 |
| Robustness | Robust (Std=0.036) |
| Fairness | Fair (Range=0.126) |

---

## Phase-by-Phase Summary

### Phase 01: Problem Definition & Understanding

**Status**: COMPLETE

**Key Outputs**:

- Problem: Boiler Efficiency Prediction + Anomaly Detection
- Dual Target: Boiler_Eff_ (Regression) + Anomaly_Label (Classification)
- Success Criteria: R2 >= 0.90 (Reg), F1 >= 0.85 (Clf)

### Phase 02: Data Collection & Preparation

**Status**: COMPLETE

**Key Outputs**:

- Dataset: Kaggle Power Plant Data
- Rows: 50,091
- Columns: 55 (initial) -> 56 (final)
- Missing: 0

### Phase 03: Data Transformation & Encoding

**Status**: COMPLETE

**Key Outputs**:

- Temporal features: 12
- Domain features: 12
- Discretization: 2 columns

### Phase 04: Exploratory Data Analysis

**Status**: COMPLETE

**Key Outputs**:

- Target range: 93.54 - 93.81 (Std=0.05)
- Correlation with Boiler_Eff_: Max = -0.017
- Anomaly rate: 1.51%

### Phase 05: Feature Engineering

**Status**: COMPLETE

**Key Outputs**:

- Temporal features: 12
- Domain features: 12
- Log transformed: 1
- Total features: 83

### Phase 06: Data Splitting & Pipeline

**Status**: COMPLETE

**Key Outputs**:

- Split: Time-Based + Stratified (70/15/15)
- Train: 35,063
- Val: 7,487
- Test: 7,514
- Temporal order preserved

### Phase 07: Assumption Checking

**Status**: COMPLETE

**Key Outputs**:

- Linearity: NOT satisfied
- Normality: NOT satisfied
- Independence: NOT satisfied
- Homoscedasticity: SATISFIED
- Multicollinearity: SEVERE
- Stationarity: SATISFIED

### Phase 08: Model Selection

**Status**: COMPLETE

**Key Outputs**:

- Linear: R2=0.9999 (LEAKAGE)
- Best Regression: Linear (Leakage)
- Data Leakage detected: turbine_to_boiler_eff

### Phase 08B: Model Selection (Fixed)

**Status**: COMPLETE

**Key Outputs**:

- Regression: R2 < 0 (FAILED)
- Classification: AdaBoost F1=0.988 (Leakage)
- Realistic Classification: F1=0.23 (Isolation Forest)

### Phase 08C: Binning Classification

**Status**: COMPLETE

**Key Outputs**:

- Binned: 3 classes (Low/Medium/High)
- Best F1: 0.336 (Random Forest)
- Result: Not predictable

### Phase 09: Architecture Design

**Status**: COMPLETE

**Key Outputs**:

- MLP: 128-64-32
- Params: 21,377
- Dropout: 0.3/0.3/0.2
- BatchNorm: Yes

### Phase 10: Loss / Objective Function

**Status**: COMPLETE

**Key Outputs**:

- Anomaly: Binary Crossentropy
- Binned: Categorical Crossentropy
- Focal Loss implemented
- Class Weight: 63.57

### Phase 11: Optimizer Selection

**Status**: COMPLETE

**Key Outputs**:

- Best Anomaly: RMSprop (Val Loss=0.375)
- Best Binned: Adam (Val Loss=1.121)
- Adam failed with Class Weight

### Phase 12: Regularization

**Status**: COMPLETE

**Key Outputs**:

- Best: Low_Reg (L2=1e-5, Dropout=0.2/0.2/0.1)
- Best Val Loss: 0.065
- Overfit: -0.484 (Val > Train)

### Phase 13: Hyperparameters

**Status**: COMPLETE

**Key Outputs**:

- Best LR: 0.002347
- Best Batch: 256
- Best L2: 1.98e-07
- Best Hidden: 128/32/64
- Best Val Loss: 0.0628

### Phase 14: Training

**Status**: COMPLETE

**Key Outputs**:

- MLP: F1=0.0 (FAILED)
- Class Weight = 63.57 caused Local Minima
- Epochs: 24 (best: 4)

### Phase 14B_G: Training Refinement

**Status**: COMPLETE

**Key Outputs**:

- Class Weight testing: All F1=0.0
- Raw Features: RF F1=1.0 (Leakage)
- Isolation Forest: F1=0.23 (Realistic)
- RCA: Data Leakage confirmed

### Phase 15: Validation

**Status**: COMPLETE

**Key Outputs**:

- RF: F1=1.0 (Leakage)
- Isolation Forest: F1=0.16 (Realistic)
- MLP: F1=0.0 (Failed)

### Phase 16: Hyperparameter Tuning & Refinement

**Status**: COMPLETE

**Key Outputs**:

- Isolation Forest Tuned: F1=0.31 (Val)
- Best: n_estimators=300, contamination=0.02
- Stability: ROBUST (Std=0.020)

### Phase 17: Model Evaluation

**Status**: COMPLETE

**Key Outputs**:

- Test F1: 0.4437
- Test ROC-AUC: 0.9710
- Test Precision: 0.3679
- Test Recall: 0.5591
- PR-AUC: 0.3593
- MCC: 0.4421

### Phase 18: Diagnostics & Assumption Validation

**Status**: COMPLETE

**Key Outputs**:

- Normality: NOT NORMAL
- Heteroscedasticity: HETEROSCEDASTIC
- Autocorrelation: PRESENT
- Multicollinearity: OK (VIF < 10)

### Phase 19: Hypothesis Testing

**Status**: COMPLETE

**Key Outputs**:

- H1 (Regression): REJECTED
- H5 (CO-Anomaly): CONFIRMED
- H7 (Shift): CONFIRMED
- Effect Size: LARGE (Cohen's d = 2.99)
- Power: 1.000

### Phase 20: Interpretability & Explainability

**Status**: COMPLETE

**Key Outputs**:

- Top Feature: Reheater_desuperheating (SHAP=0.542)
- SHAP: Available
- LIME: Available
- Partial Dependence: Computed

### Phase 21: Error & Fairness Analysis

**Status**: COMPLETE

**Key Outputs**:

- Test FN: 56, FP: 122
- FNR: 44.1%, FPR: 1.65%
- Fairness: FAIR (F1 Range=0.126)
- Robustness: ROBUST (Std=0.036)
- 10 Limitations identified

### Phase 22: Model Comparison & Final Selection

**Status**: COMPLETE

**Key Outputs**:

- Final Model: Isolation Forest (Tuned)
- F1: 0.4437, ROC-AUC: 0.9710
- No Data Leakage
- Within SOTA range

### Phase 23: Finalization & Deployment

**Status**: COMPLETE

**Key Outputs**:

- Model Card created
- Documentation complete
- Model Frozen (SHA-256)
- Compliance checked
- Sign-off: COMPLETE

---

## Results Tables

### Model Comparison (Test Set)

| Model                      | Type         | Feature_Set   |   N_Features |   Training_Time_s |   Inference_Time_s |       F1 |   Precision |   Recall |   ROC_AUC |   PR_AUC |      MCC |    Kappa | Leakage   |
|:---------------------------|:-------------|:--------------|-------------:|------------------:|-------------------:|---------:|------------:|---------:|----------:|---------:|---------:|---------:|:----------|
| Isolation Forest (Tuned)   | Unsupervised | Raw_Only      |            4 |            1.4682 |             0.3904 | 0.44375  |    0.367876 | 0.559055 |  0.971015 | 0.35934  | 0.442083 | 0.432173 | False     |
| Isolation Forest (Default) | Unsupervised | Raw_Only      |            4 |            0.1577 |             0.0576 | 0.170847 |    0.093472 | 0.992126 |  0.950769 | 0.213511 | 0.277739 | 0.144416 | False     |
| Random Forest (Raw_Only)   | Supervised   | Raw_Only      |            4 |            0.6344 |             0.0513 | 1        |    1        | 1        |  1        | 1        | 1        | 1        | True      |
| AdaBoost (Leaky)           | Supervised   | Main_Plus_Raw |           78 |           49.0167 |             0.6843 | 1        |    1        | 1        |  1        | 1        | 1        | 1        | True      |
| MLP                        | Supervised   | Main_Plus_Raw |           78 |          nan      |             0.9578 | 0        |    0        | 0        |  0.51985  | 0.018514 | 0        | 0        | False     |
| Dummy (All Normal)         | Baseline     | Raw_Only      |            4 |            0      |             0      | 0        |    0        | 0        |  0.5      | 0.016902 | 0        | 0        | False     |

### Error Analysis

| Dataset   |   N_Samples |   N_Anomalies |    TN |   FP |   FN |   TP |      FPR |      FNR |      TPR |      TNR |   Accuracy |   Precision |   Recall |       F1 |
|:----------|------------:|--------------:|------:|-----:|-----:|-----:|---------:|---------:|---------:|---------:|-----------:|------------:|---------:|---------:|
| Train     |       35063 |           543 | 34030 |  490 |  331 |  212 | 0.014195 | 0.609576 | 0.390424 | 0.985805 |   0.976585 |    0.301994 | 0.390424 | 0.340562 |
| Val       |        7487 |            86 |  7291 |  110 |   50 |   36 | 0.014863 | 0.581395 | 0.418605 | 0.985137 |   0.97863  |    0.246575 | 0.418605 | 0.310345 |
| Test      |        7514 |           127 |  7265 |  122 |   56 |   71 | 0.016516 | 0.440945 | 0.559055 | 0.983484 |   0.976311 |    0.367876 | 0.559055 | 0.44375  |

### Limitations

| ID     | Category     | Limitation                                   | Impact   | Mitigation                                    |
|:-------|:-------------|:---------------------------------------------|:---------|:----------------------------------------------|
| LIM-01 | Performance  | F1 = 0.4437 (not perfect)                    | HIGH     | Use ensemble or redefine Anomaly              |
| LIM-02 | Recall       | Recall = 0.5591 (55.9% detected)             | HIGH     | Lower threshold or use class weights          |
| LIM-03 | Precision    | Precision = 0.3679 (36.8% correct)           | MEDIUM   | Raise threshold or add features               |
| LIM-04 | Bias         | F1 varies by month (Std = 0.0877)            | MEDIUM   | Add temporal features                         |
| LIM-05 | Fairness     | F1 varies by shift (Range = 0.1259)          | LOW      | Add shift-specific features                   |
| LIM-06 | Robustness   | F1 Std across seeds = 0.0356                 | LOW      | Use more estimators or ensemble               |
| LIM-07 | Data Leakage | RF with Raw_Only achieves F1=1.0 (Leakage)   | HIGH     | Report both Leakage and Realistic results     |
| LIM-08 | Method       | Isolation Forest is Unsupervised (no labels) | MEDIUM   | Combine with Supervised when labels available |
| LIM-09 | Features     | Only 4 raw anomaly features                  | MEDIUM   | Add more sensor features                      |
| LIM-10 | Data         | 63:1 class imbalance                         | HIGH     | Use anomaly-specific methods                  |

---

## Key Insights

### 1. Data Leakage

- Random Forest (Raw_Only) F1 = 1.0 is **artificial**
- AdaBoost (Leaky) F1 = 1.0 is **artificial**
- Reason: Features derived from Target

### 2. Anomaly Detection is Hard

- Weak correlation (Max = 0.28)
- High overlap (KS p-value < 0.05)
- Severe imbalance (63:1)
- Random pattern (no temporal pattern)

### 3. Isolation Forest is the Best Choice

- Unsupervised (no labels needed)
- Realistic (F1 = 0.44)
- Interpretable (SHAP, Feature Importance)
- Robust (Std = 0.036)

### 4. Boiler_Eff_ is Not Predictable

- Regression: R² < 0
- Binned Classification: F1 = 0.33
- Reason: Features are not informative

---

## Recommendations

### For Deployment

1. **Use with Human Oversight** — F1 = 0.44 is not perfect
2. **Retrain Every 6 Months** — or when F1 < 0.35
3. **Monitor Weekly** — F1, Precision, Recall
4. **Add New Features** — to improve Precision
5. **Review Data License** — Kaggle License unspecified

### For Future Work

1. **Redefine Anomaly** using statistical methods (Z-Score, IQR)
2. **Add More Sensor Features** — temperature, pressure, flow
3. **Use Ensemble Methods** — combine Isolation Forest with Autoencoder
4. **Collect More Data** — to improve generalization
5. **Domain-Specific Features** — based on 17 years of experience

---

## Conclusion

This project successfully followed the **23-Stage ML Model Lifecycle** framework 
and produced a working **Anomaly Detection** system for power plant sensor data. 
While **Boiler Efficiency Prediction** was not feasible, **Anomaly Detection** 
achieved F1 = 0.44 and ROC-AUC = 0.97, which is within the State-of-the-Art range 
for unsupervised anomaly detection.

The project also identified **Data Leakage** as a critical issue and provided 
**realistic evaluation** of model performance. The final model is **documented, 
versioned, and ready for deployment** with human oversight.

---

## Project Artifacts

### Model

- `deployment/anomaly_model_v1.0.0.pkl`
- `models/isolation_forest_tuned.pkl`

### Documentation

- `docs/model_documentation.md`
- `docs/model_card.md`
- `docs/model_version.json`
- `docs/model_governance.json`
- `docs/model_compliance.json`
- `docs/model_justification.json`

### Reports

- `reports/README_FINAL.md`
- `reports/tables/*.csv`
- `reports/figures/*.png`

---

*End of Report*
