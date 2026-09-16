# Assumptions & Constraints

## 1. Constraints

| Constraint | Description | Impact |
|-----------|-------------|--------|
| Data Volume | 50,000+ records (one month) | Sufficient for ML, insufficient for seasonal analysis |
| Time Range | January 2022 (one month) | No seasonal coverage |
| Data Source | Anonymous plant | No precise citation possible |
| License | Unspecified on Kaggle | Commercial publication restricted |
| Missing Values | Unknown | Requires handling in preparation phase |
| Sensor Anomalies | Negative CO values | Requires cleaning |
| Computational Resources | Limited (personal laptop) | Need lightweight models |
| Time | 2-3 weeks | Requires careful planning |
| Domain Knowledge | From your experience | Must document for non-industrial audience |

## 2. Data Assumptions

| # | Assumption | Risk if Violated |
|---|-----------|------------------|
| A1 | Data recorded with acceptable quality | Extensive cleaning needed |
| A2 | Missing values are MCAR | Bias in results |
| A3 | Negative CO values are sensor anomalies, not reality | Removing valid data |
| A4 | 10-minute resolution sufficient for analysis | Missing fast fluctuations |
| A5 | One month represents full year | Poor generalizability |

## 3. Modeling Assumptions

| # | Assumption | Risk if Violated |
|---|-----------|------------------|
| A6 | Relationship between parameters is non-linear | Need non-linear models |
| A7 | Data is not stationary | Need Differencing |
| A8 | Anomalies are rare (< 5% of data) | Class imbalance |
| A9 | Features are not independent | Multicollinearity |

## 4. Domain Assumptions

| # | Assumption | Source |
|---|-----------|--------|
| A10 | Boiler efficiency is between 70% and 95% | Domain knowledge |
| A11 | Flue gas temperature is between 120 and 180 C | Domain knowledge |
| A12 | Boiler oxygen is between 2% and 6% | Domain knowledge |

---
*Generated on: 2026-09-16*
