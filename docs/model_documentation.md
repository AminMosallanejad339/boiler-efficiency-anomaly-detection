# Model Documentation

## Model Overview
- **Name**: Anomaly Detection - Isolation Forest (Tuned)
- **Version**: v1.0.0
- **Type**: Unsupervised Anomaly Detection
- **Algorithm**: Isolation Forest
- **Author**: Amin Mosallanejad
- **Date**: 2026-09-16

## Model Parameters
| Parameter | Value |
|-----------|-------|
| n_estimators | 300 |
| max_samples | 0.9 |
| contamination | 0.02 |
| max_features | 0.5 |
| bootstrap | False |
| random_state | 42 |

## Input Features
- APH_Leakage__raw
- CO_mgm3_raw
- Dust_mgm3_raw
- Reheater_desuperheating_water_flow_th_raw

## Performance Metrics (Test Set)
| Metric | Value |
|--------|-------|
| F1 | 0.4437 |
| Precision | 0.3679 |
| Recall | 0.5591 |
| Accuracy | 0.9763 |
| FPR | 0.0165 |
| FNR | 0.4409 |

## Robustness
- F1 Std across seeds: 0.0356
- Status: ROBUST

## Fairness
- F1 Std across shifts: 0.0522
- Status: FAIR

## Limitations
- F1 = 0.4437 (not perfect)
- Recall = 0.5591 (55.9% detected)
- Precision = 0.3679 (36.8% correct)
- F1 varies by month (Std = 0.0877)
- F1 varies by shift (Range = 0.1259)
- F1 Std across seeds = 0.0356
- RF with Raw_Only achieves F1=1.0 (Leakage)
- Isolation Forest is Unsupervised (no labels)
- Only 4 raw anomaly features
- 63:1 class imbalance
