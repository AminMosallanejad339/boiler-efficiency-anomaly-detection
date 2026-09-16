# Model Card: Anomaly Detection - Isolation Forest

## Model Details
- **Developed by**: Amin Mosallanejad
- **Model date**: 2026-09-16
- **Model version**: v1.0.0
- **Model type**: Unsupervised Anomaly Detection
- **Algorithm**: Isolation Forest (scikit-learn)
- **License**: For educational/research purposes

## Intended Use
- **Primary intended uses**: Detect anomalies in power plant sensor data
- **Primary intended users**: Power plant operators, maintenance engineers
- **Out-of-scope uses**: Real-time control systems, safety-critical decisions

## Metrics
| Metric | Value |
|--------|-------|
| F1 | 0.4437 |
| Precision | 0.3679 |
| Recall | 0.5591 |
| ROC-AUC | 0.9710 |
| PR-AUC | 0.3593 |

## Training Data
- **Source**: Kaggle - Power Plant Data
- **Samples**: 35,063 (Train), 7,487 (Val), 7,514 (Test)
- **Features**: 4 raw anomaly features
- **Anomaly Rate**: 1.55% (Train), 1.15% (Val), 1.69% (Test)

## Caveats and Recommendations
- F1 = 0.44 is not perfect
- 44% of anomalies are missed (FNR)
- 63% of alarms are false (1-Precision)
- Only 4 raw features available
- Data is synthetic/anonymized
- **Recommended**: Use with human oversight
