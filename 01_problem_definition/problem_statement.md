# Problem Statement

## 1. Problem Identification

### Main Problem
Coal-fired thermal power plants face two critical challenges:

**Challenge 1: Boiler Efficiency Degradation**
- Boiler efficiency decreases over time due to scaling, leakage, and improper combustion tuning.
- Operators typically notice efficiency loss **after** it becomes significant, not before.
- Every 1% drop in boiler efficiency significantly increases fuel costs.

**Challenge 2: Sensor Anomalies**
- Emission sensors (CO, SO2, NOx) and operational sensors occasionally record faulty values.
- According to the dataset documentation, **negative CO values** indicate sensor anomalies.
- Faulty data disrupts operator decision-making and may cause false alarms.

### Why This Problem Matters

| Aspect | Importance |
|--------|-----------|
| Economic | Every 1% boiler efficiency improvement saves millions in fuel annually |
| Environmental | Lower efficiency = higher fuel consumption = higher CO2 emissions |
| Operational | Early prediction of efficiency loss enables corrective action |
| Safety | Sensor anomalies can lead to dangerous decisions |
| Data-driven | Shifts decision-making from Reactive to Proactive |

## 2. Problem Definition

### Formal Problem Statement
> **Problem:** In a coal-fired thermal power plant, boiler efficiency and sensor data integrity are not monitored in real-time, preventing operators from taking corrective action before significant efficiency loss or sensor failure occurs.

> **Proposed Solution:** Develop a machine learning system using 58 operational parameters recorded at 10-minute intervals to (1) predict boiler efficiency and (2) detect sensor anomalies.

### Scope

**In-Scope:**
- Prediction of `Boiler Efficiency (%)` from other parameters
- Anomaly detection in `CO, SO2, NOx, Dust` values
- Feature importance analysis
- Interactive dashboard development

**Out-of-Scope:**
- Automatic boiler control system design
- Rotating equipment failure prediction
- Economic fuel optimization
- Real DCS system integration

## 3. Problem Type Determination

| Aspect | Type | Explanation |
|--------|------|-------------|
| Primary Type | Supervised Learning | Labeled target: `Boiler Efficiency` |
| Secondary Type | Regression | Predicting a continuous value |
| Tertiary Type | Anomaly Detection | Unsupervised anomaly identification |
| Data Nature | Time Series | 10-minute resolution data |
| Learning Type | Batch Learning | Data processed in batches |
| Model Output | Continuous + Binary | Efficiency + Anomaly label |

### Why This Classification Matters
- **Regression:** `Boiler Efficiency` is a continuous value between 0-100.
- **Time Series:** Data has temporal order; use **Time Series Split** for CV, not standard K-Fold.
- **Anomaly Detection:** No pre-existing anomaly labels; use Unsupervised methods.

---
*Generated on: 2026-09-16*
