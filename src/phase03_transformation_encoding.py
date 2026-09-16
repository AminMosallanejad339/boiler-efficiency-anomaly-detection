"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 03: Data Transformation & Encoding (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler,
    PowerTransformer, QuantileTransformer, KBinsDiscretizer
)

# ============================================================
# NOTE: imbalanced-learn (SMOTE) will be used in Phase 05
#       For now, we only report class imbalance.
# ============================================================

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
PROCESSED = BASE_DIR / "data" / "processed" / "industrial_dataset_phase02_final.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری
# ============================================================
log("=" * 70)
log("PHASE 03: DATA TRANSFORMATION & ENCODING")
log("=" * 70)

df = pd.read_csv(PROCESSED)
log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ============================================================
# 3.1 DATA TRANSFORMATION
# ============================================================
def stage_3_1_data_transformation(df):
    log("\n" + "=" * 70)
    log("STAGE 3.1: DATA TRANSFORMATION")
    log("=" * 70)

    df_transformed = df.copy()
    numeric_cols = df_transformed.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Anomaly_Label"]

    # بررسی Skewness
    skewness = df_transformed[numeric_cols].skew().sort_values(ascending=False)
    highly_skewed = skewness[abs(skewness) > 1].index.tolist()
    log(f"  Highly skewed columns (|skew| > 1): {len(highly_skewed)}")

    if len(highly_skewed) > 0:
        log(f"  Top 5 skewed columns:")
        for col in highly_skewed[:5]:
            log(f"    {col}: skew = {skewness[col]:.4f}")

    # اعمال Yeo-Johnson
    transform_report = []
    for col in numeric_cols:
        sk = skewness[col]
        if abs(sk) > 1:
            try:
                pt = PowerTransformer(method="yeo-johnson", standardize=False)
                df_transformed[f"{col}_yj"] = pt.fit_transform(df_transformed[[col]])
                transform_report.append({
                    "Column": col,
                    "Original_Skew": round(sk, 4),
                    "Transformed_Skew": round(df_transformed[f"{col}_yj"].skew(), 4),
                    "Method": "Yeo-Johnson"
                })
            except Exception as e:
                log(f"  [WARNING] Could not transform {col}: {e}")
                transform_report.append({
                    "Column": col,
                    "Original_Skew": round(sk, 4),
                    "Transformed_Skew": round(sk, 4),
                    "Method": f"Failed: {e}"
                })
        else:
            transform_report.append({
                "Column": col,
                "Original_Skew": round(sk, 4),
                "Transformed_Skew": round(sk, 4),
                "Method": "None (already normal)"
            })

    report_df = pd.DataFrame(transform_report)
    report_df.to_csv(TABLES_DIR / "03_01_transformation_report.csv", index=False)
    log(f"  [OK] Saved: 03_01_transformation_report.csv")
    log(f"  Transformed columns: {(report_df['Method'] == 'Yeo-Johnson').sum()}")

    return df_transformed

# ============================================================
# 3.2 DATA NORMALIZATION
# ============================================================
def stage_3_2_data_normalization(df):
    log("\n" + "=" * 70)
    log("STAGE 3.2: DATA NORMALIZATION")
    log("=" * 70)

    df_norm = df.copy()
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Anomaly_Label"]

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(df_norm[numeric_cols])

    df_normalized = pd.DataFrame(
        normalized,
        columns=[f"{c}_minmax" for c in numeric_cols],
        index=df_norm.index
    )

    log(f"  [OK] MinMax normalization applied to {len(numeric_cols)} columns")
    log(f"  Range: [0, 1]")

    return df_normalized

# ============================================================
# 3.3 DATA STANDARDIZATION
# ============================================================
def stage_3_3_data_standardization(df):
    log("\n" + "=" * 70)
    log("STAGE 3.3: DATA STANDARDIZATION")
    log("=" * 70)

    df_std = df.copy()
    numeric_cols = df_std.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "Anomaly_Label"]

    scaler = StandardScaler()
    standardized = scaler.fit_transform(df_std[numeric_cols])

    df_standardized = pd.DataFrame(
        standardized,
        columns=[f"{c}_std" for c in numeric_cols],
        index=df_std.index
    )

    log(f"  [OK] Standardization applied to {len(numeric_cols)} columns")
    log(f"  Mean: 0, Std: 1")

    robust_scaler = RobustScaler()
    robust_scaled = robust_scaler.fit_transform(df_std[numeric_cols])
    df_robust = pd.DataFrame(
        robust_scaled,
        columns=[f"{c}_robust" for c in numeric_cols],
        index=df_std.index
    )

    log(f"  [OK] Robust scaling applied to {len(numeric_cols)} columns")

    return df_standardized, df_robust

# ============================================================
# 3.4 DATA ENCODING
# ============================================================
def stage_3_4_data_encoding(df):
    log("\n" + "=" * 70)
    log("STAGE 3.4: DATA ENCODING")
    log("=" * 70)

    df_encoded = df.copy()

    cat_cols = df_encoded.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Timestamp"]
    log(f"  Categorical columns (excluding Timestamp): {len(cat_cols)}")

    if cat_cols:
        for col in cat_cols:
            log(f"    {col}: {df_encoded[col].nunique()} unique values")
    else:
        log(f"  [OK] No categorical columns to encode (only Timestamp)")

    log(f"  Timestamp will be handled in Feature Engineering (Phase 05)")

    return df_encoded

# ============================================================
# 3.5 DATA DISCRETIZATION
# ============================================================
def stage_3_5_data_discretization(df):
    log("\n" + "=" * 70)
    log("STAGE 3.5: DATA DISCRETIZATION")
    log("=" * 70)

    df_disc = df.copy()

    key_cols = [
        "Boiler_efficiency_",
        "Flue_gas_temperature_",
        "Boiler_oxygen_level_",
    ]
    available_key_cols = [c for c in key_cols if c in df_disc.columns]

    if not available_key_cols:
        log("  [INFO] No key columns found for discretization")
        log(f"  Available columns sample: {list(df_disc.columns[:10])}")
        return df_disc

    discretization_report = []
    for col in available_key_cols:
        try:
            kb = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
            df_disc[f"{col}_bin"] = kb.fit_transform(df_disc[[col]]).astype(int)

            bin_counts = df_disc[f"{col}_bin"].value_counts().sort_index().to_dict()
            discretization_report.append({
                "Column": col,
                "Bins": 5,
                "Strategy": "quantile",
                "Bin_Counts": str(bin_counts)
            })
            log(f"  [OK] {col}: discretized into 5 bins")
        except Exception as e:
            log(f"  [WARNING] Could not discretize {col}: {e}")

    if discretization_report:
        pd.DataFrame(discretization_report).to_csv(
            TABLES_DIR / "03_05_discretization_report.csv", index=False
        )
        log(f"  [OK] Saved: 03_05_discretization_report.csv")

    return df_disc

# ============================================================
# 3.6 DATA BALANCING
# ============================================================
def stage_3_6_data_balancing(df):
    log("\n" + "=" * 70)
    log("STAGE 3.6: DATA BALANCING")
    log("=" * 70)

    if "Anomaly_Label" not in df.columns:
        log("  [WARNING] Anomaly_Label not found. Skipping balancing.")
        return df

    class_dist = df["Anomaly_Label"].value_counts()
    log(f"  Class distribution:")
    for label, count in class_dist.items():
        log(f"    Class {label}: {count} ({count/len(df)*100:.4f}%)")

    imbalance_ratio = class_dist.max() / class_dist.min()
    log(f"  Imbalance ratio: {imbalance_ratio:.2f}")

    balance_report = {
        "Original_Class_0": int(class_dist.get(0, 0)),
        "Original_Class_1": int(class_dist.get(1, 0)),
        "Imbalance_Ratio": round(imbalance_ratio, 2),
        "Strategy": "Will apply SMOTE in Phase 06 (Data Splitting)",
    }
    pd.DataFrame(list(balance_report.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "03_06_balancing_report.csv", index=False
    )
    log(f"  [OK] Saved: 03_06_balancing_report.csv")
    log(f"  [NOTE] Balancing will be applied in Phase 06 (Data Splitting)")

    return df

# ============================================================
# 3.7 DATA AUGMENTATION
# ============================================================
def stage_3_7_data_augmentation(df):
    log("\n" + "=" * 70)
    log("STAGE 3.7: DATA AUGMENTATION")
    log("=" * 70)

    log(f"  [INFO] Time Series data - augmentation strategies:")
    log(f"    - Window slicing (for deep learning)")
    log(f"    - Jittering (adding noise)")
    log(f"    - Scaling (magnitude warping)")
    log(f"  [NOTE] Augmentation will be applied in Phase 09 (Architecture Design)")

    return df

# ============================================================
# 3.8 DATA LABELING
# ============================================================
def stage_3_8_data_labeling(df):
    log("\n" + "=" * 70)
    log("STAGE 3.8: DATA LABELING")
    log("=" * 70)

    if "Anomaly_Label" in df.columns:
        log(f"  [OK] Anomaly_Label exists: {df['Anomaly_Label'].sum()} anomalies")
        log(f"  Anomaly rate: {df['Anomaly_Label'].mean()*100:.4f}%")
    else:
        log(f"  [WARNING] Anomaly_Label not found")

    return df

# ============================================================
# 3.9 DATA ANNOTATION
# ============================================================
def stage_3_9_data_annotation(df):
    log("\n" + "=" * 70)
    log("STAGE 3.9: DATA ANNOTATION")
    log("=" * 70)

    metadata = {
        "Dataset_Name": "Power Plant Data: Steam Turbine & Boiler Metrics",
        "Source": "Kaggle - Pavanjit Subash",
        "Original_Rows": 50091,
        "Original_Columns": 55,
        "Final_Columns": df.shape[1],
        "Time_Range": "January 2022",
        "Time_Resolution": "10 minutes",
        "Anomaly_Count": int(df["Anomaly_Label"].sum()) if "Anomaly_Label" in df.columns else 0,
        "Anomaly_Rate": round(df["Anomaly_Label"].mean() * 100, 4) if "Anomaly_Label" in df.columns else 0,
        "Phase": "03 - Data Transformation & Encoding",
        "Date": datetime.now().strftime("%Y-%m-%d"),
    }

    metadata_df = pd.DataFrame(list(metadata.items()), columns=["Key", "Value"])
    metadata_df.to_csv(TABLES_DIR / "03_09_data_annotation.csv", index=False)
    log(f"  [OK] Saved: 03_09_data_annotation.csv")

    for k, v in metadata.items():
        log(f"    {k}: {v}")

    return df

# ============================================================
# 3.10 DATA VERSIONING
# ============================================================
def stage_3_10_data_versioning(df):
    log("\n" + "=" * 70)
    log("STAGE 3.10: DATA VERSIONING")
    log("=" * 70)

    version_info = {
        "Version": "v0.3.0",
        "Phase": "03 - Data Transformation & Encoding",
        "Parent_Version": "v0.2.0 (Phase 02)",
        "Changes": "Added transformation, normalization, standardization, discretization",
        "Shape": f"{df.shape[0]} x {df.shape[1]}",
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    version_df = pd.DataFrame(list(version_info.items()), columns=["Key", "Value"])
    version_df.to_csv(TABLES_DIR / "03_10_data_version.csv", index=False)
    log(f"  [OK] Saved: 03_10_data_version.csv")
    log(f"  Version: {version_info['Version']}")

    return df

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 03: DATA TRANSFORMATION & ENCODING")
    log("=" * 70)

    df = pd.read_csv(PROCESSED)

    df = stage_3_1_data_transformation(df)
    df_normalized = stage_3_2_data_normalization(df)
    df_standardized, df_robust = stage_3_3_data_standardization(df)
    df = stage_3_4_data_encoding(df)
    df = stage_3_5_data_discretization(df)
    df = stage_3_6_data_balancing(df)
    df = stage_3_7_data_augmentation(df)
    df = stage_3_8_data_labeling(df)
    df = stage_3_9_data_annotation(df)
    df = stage_3_10_data_versioning(df)

    output_path = OUTPUT_DIR / "industrial_dataset_phase03_final.csv"
    df.to_csv(output_path, index=False)
    log("\n" + "=" * 70)
    log(f"[OK] FINAL DATASET SAVED: {output_path}")
    log(f"[OK] Final shape: {df.shape}")
    log("=" * 70)

    LOG_FILE = BASE_DIR / "reports" / "phase03_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()
