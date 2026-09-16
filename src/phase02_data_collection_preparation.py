"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 02: Data Collection & Preparation (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Author: Amin Mosallanejad
Date: 2026
"""

import os
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
RAW_DATA = BASE_DIR / "data" / "raw" / "industrial_dataset.csv"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
REPORTS_DIR = BASE_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

for d in [PROCESSED_DIR, FIGURES_DIR, TABLES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# گزارش‌گیری
# ============================================================
LOG_FILE = REPORTS_DIR / "phase02_log.txt"
log_lines = []

def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    log_lines.append(line)
    print(line)

def save_log():
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))

# ============================================================
# 2.1 DATA COLLECTION
# ============================================================
def stage_2_1_data_collection():
    log("=" * 70)
    log("STAGE 2.1: DATA COLLECTION")
    log("=" * 70)
    plan = {
        "Source": "Kaggle - Power Plant Data: Steam Turbine & Boiler Metrics",
        "URL": "https://www.kaggle.com/datasets/pavanjitsubash/power-plant-data-steam-turbine-and-boiler-metrics",
        "Author": "Pavanjit Subash",
        "Data Type": "Time Series (10-minute interval)",
        "Time Range": "January 2022",
        "Records": "50,000+ rows",
        "Features": "58 operational parameters",
        "File Format": "CSV",
        "File Size": "50.64 MB",
        "License": "Unspecified",
    }
    for k, v in plan.items():
        log(f"  {k}: {v}")
    pd.DataFrame(list(plan.items()), columns=["Item", "Value"]).to_csv(
        TABLES_DIR / "02_01_data_collection_plan.csv", index=False
    )
    return plan

# ============================================================
# 2.2 DATA ACQUISITION
# ============================================================
def stage_2_2_data_acquisition():
    log("=" * 70)
    log("STAGE 2.2: DATA ACQUISITION")
    log("=" * 70)
    if not RAW_DATA.exists():
        log(f"  [ERROR] File not found: {RAW_DATA}")
        log("  Please verify the dataset path and filename.")
        return None
    df = pd.read_csv(RAW_DATA)
    log(f"  [OK] Data loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    log(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    log(f"  First 10 columns: {list(df.columns[:10])}")
    log(f"  Last 5 columns: {list(df.columns[-5:])}")
    return df

# ============================================================
# 2.3 DATA INTEGRATION
# ============================================================
def stage_2_3_data_integration(df):
    log("=" * 70)
    log("STAGE 2.3: DATA INTEGRATION")
    log("=" * 70)
    log(f"  Single-source dataset - no integration required")
    log(f"  Duplicate columns: {df.columns.duplicated().sum()}")
    log(f"  Duplicate rows: {df.duplicated().sum()}")
    return df

# ============================================================
# 2.4 DATA CLEANING
# ============================================================
def stage_2_4_data_cleaning(df):
    log("=" * 70)
    log("STAGE 2.4: DATA CLEANING")
    log("=" * 70)
    df_clean = df.copy()

    # 1. استانداردسازی نام ستون‌ها
    original_cols = df_clean.columns.tolist()
    df_clean.columns = (
        df_clean.columns.astype(str)
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    renamed = sum(1 for a, b in zip(original_cols, df_clean.columns) if a != b)
    log(f"  [OK] Renamed {renamed} columns")

    # 2. تشخیص و تبدیل ستون زمان
    time_cols = [c for c in df_clean.columns if any(k in c.lower() for k in ["time", "date", "timestamp"])]
    log(f"  Time columns detected: {time_cols}")
    if time_cols:
        tcol = time_cols[0]
        try:
            df_clean[tcol] = pd.to_datetime(df_clean[tcol], errors="coerce")
            df_clean = df_clean.sort_values(tcol).reset_index(drop=True)
            log(f"  [OK] Converted '{tcol}' to datetime and sorted chronologically")
        except Exception as e:
            log(f"  [WARNING] Failed to parse time column: {e}")

    # 3. حذف ستون‌های ثابت
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    constant_cols = [c for c in numeric_cols if df_clean[c].nunique(dropna=False) <= 1]
    if constant_cols:
        df_clean = df_clean.drop(columns=constant_cols)
        log(f"  [OK] Dropped {len(constant_cols)} constant columns: {constant_cols}")
    else:
        log(f"  [OK] No constant columns found")

    log(f"  Final shape after cleaning: {df_clean.shape}")
    return df_clean

# ============================================================
# 2.5 DATA VALIDATION
# ============================================================
def stage_2_5_data_validation(df):
    log("=" * 70)
    log("STAGE 2.5: DATA VALIDATION")
    log("=" * 70)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    log(f"  Numeric columns: {len(numeric_cols)}")

    stats_df = df[numeric_cols].describe().T
    stats_df["missing"] = df[numeric_cols].isnull().sum()
    stats_df["missing_pct"] = (stats_df["missing"] / len(df)) * 100
    stats_df["negative_count"] = (df[numeric_cols] < 0).sum()
    stats_df["zero_count"] = (df[numeric_cols] == 0).sum()
    stats_df.to_csv(TABLES_DIR / "02_05_validation_stats.csv")
    log(f"  [OK] Saved validation stats: 02_05_validation_stats.csv")

    # بررسی مقادیر منفی غیرمنطقی
    neg_summary = stats_df[stats_df["negative_count"] > 0][["negative_count", "missing_pct"]]
    if not neg_summary.empty:
        log(f"  [WARNING] Columns with negative values:")
        for col, row in neg_summary.iterrows():
            log(f"    {col}: {int(row['negative_count'])} negative values")
    else:
        log(f"  [OK] No negative values detected")

    return df

# ============================================================
# 2.6 MISSING VALUE HANDLING
# ============================================================
def stage_2_6_missing_values(df):
    log("=" * 70)
    log("STAGE 2.6: MISSING VALUE HANDLING")
    log("=" * 70)
    df_clean = df.copy()
    missing = df_clean.isnull().sum()
    missing_pct = (missing / len(df_clean)) * 100

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing_Count": missing.values,
        "Missing_Pct": missing_pct.values,
    })
    missing_df = missing_df[missing_df["Missing_Count"] > 0].sort_values("Missing_Pct", ascending=False)
    missing_df.to_csv(TABLES_DIR / "02_06_missing_values.csv", index=False)

    if missing_df.empty:
        log("  [OK] No missing values found")
        return df_clean

    log(f"  Columns with missing values: {len(missing_df)}")
    log(f"  Total missing cells: {int(missing.sum())}")

    for col in missing_df["Column"]:
        pct = missing_pct[col]
        if pct > 50:
            df_clean = df_clean.drop(columns=[col])
            log(f"    [DROP] {col}: {pct:.2f}% missing")
        elif df_clean[col].dtype.kind in "biufc":
            if pct > 5:
                df_clean[col] = df_clean[col].interpolate(method="linear", limit_direction="both")
                log(f"    [INTERPOLATE] {col}: {pct:.2f}%")
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                log(f"    [MEDIAN] {col}: {pct:.2f}%")
        else:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
            log(f"    [MODE] {col}: {pct:.2f}%")

    log(f"  [OK] Remaining missing values: {int(df_clean.isnull().sum().sum())}")
    return df_clean

# ============================================================
# 2.7 OUTLIER DETECTION
# ============================================================
def stage_2_7_outlier_detection(df):
    log("=" * 70)
    log("STAGE 2.7: OUTLIER DETECTION")
    log("=" * 70)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    rows = []
    for col in numeric_cols:
        data = df[col].dropna()
        if len(data) < 10:
            continue
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        iqr_out = int(((data < lower) | (data > upper)).sum())
        z_out = int((np.abs(stats.zscore(data)) > 3).sum())
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        mad_out = int((np.abs(0.6745 * (data - median) / mad) > 3.5).sum()) if mad > 0 else 0
        rows.append({
            "Column": col,
            "IQR_Outliers": iqr_out,
            "ZScore_Outliers": z_out,
            "MAD_Outliers": mad_out,
            "Total": len(data),
            "IQR_Pct": round(iqr_out / len(data) * 100, 3),
        })
    out_df = pd.DataFrame(rows).sort_values("IQR_Pct", ascending=False)
    out_df.to_csv(TABLES_DIR / "02_07_outlier_detection.csv", index=False)
    log(f"  [OK] Saved: 02_07_outlier_detection.csv")
    log(f"  Columns analyzed: {len(out_df)}")
    log(f"  Columns with IQR outliers: {(out_df['IQR_Outliers'] > 0).sum()}")
    log(f"  Columns with Z-Score outliers: {(out_df['ZScore_Outliers'] > 0).sum()}")
    return df, out_df

# ============================================================
# 2.8 OUTLIER TREATMENT
# ============================================================
def stage_2_8_outlier_treatment(df):
    log("=" * 70)
    log("STAGE 2.8: OUTLIER TREATMENT")
    log("=" * 70)
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    capped = 0
    for col in numeric_cols:
        data = df_clean[col].dropna()
        if len(data) < 10:
            continue
        Q1, Q3 = data.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        n_out = int(((data < lower) | (data > upper)).sum())
        if 0 < n_out / len(data) < 0.05:
            df_clean[col] = df_clean[col].clip(lower=lower, upper=upper)
            capped += 1
    log(f"  [OK] Winsorization applied to {capped} columns (threshold: <5% outliers)")
    log(f"  Strategy: IQR capping at 1.5 * IQR")
    return df_clean

# ============================================================
# 2.9 NOISE REDUCTION
# ============================================================
def stage_2_9_noise_reduction(df):
    log("=" * 70)
    log("STAGE 2.9: NOISE REDUCTION")
    log("=" * 70)
    df_clean = df.copy()
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    smoothed = 0
    for col in numeric_cols:
        if df_clean[col].nunique() > 100:
            df_clean[f"{col}_smooth3"] = (
                df_clean[col].rolling(window=3, center=True, min_periods=1).mean()
            )
            smoothed += 1
    log(f"  [OK] Rolling-mean smoothing applied to {smoothed} high-variance columns")
    log(f"  Window: 3 samples = 30 minutes (centered)")
    log(f"  Original columns preserved")
    return df_clean

# ============================================================
# 2.10 DATA DOCUMENTATION
# ============================================================
def stage_2_10_data_documentation(df_orig, df_final):
    log("=" * 70)
    log("STAGE 2.10: DATA DOCUMENTATION")
    log("=" * 70)
    rows = []
    for col in df_final.columns:
        s = df_final[col]
        rows.append({
            "Column": col,
            "Dtype": str(s.dtype),
            "Non_Null": int(s.notna().sum()),
            "Null": int(s.isna().sum()),
            "Unique": int(s.nunique()),
            "Min": s.min() if s.dtype.kind in "biufc" else "N/A",
            "Max": s.max() if s.dtype.kind in "biufc" else "N/A",
            "Mean": round(float(s.mean()), 4) if s.dtype.kind in "biufc" else "N/A",
            "Std": round(float(s.std()), 4) if s.dtype.kind in "biufc" else "N/A",
        })
    pd.DataFrame(rows).to_csv(TABLES_DIR / "02_10_data_dictionary.csv", index=False)
    log(f"  [OK] Saved: 02_10_data_dictionary.csv")

    quality = {
        "Original Shape": f"{df_orig.shape[0]} x {df_orig.shape[1]}",
        "Final Shape": f"{df_final.shape[0]} x {df_final.shape[1]}",
        "Original Missing": int(df_orig.isnull().sum().sum()),
        "Final Missing": int(df_final.isnull().sum().sum()),
        "Memory (MB)": round(df_final.memory_usage(deep=True).sum() / 1024**2, 2),
        "Numeric Columns": len(df_final.select_dtypes(include=[np.number]).columns),
        "Non-Numeric Columns": len(df_final.select_dtypes(exclude=[np.number]).columns),
        "Processing Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    pd.DataFrame(list(quality.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "02_10_data_quality_report.csv", index=False
    )
    log(f"  [OK] Saved: 02_10_data_quality_report.csv")
    for k, v in quality.items():
        log(f"    {k}: {v}")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 02: DATA COLLECTION & PREPARATION")
    log("=" * 70)

    stage_2_1_data_collection()

    df_raw = stage_2_2_data_acquisition()
    if df_raw is None:
        log("[FATAL] Cannot proceed without dataset.")
        save_log()
        return

    df = stage_2_3_data_integration(df_raw)
    df = stage_2_4_data_cleaning(df)
    df = stage_2_5_data_validation(df)
    df = stage_2_6_missing_values(df)
    df, _ = stage_2_7_outlier_detection(df)
    df = stage_2_8_outlier_treatment(df)
    df = stage_2_9_noise_reduction(df)
    stage_2_10_data_documentation(df_raw, df)

    out_path = PROCESSED_DIR / "industrial_dataset_cleaned.csv"
    df.to_csv(out_path, index=False)
    log("=" * 70)
    log(f"[OK] FINAL DATASET SAVED: {out_path}")
    log(f"[OK] Final shape: {df.shape}")
    log("=" * 70)

    save_log()
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()