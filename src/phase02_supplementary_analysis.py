"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 02: Supplementary Analysis for Decision Making
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
PROCESSED = BASE_DIR / "data" / "processed" / "industrial_dataset_cleaned.csv"
TABLES_DIR = BASE_DIR / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری دیتاست تمیز
# ============================================================
log("=" * 70)
log("PHASE 02: SUPPLEMENTARY ANALYSIS")
log("=" * 70)

df = pd.read_csv(PROCESSED)
log(f"Dataset loaded: {df.shape}")

# ============================================================
# تحلیل ۱: مقادیر منفی — دسته‌بندی دقیق
# ============================================================
log("\n" + "=" * 70)
log("ANALYSIS 1: NEGATIVE VALUES CLASSIFICATION")
log("=" * 70)

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
neg_report = []

for col in numeric_cols:
    neg = (df[col] < 0).sum()
    if neg > 0:
        pct = neg / len(df) * 100
        if pct > 99:
            category = "BY_DESIGN (always negative)"
        elif pct > 10:
            category = "HIGH_NEGATIVE (investigate)"
        elif pct > 1:
            category = "MODERATE_NEGATIVE (likely anomaly)"
        else:
            category = "RARE_NEGATIVE (sensor anomaly)"
        neg_report.append({
            "Column": col,
            "Negative_Count": neg,
            "Negative_Pct": round(pct, 3),
            "Category": category,
            "Min": round(df[col].min(), 4),
            "Max": round(df[col].max(), 4),
            "Mean": round(df[col].mean(), 4),
        })

neg_df = pd.DataFrame(neg_report).sort_values("Negative_Pct", ascending=False)
neg_df.to_csv(TABLES_DIR / "02_11_negative_values_classification.csv", index=False)
log(f"[OK] Saved: 02_11_negative_values_classification.csv")
log(f"\nNegative values summary:")
for _, row in neg_df.iterrows():
    log(f"  {row['Column']}: {row['Negative_Count']} ({row['Negative_Pct']}%) -> {row['Category']}")

# ============================================================
# تحلیل ۲: بررسی outlier treatment — چند ستون واقعاً کپ شدند؟
# ============================================================
log("\n" + "=" * 70)
log("ANALYSIS 2: OUTLIER TREATMENT EFFECTIVENESS")
log("=" * 70)

# بارگذاری دیتاست خام برای مقایسه
RAW = BASE_DIR / "data" / "raw" / "industrial_dataset.csv"
df_raw = pd.read_csv(RAW)

# استانداردسازی نام ستون‌های خام
df_raw.columns = (
    df_raw.columns.astype(str)
    .str.strip()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(r"\s+", "_", regex=True)
)

# مقایسه min/max ستون‌های مشترک
common_cols = [c for c in df_raw.columns if c in df.columns and df_raw[c].dtype.kind in "biufc"]
treatment_report = []
for col in common_cols:
    raw_min, raw_max = df_raw[col].min(), df_raw[col].max()
    clean_min, clean_max = df[col].min(), df[col].max()
    changed = (raw_min != clean_min) or (raw_max != clean_max)
    treatment_report.append({
        "Column": col,
        "Raw_Min": round(raw_min, 4),
        "Raw_Max": round(raw_max, 4),
        "Clean_Min": round(clean_min, 4),
        "Clean_Max": round(clean_max, 4),
        "Capped": changed,
    })

treat_df = pd.DataFrame(treatment_report)
treat_df.to_csv(TABLES_DIR / "02_12_outlier_treatment_effectiveness.csv", index=False)
capped_count = treat_df["Capped"].sum()
log(f"[OK] Saved: 02_12_outlier_treatment_effectiveness.csv")
log(f"Columns capped: {capped_count} / {len(treat_df)}")
log(f"Columns NOT capped: {len(treat_df) - capped_count}")
log(f"\nColumns that were NOT capped (outliers > 5%):")
not_capped = treat_df[~treat_df["Capped"]]["Column"].tolist()
for c in not_capped[:20]:
    log(f"  {c}")

# ============================================================
# تحلیل ۳: ستون‌های Smooth — بررسی همبستگی
# ============================================================
log("\n" + "=" * 70)
log("ANALYSIS 3: SMOOTH COLUMNS CORRELATION")
log("=" * 70)

smooth_cols = [c for c in df.columns if c.endswith("_smooth3")]
original_cols = [c.replace("_smooth3", "") for c in smooth_cols if c.replace("_smooth3", "") in df.columns]
log(f"Smooth columns: {len(smooth_cols)}")
log(f"Matching originals: {len(original_cols)}")

# بررسی همبستگی بین هر جفت
corr_report = []
for s_col in smooth_cols[:20]:  # فقط 20 تا برای سرعت
    o_col = s_col.replace("_smooth3", "")
    if o_col in df.columns:
        corr = df[s_col].corr(df[o_col])
        corr_report.append({
            "Smooth_Column": s_col,
            "Original_Column": o_col,
            "Correlation": round(corr, 6),
        })

corr_df = pd.DataFrame(corr_report)
corr_df.to_csv(TABLES_DIR / "02_13_smooth_original_correlation.csv", index=False)
log(f"[OK] Saved: 02_13_smooth_original_correlation.csv")
log(f"Mean correlation (smooth vs original): {corr_df['Correlation'].mean():.6f}")
log(f"Min correlation: {corr_df['Correlation'].min():.6f}")

# ============================================================
# تحلیل ۴: لیست کامل ستون‌ها برای Data Dictionary
# ============================================================
log("\n" + "=" * 70)
log("ANALYSIS 4: COMPLETE COLUMN LIST")
log("=" * 70)

col_info = []
for col in df.columns:
    col_info.append({
        "Index": df.columns.get_loc(col),
        "Column": col,
        "Dtype": str(df[col].dtype),
        "Is_Numeric": df[col].dtype.kind in "biufc",
        "Is_Smooth": col.endswith("_smooth3"),
        "Is_Timestamp": "timestamp" in col.lower() or "time" in col.lower(),
    })

col_df = pd.DataFrame(col_info)
col_df.to_csv(TABLES_DIR / "02_14_complete_column_list.csv", index=False)
log(f"[OK] Saved: 02_14_complete_column_list.csv")
log(f"Total columns: {len(col_df)}")
log(f"Numeric: {col_df['Is_Numeric'].sum()}")
log(f"Smooth: {col_df['Is_Smooth'].sum()}")
log(f"Timestamp: {col_df['Is_Timestamp'].sum()}")

# ============================================================
# ذخیره لاگ
# ============================================================
LOG_FILE = BASE_DIR / "reports" / "phase02_supplementary_log.txt"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"\n[OK] Log saved: {LOG_FILE}")