"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 02b: Rebuild Anomaly_Label from Raw Data
Framework: ML Model Lifecycle - 23 Main Stages
Author: Amin Mosallanejad
Date: 2026

Rationale:
The Winsorization in Phase 02 (Stage 2.8) capped negative values in
APH_Leakage_, CO_mgm3, Dust_mgm3, and Reheater_desuperheating_water_flow_th.
As a result, the anomaly labeling in Phase 02 missed anomalies from Dust_mgm3.

This script rebuilds Anomaly_Label from the ORIGINAL raw data to ensure
all sensor anomalies are captured.
"""

import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
RAW_DATA = BASE_DIR / "data" / "raw" / "industrial_dataset.csv"
PHASE02_FINAL = BASE_DIR / "data" / "processed" / "industrial_dataset_phase02_final.csv"
OUTPUT = BASE_DIR / "data" / "processed" / "industrial_dataset_phase02_final_v2.csv"
TABLES_DIR = BASE_DIR / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری داده خام و داده تمیز
# ============================================================
log("=" * 70)
log("PHASE 02b: REBUILD ANOMALY LABEL FROM RAW DATA")
log("=" * 70)

df_raw = pd.read_csv(RAW_DATA)
log(f"Raw data loaded: {df_raw.shape}")

df_clean = pd.read_csv(PHASE02_FINAL)
log(f"Clean data loaded: {df_clean.shape}")

# ============================================================
# استانداردسازی نام ستون‌های داده خام
# ============================================================
df_raw.columns = (
    df_raw.columns.astype(str)
    .str.strip()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.replace(r"\s+", "_", regex=True)
)
log(f"Raw columns standardized")

# ============================================================
# شناسایی ستون‌های ناهنجاری در داده خام
# ============================================================
anomaly_source_cols = [
    "APH_Leakage_",
    "CO_mgm3",
    "Reheater_desuperheating_water_flow_th",
    "Dust_mgm3",
]

# بررسی وجود ستون‌ها
available_cols = [c for c in anomaly_source_cols if c in df_raw.columns]
log(f"\nAnomaly source columns in RAW data: {available_cols}")

# ============================================================
# ساخت Anomaly_Label از داده خام
# ============================================================
log("\n" + "=" * 70)
log("REBUILDING ANOMALY LABEL FROM RAW DATA")
log("=" * 70)

anomaly_mask = pd.Series(False, index=df_raw.index)
anomaly_details = []

for col in available_cols:
    neg_mask = df_raw[col] < 0
    count = int(neg_mask.sum())
    pct = count / len(df_raw) * 100
    anomaly_mask = anomaly_mask | neg_mask
    anomaly_details.append({
        "Column": col,
        "Negative_Count": count,
        "Negative_Pct": round(pct, 4),
    })
    log(f"  {col}: {count} negative values ({pct:.4f}%)")

# ============================================================
# مقایسه با Anomaly_Label قبلی
# ============================================================
log("\n" + "=" * 70)
log("COMPARISON: OLD vs NEW ANOMALY LABEL")
log("=" * 70)

# برچسب جدید
new_anomaly = anomaly_mask.astype(int)
new_count = int(new_anomaly.sum())

# برچسب قدیمی (از فایل تمیز)
if "Anomaly_Label" in df_clean.columns:
    old_anomaly = df_clean["Anomaly_Label"]
    old_count = int(old_anomaly.sum())
    
    # مقایسه
    comparison = pd.DataFrame({
        "Old_Label": old_anomaly,
        "New_Label": new_anomaly,
    })
    
    both = ((comparison["Old_Label"] == 1) & (comparison["New_Label"] == 1)).sum()
    only_old = ((comparison["Old_Label"] == 1) & (comparison["New_Label"] == 0)).sum()
    only_new = ((comparison["Old_Label"] == 0) & (comparison["New_Label"] == 1)).sum()
    neither = ((comparison["Old_Label"] == 0) & (comparison["New_Label"] == 0)).sum()
    
    log(f"  Old Anomaly_Label count: {old_count} ({old_count/len(df_clean)*100:.4f}%)")
    log(f"  New Anomaly_Label count: {new_count} ({new_count/len(df_clean)*100:.4f}%)")
    log(f"  Difference: {new_count - old_count}")
    log(f"\n  Cross-tabulation:")
    log(f"    Both labeled anomaly:     {both}")
    log(f"    Only OLD labeled anomaly: {only_old}")
    log(f"    Only NEW labeled anomaly: {only_new}")
    log(f"    Neither:                  {neither}")
    
    # ذخیره گزارش مقایسه
    comparison_report = pd.DataFrame({
        "Metric": [
            "Old_Anomaly_Count", "New_Anomaly_Count", "Difference",
            "Both_Anomaly", "Only_Old_Anomaly", "Only_New_Anomaly", "Neither",
        ],
        "Value": [
            old_count, new_count, new_count - old_count,
            both, only_old, only_new, neither,
        ],
    })
    comparison_report.to_csv(
        TABLES_DIR / "02b_anomaly_label_comparison.csv", index=False
    )
    log(f"\n  [OK] Saved: 02b_anomaly_label_comparison.csv")
else:
    log(f"  [WARNING] Old Anomaly_Label not found in clean data")

# ============================================================
# به‌روزرسانی دیتاست تمیز با برچسب جدید
# ============================================================
log("\n" + "=" * 70)
log("UPDATING DATASET WITH NEW ANOMALY LABEL")
log("=" * 70)

df_updated = df_clean.copy()

# جایگزینی Anomaly_Label
if "Anomaly_Label" in df_updated.columns:
    df_updated = df_updated.drop(columns=["Anomaly_Label"])

df_updated["Anomaly_Label"] = new_anomaly.values
log(f"  [OK] Anomaly_Label updated: {new_count} anomalies ({new_count/len(df_updated)*100:.4f}%)")

# ============================================================
# ذخیره دیتاست به‌روزشده
# ============================================================
df_updated.to_csv(OUTPUT, index=False)
log(f"\n[OK] Saved: {OUTPUT}")
log(f"[OK] Final shape: {df_updated.shape}")

# ============================================================
# ذخیره گزارش جزئیات
# ============================================================
details_df = pd.DataFrame(anomaly_details)
details_df.to_csv(TABLES_DIR / "02b_anomaly_sources_details.csv", index=False)
log(f"[OK] Saved: 02b_anomaly_sources_details.csv")

# ============================================================
# ذخیره لاگ
# ============================================================
LOG_FILE = BASE_DIR / "reports" / "phase02b_rebuild_log.txt"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"\n[OK] Log saved: {LOG_FILE}")

log("\n" + "=" * 70)
log("PHASE 02b COMPLETE!")
log("=" * 70)
log("Next: Replace 'industrial_dataset_phase02_final.csv' with the new version,")
log("      then proceed to Phase 03.")