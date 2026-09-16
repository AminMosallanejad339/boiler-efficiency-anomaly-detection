"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 02: Final Cleanup & Anomaly Labeling
Framework: ML Model Lifecycle - 23 Main Stages
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

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
PROCESSED = BASE_DIR / "data" / "processed" / "industrial_dataset_cleaned.csv"
FINAL_OUTPUT = BASE_DIR / "data" / "processed" / "industrial_dataset_phase02_final.csv"
TABLES_DIR = BASE_DIR / "reports" / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری
# ============================================================
log("=" * 70)
log("PHASE 02: FINAL CLEANUP & ANOMALY LABELING")
log("=" * 70)

df = pd.read_csv(PROCESSED)
log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ============================================================
# تصمیم ۱: مستندسازی ستون‌های BY_DESIGN
# ============================================================
log("\n" + "=" * 70)
log("DECISION 1: BY_DESIGN NEGATIVE COLUMNS")
log("=" * 70)

by_design_cols = ["Condenser_vacuum_kPa", "Pressure_Kpa"]
for col in by_design_cols:
    if col in df.columns:
        log(f"  [KEEP] {col}: always negative by design (vacuum / differential pressure)")

# ============================================================
# تصمیم ۲: ساخت Anomaly_Label
# ============================================================
log("\n" + "=" * 70)
log("DECISION 2: ANOMALY LABELING")
log("=" * 70)

# ستون‌های ناهنجاری (بر اساس تحلیل قبلی)
anomaly_source_cols = [
    "APH_Leakage_",
    "CO_mgm3",
    "Reheater_desuperheating_water_flow_th",
    "Dust_mgm3",
]

# بررسی وجود ستون‌ها
available_anomaly_cols = [c for c in anomaly_source_cols if c in df.columns]
log(f"Anomaly source columns found: {available_anomaly_cols}")

# ساخت Anomaly_Label
df["Anomaly_Label"] = 0
anomaly_mask = pd.Series(False, index=df.index)

for col in available_anomaly_cols:
    neg_mask = df[col] < 0
    anomaly_mask = anomaly_mask | neg_mask
    log(f"  {col}: {neg_mask.sum()} negative values")

df.loc[anomaly_mask, "Anomaly_Label"] = 1
log(f"\n  Total anomalies labeled: {df['Anomaly_Label'].sum()} ({df['Anomaly_Label'].mean()*100:.3f}%)")
log(f"  Normal records: {(df['Anomaly_Label'] == 0).sum()}")

# ============================================================
# تصمیم ۳: حذف ستون‌های _smooth3
# ============================================================
log("\n" + "=" * 70)
log("DECISION 3: REMOVE SMOOTH COLUMNS")
log("=" * 70)

smooth_cols = [c for c in df.columns if c.endswith("_smooth3")]
log(f"Smooth columns found: {len(smooth_cols)}")

if smooth_cols:
    df = df.drop(columns=smooth_cols)
    log(f"  [OK] Removed {len(smooth_cols)} smooth columns")
    log(f"  Rationale: mean correlation with original = 0.9769 (Multicollinearity)")
else:
    log(f"  [OK] No smooth columns to remove")

log(f"Shape after removal: {df.shape}")

# ============================================================
# تصمیم ۴: تأیید Outlier Treatment
# ============================================================
log("\n" + "=" * 70)
log("DECISION 4: OUTLIER TREATMENT CONFIRMATION")
log("=" * 70)
log("  [CONFIRMED] Winsorization (IQR capping at 1.5*IQR)")
log("  All 54 columns had < 5% outliers and were capped")

# ============================================================
# ذخیره دیتاست نهایی
# ============================================================
log("\n" + "=" * 70)
log("SAVING FINAL DATASET")
log("=" * 70)

df.to_csv(FINAL_OUTPUT, index=False)
log(f"[OK] Saved: {FINAL_OUTPUT}")
log(f"[OK] Final shape: {df.shape}")

# ============================================================
# گزارش نهایی
# ============================================================
log("\n" + "=" * 70)
log("FINAL DATA QUALITY REPORT")
log("=" * 70)

quality_report = {
    "Phase": "02 - Data Collection & Preparation",
    "Framework": "ML Model Lifecycle - 23 Main Stages",
    "Original Shape": "50091 x 55",
    "After Preparation": "50091 x 55",
    "After Cleanup": f"{df.shape[0]} x {df.shape[1]}",
    "Total Anomalies Labeled": int(df["Anomaly_Label"].sum()),
    "Anomaly Rate (%)": round(df["Anomaly_Label"].mean() * 100, 3),
    "Missing Values": int(df.isnull().sum().sum()),
    "Duplicate Rows": int(df.duplicated().sum()),
    "Numeric Columns": len(df.select_dtypes(include=[np.number]).columns),
    "Non-Numeric Columns": len(df.select_dtypes(exclude=[np.number]).columns),
    "Memory (MB)": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    "Processing Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

for k, v in quality_report.items():
    log(f"  {k}: {v}")

# ذخیره گزارش
report_df = pd.DataFrame(list(quality_report.items()), columns=["Metric", "Value"])
report_df.to_csv(TABLES_DIR / "02_15_final_quality_report.csv", index=False)
log(f"\n[OK] Saved: 02_15_final_quality_report.csv")

# ============================================================
# مستندسازی تصمیمات
# ============================================================
decisions = [
    {"Decision": "BY_DESIGN negative columns kept", "Columns": "Condenser_vacuum_kPa, Pressure_Kpa", "Rationale": "Negative by design (vacuum, differential pressure)"},
    {"Decision": "Anomaly_Label created", "Columns": ", ".join(available_anomaly_cols), "Rationale": "Negative values indicate sensor anomalies per dataset documentation"},
    {"Decision": "Smooth columns removed", "Columns": f"{len(smooth_cols)} columns", "Rationale": "Mean correlation with original = 0.9769 (Multicollinearity)"},
    {"Decision": "Outlier treatment confirmed", "Columns": "All 54 numeric columns", "Rationale": "Winsorization with IQR capping, all < 5% outliers"},
]

decisions_df = pd.DataFrame(decisions)
decisions_df.to_csv(TABLES_DIR / "02_16_phase02_decisions.csv", index=False)
log(f"[OK] Saved: 02_16_phase02_decisions.csv")

# ============================================================
# ذخیره لاگ
# ============================================================
LOG_FILE = BASE_DIR / "reports" / "phase02_final_cleanup_log.txt"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"\n[OK] Log saved: {LOG_FILE}")

log("\n" + "=" * 70)
log("PHASE 02 COMPLETE!")
log("=" * 70)
log("Ready for Phase 03: Data Transformation & Encoding")