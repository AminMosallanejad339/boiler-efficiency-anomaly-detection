"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 06: Data Splitting & Pipeline (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Strategy: Time-Based + Stratified Split (Hybrid)
Dual Target: Boiler_Eff_ (Regression) + Anomaly_Label (Classification)
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import (
    TimeSeriesSplit, StratifiedKFold, train_test_split
)
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
PROCESSED = BASE_DIR / "data" / "processed" / "industrial_dataset_phase05_final.csv"
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [SPLITS_DIR, MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# تعریف اهداف
# ============================================================
TARGET_REG = "Boiler_Eff_"
TARGET_CLF = "Anomaly_Label"

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری
# ============================================================
log("=" * 70)
log("PHASE 06: DATA SPLITTING & PIPELINE")
log("=" * 70)

df = pd.read_csv(PROCESSED)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Time range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# ============================================================
# 6.1 DATA SPLITTING (Strategy)
# ============================================================
def stage_6_1_splitting_strategy():
    log("\n" + "=" * 70)
    log("STAGE 6.1: DATA SPLITTING STRATEGY")
    log("=" * 70)
    
    strategy = {
        "Split Type": "Hybrid (Time-Based + Stratified)",
        "Train Ratio": "70%",
        "Validation Ratio": "15%",
        "Test Ratio": "15%",
        "Time-Based": "Preserves temporal order",
        "Stratified": "Preserves class distribution",
        "Random State": 42,
        "Rationale": "Time Series data + Imbalanced classes",
    }
    
    for k, v in strategy.items():
        log(f"  {k}: {v}")
    
    return strategy

# ============================================================
# 6.2-6.4 TIME-BASED SPLIT
# ============================================================
def stage_6_2_4_time_based_split(df):
    log("\n" + "=" * 70)
    log("STAGE 6.2-6.4: TIME-BASED SPLIT (70/15/15)")
    log("=" * 70)
    
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    log(f"  Total records: {n}")
    log(f"  Train: {len(train_df)} ({len(train_df)/n*100:.2f}%)")
    log(f"  Validation: {len(val_df)} ({len(val_df)/n*100:.2f}%)")
    log(f"  Test: {len(test_df)} ({len(test_df)/n*100:.2f}%)")
    
    log(f"\n  Time ranges:")
    log(f"    Train: {train_df['Timestamp'].min()} to {train_df['Timestamp'].max()}")
    log(f"    Val:   {val_df['Timestamp'].min()} to {val_df['Timestamp'].max()}")
    log(f"    Test:  {test_df['Timestamp'].min()} to {test_df['Timestamp'].max()}")
    
    log(f"\n  Anomaly distribution:")
    log(f"    Train: {train_df[TARGET_CLF].sum()} ({train_df[TARGET_CLF].mean()*100:.4f}%)")
    log(f"    Val:   {val_df[TARGET_CLF].sum()} ({val_df[TARGET_CLF].mean()*100:.4f}%)")
    log(f"    Test:  {test_df[TARGET_CLF].sum()} ({test_df[TARGET_CLF].mean()*100:.4f}%)")
    
    return train_df, val_df, test_df

# ============================================================
# 6.5-6.6 STRATIFIED CHECK & ADJUSTMENT
# ============================================================
def stage_6_5_6_stratified_check(train_df, val_df, test_df):
    log("\n" + "=" * 70)
    log("STAGE 6.5-6.6: STRATIFIED CHECK & ADJUSTMENT")
    log("=" * 70)
    
    # بررسی توزیع کلاس در هر مجموعه
    train_ratio = train_df[TARGET_CLF].mean()
    val_ratio = val_df[TARGET_CLF].mean()
    test_ratio = test_df[TARGET_CLF].mean()
    
    overall_ratio = df[TARGET_CLF].mean()
    
    log(f"  Overall anomaly rate: {overall_ratio*100:.4f}%")
    log(f"  Train anomaly rate:   {train_ratio*100:.4f}%")
    log(f"  Val anomaly rate:     {val_ratio*100:.4f}%")
    log(f"  Test anomaly rate:    {test_ratio*100:.4f}%")
    
    # محاسبه انحراف
    deviation = {
        "Train": abs(train_ratio - overall_ratio) / overall_ratio * 100,
        "Val": abs(val_ratio - overall_ratio) / overall_ratio * 100,
        "Test": abs(test_ratio - overall_ratio) / overall_ratio * 100,
    }
    
    log(f"\n  Deviation from overall:")
    for k, v in deviation.items():
        status = "OK" if v < 20 else "ADJUST NEEDED"
        log(f"    {k}: {v:.2f}% [{status}]")
    
    # اگر انحراف زیاد بود، Stratified Split می‌کنیم
    if any(v > 20 for v in deviation.values()):
        log(f"\n  [ACTION] Applying Stratified Split...")
        # تقسیم Stratified
        train_df, temp_df = train_test_split(
            df, test_size=0.30, stratify=df[TARGET_CLF], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=0.50, stratify=temp_df[TARGET_CLF], random_state=42
        )
        log(f"  [OK] Stratified Split applied")
    else:
        log(f"\n  [OK] Time-Based Split accepted (deviation < 20%)")
    
    return train_df, val_df, test_df

# ============================================================
# 6.7 CROSS-VALIDATION SETUP
# ============================================================
def stage_6_7_cross_validation():
    log("\n" + "=" * 70)
    log("STAGE 6.7: CROSS-VALIDATION SETUP")
    log("=" * 70)
    
    cv_setup = {
        "Regression": {
            "Method": "TimeSeriesSplit",
            "N_Splits": 5,
            "Rationale": "Preserves temporal order for time series",
        },
        "Classification": {
            "Method": "StratifiedKFold",
            "N_Splits": 5,
            "Rationale": "Preserves class distribution across folds",
        },
    }
    
    for task, config in cv_setup.items():
        log(f"  {task}:")
        for k, v in config.items():
            log(f"    {k}: {v}")
    
    return cv_setup

# ============================================================
# 6.8 DATA LEAKAGE CHECK
# ============================================================
def stage_6_8_leakage_check(train_df, val_df, test_df):
    log("\n" + "=" * 70)
    log("STAGE 6.8: DATA LEAKAGE CHECK")
    log("=" * 70)
    
    # بررسی همپوشانی زمانی
    train_max = train_df["Timestamp"].max()
    val_min = val_df["Timestamp"].min()
    val_max = val_df["Timestamp"].max()
    test_min = test_df["Timestamp"].min()
    
    log(f"  Train max timestamp: {train_max}")
    log(f"  Val min timestamp:   {val_min}")
    log(f"  Test min timestamp:  {test_min}")
    
    if train_max < val_min:
        log(f"  [OK] No temporal overlap between Train and Val")
    else:
        log(f"  [WARNING] Temporal overlap detected!")
    
    if val_max < test_min:
        log(f"  [OK] No temporal overlap between Val and Test")
    else:
        log(f"  [WARNING] Temporal overlap detected!")
    
    # بررسی Target Leakage
    feature_cols = [c for c in df.columns if c not in [TARGET_REG, TARGET_CLF, "Timestamp"]]
    
    # بررسی اینکه Target در ویژگی‌ها نباشد
    for target in [TARGET_REG, TARGET_CLF]:
        if target in feature_cols:
            log(f"  [WARNING] Target '{target}' found in features!")
        else:
            log(f"  [OK] Target '{target}' not in features")
    
    # بررسی همبستگی بیش از حد
    high_corr_with_target = []
    for col in feature_cols:
        if df[col].dtype.kind in "biufc":
            corr = abs(df[col].corr(df[TARGET_REG]))
            if corr > 0.99:
                high_corr_with_target.append((col, corr))
    
    if high_corr_with_target:
        log(f"  [WARNING] Features with |corr| > 0.99 with {TARGET_REG}:")
        for col, corr in high_corr_with_target:
            log(f"    {col}: {corr:.6f}")
    else:
        log(f"  [OK] No features with |corr| > 0.99 with {TARGET_REG}")
    
    return True

# ============================================================
# 6.9 DATA PIPELINE CONSTRUCTION
# ============================================================
def stage_6_9_pipeline_construction(train_df):
    log("\n" + "=" * 70)
    log("STAGE 6.9: DATA PIPELINE CONSTRUCTION")
    log("=" * 70)
    
    feature_cols = [c for c in train_df.columns 
                    if c not in [TARGET_REG, TARGET_CLF, "Timestamp"]
                    and train_df[c].dtype.kind in "biufc"]
    
    log(f"  Feature columns: {len(feature_cols)}")
    
    # Pipeline برای Regression
    pipeline_reg = Pipeline([
        ("scaler", RobustScaler()),
        # مدل در مرحله ۰۸ اضافه می‌شود
    ])
    log(f"  [OK] Regression Pipeline created (RobustScaler)")
    
    # Pipeline برای Classification
    pipeline_clf = ImbPipeline([
        ("scaler", RobustScaler()),
        ("smote", SMOTE(random_state=42, sampling_strategy=0.3)),
        # مدل در مرحله ۰۸ اضافه می‌شود
    ])
    log(f"  [OK] Classification Pipeline created (RobustScaler + SMOTE)")
    
    # ذخیره Pipelineها
    joblib.dump(pipeline_reg, MODELS_DIR / "pipeline_regression.pkl")
    joblib.dump(pipeline_clf, MODELS_DIR / "pipeline_classification.pkl")
    log(f"  [OK] Pipelines saved to models/")
    
    return pipeline_reg, pipeline_clf, feature_cols

# ============================================================
# 6.10 DATA LOADER SETUP
# ============================================================
def stage_6_10_data_loader(train_df, val_df, test_df, feature_cols):
    log("\n" + "=" * 70)
    log("STAGE 6.10: DATA LOADER SETUP")
    log("=" * 70)
    
    # ذخیره Splitها
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "validation.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)
    
    log(f"  [OK] Saved: train.csv ({len(train_df)} rows)")
    log(f"  [OK] Saved: validation.csv ({len(val_df)} rows)")
    log(f"  [OK] Saved: test.csv ({len(test_df)} rows)")
    
    # ذخیره لیست ویژگی‌ها
    feature_df = pd.DataFrame({"Feature": feature_cols})
    feature_df.to_csv(SPLITS_DIR / "feature_columns.csv", index=False)
    log(f"  [OK] Saved: feature_columns.csv ({len(feature_cols)} features)")
    
    # گزارش نهایی
    summary = {
        "Total_Records": len(df),
        "Train_Records": len(train_df),
        "Val_Records": len(val_df),
        "Test_Records": len(test_df),
        "Feature_Count": len(feature_cols),
        "Target_Regression": TARGET_REG,
        "Target_Classification": TARGET_CLF,
        "Train_Anomaly_Rate": round(train_df[TARGET_CLF].mean() * 100, 4),
        "Val_Anomaly_Rate": round(val_df[TARGET_CLF].mean() * 100, 4),
        "Test_Anomaly_Rate": round(test_df[TARGET_CLF].mean() * 100, 4),
    }
    
    pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "06_10_split_summary.csv", index=False
    )
    log(f"  [OK] Saved: 06_10_split_summary.csv")
    
    log(f"\n  Split Summary:")
    for k, v in summary.items():
        log(f"    {k}: {v}")
    
    # نمودار تقسیم
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    # نمودار ۱: تعداد رکورد
    sizes = [len(train_df), len(val_df), len(test_df)]
    labels = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    colors = ["steelblue", "coral", "green"]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Data Split Distribution")
    
    # نمودار ۲: نرخ Anomaly
    rates = [
        train_df[TARGET_CLF].mean() * 100,
        val_df[TARGET_CLF].mean() * 100,
        test_df[TARGET_CLF].mean() * 100,
    ]
    axes[1].bar(labels, rates, color=colors)
    axes[1].axhline(y=df[TARGET_CLF].mean() * 100, color="red", linestyle="--", label="Overall")
    axes[1].set_ylabel("Anomaly Rate (%)")
    axes[1].set_title("Anomaly Rate by Split")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_10_split_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 06_10_split_distribution.png")
    
    return summary

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 06: DATA SPLITTING & PIPELINE")
    log("=" * 70)
    
    df = pd.read_csv(PROCESSED)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # 6.1
    strategy = stage_6_1_splitting_strategy()
    
    # 6.2-6.4
    train_df, val_df, test_df = stage_6_2_4_time_based_split(df)
    
    # 6.5-6.6
    train_df, val_df, test_df = stage_6_5_6_stratified_check(train_df, val_df, test_df)
    
    # 6.7
    cv_setup = stage_6_7_cross_validation()
    
    # 6.8
    stage_6_8_leakage_check(train_df, val_df, test_df)
    
    # 6.9
    pipeline_reg, pipeline_clf, feature_cols = stage_6_9_pipeline_construction(train_df)
    
    # 6.10
    summary = stage_6_10_data_loader(train_df, val_df, test_df, feature_cols)
    
    log("\n" + "=" * 70)
    log("PHASE 06 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase06_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()