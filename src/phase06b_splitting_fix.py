"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 06b: Data Splitting Fix - Preserve Temporal Order
Framework: ML Model Lifecycle - 23 Main Stages
Strategy: Time-Based Split + Stratified Resampling (within each set)
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
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

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
log("PHASE 06b: DATA SPLITTING FIX (Temporal Order Preserved)")
log("=" * 70)

df = pd.read_csv(PROCESSED)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)
log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Time range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# ============================================================
# STRATIFIED RESAMPLING WITHIN TIME-BASED SPLIT
# ============================================================
def stratified_resample_within_time(df_subset, target_ratio, random_state=42):
    """
    نمونه‌برداری Stratified درون یک مجموعه زمانی.
    ترتیب زمانی حفظ نمی‌شود (چون نمونه‌برداری تصادفی است)،
    اما مجموعه همچنان فقط شامل داده‌های همان بازه زمانی است.
    """
    df_subset = df_subset.reset_index(drop=True)
    
    # جدا کردن کلاس‌ها
    df_normal = df_subset[df_subset[TARGET_CLF] == 0]
    df_anomaly = df_subset[df_subset[TARGET_CLF] == 1]
    
    n_total = len(df_subset)
    n_anomaly_target = int(n_total * target_ratio)
    n_normal_target = n_total - n_anomaly_target
    
    # نمونه‌برداری
    if len(df_anomaly) >= n_anomaly_target:
        df_anomaly_sampled = df_anomaly.sample(n=n_anomaly_target, random_state=random_state)
    else:
        df_anomaly_sampled = df_anomaly  # همه
    
    if len(df_normal) >= n_normal_target:
        df_normal_sampled = df_normal.sample(n=n_normal_target, random_state=random_state)
    else:
        df_normal_sampled = df_normal
    
    # ادغام و مرتب‌سازی بر اساس زمان
    df_result = pd.concat([df_normal_sampled, df_anomaly_sampled])
    df_result = df_result.sort_values("Timestamp").reset_index(drop=True)
    
    return df_result

# ============================================================
# MAIN SPLITTING
# ============================================================
def time_based_split_preserved(df):
    log("\n" + "=" * 70)
    log("TIME-BASED SPLIT (Temporal Order Preserved)")
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
    
    log(f"\n  Anomaly distribution (BEFORE resampling):")
    log(f"    Train: {train_df[TARGET_CLF].sum()} ({train_df[TARGET_CLF].mean()*100:.4f}%)")
    log(f"    Val:   {val_df[TARGET_CLF].sum()} ({val_df[TARGET_CLF].mean()*100:.4f}%)")
    log(f"    Test:  {test_df[TARGET_CLF].sum()} ({test_df[TARGET_CLF].mean()*100:.4f}%)")
    
    return train_df, val_df, test_df

def stratified_resampling_if_needed(train_df, val_df, test_df, df):
    log("\n" + "=" * 70)
    log("STRATIFIED RESAMPLING (Within Time-Based Split)")
    log("=" * 70)
    
    overall_ratio = df[TARGET_CLF].mean()
    log(f"  Overall anomaly rate: {overall_ratio*100:.4f}%")
    
    results = {}
    for name, subset in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        current_ratio = subset[TARGET_CLF].mean()
        deviation = abs(current_ratio - overall_ratio) / overall_ratio * 100
        
        log(f"\n  {name}:")
        log(f"    Current ratio: {current_ratio*100:.4f}%")
        log(f"    Deviation: {deviation:.2f}%")
        
        if deviation > 20:
            log(f"    [ACTION] Deviation > 20% -> Applying stratified resampling")
            resampled = stratified_resample_within_time(subset, overall_ratio)
            new_ratio = resampled[TARGET_CLF].mean()
            log(f"    [OK] New ratio: {new_ratio*100:.4f}%")
            log(f"    [OK] New size: {len(resampled)}")
            results[name] = resampled
        else:
            log(f"    [OK] Deviation < 20% -> Keeping as-is")
            results[name] = subset
    
    return results["Train"], results["Val"], results["Test"]

def verify_temporal_order(train_df, val_df, test_df):
    log("\n" + "=" * 70)
    log("VERIFY TEMPORAL ORDER")
    log("=" * 70)
    
    train_max = train_df["Timestamp"].max()
    val_min = val_df["Timestamp"].min()
    val_max = val_df["Timestamp"].max()
    test_min = test_df["Timestamp"].min()
    
    log(f"  Train max: {train_max}")
    log(f"  Val min:   {val_min}")
    log(f"  Val max:   {val_max}")
    log(f"  Test min:  {test_min}")
    
    if train_max < val_min:
        log(f"  [OK] No temporal overlap between Train and Val")
    else:
        log(f"  [WARNING] Temporal overlap detected!")
    
    if val_max < test_min:
        log(f"  [OK] No temporal overlap between Val and Test")
    else:
        log(f"  [WARNING] Temporal overlap detected!")
    
    return True

# ============================================================
# PIPELINE & SAVE
# ============================================================
def create_pipelines():
    log("\n" + "=" * 70)
    log("PIPELINE CONSTRUCTION")
    log("=" * 70)
    
    pipeline_reg = Pipeline([("scaler", RobustScaler())])
    pipeline_clf = Pipeline([("scaler", RobustScaler())])
    
    joblib.dump(pipeline_reg, MODELS_DIR / "pipeline_regression.pkl")
    joblib.dump(pipeline_clf, MODELS_DIR / "pipeline_classification.pkl")
    
    log(f"  [OK] Regression Pipeline saved")
    log(f"  [OK] Classification Pipeline saved")
    log(f"  [NOTE] SMOTE will be applied in Phase 08 via Class Weight")
    
    return pipeline_reg, pipeline_clf

def save_splits(train_df, val_df, test_df, feature_cols, df):
    log("\n" + "=" * 70)
    log("SAVING SPLITS")
    log("=" * 70)
    
    train_df.to_csv(SPLITS_DIR / "train.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "validation.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test.csv", index=False)
    
    log(f"  [OK] train.csv ({len(train_df)} rows)")
    log(f"  [OK] validation.csv ({len(val_df)} rows)")
    log(f"  [OK] test.csv ({len(test_df)} rows)")
    
    pd.DataFrame({"Feature": feature_cols}).to_csv(
        SPLITS_DIR / "feature_columns.csv", index=False
    )
    log(f"  [OK] feature_columns.csv ({len(feature_cols)} features)")
    
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
        "Train_Time_Start": str(train_df["Timestamp"].min()),
        "Train_Time_End": str(train_df["Timestamp"].max()),
        "Val_Time_Start": str(val_df["Timestamp"].min()),
        "Val_Time_End": str(val_df["Timestamp"].max()),
        "Test_Time_Start": str(test_df["Timestamp"].min()),
        "Test_Time_End": str(test_df["Timestamp"].max()),
    }
    
    pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "06_10_split_summary.csv", index=False
    )
    log(f"  [OK] 06_10_split_summary.csv")
    
    log(f"\n  Final Split Summary:")
    for k, v in summary.items():
        log(f"    {k}: {v}")
    
    # نمودار
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    
    sizes = [len(train_df), len(val_df), len(test_df)]
    labels = ["Train (70%)", "Validation (15%)", "Test (15%)"]
    colors = ["steelblue", "coral", "green"]
    axes[0].pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    axes[0].set_title("Data Split Distribution")
    
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
    log(f"  [OK] 06_10_split_distribution.png")
    
    return summary

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 06b: DATA SPLITTING FIX")
    log("=" * 70)
    
    # 1. بارگذاری
    df = pd.read_csv(PROCESSED)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)
    
    # 2. Time-Based Split (حفظ ترتیب زمانی)
    train_df, val_df, test_df = time_based_split_preserved(df)
    
    # 3. Stratified Resampling (درون هر مجموعه، فقط اگر نیاز باشد)
    train_df, val_df, test_df = stratified_resampling_if_needed(train_df, val_df, test_df, df)
    
    # 4. بررسی ترتیب زمانی
    verify_temporal_order(train_df, val_df, test_df)
    
    # 5. CV Setup
    log("\n" + "=" * 70)
    log("CROSS-VALIDATION SETUP")
    log("=" * 70)
    log(f"  Regression: TimeSeriesSplit (5 folds)")
    log(f"  Classification: StratifiedKFold (5 folds)")
    
    # 6. Pipeline
    pipeline_reg, pipeline_clf = create_pipelines()
    
    # 7. ویژگی‌ها
    feature_cols = [c for c in df.columns 
                    if c not in [TARGET_REG, TARGET_CLF, "Timestamp"]
                    and df[c].dtype.kind in "biufc"]
    
    # 8. ذخیره
    save_splits(train_df, val_df, test_df, feature_cols, df)
    
    log("\n" + "=" * 70)
    log("PHASE 06 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase06b_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()