"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 08c: Binning - Convert Regression to Classification
Framework: ML Model Lifecycle - 23 Main Stages
Target: Boiler_Eff_Class (3 classes: Low, Medium, High)
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    AdaBoostClassifier, ExtraTreesClassifier,
)
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
SPLITS_DIR = BASE_DIR / "data" / "splits"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_REG = "Boiler_Eff_"
TARGET_CLF_ANOMALY = "Anomaly_Label"
TARGET_CLF_BINNED = "Boiler_Eff_Class"

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# 8c.1 BINNING STRATEGY
# ============================================================
def stage_8c_1_binning_strategy():
    log("=" * 70)
    log("PHASE 08c: BINNING - CONVERT REGRESSION TO CLASSIFICATION")
    log("=" * 70)
    
    strategy = {
        "Original Target": TARGET_REG,
        "New Target": TARGET_CLF_BINNED,
        "Binning Method": "Quantile-based (33/34/33)",
        "Classes": {
            "0 (Low)": f"{TARGET_REG} < Q33",
            "1 (Medium)": f"Q33 <= {TARGET_REG} <= Q67",
            "2 (High)": f"{TARGET_REG} > Q67",
        },
        "Rationale": [
            "Regression failed (R2 negative)",
            "Target has narrow range (Std=0.05)",
            "Binning creates balanced classes",
            "Classification is more robust",
        ],
        "Leaky Features to Remove": [
            "turbine_to_boiler_eff (used target)",
        ],
    }
    
    for k, v in strategy.items():
        if isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                log(f"    {kk}: {vv}")
        elif isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        else:
            log(f"  {k}: {v}")
    
    return strategy

# ============================================================
# 8c.2 CREATE BINNED TARGET
# ============================================================
def stage_8c_2_create_binned_target():
    log("\n" + "=" * 70)
    log("STAGE 8c.2: CREATE BINNED TARGET")
    log("=" * 70)
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    # محاسبه Quantiles از Train (نه کل داده - جلوگیری از نشت)
    q33 = train_df[TARGET_REG].quantile(0.33)
    q67 = train_df[TARGET_REG].quantile(0.67)
    
    log(f"  Quantiles from Train:")
    log(f"    Q33 = {q33:.6f}")
    log(f"    Q67 = {q67:.6f}")
    log(f"    Min = {train_df[TARGET_REG].min():.6f}")
    log(f"    Max = {train_df[TARGET_REG].max():.6f}")
    
    def bin_target(df):
        df = df.copy()
        df[TARGET_CLF_BINNED] = pd.cut(
            df[TARGET_REG],
            bins=[-np.inf, q33, q67, np.inf],
            labels=[0, 1, 2],
        ).astype(int)
        return df
    
    train_df = bin_target(train_df)
    val_df = bin_target(val_df)
    test_df = bin_target(test_df)
    
    # توزیع کلاس
    log(f"\n  Class distribution:")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        dist = df[TARGET_CLF_BINNED].value_counts().sort_index()
        log(f"    {name}:")
        for cls, count in dist.items():
            pct = count / len(df) * 100
            log(f"      Class {cls}: {count} ({pct:.2f}%)")
    
    # ذخیره
    train_df.to_csv(SPLITS_DIR / "train_binned.csv", index=False)
    val_df.to_csv(SPLITS_DIR / "validation_binned.csv", index=False)
    test_df.to_csv(SPLITS_DIR / "test_binned.csv", index=False)
    
    log(f"\n  [OK] Saved: train_binned.csv")
    log(f"  [OK] Saved: validation_binned.csv")
    log(f"  [OK] Saved: test_binned.csv")
    
    # ذخیره Quantiles
    quantiles_df = pd.DataFrame({
        "Quantile": ["Q33", "Q67"],
        "Value": [q33, q67],
    })
    quantiles_df.to_csv(TABLES_DIR / "08c_2_quantiles.csv", index=False)
    log(f"  [OK] Saved: 08c_2_quantiles.csv")
    
    # نمودار
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, df) in enumerate([("Train", train_df), ("Val", val_df), ("Test", test_df)]):
        dist = df[TARGET_CLF_BINNED].value_counts().sort_index()
        axes[i].bar(dist.index, dist.values, color=["coral", "steelblue", "green"])
        axes[i].set_title(f"{name}: Class Distribution")
        axes[i].set_xlabel("Class")
        axes[i].set_ylabel("Count")
        axes[i].set_xticks([0, 1, 2])
        axes[i].set_xticklabels(["Low", "Medium", "High"])
        axes[i].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08c_2_binned_target.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 08c_2_binned_target.png")
    
    return train_df, val_df, test_df, q33, q67

# ============================================================
# 8c.3 MODEL COMPARISON
# ============================================================
def stage_8c_3_model_comparison(train_df, val_df):
    log("\n" + "=" * 70)
    log("STAGE 8c.3: MODEL COMPARISON FOR BINNED TARGET")
    log("=" * 70)
    
    # ویژگی‌ها (حذف نشت‌دهنده‌ها)
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    LEAKY = ["turbine_to_boiler_eff"]
    FEATURES = [c for c in feature_df["Feature"].tolist() 
                if c in train_df.columns and c not in LEAKY]
    
    log(f"  Features: {len(FEATURES)}")
    log(f"  Removed leaky: {LEAKY}")
    
    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df[TARGET_CLF_BINNED]
    X_val = val_df[FEATURES].fillna(0)
    y_val = val_df[TARGET_CLF_BINNED]
    
    models = {
        "Dummy_Stratified": DummyClassifier(strategy="stratified", random_state=42),
        "Logistic": LogisticRegression(max_iter=1000, random_state=42, multi_class="multinomial"),
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=15),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }
    
    results = []
    for name, model in models.items():
        log(f"\n  Training {name}...")
        start = time.time()
        try:
            pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start
            
            y_pred = pipeline.predict(X_val)
            
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
            f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)
            f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)
            
            results.append({
                "Model": name,
                "Train_Time_s": round(train_time, 3),
                "Accuracy": round(acc, 6),
                "Precision_Macro": round(prec, 6),
                "Recall_Macro": round(rec, 6),
                "F1_Macro": round(f1, 6),
                "F1_Weighted": round(f1_weighted, 6),
            })
            
            log(f"    Accuracy: {acc:.6f} | F1 Macro: {f1:.6f} | Time: {train_time:.2f}s")
        
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    results_df = pd.DataFrame(results).sort_values("F1_Macro", ascending=False)
    results_df.to_csv(TABLES_DIR / "08c_3_binned_comparison.csv", index=False)
    log(f"\n  [OK] Saved: 08c_3_binned_comparison.csv")
    
    log(f"\n  Binned Classification Ranking (by F1 Macro):")
    for _, row in results_df.iterrows():
        log(f"    {row['Model']}: F1={row['F1_Macro']:.6f}, Acc={row['Accuracy']:.6f}")
    
    return results_df, FEATURES

# ============================================================
# 8c.4 BEST MODEL DETAILED ANALYSIS
# ============================================================
def stage_8c_4_best_model_analysis(train_df, val_df, best_model_name):
    log("\n" + "=" * 70)
    log(f"STAGE 8c.4: BEST MODEL ANALYSIS ({best_model_name})")
    log("=" * 70)
    
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    LEAKY = ["turbine_to_boiler_eff"]
    FEATURES = [c for c in feature_df["Feature"].tolist() 
                if c in train_df.columns and c not in LEAKY]
    
    X_train = train_df[FEATURES].fillna(0)
    y_train = train_df[TARGET_CLF_BINNED]
    X_val = val_df[FEATURES].fillna(0)
    y_val = val_df[TARGET_CLF_BINNED]
    
    models = {
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=15),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    }
    
    model = models[best_model_name]
    pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)
    
    # Classification Report
    report = classification_report(y_val, y_pred, 
                                    target_names=["Low", "Medium", "High"],
                                    digits=4)
    log(f"\n  Classification Report:")
    for line in report.split("\n"):
        log(f"    {line}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    log(f"\n  Confusion Matrix:")
    log(f"    {cm}")
    
    # نمودار
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Low", "Medium", "High"],
                yticklabels=["Low", "Medium", "High"])
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")
    axes[0].set_title(f"Confusion Matrix: {best_model_name}")
    
    # Feature Importance (اگر درختی باشد)
    try:
        if hasattr(pipeline.named_steps["model"], "feature_importances_"):
            imp = pipeline.named_steps["model"].feature_importances_
            imp_df = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": imp,
            }).sort_values("Importance", ascending=False)
            imp_df.to_csv(TABLES_DIR / "08c_4_feature_importance.csv", index=False)
            
            top15 = imp_df.head(15)
            axes[1].barh(top15["Feature"][::-1], top15["Importance"][::-1], color="steelblue")
            axes[1].set_title(f"Top 15 Feature Importance")
            axes[1].set_xlabel("Importance")
            axes[1].grid(True, alpha=0.3)
            
            log(f"\n  Top 10 Features:")
            for _, row in imp_df.head(10).iterrows():
                log(f"    {row['Feature']}: {row['Importance']:.6f}")
    except Exception as e:
        log(f"  [WARNING] Feature importance failed: {e}")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08c_4_best_model_analysis.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 08c_4_best_model_analysis.png")
    
    return pipeline

# ============================================================
# 8c.5 SUMMARY & COMPARISON
# ============================================================
def stage_8c_5_summary(reg_results_old, clf_results_old, binned_results):
    log("\n" + "=" * 70)
    log("STAGE 8c.5: SUMMARY & COMPARISON")
    log("=" * 70)
    
    log(f"  Original Anomaly Classification (Phase 08b):")
    if not clf_results_old.empty:
        best_anomaly = clf_results_old.iloc[0]
        log(f"    Best Model: {best_anomaly['Model']}")
        log(f"    F1: {best_anomaly['F1']:.6f}")
        log(f"    ROC-AUC: {best_anomaly['ROC_AUC']:.6f}")
    
    log(f"\n  New Binned Classification (Phase 08c):")
    best_binned = binned_results.iloc[0]
    log(f"    Best Model: {best_binned['Model']}")
    log(f"    Accuracy: {best_binned['Accuracy']:.6f}")
    log(f"    F1 Macro: {best_binned['F1_Macro']:.6f}")
    log(f"    F1 Weighted: {best_binned['F1_Weighted']:.6f}")
    
    log(f"\n  Comparison:")
    log(f"    - Anomaly Classification: Binary, imbalanced (1.5%)")
    log(f"    - Binned Classification: 3-class, balanced (33/33/33)")
    log(f"    - Both can be used for different purposes")
    
    summary = {
        "Anomaly_Best_Model": clf_results_old.iloc[0]["Model"] if not clf_results_old.empty else "N/A",
        "Anomaly_F1": clf_results_old.iloc[0]["F1"] if not clf_results_old.empty else 0,
        "Binned_Best_Model": best_binned["Model"],
        "Binned_F1_Macro": best_binned["F1_Macro"],
        "Binned_Accuracy": best_binned["Accuracy"],
    }
    
    pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "08c_5_summary.csv", index=False
    )
    log(f"  [OK] Saved: 08c_5_summary.csv")
    
    return summary

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 08c: BINNING - CONVERT REGRESSION TO CLASSIFICATION")
    log("=" * 70)
    
    # 8c.1
    strategy = stage_8c_1_binning_strategy()
    
    # 8c.2
    train_df, val_df, test_df, q33, q67 = stage_8c_2_create_binned_target()
    
    # 8c.3
    binned_results, FEATURES = stage_8c_3_model_comparison(train_df, val_df)
    
    # 8c.4
    best_model_name = binned_results.iloc[0]["Model"]
    pipeline = stage_8c_4_best_model_analysis(train_df, val_df, best_model_name)
    
    # 8c.5
    clf_results_old = pd.read_csv(TABLES_DIR / "08b_classification_comparison.csv")
    reg_results_old = pd.DataFrame()
    summary = stage_8c_5_summary(reg_results_old, clf_results_old, binned_results)
    
    log("\n" + "=" * 70)
    log("PHASE 08c COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase08c_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()