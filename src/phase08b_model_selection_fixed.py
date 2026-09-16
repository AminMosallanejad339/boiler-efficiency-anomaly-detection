"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 08b: Model Selection - Fixed (No Data Leakage)
Framework: ML Model Lifecycle - 23 Main Stages
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    AdaBoostRegressor, AdaBoostClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
)
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")

BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
SPLITS_DIR = BASE_DIR / "data" / "splits"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

TARGET_REG = "Boiler_Eff_"
TARGET_CLF = "Anomaly_Label"

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

log("=" * 70)
log("PHASE 08b: MODEL SELECTION - FIXED (No Data Leakage)")
log("=" * 70)

train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")

# حذف ویژگی‌های نشت‌دهنده
LEAKY_FEATURES_REG = ["turbine_to_boiler_eff"]
LEAKY_FEATURES_CLF = ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]

FEATURES_REG = [c for c in feature_df["Feature"].tolist() 
                if c in train_df.columns and c not in LEAKY_FEATURES_REG]
FEATURES_CLF = [c for c in feature_df["Feature"].tolist() 
                if c in train_df.columns and c not in LEAKY_FEATURES_CLF]

log(f"Original features: {len(feature_df)}")
log(f"Features for Regression (after removing leaky): {len(FEATURES_REG)}")
log(f"Features for Classification (after removing leaky): {len(FEATURES_CLF)}")
log(f"Removed for Regression: {LEAKY_FEATURES_REG}")
log(f"Removed for Classification: {LEAKY_FEATURES_CLF}")

# Regression Models
reg_models = {
    "Dummy_Mean": DummyRegressor(strategy="mean"),
    "Linear": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.001),
    "DecisionTree": DecisionTreeRegressor(random_state=42, max_depth=10),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
}

# Classification Models
clf_models = {
    "Dummy_MostFrequent": DummyClassifier(strategy="most_frequent"),
    "Logistic": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
    "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
    "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
    "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
}

def run_regression():
    log("\n" + "=" * 70)
    log("REGRESSION MODEL COMPARISON (No Leakage)")
    log("=" * 70)
    
    X_train = train_df[FEATURES_REG].fillna(0)
    y_train = train_df[TARGET_REG]
    X_val = val_df[FEATURES_REG].fillna(0)
    y_val = val_df[TARGET_REG]
    
    results = []
    for name, model in reg_models.items():
        log(f"\n  Training {name}...")
        start = time.time()
        try:
            pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start
            
            y_pred_train = pipeline.predict(X_train)
            y_pred_val = pipeline.predict(X_val)
            
            rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
            mae_val = mean_absolute_error(y_val, y_pred_val)
            r2_train = r2_score(y_train, y_pred_train)
            r2_val = r2_score(y_val, y_pred_val)
            
            results.append({
                "Model": name,
                "Train_Time_s": round(train_time, 3),
                "RMSE_Val": round(rmse_val, 6),
                "MAE_Val": round(mae_val, 6),
                "R2_Train": round(r2_train, 6),
                "R2_Val": round(r2_val, 6),
                "Overfit": round(r2_train - r2_val, 6),
            })
            log(f"    R2 Val: {r2_val:.6f} | RMSE: {rmse_val:.6f} | Time: {train_time:.2f}s")
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    results_df = pd.DataFrame(results).sort_values("R2_Val", ascending=False)
    results_df.to_csv(TABLES_DIR / "08b_regression_comparison.csv", index=False)
    
    log(f"\n  Regression Ranking (by R2 Val):")
    for _, row in results_df.iterrows():
        log(f"    {row['Model']}: R2={row['R2_Val']:.6f}, RMSE={row['RMSE_Val']:.6f}, Overfit={row['Overfit']:.6f}")
    
    return results_df

def run_classification():
    log("\n" + "=" * 70)
    log("CLASSIFICATION MODEL COMPARISON (No Leakage)")
    log("=" * 70)
    
    X_train = train_df[FEATURES_CLF].fillna(0)
    y_train = train_df[TARGET_CLF]
    X_val = val_df[FEATURES_CLF].fillna(0)
    y_val = val_df[TARGET_CLF]
    
    results = []
    for name, model in clf_models.items():
        log(f"\n  Training {name}...")
        start = time.time()
        try:
            pipeline = Pipeline([("scaler", RobustScaler()), ("model", model)])
            pipeline.fit(X_train, y_train)
            train_time = time.time() - start
            
            y_pred = pipeline.predict(X_val)
            try:
                y_proba = pipeline.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val, y_proba)
                pr_auc = average_precision_score(y_val, y_proba)
            except Exception:
                roc_auc, pr_auc = np.nan, np.nan
            
            acc = accuracy_score(y_val, y_pred)
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            results.append({
                "Model": name,
                "Train_Time_s": round(train_time, 3),
                "Accuracy": round(acc, 6),
                "Precision": round(prec, 6),
                "Recall": round(rec, 6),
                "F1": round(f1, 6),
                "ROC_AUC": round(roc_auc, 6) if not np.isnan(roc_auc) else np.nan,
                "PR_AUC": round(pr_auc, 6) if not np.isnan(pr_auc) else np.nan,
            })
            log(f"    F1: {f1:.6f} | Recall: {rec:.6f} | ROC-AUC: {roc_auc:.6f} | Time: {train_time:.2f}s")
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(TABLES_DIR / "08b_classification_comparison.csv", index=False)
    
    log(f"\n  Classification Ranking (by F1):")
    for _, row in results_df.iterrows():
        log(f"    {row['Model']}: F1={row['F1']:.6f}, Recall={row['Recall']:.6f}, ROC-AUC={row['ROC_AUC']:.6f}")
    
    return results_df

def visualize(reg_results, clf_results):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    reg_sorted = reg_results.dropna(subset=["R2_Val"]).sort_values("R2_Val", ascending=True)
    axes[0, 0].barh(reg_sorted["Model"], reg_sorted["R2_Val"], color="steelblue")
    axes[0, 0].set_title("Regression: R2 Validation (No Leakage)", fontsize=12)
    axes[0, 0].set_xlabel("R2")
    axes[0, 0].grid(True, alpha=0.3)
    
    reg_sorted2 = reg_results.dropna(subset=["RMSE_Val"]).sort_values("RMSE_Val", ascending=False)
    axes[0, 1].barh(reg_sorted2["Model"], reg_sorted2["RMSE_Val"], color="coral")
    axes[0, 1].set_title("Regression: RMSE Validation (No Leakage)", fontsize=12)
    axes[0, 1].set_xlabel("RMSE")
    axes[0, 1].grid(True, alpha=0.3)
    
    clf_sorted = clf_results.dropna(subset=["F1"]).sort_values("F1", ascending=True)
    axes[1, 0].barh(clf_sorted["Model"], clf_sorted["F1"], color="green")
    axes[1, 0].set_title("Classification: F1 Score (No Leakage)", fontsize=12)
    axes[1, 0].set_xlabel("F1 Score")
    axes[1, 0].grid(True, alpha=0.3)
    
    clf_sorted2 = clf_results.dropna(subset=["ROC_AUC"]).sort_values("ROC_AUC", ascending=True)
    axes[1, 1].barh(clf_sorted2["Model"], clf_sorted2["ROC_AUC"], color="purple")
    axes[1, 1].set_title("Classification: ROC-AUC (No Leakage)", fontsize=12)
    axes[1, 1].set_xlabel("ROC-AUC")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08b_model_comparison_fixed.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"\n[OK] Saved: 08b_model_comparison_fixed.png")

log("=" * 70)
log("PHASE 08b: MODEL SELECTION - FIXED")
log("=" * 70)

reg_results = run_regression()
clf_results = run_classification()
visualize(reg_results, clf_results)

log("\n" + "=" * 70)
log("PHASE 08b COMPLETE!")
log("=" * 70)

LOG_FILE = BASE_DIR / "reports" / "phase08b_log.txt"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"[OK] Log saved: {LOG_FILE}")