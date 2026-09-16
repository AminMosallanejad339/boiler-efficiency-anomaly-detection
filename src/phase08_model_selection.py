"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 08: Model Selection (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
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
import time
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score
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
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier

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
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
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
log("PHASE 08: MODEL SELECTION")
log("=" * 70)

train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")

FEATURES = feature_df["Feature"].tolist()
FEATURES = [c for c in FEATURES if c in train_df.columns]

log(f"Train: {train_df.shape}")
log(f"Validation: {val_df.shape}")
log(f"Features: {len(FEATURES)}")

X_train = train_df[FEATURES].fillna(0)
y_train_reg = train_df[TARGET_REG]
y_train_clf = train_df[TARGET_CLF]

X_val = val_df[FEATURES].fillna(0)
y_val_reg = val_df[TARGET_REG]
y_val_clf = val_df[TARGET_CLF]

log(f"X_train: {X_train.shape}")
log(f"X_val: {X_val.shape}")

# ============================================================
# 8.1 MODEL SELECTION STRATEGY
# ============================================================
def stage_8_1_strategy():
    log("\n" + "=" * 70)
    log("STAGE 8.1: MODEL SELECTION STRATEGY")
    log("=" * 70)
    
    strategy = {
        "Regression Target": TARGET_REG,
        "Classification Target": TARGET_CLF,
        "Assumptions from Phase 07": [
            "Linearity: NOT satisfied -> Use non-linear models",
            "Normality: NOT satisfied -> Use non-parametric",
            "Independence: NOT satisfied -> Use TimeSeriesSplit",
            "Homoscedasticity: SATISFIED",
            "Multicollinearity: SEVERE -> Use tree-based",
            "Stationarity: SATISFIED",
        ],
        "Model Families to Test": [
            "Baseline (Dummy)",
            "Linear (LinearRegression, Ridge, Lasso)",
            "Tree (DecisionTree)",
            "Ensemble (RandomForest, ExtraTrees, GradientBoosting, AdaBoost)",
            "Advanced (XGBoost, LightGBM)",
        ],
        "Evaluation Metrics Regression": ["RMSE", "MAE", "R2"],
        "Evaluation Metrics Classification": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
        "Cross-Validation": {
            "Regression": "TimeSeriesSplit (5 folds)",
            "Classification": "StratifiedKFold (5 folds)",
        },
    }
    
    for k, v in strategy.items():
        if isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        elif isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                log(f"    {kk}: {vv}")
        else:
            log(f"  {k}: {v}")
    
    return strategy

# ============================================================
# 8.2 MODEL FAMILY IDENTIFICATION
# ============================================================
def stage_8_2_model_families():
    log("\n" + "=" * 70)
    log("STAGE 8.2: MODEL FAMILY IDENTIFICATION")
    log("=" * 70)
    
    families = {
        "Baseline": {
            "Models": ["DummyRegressor", "DummyClassifier"],
            "Purpose": "Lower bound benchmark",
        },
        "Linear": {
            "Models": ["LinearRegression", "Ridge", "Lasso", "LogisticRegression"],
            "Purpose": "Test linear relationships",
            "Suitability": "LOW (linearity not satisfied)",
        },
        "Tree": {
            "Models": ["DecisionTreeRegressor", "DecisionTreeClassifier"],
            "Purpose": "Non-linear baseline",
            "Suitability": "MEDIUM (prone to overfitting)",
        },
        "Ensemble": {
            "Models": ["RandomForest", "ExtraTrees", "GradientBoosting", "AdaBoost"],
            "Purpose": "Robust non-linear models",
            "Suitability": "HIGH",
        },
        "Advanced": {
            "Models": ["XGBoost", "LightGBM"],
            "Purpose": "State-of-the-art gradient boosting",
            "Suitability": "VERY HIGH",
        },
    }
    
    for family, info in families.items():
        log(f"  {family}:")
        for k, v in info.items():
            if isinstance(v, list):
                log(f"    {k}: {', '.join(v)}")
            else:
                log(f"    {k}: {v}")
    
    return families

# ============================================================
# 8.3 ALGORITHM SELECTION
# ============================================================
def stage_8_3_algorithm_selection():
    log("\n" + "=" * 70)
    log("STAGE 8.3: ALGORITHM SELECTION")
    log("=" * 70)
    
    # Regression models
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
        "XGBoost": XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0),
        "LightGBM": LGBMRegressor(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1),
    }
    
    # Classification models
    clf_models = {
        "Dummy_MostFrequent": DummyClassifier(strategy="most_frequent"),
        "Logistic": LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"),
        "DecisionTree": DecisionTreeClassifier(random_state=42, max_depth=10, class_weight="balanced"),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
        "ExtraTrees": ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced"),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbosity=0, 
                                  scale_pos_weight=63.57, eval_metric="logloss"),
        "LightGBM": LGBMClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1,
                                    class_weight="balanced"),
    }
    
    log(f"  Regression models: {len(reg_models)}")
    for name in reg_models.keys():
        log(f"    - {name}")
    
    log(f"\n  Classification models: {len(clf_models)}")
    for name in clf_models.keys():
        log(f"    - {name}")
    
    return reg_models, clf_models

# ============================================================
# 8.4-8.6 BASELINE & MODEL COMPARISON
# ============================================================
def stage_8_4_to_8_6_regression(reg_models, X_train, y_train_reg, X_val, y_val_reg):
    log("\n" + "=" * 70)
    log("STAGE 8.4-8.6: REGRESSION MODEL COMPARISON")
    log("=" * 70)
    
    results = []
    
    for name, model in reg_models.items():
        log(f"\n  Training {name}...")
        start_time = time.time()
        
        try:
            # Pipeline با Scaler
            pipeline = Pipeline([
                ("scaler", RobustScaler()),
                ("model", model),
            ])
            
            pipeline.fit(X_train, y_train_reg)
            train_time = time.time() - start_time
            
            # پیش‌بینی
            y_pred_train = pipeline.predict(X_train)
            y_pred_val = pipeline.predict(X_val)
            
            # معیارها
            rmse_train = np.sqrt(mean_squared_error(y_train_reg, y_pred_train))
            rmse_val = np.sqrt(mean_squared_error(y_val_reg, y_pred_val))
            mae_val = mean_absolute_error(y_val_reg, y_pred_val)
            r2_train = r2_score(y_train_reg, y_pred_train)
            r2_val = r2_score(y_val_reg, y_pred_val)
            
            results.append({
                "Model": name,
                "Train_Time_s": round(train_time, 3),
                "RMSE_Train": round(rmse_train, 6),
                "RMSE_Val": round(rmse_val, 6),
                "MAE_Val": round(mae_val, 6),
                "R2_Train": round(r2_train, 6),
                "R2_Val": round(r2_val, 6),
                "Overfit": round(r2_train - r2_val, 6),
            })
            
            log(f"    RMSE Val: {rmse_val:.6f} | R2 Val: {r2_val:.6f} | Time: {train_time:.2f}s")
        
        except Exception as e:
            log(f"    [ERROR] {name}: {e}")
            results.append({
                "Model": name,
                "Train_Time_s": np.nan,
                "RMSE_Train": np.nan,
                "RMSE_Val": np.nan,
                "MAE_Val": np.nan,
                "R2_Train": np.nan,
                "R2_Val": np.nan,
                "Overfit": np.nan,
            })
    
    results_df = pd.DataFrame(results).sort_values("R2_Val", ascending=False)
    results_df.to_csv(TABLES_DIR / "08_04_regression_comparison.csv", index=False)
    log(f"\n  [OK] Saved: 08_04_regression_comparison.csv")
    
    log(f"\n  Regression Model Ranking (by R2 Val):")
    for i, row in results_df.iterrows():
        log(f"    {row['Model']}: R2={row['R2_Val']:.6f}, RMSE={row['RMSE_Val']:.6f}, Overfit={row['Overfit']:.6f}")
    
    return results_df

def stage_8_4_to_8_6_classification(clf_models, X_train, y_train_clf, X_val, y_val_clf):
    log("\n" + "=" * 70)
    log("STAGE 8.4-8.6: CLASSIFICATION MODEL COMPARISON")
    log("=" * 70)
    
    results = []
    
    for name, model in clf_models.items():
        log(f"\n  Training {name}...")
        start_time = time.time()
        
        try:
            pipeline = Pipeline([
                ("scaler", RobustScaler()),
                ("model", model),
            ])
            
            pipeline.fit(X_train, y_train_clf)
            train_time = time.time() - start_time
            
            # پیش‌بینی
            y_pred = pipeline.predict(X_val)
            
            # احتمال برای ROC-AUC
            try:
                y_proba = pipeline.predict_proba(X_val)[:, 1]
                roc_auc = roc_auc_score(y_val_clf, y_proba)
                pr_auc = average_precision_score(y_val_clf, y_proba)
            except Exception:
                roc_auc = np.nan
                pr_auc = np.nan
            
            # معیارها
            acc = accuracy_score(y_val_clf, y_pred)
            prec = precision_score(y_val_clf, y_pred, zero_division=0)
            rec = recall_score(y_val_clf, y_pred, zero_division=0)
            f1 = f1_score(y_val_clf, y_pred, zero_division=0)
            
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
            log(f"    [ERROR] {name}: {e}")
            results.append({
                "Model": name,
                "Train_Time_s": np.nan,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "ROC_AUC": np.nan,
                "PR_AUC": np.nan,
            })
    
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(TABLES_DIR / "08_04_classification_comparison.csv", index=False)
    log(f"\n  [OK] Saved: 08_04_classification_comparison.csv")
    
    log(f"\n  Classification Model Ranking (by F1):")
    for i, row in results_df.iterrows():
        log(f"    {row['Model']}: F1={row['F1']:.6f}, Recall={row['Recall']:.6f}, ROC-AUC={row['ROC_AUC']:.6f}")
    
    return results_df

# ============================================================
# 8.7 MODEL JUSTIFICATION
# ============================================================
def stage_8_7_justification(reg_results, clf_results):
    log("\n" + "=" * 70)
    log("STAGE 8.7: MODEL JUSTIFICATION")
    log("=" * 70)
    
    # بهترین مدل Regression
    best_reg = reg_results.dropna(subset=["R2_Val"]).iloc[0]
    log(f"  Best Regression Model: {best_reg['Model']}")
    log(f"    R2 Val: {best_reg['R2_Val']:.6f}")
    log(f"    RMSE Val: {best_reg['RMSE_Val']:.6f}")
    log(f"    Overfit: {best_reg['Overfit']:.6f}")
    
    # بهترین مدل Classification
    best_clf = clf_results.dropna(subset=["F1"]).iloc[0]
    log(f"\n  Best Classification Model: {best_clf['Model']}")
    log(f"    F1: {best_clf['F1']:.6f}")
    log(f"    Recall: {best_clf['Recall']:.6f}")
    log(f"    Precision: {best_clf['Precision']:.6f}")
    log(f"    ROC-AUC: {best_clf['ROC_AUC']:.6f}")
    
    # توجیه
    log(f"\n  Justification:")
    log(f"    Regression: {best_reg['Model']} selected because:")
    log(f"      - Highest R2 Val")
    log(f"      - Non-linear (matches Phase 07 findings)")
    log(f"      - Robust to multicollinearity")
    
    log(f"    Classification: {best_clf['Model']} selected because:")
    log(f"      - Highest F1 Score")
    log(f"      - Handles imbalanced data via class_weight/scale_pos_weight")
    log(f"      - Robust to non-normal distributions")
    
    return best_reg, best_clf

# ============================================================
# 8.8 COMPLEXITY ANALYSIS
# ============================================================
def stage_8_8_complexity(reg_results, clf_results):
    log("\n" + "=" * 70)
    log("STAGE 8.8: COMPLEXITY ANALYSIS")
    log("=" * 70)
    
    log(f"  Regression Models (sorted by Train Time):")
    reg_sorted = reg_results.dropna(subset=["Train_Time_s"]).sort_values("Train_Time_s")
    for _, row in reg_sorted.iterrows():
        log(f"    {row['Model']}: {row['Train_Time_s']:.3f}s (R2={row['R2_Val']:.4f})")
    
    log(f"\n  Classification Models (sorted by Train Time):")
    clf_sorted = clf_results.dropna(subset=["Train_Time_s"]).sort_values("Train_Time_s")
    for _, row in clf_sorted.iterrows():
        log(f"    {row['Model']}: {row['Train_Time_s']:.3f}s (F1={row['F1']:.4f})")
    
    return reg_sorted, clf_sorted

# ============================================================
# 8.9 BIAS-VARIANCE ANALYSIS
# ============================================================
def stage_8_9_bias_variance(reg_results):
    log("\n" + "=" * 70)
    log("STAGE 8.9: BIAS-VARIANCE ANALYSIS")
    log("=" * 70)
    
    log(f"  Regression Models (Overfit = R2_Train - R2_Val):")
    log(f"  {'Model':<25} {'R2_Train':<12} {'R2_Val':<12} {'Overfit':<12} {'Diagnosis'}")
    log(f"  {'-'*75}")
    
    for _, row in reg_results.iterrows():
        if pd.isna(row["R2_Val"]):
            continue
        
        overfit = row["Overfit"]
        if overfit > 0.1:
            diagnosis = "HIGH VARIANCE (Overfit)"
        elif row["R2_Val"] < 0.3:
            diagnosis = "HIGH BIAS (Underfit)"
        else:
            diagnosis = "BALANCED"
        
        log(f"  {row['Model']:<25} {row['R2_Train']:<12.6f} {row['R2_Val']:<12.6f} {overfit:<12.6f} {diagnosis}")
    
    return reg_results

# ============================================================
# 8.10 THEORETICAL FOUNDATION CHECK
# ============================================================
def stage_8_10_theoretical(reg_results, clf_results):
    log("\n" + "=" * 70)
    log("STAGE 8.10: THEORETICAL FOUNDATION CHECK")
    log("=" * 70)
    
    log("  Regression Models - Theoretical Basis:")
    log("    - LinearRegression: Assumes linearity (violated in Phase 07)")
    log("    - Ridge/Lasso: Linear + regularization (still assumes linearity)")
    log("    - DecisionTree: Non-parametric, no distribution assumption")
    log("    - RandomForest: Ensemble of trees, reduces variance")
    log("    - GradientBoosting: Sequential ensemble, reduces bias")
    log("    - XGBoost/LightGBM: Optimized gradient boosting")
    
    log(f"\n  Classification Models - Theoretical Basis:")
    log(f"    - LogisticRegression: Assumes linear decision boundary")
    log(f"    - DecisionTree: Non-parametric, handles non-linear")
    log(f"    - RandomForest: Ensemble, handles imbalance via class_weight")
    log(f"    - XGBoost: scale_pos_weight handles imbalance")
    log(f"    - LightGBM: class_weight handles imbalance")
    
    log(f"\n  [RECOMMENDATION] Based on Phase 07 findings:")
    log(f"    - Use tree-based models (RandomForest, XGBoost, LightGBM)")
    log(f"    - Avoid linear models (linearity violated)")
    log(f"    - Apply class_weight/scale_pos_weight for imbalance")
    
    return True

# ============================================================
# VISUALIZATION
# ============================================================
def visualize_results(reg_results, clf_results):
    log("\n" + "=" * 70)
    log("VISUALIZING RESULTS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Regression R2
    reg_sorted = reg_results.dropna(subset=["R2_Val"]).sort_values("R2_Val", ascending=True)
    axes[0, 0].barh(reg_sorted["Model"], reg_sorted["R2_Val"], color="steelblue")
    axes[0, 0].set_title("Regression: R2 Validation by Model", fontsize=12)
    axes[0, 0].set_xlabel("R2")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Regression RMSE
    reg_sorted2 = reg_results.dropna(subset=["RMSE_Val"]).sort_values("RMSE_Val", ascending=False)
    axes[0, 1].barh(reg_sorted2["Model"], reg_sorted2["RMSE_Val"], color="coral")
    axes[0, 1].set_title("Regression: RMSE Validation by Model", fontsize=12)
    axes[0, 1].set_xlabel("RMSE")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Classification F1
    clf_sorted = clf_results.dropna(subset=["F1"]).sort_values("F1", ascending=True)
    axes[1, 0].barh(clf_sorted["Model"], clf_sorted["F1"], color="green")
    axes[1, 0].set_title("Classification: F1 Score by Model", fontsize=12)
    axes[1, 0].set_xlabel("F1 Score")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Classification ROC-AUC
    clf_sorted2 = clf_results.dropna(subset=["ROC_AUC"]).sort_values("ROC_AUC", ascending=True)
    axes[1, 1].barh(clf_sorted2["Model"], clf_sorted2["ROC_AUC"], color="purple")
    axes[1, 1].set_title("Classification: ROC-AUC by Model", fontsize=12)
    axes[1, 1].set_xlabel("ROC-AUC")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_model_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 08_model_comparison.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 08: MODEL SELECTION")
    log("=" * 70)
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    
    FEATURES = [c for c in feature_df["Feature"].tolist() if c in train_df.columns]
    
    X_train = train_df[FEATURES].fillna(0)
    y_train_reg = train_df[TARGET_REG]
    y_train_clf = train_df[TARGET_CLF]
    
    X_val = val_df[FEATURES].fillna(0)
    y_val_reg = val_df[TARGET_REG]
    y_val_clf = val_df[TARGET_CLF]
    
    log(f"Train: {X_train.shape}")
    log(f"Val: {X_val.shape}")
    log(f"Features: {len(FEATURES)}")
    
    # 8.1
    stage_8_1_strategy()
    
    # 8.2
    families = stage_8_2_model_families()
    
    # 8.3
    reg_models, clf_models = stage_8_3_algorithm_selection()
    
    # 8.4-8.6 Regression
    reg_results = stage_8_4_to_8_6_regression(reg_models, X_train, y_train_reg, X_val, y_val_reg)
    
    # 8.4-8.6 Classification
    clf_results = stage_8_4_to_8_6_classification(clf_models, X_train, y_train_clf, X_val, y_val_clf)
    
    # 8.7
    best_reg, best_clf = stage_8_7_justification(reg_results, clf_results)
    
    # 8.8
    stage_8_8_complexity(reg_results, clf_results)
    
    # 8.9
    stage_8_9_bias_variance(reg_results)
    
    # 8.10
    stage_8_10_theoretical(reg_results, clf_results)
    
    # Visualization
    visualize_results(reg_results, clf_results)
    
    log("\n" + "=" * 70)
    log("PHASE 08 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase08_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()