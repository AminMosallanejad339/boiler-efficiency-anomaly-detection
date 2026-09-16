"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 22: Model Comparison & Final Selection (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Final Model Selection
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import time
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest, RandomForestClassifier, AdaBoostClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score,
)

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
SPLITS_DIR = BASE_DIR / "data" / "splits"
RAW_DATA = BASE_DIR / "data" / "raw" / "industrial_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_ANOMALY = "Anomaly_Label"
SEED = 42
np.random.seed(SEED)

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# LOAD DATA
# ============================================================
def load_data():
    log("=" * 70)
    log("PHASE 22: MODEL COMPARISON & FINAL SELECTION")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("LOADING DATA")
    log("=" * 70)
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    
    df_raw = pd.read_csv(RAW_DATA)
    df_raw.columns = (
        df_raw.columns.astype(str)
        .str.strip()
        .str.replace(r"[^\w\s]", "", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    df_raw["Timestamp"] = pd.to_datetime(df_raw["Timestamp"])
    
    anomaly_features_raw = [
        "APH_Leakage_", "CO_mgm3", "Dust_mgm3",
        "Reheater_desuperheating_water_flow_th",
    ]
    available_raw = [c for c in anomaly_features_raw if c in df_raw.columns]
    
    raw_features = df_raw[["Timestamp"] + available_raw].copy()
    raw_features = raw_features.rename(columns={c: f"{c}_raw" for c in available_raw})
    
    for df in [train_df, val_df, test_df]:
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    
    train_df = train_df.merge(raw_features, on="Timestamp", how="left")
    val_df = val_df.merge(raw_features, on="Timestamp", how="left")
    test_df = test_df.merge(raw_features, on="Timestamp", how="left")
    
    cols_to_drop = [c for c in available_raw if c in train_df.columns]
    if cols_to_drop:
        train_df = train_df.drop(columns=cols_to_drop)
        val_df = val_df.drop(columns=cols_to_drop)
        test_df = test_df.drop(columns=cols_to_drop)
    
    raw_features_new = [f"{c}_raw" for c in available_raw]
    
    log(f"  Train: {train_df.shape}")
    log(f"  Val: {val_df.shape}")
    log(f"  Test: {test_df.shape}")
    log(f"  Raw anomaly features: {raw_features_new}")
    
    return train_df, val_df, test_df, feature_df, raw_features_new

# ============================================================
# 22.1 MODEL COMPARISON STRATEGY
# ============================================================
def stage_22_1_strategy():
    log("\n" + "=" * 70)
    log("STAGE 22.1: MODEL COMPARISON STRATEGY")
    log("=" * 70)
    
    strategy = {
        "Models to Compare": [
            "Isolation Forest (Tuned) - Realistic",
            "Isolation Forest (Default) - Realistic",
            "Random Forest (Raw_Only) - Leakage",
            "AdaBoost (Leaky) - Leakage",
            "MLP - Failed",
            "Dummy (All Normal) - Baseline",
        ],
        "Comparison Dimensions": [
            "Performance (F1, ROC-AUC, PR-AUC)",
            "Computational (Training Time, Inference Time)",
            "Complexity (Parameters, Depth)",
            "Interpretability (SHAP, Feature Importance)",
            "Robustness (Stability, Noise Resistance)",
            "Fairness (Shift, Month)",
        ],
        "Selection Criteria": {
            "Primary": "F1 Score (Realistic)",
            "Secondary": "ROC-AUC, Robustness",
            "Tertiary": "Interpretability, Fairness",
            "Constraint": "No Data Leakage",
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
# TRAIN ALL MODELS
# ============================================================
def train_all_models(train_df, val_df, test_df, feature_df, raw_features):
    log("\n" + "=" * 70)
    log("TRAINING ALL MODELS FOR COMPARISON")
    log("=" * 70)
    
    results = []
    
    # داده‌ها
    X_train_raw = train_df[raw_features].fillna(0).values
    X_val_raw = val_df[raw_features].fillna(0).values
    X_test_raw = test_df[raw_features].fillna(0).values
    
    main_features = [c for c in feature_df["Feature"].tolist()
                     if c in train_df.columns and c not in ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]]
    mlp_features = main_features + raw_features
    
    X_train_full = train_df[mlp_features].fillna(0).values
    X_val_full = val_df[mlp_features].fillna(0).values
    X_test_full = test_df[mlp_features].fillna(0).values
    
    y_train = train_df[TARGET_ANOMALY].values
    y_val = val_df[TARGET_ANOMALY].values
    y_test = test_df[TARGET_ANOMALY].values
    
    # ============================================================
    # 1. Isolation Forest (Tuned)
    # ============================================================
    log("\n  [1] Isolation Forest (Tuned)...")
    try:
        start = time.time()
        iso_tuned = IsolationForest(
            n_estimators=300, max_samples=0.9, contamination=0.02,
            max_features=0.5, bootstrap=False,
            random_state=SEED, n_jobs=-1,
        )
        iso_tuned.fit(X_train_raw)
        train_time = time.time() - start
        
        # Inference
        start = time.time()
        y_pred = (iso_tuned.predict(X_test_raw) == -1).astype(int)
        y_score = -iso_tuned.score_samples(X_test_raw)
        inference_time = time.time() - start
        
        y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-10)
        
        results.append({
            "Model": "Isolation Forest (Tuned)",
            "Type": "Unsupervised",
            "Feature_Set": "Raw_Only",
            "N_Features": len(raw_features),
            "Training_Time_s": round(train_time, 4),
            "Inference_Time_s": round(inference_time, 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 6),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
            "ROC_AUC": round(roc_auc_score(y_test, y_score_norm), 6),
            "PR_AUC": round(average_precision_score(y_test, y_score_norm), 6),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 6),
            "Kappa": round(cohen_kappa_score(y_test, y_pred), 6),
            "Leakage": False,
        })
        log(f"    F1={results[-1]['F1']:.4f}, ROC-AUC={results[-1]['ROC_AUC']:.4f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # 2. Isolation Forest (Default)
    # ============================================================
    log("\n  [2] Isolation Forest (Default)...")
    try:
        start = time.time()
        iso_default = IsolationForest(random_state=SEED, n_jobs=-1)
        iso_default.fit(X_train_raw)
        train_time = time.time() - start
        
        start = time.time()
        y_pred = (iso_default.predict(X_test_raw) == -1).astype(int)
        y_score = -iso_default.score_samples(X_test_raw)
        inference_time = time.time() - start
        
        y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-10)
        
        results.append({
            "Model": "Isolation Forest (Default)",
            "Type": "Unsupervised",
            "Feature_Set": "Raw_Only",
            "N_Features": len(raw_features),
            "Training_Time_s": round(train_time, 4),
            "Inference_Time_s": round(inference_time, 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 6),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
            "ROC_AUC": round(roc_auc_score(y_test, y_score_norm), 6),
            "PR_AUC": round(average_precision_score(y_test, y_score_norm), 6),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 6),
            "Kappa": round(cohen_kappa_score(y_test, y_pred), 6),
            "Leakage": False,
        })
        log(f"    F1={results[-1]['F1']:.4f}, ROC-AUC={results[-1]['ROC_AUC']:.4f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # 3. Random Forest (Raw_Only)
    # ============================================================
    log("\n  [3] Random Forest (Raw_Only) - Leakage...")
    try:
        start = time.time()
        rf = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=SEED,
            n_jobs=-1, class_weight="balanced_subsample",
        )
        rf.fit(X_train_raw, y_train)
        train_time = time.time() - start
        
        start = time.time()
        y_pred = rf.predict(X_test_raw)
        y_score = rf.predict_proba(X_test_raw)[:, 1]
        inference_time = time.time() - start
        
        results.append({
            "Model": "Random Forest (Raw_Only)",
            "Type": "Supervised",
            "Feature_Set": "Raw_Only",
            "N_Features": len(raw_features),
            "Training_Time_s": round(train_time, 4),
            "Inference_Time_s": round(inference_time, 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 6),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
            "ROC_AUC": round(roc_auc_score(y_test, y_score), 6),
            "PR_AUC": round(average_precision_score(y_test, y_score), 6),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 6),
            "Kappa": round(cohen_kappa_score(y_test, y_pred), 6),
            "Leakage": True,
        })
        log(f"    F1={results[-1]['F1']:.4f}, ROC-AUC={results[-1]['ROC_AUC']:.4f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # 4. AdaBoost (Leaky)
    # ============================================================
    log("\n  [4] AdaBoost (Leaky) - Leakage...")
    try:
        start = time.time()
        ada = AdaBoostClassifier(n_estimators=100, random_state=SEED)
        ada.fit(X_train_full, y_train)
        train_time = time.time() - start
        
        start = time.time()
        y_pred = ada.predict(X_test_full)
        y_score = ada.predict_proba(X_test_full)[:, 1]
        inference_time = time.time() - start
        
        results.append({
            "Model": "AdaBoost (Leaky)",
            "Type": "Supervised",
            "Feature_Set": "Main_Plus_Raw",
            "N_Features": len(mlp_features),
            "Training_Time_s": round(train_time, 4),
            "Inference_Time_s": round(inference_time, 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 6),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
            "ROC_AUC": round(roc_auc_score(y_test, y_score), 6),
            "PR_AUC": round(average_precision_score(y_test, y_score), 6),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 6),
            "Kappa": round(cohen_kappa_score(y_test, y_pred), 6),
            "Leakage": True,
        })
        log(f"    F1={results[-1]['F1']:.4f}, ROC-AUC={results[-1]['ROC_AUC']:.4f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # 5. MLP
    # ============================================================
    log("\n  [5] MLP...")
    try:
        import tensorflow as tf
        mlp = tf.keras.models.load_model(MODELS_DIR / "anomaly_mlp_final_v3.keras")
        
        start = time.time()
        y_score = mlp.predict(X_test_full, verbose=0).flatten()
        y_pred = (y_score > 0.5).astype(int)
        inference_time = time.time() - start
        
        results.append({
            "Model": "MLP",
            "Type": "Supervised",
            "Feature_Set": "Main_Plus_Raw",
            "N_Features": len(mlp_features),
            "Training_Time_s": np.nan,
            "Inference_Time_s": round(inference_time, 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 6),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
            "ROC_AUC": round(roc_auc_score(y_test, y_score), 6),
            "PR_AUC": round(average_precision_score(y_test, y_score), 6),
            "MCC": round(matthews_corrcoef(y_test, y_pred), 6),
            "Kappa": round(cohen_kappa_score(y_test, y_pred), 6),
            "Leakage": False,
        })
        log(f"    F1={results[-1]['F1']:.4f}, ROC-AUC={results[-1]['ROC_AUC']:.4f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # 6. Dummy (All Normal)
    # ============================================================
    log("\n  [6] Dummy (All Normal)...")
    try:
        dummy = DummyClassifier(strategy="most_frequent")
        dummy.fit(X_train_raw, y_train)
        
        y_pred = dummy.predict(X_test_raw)
        y_score = dummy.predict_proba(X_test_raw)[:, 1]
        
        results.append({
            "Model": "Dummy (All Normal)",
            "Type": "Baseline",
            "Feature_Set": "Raw_Only",
            "N_Features": len(raw_features),
            "Training_Time_s": 0.0,
            "Inference_Time_s": 0.0,
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 6),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 6),
            "ROC_AUC": 0.5,
            "PR_AUC": round(y_test.mean(), 6),
            "MCC": 0.0,
            "Kappa": 0.0,
            "Leakage": False,
        })
        log(f"    F1={results[-1]['F1']:.4f}, ROC-AUC={results[-1]['ROC_AUC']:.4f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "22_1_all_models_comparison.csv", index=False)
    log(f"\n  [OK] Saved: 22_1_all_models_comparison.csv")
    
    return results_df

# ============================================================
# 22.2 BASELINE COMPARISON
# ============================================================
def stage_22_2_baseline(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.2: BASELINE COMPARISON")
    log("=" * 70)
    
    dummy_f1 = results_df[results_df["Model"] == "Dummy (All Normal)"]["F1"].values[0]
    dummy_roc = results_df[results_df["Model"] == "Dummy (All Normal)"]["ROC_AUC"].values[0]
    
    log(f"  Baseline (Dummy):")
    log(f"    F1: {dummy_f1:.6f}")
    log(f"    ROC-AUC: {dummy_roc:.6f}")
    
    log(f"\n  Improvement over Baseline:")
    for _, row in results_df.iterrows():
        if row["Model"] == "Dummy (All Normal)":
            continue
        
        f1_improvement = (row["F1"] - dummy_f1) / (dummy_f1 + 1e-10) * 100 if dummy_f1 > 0 else 0
        roc_improvement = (row["ROC_AUC"] - dummy_roc) / (dummy_roc + 1e-10) * 100 if dummy_roc > 0 else 0
        
        log(f"    {row['Model']}: F1 +{f1_improvement:.2f}%, ROC-AUC +{roc_improvement:.2f}%")
    
    return {"dummy_f1": dummy_f1, "dummy_roc": dummy_roc}

# ============================================================
# 22.3 SOTA COMPARISON
# ============================================================
def stage_22_3_sota(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.3: STATE-OF-THE-ART COMPARISON")
    log("=" * 70)
    
    log(f"  SOTA Benchmarks for Anomaly Detection:")
    log(f"    - Isolation Forest: F1 ~0.4-0.6 (typical)")
    log(f"    - One-Class SVM: F1 ~0.3-0.5")
    log(f"    - Autoencoder: F1 ~0.5-0.7")
    log(f"    - LOF: F1 ~0.3-0.5")
    
    log(f"\n  Our Model vs SOTA:")
    for _, row in results_df.iterrows():
        if row["Leakage"]:
            status = "LEAKAGE (not comparable)"
        elif row["F1"] > 0.5:
            status = "Above SOTA"
        elif row["F1"] > 0.3:
            status = "Within SOTA"
        else:
            status = "Below SOTA"
        
        log(f"    {row['Model']}: F1={row['F1']:.4f} ({status})")
    
    return True

# ============================================================
# 22.4 STATISTICAL COMPARISON
# ============================================================
def stage_22_4_statistical(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.4: STATISTICAL COMPARISON")
    log("=" * 70)
    
    from scipy import stats
    
    # مقایسه F1 بین مدل‌ها
    models = results_df[results_df["Model"] != "Dummy (All Normal)"]["Model"].tolist()
    f1_scores = results_df[results_df["Model"] != "Dummy (All Normal)"]["F1"].tolist()
    
    log(f"  F1 Scores:")
    for model, f1 in zip(models, f1_scores):
        log(f"    {model}: {f1:.6f}")
    
    # ANOVA
    if len(f1_scores) >= 2:
        # برای ANOVA نیاز به گروه‌های مستقل داریم
        # اینجا فقط توصیفی
        log(f"\n  Descriptive Statistics:")
        log(f"    Mean F1: {np.mean(f1_scores):.6f}")
        log(f"    Std F1: {np.std(f1_scores):.6f}")
        log(f"    Min F1: {np.min(f1_scores):.6f}")
        log(f"    Max F1: {np.max(f1_scores):.6f}")
    
    return True

# ============================================================
# 22.5 COMPUTATIONAL COMPARISON
# ============================================================
def stage_22_5_computational(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.5: COMPUTATIONAL COMPARISON")
    log("=" * 70)
    
    log(f"  Computational Metrics:")
    log(f"  {'Model':<30} {'Train_Time':<12} {'Inference_Time':<15}")
    log(f"  {'-'*60}")
    
    for _, row in results_df.iterrows():
        train_time = f"{row['Training_Time_s']:.4f}s" if not pd.isna(row['Training_Time_s']) else "N/A"
        inf_time = f"{row['Inference_Time_s']:.4f}s"
        log(f"  {row['Model']:<30} {train_time:<12} {inf_time:<15}")
    
    # سریع‌ترین
    valid = results_df[results_df["Training_Time_s"].notna()]
    fastest = valid.loc[valid["Training_Time_s"].idxmin()]
    log(f"\n  Fastest Training: {fastest['Model']} ({fastest['Training_Time_s']:.4f}s)")
    
    return True

# ============================================================
# 22.6 TRADE-OFF ANALYSIS
# ============================================================
def stage_22_6_tradeoff(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.6: TRADE-OFF ANALYSIS")
    log("=" * 70)
    
    log(f"  Trade-off Analysis:")
    log(f"  {'Model':<30} {'F1':<10} {'ROC_AUC':<10} {'Leakage':<10} {'Type':<15}")
    log(f"  {'-'*75}")
    
    for _, row in results_df.iterrows():
        log(f"  {row['Model']:<30} {row['F1']:<10.4f} {row['ROC_AUC']:<10.4f} "
            f"{str(row['Leakage']):<10} {row['Type']:<15}")
    
    log(f"\n  Key Trade-offs:")
    log(f"    1. F1 vs Leakage: RF has F1=1.0 but Leakage=True")
    log(f"    2. F1 vs Interpretability: Iso Forest is interpretable")
    log(f"    3. F1 vs Complexity: Simple models are more robust")
    log(f"    4. F1 vs Realism: Iso Forest is realistic")
    
    return True

# ============================================================
# 22.7 COST-BENEFIT ANALYSIS
# ============================================================
def stage_22_7_cost_benefit(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.7: COST-BENEFIT ANALYSIS")
    log("=" * 70)
    
    # فرضیات
    cost_fp = 100  # هزینه هر False Positive
    cost_fn = 1000  # هزینه هر False Negative
    
    log(f"  Assumptions:")
    log(f"    Cost per FP: ${cost_fp}")
    log(f"    Cost per FN: ${cost_fn}")
    
    log(f"\n  Cost Analysis (based on Test Set):")
    
    for _, row in results_df.iterrows():
        if row["Model"] == "Dummy (All Normal)":
            continue
        
        # محاسبه هزینه
        # FN = (1 - Recall) * N_Anomalies
        # FP = ... (تقریبی)
        
        # اینجا فقط نمایش
        log(f"    {row['Model']}:")
        log(f"      F1: {row['F1']:.4f}")
        log(f"      Recall: {row['Recall']:.4f}")
        log(f"      Cost: ${cost_fn * (1 - row['Recall']) * 127:.0f} (approx)")
    
    return True

# ============================================================
# 22.8 COMPLEXITY COMPARISON
# ============================================================
def stage_22_8_complexity(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.8: COMPLEXITY COMPARISON")
    log("=" * 70)
    
    complexity = {
        "Isolation Forest (Tuned)": {"N_Estimators": 300, "Max_Samples": 0.9, "Complexity": "Medium"},
        "Isolation Forest (Default)": {"N_Estimators": 100, "Max_Samples": "auto", "Complexity": "Low"},
        "Random Forest (Raw_Only)": {"N_Estimators": 100, "Max_Depth": 10, "Complexity": "Medium"},
        "AdaBoost (Leaky)": {"N_Estimators": 100, "Complexity": "Medium"},
        "MLP": {"Layers": 3, "Params": 17313, "Complexity": "High"},
        "Dummy (All Normal)": {"Complexity": "Trivial"},
    }
    
    log(f"  Model Complexity:")
    for model, config in complexity.items():
        log(f"    {model}:")
        for k, v in config.items():
            log(f"      {k}: {v}")
    
    return complexity

# ============================================================
# 22.9 INTERPRETABILITY COMPARISON
# ============================================================
def stage_22_9_interpretability(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.9: INTERPRETABILITY COMPARISON")
    log("=" * 70)
    
    interpretability = {
        "Isolation Forest (Tuned)": {"SHAP": "Yes", "Feature_Importance": "Yes", "Global": "Yes", "Local": "Yes", "Score": 9},
        "Isolation Forest (Default)": {"SHAP": "Yes", "Feature_Importance": "Yes", "Global": "Yes", "Local": "Yes", "Score": 9},
        "Random Forest (Raw_Only)": {"SHAP": "Yes", "Feature_Importance": "Yes", "Global": "Yes", "Local": "Yes", "Score": 9},
        "AdaBoost (Leaky)": {"SHAP": "Yes", "Feature_Importance": "Yes", "Global": "Yes", "Local": "Partial", "Score": 8},
        "MLP": {"SHAP": "Partial", "Feature_Importance": "No", "Global": "No", "Local": "No", "Score": 3},
        "Dummy (All Normal)": {"SHAP": "N/A", "Feature_Importance": "N/A", "Global": "N/A", "Local": "N/A", "Score": 0},
    }
    
    log(f"  Interpretability Scores:")
    for model, config in interpretability.items():
        log(f"    {model}: Score={config['Score']}/10")
    
    return interpretability

# ============================================================
# 22.10 FINAL MODEL RANKING
# ============================================================
def stage_22_10_final_ranking(results_df):
    log("\n" + "=" * 70)
    log("STAGE 22.10: FINAL MODEL RANKING")
    log("=" * 70)
    
    # فیلتر مدل‌های بدون Leakage
    valid = results_df[~results_df["Leakage"]].copy()
    
    # رتبه‌بندی بر اساس F1
    valid_sorted = valid.sort_values("F1", ascending=False)
    
    log(f"  Ranking (Realistic Models - No Leakage):")
    for i, (_, row) in enumerate(valid_sorted.iterrows(), 1):
        log(f"    {i}. {row['Model']}")
        log(f"       F1={row['F1']:.4f}, ROC-AUC={row['ROC_AUC']:.4f}, "
            f"Precision={row['Precision']:.4f}, Recall={row['Recall']:.4f}")
    
    # انتخاب نهایی
    best = valid_sorted.iloc[0]
    
    log(f"\n  [FINAL SELECTION]")
    log(f"    Model: {best['Model']}")
    log(f"    F1: {best['F1']:.6f}")
    log(f"    ROC-AUC: {best['ROC_AUC']:.6f}")
    log(f"    Precision: {best['Precision']:.6f}")
    log(f"    Recall: {best['Recall']:.6f}")
    log(f"    Type: {best['Type']}")
    log(f"    Leakage: {best['Leakage']}")
    
    # ذخیره
    selection = {
        "Final_Model": best["Model"],
        "F1": round(best["F1"], 6),
        "ROC_AUC": round(best["ROC_AUC"], 6),
        "Precision": round(best["Precision"], 6),
        "Recall": round(best["Recall"], 6),
        "Type": best["Type"],
        "Leakage": bool(best["Leakage"]),
        "N_Features": int(best["N_Features"]),
        "Training_Time_s": round(best["Training_Time_s"], 4) if not pd.isna(best["Training_Time_s"]) else None,
        "Rationale": "Best F1 among models without Data Leakage",
    }
    
    pd.DataFrame(list(selection.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "22_10_final_selection.csv", index=False
    )
    log(f"\n  [OK] Saved: 22_10_final_selection.csv")
    
    return selection, valid_sorted

# ============================================================
# VISUALIZE
# ============================================================
def visualize_comparison(results_df, valid_sorted, selection):
    log("\n" + "=" * 70)
    log("VISUALIZING MODEL COMPARISON")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. F1 Comparison
    colors = ["red" if l else "green" for l in results_df["Leakage"]]
    axes[0, 0].barh(results_df["Model"], results_df["F1"], color=colors)
    axes[0, 0].set_xlabel("F1 Score")
    axes[0, 0].set_title("F1 Score by Model (Red=Leakage, Green=Realistic)")
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC-AUC Comparison
    axes[0, 1].barh(results_df["Model"], results_df["ROC_AUC"], color="steelblue")
    axes[0, 1].set_xlabel("ROC-AUC")
    axes[0, 1].set_title("ROC-AUC by Model")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision vs Recall
    axes[1, 0].scatter(results_df["Precision"], results_df["Recall"],
                       s=200, c=colors, alpha=0.7)
    for _, row in results_df.iterrows():
        axes[1, 0].annotate(row["Model"][:15],
                            (row["Precision"], row["Recall"]),
                            fontsize=8, ha="right")
    axes[1, 0].set_xlabel("Precision")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].set_title("Precision vs Recall")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Final Selection Summary
    axes[1, 1].axis("off")
    
    summary_text = "FINAL MODEL SELECTION\n" + "=" * 35 + "\n\n"
    summary_text += f"Selected: {selection['Final_Model']}\n\n"
    summary_text += f"F1: {selection['F1']:.4f}\n"
    summary_text += f"ROC-AUC: {selection['ROC_AUC']:.4f}\n"
    summary_text += f"Precision: {selection['Precision']:.4f}\n"
    summary_text += f"Recall: {selection['Recall']:.4f}\n"
    summary_text += f"Type: {selection['Type']}\n"
    summary_text += f"Leakage: {selection['Leakage']}\n"
    summary_text += f"Features: {selection['N_Features']}\n\n"
    summary_text += f"Rationale:\n{selection['Rationale']}"
    
    axes[1, 1].text(0.05, 0.5, summary_text, fontsize=11, verticalalignment="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "22_model_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 22_model_comparison.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 22: MODEL COMPARISON & FINAL SELECTION")
    log("=" * 70)
    
    train_df, val_df, test_df, feature_df, raw_features = load_data()
    
    # 22.1
    strategy = stage_22_1_strategy()
    
    # Train all models
    results_df = train_all_models(train_df, val_df, test_df, feature_df, raw_features)
    
    # 22.2
    baseline = stage_22_2_baseline(results_df)
    
    # 22.3
    stage_22_3_sota(results_df)
    
    # 22.4
    stage_22_4_statistical(results_df)
    
    # 22.5
    stage_22_5_computational(results_df)
    
    # 22.6
    stage_22_6_tradeoff(results_df)
    
    # 22.7
    stage_22_7_cost_benefit(results_df)
    
    # 22.8
    complexity = stage_22_8_complexity(results_df)
    
    # 22.9
    interpretability = stage_22_9_interpretability(results_df)
    
    # 22.10
    selection, valid_sorted = stage_22_10_final_ranking(results_df)
    
    # Visualize
    visualize_comparison(results_df, valid_sorted, selection)
    
    log("\n" + "=" * 70)
    log("PHASE 22 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase22_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()