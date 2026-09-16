"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 15: Validation (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Anomaly Detection - Comprehensive Validation
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
import tensorflow as tf
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
)
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold

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

for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
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
    log("PHASE 15: VALIDATION")
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
# 15.1 VALIDATION STRATEGY
# ============================================================
def stage_15_1_strategy():
    log("\n" + "=" * 70)
    log("STAGE 15.1: VALIDATION STRATEGY")
    log("=" * 70)
    
    strategy = {
        "Models to Validate": [
            "Random Forest (Raw_Only) - Leakage",
            "Isolation Forest (Raw_Only) - Realistic",
            "MLP (Main_Plus_Raw) - Failed",
        ],
        "Validation Metrics": [
            "F1 Score", "Precision", "Recall",
            "ROC-AUC", "PR-AUC", "Accuracy",
            "Confusion Matrix",
        ],
        "Validation Methods": [
            "Hold-out Validation (70/15/15)",
            "Cross-Validation (StratifiedKFold)",
            "Learning Curve Analysis",
            "Validation Curve Analysis",
        ],
        "Overfitting Detection": [
            "Train vs Val F1 Gap",
            "Train vs Val Loss Gap",
            "Learning Curve Convergence",
        ],
        "Underfitting Detection": [
            "Both Train and Val Performance Low",
            "Learning Curve Plateau at Low Performance",
        ],
    }
    
    for k, v in strategy.items():
        if isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        else:
            log(f"  {k}: {v}")
    
    return strategy

# ============================================================
# 15.2-15.4 VALIDATION EVALUATION
# ============================================================
def stage_15_2_to_15_4_evaluation(train_df, val_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 15.2-15.4: VALIDATION SET EVALUATION")
    log("=" * 70)
    
    results = []
    trained_models = {}
    
    # ============================================================
    # Random Forest (Raw_Only)
    # ============================================================
    log("\n  [1] Random Forest (Raw_Only)...")
    try:
        rf = joblib.load(MODELS_DIR / "random_forest_anomaly.pkl")
        
        X_train = train_df[raw_features].fillna(0).values
        X_val = val_df[raw_features].fillna(0).values
        X_test = test_df[raw_features].fillna(0).values
        y_train = train_df[TARGET_ANOMALY].values
        y_val = val_df[TARGET_ANOMALY].values
        y_test = test_df[TARGET_ANOMALY].values
        
        # Loss (Binary Crossentropy approximation)
        for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
            y_pred = rf.predict(X)
            y_proba = rf.predict_proba(X)[:, 1]
            y_proba = np.clip(y_proba, 1e-10, 1 - 1e-10)
            
            # Loss
            bce_loss = -np.mean(y * np.log(y_proba) + (1 - y) * np.log(1 - y_proba))
            
            results.append({
                "Model": "Random_Forest",
                "Dataset": name,
                "N_Samples": len(y),
                "N_Features": len(raw_features),
                "Loss": round(bce_loss, 6),
                "Accuracy": round(accuracy_score(y, y_pred), 6),
                "Precision": round(precision_score(y, y_pred, zero_division=0), 6),
                "Recall": round(recall_score(y, y_pred, zero_division=0), 6),
                "F1": round(f1_score(y, y_pred, zero_division=0), 6),
                "ROC_AUC": round(roc_auc_score(y, y_proba), 6),
                "PR_AUC": round(average_precision_score(y, y_proba), 6),
            })
            
            log(f"    {name}: F1={results[-1]['F1']:.6f}, Loss={bce_loss:.6f}")
        
        trained_models["Random_Forest"] = rf
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # Isolation Forest (Raw_Only)
    # ============================================================
    log("\n  [2] Isolation Forest (Raw_Only)...")
    try:
        iso = joblib.load(MODELS_DIR / "isolation_forest_final.pkl")
        
        X_train = train_df[raw_features].fillna(0).values
        X_val = val_df[raw_features].fillna(0).values
        X_test = test_df[raw_features].fillna(0).values
        y_train = train_df[TARGET_ANOMALY].values
        y_val = val_df[TARGET_ANOMALY].values
        y_test = test_df[TARGET_ANOMALY].values
        
        for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
            y_pred = (iso.predict(X) == -1).astype(int)
            y_proba = -iso.score_samples(X)
            # نرمال‌سازی
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
            y_proba = np.clip(y_proba, 1e-10, 1 - 1e-10)
            
            # Loss
            bce_loss = -np.mean(y * np.log(y_proba) + (1 - y) * np.log(1 - y_proba))
            
            results.append({
                "Model": "Isolation_Forest",
                "Dataset": name,
                "N_Samples": len(y),
                "N_Features": len(raw_features),
                "Loss": round(bce_loss, 6),
                "Accuracy": round(accuracy_score(y, y_pred), 6),
                "Precision": round(precision_score(y, y_pred, zero_division=0), 6),
                "Recall": round(recall_score(y, y_pred, zero_division=0), 6),
                "F1": round(f1_score(y, y_pred, zero_division=0), 6),
                "ROC_AUC": round(roc_auc_score(y, y_proba), 6),
                "PR_AUC": round(average_precision_score(y, y_proba), 6),
            })
            
            log(f"    {name}: F1={results[-1]['F1']:.6f}, Loss={bce_loss:.6f}")
        
        trained_models["Isolation_Forest"] = iso
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # MLP
    # ============================================================
    log("\n  [3] MLP (Main_Plus_Raw)...")
    try:
        mlp = tf.keras.models.load_model(MODELS_DIR / "anomaly_mlp_final_v3.keras")
        feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
        main_features = [c for c in feature_df["Feature"].tolist()
                         if c in train_df.columns and c not in ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]]
        mlp_features = main_features + raw_features
        
        X_train = train_df[mlp_features].fillna(0).values
        X_val = val_df[mlp_features].fillna(0).values
        X_test = test_df[mlp_features].fillna(0).values
        y_train = train_df[TARGET_ANOMALY].values
        y_val = val_df[TARGET_ANOMALY].values
        y_test = test_df[TARGET_ANOMALY].values
        
        for name, X, y in [("Train", X_train, y_train), ("Val", X_val, y_val), ("Test", X_test, y_test)]:
            y_proba = mlp.predict(X, verbose=0).flatten()
            y_pred = (y_proba > 0.5).astype(int)
            y_proba = np.clip(y_proba, 1e-10, 1 - 1e-10)
            
            bce_loss = -np.mean(y * np.log(y_proba) + (1 - y) * np.log(1 - y_proba))
            
            results.append({
                "Model": "MLP",
                "Dataset": name,
                "N_Samples": len(y),
                "N_Features": len(mlp_features),
                "Loss": round(bce_loss, 6),
                "Accuracy": round(accuracy_score(y, y_pred), 6),
                "Precision": round(precision_score(y, y_pred, zero_division=0), 6),
                "Recall": round(recall_score(y, y_pred, zero_division=0), 6),
                "F1": round(f1_score(y, y_pred, zero_division=0), 6),
                "ROC_AUC": round(roc_auc_score(y, y_proba), 6),
                "PR_AUC": round(average_precision_score(y, y_proba), 6),
            })
            
            log(f"    {name}: F1={results[-1]['F1']:.6f}, Loss={bce_loss:.6f}")
        
        trained_models["MLP"] = mlp
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "15_2_validation_results.csv", index=False)
    log(f"\n  [OK] Saved: 15_2_validation_results.csv")
    
    return results_df, trained_models

# ============================================================
# 15.5 OVERFITTING DETECTION
# ============================================================
def stage_15_5_overfitting(results_df):
    log("\n" + "=" * 70)
    log("STAGE 15.5: OVERFITTING DETECTION")
    log("=" * 70)
    
    overfitting_report = []
    
    models = results_df["Model"].unique()
    for model in models:
        model_df = results_df[results_df["Model"] == model]
        
        train_row = model_df[model_df["Dataset"] == "Train"]
        val_row = model_df[model_df["Dataset"] == "Val"]
        
        if train_row.empty or val_row.empty:
            continue
        
        train_f1 = train_row.iloc[0]["F1"]
        val_f1 = val_row.iloc[0]["F1"]
        train_loss = train_row.iloc[0]["Loss"]
        val_loss = val_row.iloc[0]["Loss"]
        
        f1_gap = train_f1 - val_f1
        loss_gap = val_loss - train_loss
        
        if f1_gap > 0.1:
            status = "OVERFITTING"
        elif f1_gap < -0.1:
            status = "UNDERFITTING (Val > Train)"
        else:
            status = "OK"
        
        overfitting_report.append({
            "Model": model,
            "Train_F1": round(train_f1, 6),
            "Val_F1": round(val_f1, 6),
            "F1_Gap": round(f1_gap, 6),
            "Train_Loss": round(train_loss, 6),
            "Val_Loss": round(val_loss, 6),
            "Loss_Gap": round(loss_gap, 6),
            "Status": status,
        })
        
        log(f"  {model}:")
        log(f"    Train F1: {train_f1:.6f}, Val F1: {val_f1:.6f}, Gap: {f1_gap:.6f}")
        log(f"    Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Gap: {loss_gap:.6f}")
        log(f"    Status: {status}")
    
    overfitting_df = pd.DataFrame(overfitting_report)
    overfitting_df.to_csv(TABLES_DIR / "15_5_overfitting.csv", index=False)
    log(f"\n  [OK] Saved: 15_5_overfitting.csv")
    
    return overfitting_df

# ============================================================
# 15.6 UNDERFITTING DETECTION
# ============================================================
def stage_15_6_underfitting(results_df):
    log("\n" + "=" * 70)
    log("STAGE 15.6: UNDERFITTING DETECTION")
    log("=" * 70)
    
    underfitting_report = []
    
    models = results_df["Model"].unique()
    for model in models:
        model_df = results_df[results_df["Model"] == model]
        
        train_row = model_df[model_df["Dataset"] == "Train"]
        val_row = model_df[model_df["Dataset"] == "Val"]
        
        if train_row.empty or val_row.empty:
            continue
        
        train_f1 = train_row.iloc[0]["F1"]
        val_f1 = val_row.iloc[0]["F1"]
        
        # Underfitting: both train and val are low
        if train_f1 < 0.5 and val_f1 < 0.5:
            status = "UNDERFITTING"
        elif train_f1 < 0.3:
            status = "SEVERE UNDERFITTING"
        else:
            status = "OK"
        
        underfitting_report.append({
            "Model": model,
            "Train_F1": round(train_f1, 6),
            "Val_F1": round(val_f1, 6),
            "Status": status,
        })
        
        log(f"  {model}: Train F1={train_f1:.6f}, Val F1={val_f1:.6f} -> {status}")
    
    underfitting_df = pd.DataFrame(underfitting_report)
    underfitting_df.to_csv(TABLES_DIR / "15_6_underfitting.csv", index=False)
    log(f"\n  [OK] Saved: 15_6_underfitting.csv")
    
    return underfitting_df

# ============================================================
# 15.7 LEARNING CURVE ANALYSIS
# ============================================================
def stage_15_7_learning_curve(train_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 15.7: LEARNING CURVE ANALYSIS")
    log("=" * 70)
    
    X = train_df[raw_features].fillna(0).values
    y = train_df[TARGET_ANOMALY].values
    
    # Learning Curve for Random Forest
    log(f"  Computing Learning Curve for Random Forest...")
    
    try:
        rf = RandomForestClassifier(
            n_estimators=100, random_state=SEED, n_jobs=-1,
            class_weight="balanced", max_depth=10,
        )
        
        train_sizes, train_scores, val_scores = learning_curve(
            rf, X, y,
            train_sizes=np.linspace(0.1, 1.0, 10),
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
            scoring="f1",
            n_jobs=-1,
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        lc_df = pd.DataFrame({
            "Train_Size": train_sizes,
            "Train_F1_Mean": train_mean,
            "Train_F1_Std": train_std,
            "Val_F1_Mean": val_mean,
            "Val_F1_Std": val_std,
        })
        lc_df.to_csv(TABLES_DIR / "15_7_learning_curve.csv", index=False)
        log(f"  [OK] Saved: 15_7_learning_curve.csv")
        
        # نمودار
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(train_sizes, train_mean, "o-", color="steelblue", label="Train F1")
        ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2, color="steelblue")
        ax.plot(train_sizes, val_mean, "o-", color="coral", label="Val F1")
        ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2, color="coral")
        ax.set_xlabel("Training Size")
        ax.set_ylabel("F1 Score")
        ax.set_title("Learning Curve - Random Forest")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "15_7_learning_curve.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 15_7_learning_curve.png")
        
        log(f"\n  Learning Curve Results:")
        for i, size in enumerate(train_sizes):
            log(f"    Size={size:.0f}: Train F1={train_mean[i]:.4f}, Val F1={val_mean[i]:.4f}")
        
        return lc_df
    except Exception as e:
        log(f"  [ERROR] {e}")
        return pd.DataFrame()

# ============================================================
# 15.8-15.9 EARLY STOPPING & CHECKPOINTING
# ============================================================
def stage_15_8_to_15_9(train_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 15.8-15.9: EARLY STOPPING & CHECKPOINTING")
    log("=" * 70)
    
    log(f"  [INFO] Early Stopping and Checkpointing are for Neural Networks")
    log(f"  [INFO] Used in Phase 14 (MLP Training)")
    log(f"  [INFO] For Random Forest and Isolation Forest, these are not applicable")
    log(f"  [INFO] Random Forest uses n_estimators (fixed)")
    log(f"  [INFO] Isolation Forest uses n_estimators (fixed)")
    
    config = {
        "MLP": {
            "EarlyStopping": "patience=20 (used in Phase 14)",
            "ModelCheckpoint": "best_model_during_training.keras",
        },
        "RandomForest": {
            "EarlyStopping": "Not applicable",
            "ModelCheckpoint": "Not applicable",
        },
        "IsolationForest": {
            "EarlyStopping": "Not applicable",
            "ModelCheckpoint": "Not applicable",
        },
    }
    
    for model, cfg in config.items():
        log(f"  {model}:")
        for k, v in cfg.items():
            log(f"    {k}: {v}")
    
    return config

# ============================================================
# 15.10 BEST MODEL SELECTION
# ============================================================
def stage_15_10_best_model(results_df, overfitting_df):
    log("\n" + "=" * 70)
    log("STAGE 15.10: BEST MODEL SELECTION")
    log("=" * 70)
    
    # بهترین مدل بر اساس Val F1
    val_results = results_df[results_df["Dataset"] == "Val"].copy()
    val_results = val_results.sort_values("F1", ascending=False)
    
    log(f"\n  Model Ranking by Val F1:")
    for i, (_, row) in enumerate(val_results.iterrows()):
        log(f"    {i+1}. {row['Model']}: F1={row['F1']:.6f}, "
            f"Precision={row['Precision']:.6f}, Recall={row['Recall']:.6f}")
    
    best_model = val_results.iloc[0]
    
    log(f"\n  [BEST] Model: {best_model['Model']}")
    log(f"  Val F1: {best_model['F1']:.6f}")
    log(f"  Val Precision: {best_model['Precision']:.6f}")
    log(f"  Val Recall: {best_model['Recall']:.6f}")
    log(f"  Val ROC-AUC: {best_model['ROC_AUC']:.6f}")
    
    # بررسی Overfitting
    overfit_row = overfitting_df[overfitting_df["Model"] == best_model["Model"]]
    if not overfit_row.empty:
        log(f"\n  Overfitting Status: {overfit_row.iloc[0]['Status']}")
    
    # انتخاب نهایی
    log(f"\n  [FINAL SELECTION]")
    log(f"  Based on Val F1:")
    log(f"    Best Supervised: Random Forest (F1 = 1.000) - LEAKAGE")
    log(f"    Best Realistic: Isolation Forest (F1 = 0.161)")
    log(f"    Best Overall for Deployment: Isolation Forest")
    
    selection = {
        "Best_Val_F1_Model": best_model["Model"],
        "Best_Val_F1": round(best_model["F1"], 6),
        "Best_Realistic_Model": "Isolation_Forest",
        "Best_Realistic_F1": round(
            results_df[(results_df["Model"] == "Isolation_Forest") & 
                       (results_df["Dataset"] == "Val")].iloc[0]["F1"], 6
        ),
        "Deployment_Recommendation": "Isolation Forest (Unsupervised)",
        "Rationale": "No Data Leakage, Realistic Performance",
    }
    
    pd.DataFrame(list(selection.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "15_10_best_model.csv", index=False
    )
    log(f"\n  [OK] Saved: 15_10_best_model.csv")
    
    return selection, best_model

# ============================================================
# VISUALIZE
# ============================================================
def visualize_validation(results_df, overfitting_df, best_model_name):
    log("\n" + "=" * 70)
    log("VISUALIZING VALIDATION RESULTS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. F1 by Dataset and Model
    models = results_df["Model"].unique()
    datasets = ["Train", "Val", "Test"]
    x = np.arange(len(models))
    width = 0.25
    
    for i, dataset in enumerate(datasets):
        dataset_df = results_df[results_df["Dataset"] == dataset]
        values = [dataset_df[dataset_df["Model"] == m]["F1"].values[0] if not dataset_df[dataset_df["Model"] == m].empty else 0 for m in models]
        axes[0, 0].bar(x + i * width - width, values, width, label=dataset)
    
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha="right")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 by Model and Dataset")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # 2. Loss by Dataset and Model
    for i, dataset in enumerate(datasets):
        dataset_df = results_df[results_df["Dataset"] == dataset]
        values = [dataset_df[dataset_df["Model"] == m]["Loss"].values[0] if not dataset_df[dataset_df["Model"] == m].empty else 0 for m in models]
        axes[0, 1].bar(x + i * width - width, values, width, label=dataset)
    
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(models, rotation=45, ha="right")
    axes[0, 1].set_ylabel("Loss (BCE)")
    axes[0, 1].set_title("Loss by Model and Dataset")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Overfitting Gap
    axes[0, 2].bar(overfitting_df["Model"], overfitting_df["F1_Gap"], color="coral")
    axes[0, 2].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    axes[0, 2].axhline(y=0.1, color="red", linestyle="--", label="Overfit Threshold")
    axes[0, 2].axhline(y=-0.1, color="blue", linestyle="--", label="Underfit Threshold")
    axes[0, 2].set_ylabel("F1 Gap (Train - Val)")
    axes[0, 2].set_title("Overfitting Detection")
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Precision vs Recall
    val_results = results_df[results_df["Dataset"] == "Val"]
    axes[1, 0].scatter(val_results["Precision"], val_results["Recall"],
                       s=200, c="purple", alpha=0.7)
    for _, row in val_results.iterrows():
        axes[1, 0].annotate(row["Model"], (row["Precision"], row["Recall"]),
                            fontsize=10, ha="right")
    axes[1, 0].set_xlabel("Precision")
    axes[1, 0].set_ylabel("Recall")
    axes[1, 0].set_title("Precision vs Recall (Validation)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim(-0.05, 1.05)
    axes[1, 0].set_ylim(-0.05, 1.05)
    
    # 5. ROC-AUC Comparison
    val_results = val_results.sort_values("ROC_AUC", ascending=True)
    axes[1, 1].barh(val_results["Model"], val_results["ROC_AUC"], color="steelblue")
    axes[1, 1].set_xlabel("ROC-AUC")
    axes[1, 1].set_title("ROC-AUC by Model (Validation)")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim(0, 1.1)
    
    # 6. Summary Text
    axes[1, 2].axis("off")
    
    summary_text = "VALIDATION SUMMARY\n" + "=" * 40 + "\n\n"
    for _, row in val_results.sort_values("F1", ascending=False).iterrows():
        summary_text += f"{row['Model']}:\n"
        summary_text += f"  F1 = {row['F1']:.4f}\n"
        summary_text += f"  P = {row['Precision']:.4f}\n"
        summary_text += f"  R = {row['Recall']:.4f}\n\n"
    
    summary_text += "RECOMMENDATION:\n"
    summary_text += "Isolation Forest\n"
    summary_text += "for deployment\n"
    summary_text += "(no Data Leakage)"
    
    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=10, verticalalignment="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "15_validation_results.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 15_validation_results.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 15: VALIDATION")
    log("=" * 70)
    
    train_df, val_df, test_df, feature_df, raw_features = load_data()
    
    # 15.1
    strategy = stage_15_1_strategy()
    
    # 15.2-15.4
    results_df, trained_models = stage_15_2_to_15_4_evaluation(train_df, val_df, test_df, raw_features)
    
    # 15.5
    overfitting_df = stage_15_5_overfitting(results_df)
    
    # 15.6
    underfitting_df = stage_15_6_underfitting(results_df)
    
    # 15.7
    lc_df = stage_15_7_learning_curve(train_df, raw_features)
    
    # 15.8-15.9
    early_stopping_config = stage_15_8_to_15_9(train_df, raw_features)
    
    # 15.10
    selection, best_model = stage_15_10_best_model(results_df, overfitting_df)
    
    # Visualize
    visualize_validation(results_df, overfitting_df, best_model["Model"])
    
    log("\n" + "=" * 70)
    log("PHASE 15 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase15_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()