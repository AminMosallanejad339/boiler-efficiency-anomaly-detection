"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 14f: Random Forest for Anomaly Detection
Framework: ML Model Lifecycle - 23 Main Stages
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
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

for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_ANOMALY = "Anomaly_Label"
LEAKY_CLF = ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]

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
    log("PHASE 14f: RANDOM FOREST FOR ANOMALY DETECTION")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("LOADING DATA")
    log("=" * 70)
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    
    # اضافه کردن ویژگی‌های خام Anomaly
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
    
    # حذف ستون‌های تکراری
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
# FEATURE SELECTION
# ============================================================
def select_features(train_df, feature_df, raw_features):
    log("\n" + "=" * 70)
    log("FEATURE SELECTION")
    log("=" * 70)
    
    # ویژگی‌های اصلی (بدون Leaky)
    main_features = [c for c in feature_df["Feature"].tolist()
                     if c in train_df.columns and c not in LEAKY_CLF]
    
    log(f"  Main features: {len(main_features)}")
    log(f"  Raw anomaly features: {len(raw_features)}")
    
    # گزینه‌های مختلف
    feature_sets = {
        "Raw_Only": raw_features,
        "Main_Only": main_features,
        "Main_Plus_Raw": main_features + raw_features,
        "Top_Main_Plus_Raw": None,  # بعداً پر می‌شود
    }
    
    # Top features از RCA
    top_main = [
        "APH_Leakage_", "CO_mgm3", "Dust_mgm3",
        "Reheater_desuperheating_water_flow_th",
        "Boiler_oxygen_level_", "Flue_gas_temperature_",
        "Main_steam_flow_th", "Coal_Flow_th",
    ]
    top_main = [c for c in top_main if c in train_df.columns]
    feature_sets["Top_Main_Plus_Raw"] = top_main + raw_features
    
    for name, features in feature_sets.items():
        log(f"    {name}: {len(features)} features")
    
    return feature_sets

# ============================================================
# TEST FEATURE SETS
# ============================================================
def test_feature_sets(train_df, val_df, test_df, feature_sets):
    log("\n" + "=" * 70)
    log("TESTING FEATURE SETS")
    log("=" * 70)
    
    y_train = train_df[TARGET_ANOMALY].values
    y_val = val_df[TARGET_ANOMALY].values
    y_test = test_df[TARGET_ANOMALY].values
    
    results = []
    
    for set_name, features in feature_sets.items():
        log(f"\n  Testing {set_name} ({len(features)} features)...")
        
        X_train = train_df[features].fillna(0).values
        X_val = val_df[features].fillna(0).values
        X_test = test_df[features].fillna(0).values
        
        start = time.time()
        
        try:
            rf = RandomForestClassifier(
                n_estimators=200,
                random_state=SEED,
                n_jobs=-1,
                class_weight="balanced",
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
            )
            rf.fit(X_train, y_train)
            
            train_time = time.time() - start
            
            # پیش‌بینی
            y_val_pred = rf.predict(X_val)
            y_test_pred = rf.predict(X_test)
            
            # احتمال
            y_val_proba = rf.predict_proba(X_val)[:, 1]
            y_test_proba = rf.predict_proba(X_test)[:, 1]
            
            # معیارها
            for name, y_true, y_pred, y_proba in [
                ("Val", y_val, y_val_pred, y_val_proba),
                ("Test", y_test, y_test_pred, y_test_proba),
            ]:
                prec = precision_score(y_true, y_pred, zero_division=0)
                rec = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                roc_auc = roc_auc_score(y_true, y_proba)
                pr_auc = average_precision_score(y_true, y_proba)
                
                results.append({
                    "Feature_Set": set_name,
                    "N_Features": len(features),
                    "Dataset": name,
                    "Precision": round(prec, 6),
                    "Recall": round(rec, 6),
                    "F1": round(f1, 6),
                    "ROC_AUC": round(roc_auc, 6),
                    "PR_AUC": round(pr_auc, 6),
                })
            
            val_results = [r for r in results if r["Feature_Set"] == set_name and r["Dataset"] == "Val"][0]
            test_results = [r for r in results if r["Feature_Set"] == set_name and r["Dataset"] == "Test"][0]
            
            log(f"    Val: F1={val_results['F1']:.4f}, Precision={val_results['Precision']:.4f}, Recall={val_results['Recall']:.4f}")
            log(f"    Test: F1={test_results['F1']:.4f}, Precision={test_results['Precision']:.4f}, Recall={test_results['Recall']:.4f}")
            log(f"    Time: {train_time:.2f}s")
        
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    return pd.DataFrame(results)

# ============================================================
# HYPERPARAMETER TUNING
# ============================================================
def tune_random_forest(train_df, val_df, features):
    log("\n" + "=" * 70)
    log("HYPERPARAMETER TUNING")
    log("=" * 70)
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    X_val = val_df[features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values
    
    # Grid Search
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [10, 15, 20, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "class_weight": ["balanced", "balanced_subsample", None],
    }
    
    results = []
    total = len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * \
            len(param_grid["min_samples_split"]) * len(param_grid["min_samples_leaf"]) * \
            len(param_grid["class_weight"])
    
    log(f"  Total combinations: {total}")
    log(f"  Running Randomized Search (30 iterations)...")
    
    import random
    random.seed(SEED)
    
    n_iter = 30
    for i in range(n_iter):
        params = {
            "n_estimators": random.choice(param_grid["n_estimators"]),
            "max_depth": random.choice(param_grid["max_depth"]),
            "min_samples_split": random.choice(param_grid["min_samples_split"]),
            "min_samples_leaf": random.choice(param_grid["min_samples_leaf"]),
            "class_weight": random.choice(param_grid["class_weight"]),
        }
        
        try:
            rf = RandomForestClassifier(
                random_state=SEED,
                n_jobs=-1,
                **params,
            )
            rf.fit(X_train, y_train)
            
            y_val_pred = rf.predict(X_val)
            
            prec = precision_score(y_val, y_val_pred, zero_division=0)
            rec = recall_score(y_val, y_val_pred, zero_division=0)
            f1 = f1_score(y_val, y_val_pred, zero_division=0)
            
            results.append({
                "Iteration": i + 1,
                **params,
                "Precision": round(prec, 6),
                "Recall": round(rec, 6),
                "F1": round(f1, 6),
            })
            
            if (i + 1) % 5 == 0:
                log(f"    Iteration {i+1}/{n_iter}: F1={f1:.4f}")
        
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(TABLES_DIR / "14f_rf_tuning_results.csv", index=False)
    log(f"\n  [OK] Saved: 14f_rf_tuning_results.csv")
    
    log(f"\n  Top 5 Configurations:")
    for _, row in results_df.head(5).iterrows():
        log(f"    F1={row['F1']:.4f}, n_est={row['n_estimators']}, "
            f"depth={row['max_depth']}, split={row['min_samples_split']}, "
            f"leaf={row['min_samples_leaf']}, cw={row['class_weight']}")
    
    return results_df

# ============================================================
# TRAIN FINAL MODEL
# ============================================================
def train_final_model(train_df, val_df, test_df, features, best_params):
    log("\n" + "=" * 70)
    log("TRAINING FINAL MODEL")
    log("=" * 70)
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    X_val = val_df[features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values
    X_test = test_df[features].fillna(0).values
    y_test = test_df[TARGET_ANOMALY].values
    
    log(f"  Features: {len(features)}")
    log(f"  Best Params: {best_params}")
    
    rf = RandomForestClassifier(
        random_state=SEED,
        n_jobs=-1,
        **best_params,
    )
    rf.fit(X_train, y_train)
    
    # پیش‌بینی
    y_val_pred = rf.predict(X_val)
    y_test_pred = rf.predict(X_test)
    
    # گزارش
    log(f"\n  Validation Report:")
    log(f"    {classification_report(y_val, y_val_pred, target_names=['Normal', 'Anomaly'], digits=4)}")
    
    log(f"\n  Test Report:")
    log(f"    {classification_report(y_test, y_test_pred, target_names=['Normal', 'Anomaly'], digits=4)}")
    
    # Confusion Matrix
    cm_val = confusion_matrix(y_val, y_val_pred)
    cm_test = confusion_matrix(y_test, y_test_pred)
    
    log(f"\n  Validation Confusion Matrix:")
    log(f"    {cm_val}")
    log(f"\n  Test Confusion Matrix:")
    log(f"    {cm_test}")
    
    # Feature Importance
    importance = pd.DataFrame({
        "Feature": features,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)
    
    importance.to_csv(TABLES_DIR / "14f_rf_feature_importance.csv", index=False)
    log(f"\n  [OK] Saved: 14f_rf_feature_importance.csv")
    
    log(f"\n  Top 10 Features:")
    for _, row in importance.head(10).iterrows():
        log(f"    {row['Feature']}: {row['Importance']:.6f}")
    
    return rf, importance

# ============================================================
# VISUALIZE
# ============================================================
def visualize_results(results_df, importance_df, rf, val_df, test_df, features):
    log("\n" + "=" * 70)
    log("VISUALIZING RESULTS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Feature Set Comparison
    val_results = results_df[results_df["Dataset"] == "Val"]
    test_results = results_df[results_df["Dataset"] == "Test"]
    
    x = np.arange(len(val_results))
    width = 0.35
    axes[0, 0].bar(x - width/2, val_results["F1"], width, label="Val F1", color="steelblue")
    axes[0, 0].bar(x + width/2, test_results["F1"], width, label="Test F1", color="coral")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(val_results["Feature_Set"], rotation=45, ha="right")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 by Feature Set")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision vs Recall
    axes[0, 1].scatter(val_results["Precision"], val_results["Recall"], s=100, c="purple", label="Val")
    axes[0, 1].scatter(test_results["Precision"], test_results["Recall"], s=100, c="green", label="Test")
    for _, row in val_results.iterrows():
        axes[0, 1].annotate(row["Feature_Set"][:15], (row["Precision"], row["Recall"]), fontsize=8)
    axes[0, 1].set_xlabel("Precision")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].set_title("Precision vs Recall")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature Importance
    top15 = importance_df.head(15)
    axes[1, 0].barh(top15["Feature"][::-1], top15["Importance"][::-1], color="steelblue")
    axes[1, 0].set_xlabel("Importance")
    axes[1, 0].set_title("Top 15 Feature Importance")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Confusion Matrix (Test)
    y_test = test_df[TARGET_ANOMALY].values
    y_test_pred = rf.predict(test_df[features].fillna(0).values)
    cm = confusion_matrix(y_test, y_test_pred)
    
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[1, 1],
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("Actual")
    axes[1, 1].set_title("Test Confusion Matrix")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14f_rf_results.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14f_rf_results.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 14f: RANDOM FOREST FOR ANOMALY DETECTION")
    log("=" * 70)
    
    # بارگذاری داده
    train_df, val_df, test_df, feature_df, raw_features = load_data()
    
    # انتخاب ویژگی‌ها
    feature_sets = select_features(train_df, feature_df, raw_features)
    
    # تست Feature Setها
    results_df = test_feature_sets(train_df, val_df, test_df, feature_sets)
    results_df.to_csv(TABLES_DIR / "14f_feature_sets_results.csv", index=False)
    log(f"\n  [OK] Saved: 14f_feature_sets_results.csv")
    
    # بهترین Feature Set
    best_val = results_df[results_df["Dataset"] == "Val"].sort_values("F1", ascending=False).iloc[0]
    best_set_name = best_val["Feature_Set"]
    best_features = feature_sets[best_set_name]
    
    log(f"\n  [BEST] Feature Set: {best_set_name}")
    log(f"  F1: {best_val['F1']:.6f}")
    log(f"  N Features: {len(best_features)}")
    
    # Hyperparameter Tuning
    tuning_results = tune_random_forest(train_df, val_df, best_features)
    
    # بهترین پارامترها
    best_params_row = tuning_results.iloc[0]
    best_params = {
        "n_estimators": int(best_params_row["n_estimators"]),
        "max_depth": int(best_params_row["max_depth"]) if pd.notna(best_params_row["max_depth"]) else None,
        "min_samples_split": int(best_params_row["min_samples_split"]),
        "min_samples_leaf": int(best_params_row["min_samples_leaf"]),
        "class_weight": best_params_row["class_weight"] if pd.notna(best_params_row["class_weight"]) else None,
    }
    
    log(f"\n  Best Params: {best_params}")
    
    # آموزش مدل نهایی
    rf, importance = train_final_model(train_df, val_df, test_df, best_features, best_params)
    
    # ذخیره مدل
    joblib.dump(rf, MODELS_DIR / "random_forest_anomaly.pkl")
    log(f"  [OK] Saved: random_forest_anomaly.pkl")
    
    # Visualize
    visualize_results(results_df, importance, rf, val_df, test_df, best_features)
    
    # خلاصه
    val_pred = rf.predict(val_df[best_features].fillna(0).values)
    test_pred = rf.predict(test_df[best_features].fillna(0).values)
    
    summary = {
        "Best_Feature_Set": best_set_name,
        "N_Features": len(best_features),
        "Best_n_estimators": best_params["n_estimators"],
        "Best_max_depth": str(best_params["max_depth"]),
        "Best_min_samples_split": best_params["min_samples_split"],
        "Best_min_samples_leaf": best_params["min_samples_leaf"],
        "Best_class_weight": str(best_params["class_weight"]),
        "Val_F1": round(f1_score(val_df[TARGET_ANOMALY], val_pred, zero_division=0), 6),
        "Val_Precision": round(precision_score(val_df[TARGET_ANOMALY], val_pred, zero_division=0), 6),
        "Val_Recall": round(recall_score(val_df[TARGET_ANOMALY], val_pred, zero_division=0), 6),
        "Test_F1": round(f1_score(test_df[TARGET_ANOMALY], test_pred, zero_division=0), 6),
        "Test_Precision": round(precision_score(test_df[TARGET_ANOMALY], test_pred, zero_division=0), 6),
        "Test_Recall": round(recall_score(test_df[TARGET_ANOMALY], test_pred, zero_division=0), 6),
    }
    
    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "14f_rf_summary.csv", index=False
    )
    log(f"  [OK] Saved: 14f_rf_summary.csv")
    
    log(f"\n  Final Summary:")
    for k, v in summary.items():
        log(f"    {k}: {v}")
    
    log("\n" + "=" * 70)
    log("PHASE 14f COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase14f_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()