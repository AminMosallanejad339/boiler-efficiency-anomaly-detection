"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 16: Hyperparameter Tuning & Refinement (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Isolation Forest Refinement
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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report,
)
from sklearn.model_selection import ParameterGrid

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
    log("PHASE 16: HYPERPARAMETER TUNING & REFINEMENT")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("LOADING DATA")
    log("=" * 70)
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
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
    
    return train_df, val_df, test_df, raw_features_new

# ============================================================
# 16.1 TUNING STRATEGY
# ============================================================
def stage_16_1_strategy():
    log("\n" + "=" * 70)
    log("STAGE 16.1: HYPERPARAMETER TUNING STRATEGY")
    log("=" * 70)
    
    strategy = {
        "Target Model": "Isolation Forest",
        "Rationale": "Realistic performance (F1=0.23), needs improvement",
        "Random Forest": "Skipped (Leakage, F1=1.0)",
        "MLP": "Skipped (Failed, F1=0.0)",
        "Hyperparameters to Tune": {
            "n_estimators": [100, 200, 300, 500],
            "max_samples": [0.5, 0.7, 0.9, "auto"],
            "contamination": [0.01, 0.02, 0.03, 0.05, 0.08],
            "max_features": [0.5, 0.7, 1.0],
            "bootstrap": [True, False],
        },
        "Tuning Methods": [
            "Grid Search",
            "Random Search",
            "Ablation Study",
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
# 16.2-16.4 RE-TRAINING & TUNING
# ============================================================
def stage_16_2_to_16_4_tuning(train_df, val_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 16.2-16.4: RE-TRAINING & TUNING")
    log("=" * 70)
    
    X_train = train_df[raw_features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    X_val = val_df[raw_features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values
    
    log(f"  X_train: {X_train.shape}")
    log(f"  X_val: {X_val.shape}")
    
    # Grid Search
    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_samples": [0.5, 0.7, 0.9, "auto"],
        "contamination": [0.01, 0.02, 0.03, 0.05],
        "max_features": [0.5, 0.7, 1.0],
        "bootstrap": [True, False],
    }
    
    grid = list(ParameterGrid(param_grid))
    log(f"  Total combinations: {len(grid)}")
    
    # محدود کردن به 30 ترکیب (Random Search)
    import random
    random.seed(SEED)
    random.shuffle(grid)
    grid = grid[:30]
    
    log(f"  Testing {len(grid)} combinations...")
    
    results = []
    start_time = time.time()
    
    for i, params in enumerate(grid):
        try:
            iso = IsolationForest(
                random_state=SEED,
                n_jobs=-1,
                **params,
            )
            iso.fit(X_train)
            
            y_pred = (iso.predict(X_val) == -1).astype(int)
            
            prec = precision_score(y_val, y_pred, zero_division=0)
            rec = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            results.append({
                "Iteration": i + 1,
                "n_estimators": params["n_estimators"],
                "max_samples": str(params["max_samples"]),
                "contamination": params["contamination"],
                "max_features": params["max_features"],
                "bootstrap": params["bootstrap"],
                "Precision": round(prec, 6),
                "Recall": round(rec, 6),
                "F1": round(f1, 6),
            })
            
            if (i + 1) % 5 == 0:
                log(f"    Iteration {i+1}/{len(grid)}: F1={f1:.4f}")
        
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    elapsed = time.time() - start_time
    log(f"\n  Total time: {elapsed:.2f}s")
    
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(TABLES_DIR / "16_2_tuning_results.csv", index=False)
    log(f"  [OK] Saved: 16_2_tuning_results.csv")
    
    log(f"\n  Top 5 Configurations:")
    for _, row in results_df.head(5).iterrows():
        log(f"    F1={row['F1']:.4f}, n_est={row['n_estimators']}, "
            f"max_samples={row['max_samples']}, cont={row['contamination']}, "
            f"max_feat={row['max_features']}, bootstrap={row['bootstrap']}")
    
    return results_df

# ============================================================
# 16.5 ABLATION STUDY
# ============================================================
def stage_16_5_ablation(train_df, val_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 16.5: ABLATION STUDY")
    log("=" * 70)
    
    X_train = train_df[raw_features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    X_val = val_df[raw_features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values
    
    log(f"  Ablation: Remove one feature at a time")
    
    results = []
    
    # Baseline: همه ویژگی‌ها
    for name, features in [("All_Features", raw_features)] + \
                          [(f"Without_{f}", [x for x in raw_features if x != f]) for f in raw_features]:
        feature_indices = [raw_features.index(f) for f in features if f in raw_features]
        
        X_train_sub = X_train[:, feature_indices]
        X_val_sub = X_val[:, feature_indices]
        
        iso = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=SEED, n_jobs=-1
        )
        iso.fit(X_train_sub)
        
        y_pred = (iso.predict(X_val_sub) == -1).astype(int)
        
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            "Configuration": name,
            "N_Features": len(features),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
        })
        
        log(f"    {name}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    results_df.to_csv(TABLES_DIR / "16_5_ablation.csv", index=False)
    log(f"\n  [OK] Saved: 16_5_ablation.csv")
    
    return results_df

# ============================================================
# 16.6 SENSITIVITY ANALYSIS
# ============================================================
def stage_16_6_sensitivity(train_df, val_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 16.6: SENSITIVITY ANALYSIS")
    log("=" * 70)
    
    X_train = train_df[raw_features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    X_val = val_df[raw_features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values
    
    # Sensitivity to Contamination
    log(f"\n  Sensitivity to Contamination:")
    contamination_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]
    
    results = []
    for cont in contamination_values:
        iso = IsolationForest(
            n_estimators=200, contamination=cont, random_state=SEED, n_jobs=-1
        )
        iso.fit(X_train)
        y_pred = (iso.predict(X_val) == -1).astype(int)
        
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            "Contamination": cont,
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
        })
        
        log(f"    Contamination={cont}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "16_6_sensitivity.csv", index=False)
    log(f"  [OK] Saved: 16_6_sensitivity.csv")
    
    # نمودار
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df["Contamination"], results_df["F1"], "o-", color="steelblue", linewidth=2, label="F1")
    ax.plot(results_df["Contamination"], results_df["Precision"], "s-", color="coral", linewidth=2, label="Precision")
    ax.plot(results_df["Contamination"], results_df["Recall"], "^-", color="green", linewidth=2, label="Recall")
    ax.set_xlabel("Contamination")
    ax.set_ylabel("Score")
    ax.set_title("Sensitivity Analysis: Contamination")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_6_sensitivity.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 16_6_sensitivity.png")
    
    return results_df

# ============================================================
# 16.7 ROBUSTNESS CHECK
# ============================================================
def stage_16_7_robustness(train_df, val_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 16.7: ROBUSTNESS CHECK")
    log("=" * 70)
    
    X_train = train_df[raw_features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    
    # Test on different subsets
    log(f"  Testing on different data subsets...")
    
    results = []
    
    # Full data
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        X = df[raw_features].fillna(0).values
        y = df[TARGET_ANOMALY].values
        
        iso = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=SEED, n_jobs=-1
        )
        iso.fit(X_train)
        
        y_pred = (iso.predict(X) == -1).astype(int)
        
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
        results.append({
            "Dataset": name,
            "N_Samples": len(y),
            "N_Anomalies": int(y.sum()),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
        })
        
        log(f"    {name}: F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "16_7_robustness.csv", index=False)
    log(f"\n  [OK] Saved: 16_7_robustness.csv")
    
    return results_df

# ============================================================
# 16.8 STABILITY CHECK
# ============================================================
def stage_16_8_stability(train_df, val_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 16.8: STABILITY CHECK")
    log("=" * 70)
    
    X_train = train_df[raw_features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    X_val = val_df[raw_features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values
    
    # Test with different random seeds
    log(f"  Testing with 10 different random seeds...")
    
    results = []
    for seed in range(10):
        iso = IsolationForest(
            n_estimators=200, contamination=0.05, random_state=seed, n_jobs=-1
        )
        iso.fit(X_train)
        y_pred = (iso.predict(X_val) == -1).astype(int)
        
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            "Seed": seed,
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
        })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "16_8_stability.csv", index=False)
    log(f"  [OK] Saved: 16_8_stability.csv")
    
    log(f"\n  Stability Analysis:")
    log(f"    F1 Mean: {results_df['F1'].mean():.6f}")
    log(f"    F1 Std: {results_df['F1'].std():.6f}")
    log(f"    F1 Min: {results_df['F1'].min():.6f}")
    log(f"    F1 Max: {results_df['F1'].max():.6f}")
    log(f"    Stability: {'STABLE' if results_df['F1'].std() < 0.05 else 'UNSTABLE'}")
    
    return results_df

# ============================================================
# 16.9 CONVERGENCE CHECK
# ============================================================
def stage_16_9_convergence(train_df, val_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 16.9: CONVERGENCE CHECK")
    log("=" * 70)
    
    X_train = train_df[raw_features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    X_val = val_df[raw_features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values
    
    # Test different n_estimators
    log(f"  Testing different n_estimators...")
    
    results = []
    for n_est in [50, 100, 150, 200, 300, 400, 500]:
        iso = IsolationForest(
            n_estimators=n_est, contamination=0.05, random_state=SEED, n_jobs=-1
        )
        iso.fit(X_train)
        y_pred = (iso.predict(X_val) == -1).astype(int)
        
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        
        results.append({
            "n_estimators": n_est,
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
        })
        
        log(f"    n_estimators={n_est}: F1={f1:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "16_9_convergence.csv", index=False)
    log(f"  [OK] Saved: 16_9_convergence.csv")
    
    # نمودار
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(results_df["n_estimators"], results_df["F1"], "o-", color="steelblue", linewidth=2)
    ax.set_xlabel("n_estimators")
    ax.set_ylabel("F1 Score")
    ax.set_title("Convergence: F1 vs n_estimators")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "16_9_convergence.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 16_9_convergence.png")
    
    return results_df

# ============================================================
# 16.10 FINAL MODEL SELECTION
# ============================================================
def stage_16_10_final_selection(tuning_df, stability_df):
    log("\n" + "=" * 70)
    log("STAGE 16.10: FINAL MODEL SELECTION")
    log("=" * 70)
    
    best = tuning_df.iloc[0]
    
    log(f"  Best Configuration:")
    log(f"    n_estimators: {best['n_estimators']}")
    log(f"    max_samples: {best['max_samples']}")
    log(f"    contamination: {best['contamination']}")
    log(f"    max_features: {best['max_features']}")
    log(f"    bootstrap: {best['bootstrap']}")
    log(f"    F1: {best['F1']:.6f}")
    log(f"    Precision: {best['Precision']:.6f}")
    log(f"    Recall: {best['Recall']:.6f}")
    
    log(f"\n  Stability: F1 Std = {stability_df['F1'].std():.6f}")
    
    selection = {
        "Best_n_estimators": int(best["n_estimators"]),
        "Best_max_samples": best["max_samples"],
        "Best_contamination": best["contamination"],
        "Best_max_features": best["max_features"],
        "Best_bootstrap": bool(best["bootstrap"]),
        "Best_F1": round(best["F1"], 6),
        "Best_Precision": round(best["Precision"], 6),
        "Best_Recall": round(best["Recall"], 6),
        "Stability_Std": round(stability_df["F1"].std(), 6),
        "Stability_Status": "STABLE" if stability_df["F1"].std() < 0.05 else "UNSTABLE",
    }
    
    pd.DataFrame(list(selection.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "16_10_final_selection.csv", index=False
    )
    log(f"\n  [OK] Saved: 16_10_final_selection.csv")
    
    return selection

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 16: HYPERPARAMETER TUNING & REFINEMENT")
    log("=" * 70)
    
    train_df, val_df, test_df, raw_features = load_data()
    
    # 16.1
    strategy = stage_16_1_strategy()
    
    # 16.2-16.4
    tuning_df = stage_16_2_to_16_4_tuning(train_df, val_df, raw_features)
    
    # 16.5
    ablation_df = stage_16_5_ablation(train_df, val_df, raw_features)
    
    # 16.6
    sensitivity_df = stage_16_6_sensitivity(train_df, val_df, raw_features)
    
    # 16.7
    robustness_df = stage_16_7_robustness(train_df, val_df, test_df, raw_features)
    
    # 16.8
    stability_df = stage_16_8_stability(train_df, val_df, raw_features)
    
    # 16.9
    convergence_df = stage_16_9_convergence(train_df, val_df, raw_features)
    
    # 16.10
    selection = stage_16_10_final_selection(tuning_df, stability_df)
    
    log("\n" + "=" * 70)
    log("PHASE 16 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase16_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()