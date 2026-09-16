"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 14d: Isolation Forest (Unsupervised Anomaly Detection)
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
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import IsolationForest
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

BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
SPLITS_DIR = BASE_DIR / "data" / "splits"
RAW_DATA = BASE_DIR / "data" / "raw" / "industrial_dataset.csv"
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

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
# ISOLATION FOREST
# ============================================================
def run_isolation_forest(X_train, y_train, X_val, y_val, X_test, y_test, features):
    log("\n" + "=" * 70)
    log("ISOLATION FOREST - UNSUPERVISED ANOMALY DETECTION")
    log("=" * 70)
    
    # فقط ویژگی‌های خام Anomaly
    log(f"\n  Using features: {features}")
    
    X_train_iso = X_train[features].values
    X_val_iso = X_val[features].values
    X_test_iso = X_test[features].values
    
    log(f"  X_train: {X_train_iso.shape}")
    log(f"  X_val: {X_val_iso.shape}")
    log(f"  X_test: {X_test_iso.shape}")
    
    # تست Contaminationهای مختلف
    contamination_values = [0.01, 0.015, 0.02, 0.03, 0.05]
    
    results = []
    
    for contamination in contamination_values:
        log(f"\n  Testing contamination={contamination}...")
        
        # آموزش
        iso = IsolationForest(
            contamination=contamination,
            random_state=SEED,
            n_estimators=200,
            max_samples="auto",
            n_jobs=-1,
        )
        iso.fit(X_train_iso)
        
        # پیش‌بینی (1 = normal, -1 = anomaly)
        y_train_pred = iso.predict(X_train_iso)
        y_val_pred = iso.predict(X_val_iso)
        y_test_pred = iso.predict(X_test_iso)
        
        # تبدیل به 0/1 (0 = normal, 1 = anomaly)
        y_train_pred = (y_train_pred == -1).astype(int)
        y_val_pred = (y_val_pred == -1).astype(int)
        y_test_pred = (y_test_pred == -1).astype(int)
        
        # معیارها
        for name, y_true, y_pred in [
            ("Train", y_train, y_train_pred),
            ("Val", y_val, y_val_pred),
            ("Test", y_test, y_test_pred),
        ]:
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            
            results.append({
                "Contamination": contamination,
                "Dataset": name,
                "Precision": round(prec, 6),
                "Recall": round(rec, 6),
                "F1": round(f1, 6),
                "Anomalies_Detected": int(y_pred.sum()),
                "Total": len(y_pred),
                "Anomaly_Rate": round(y_pred.mean() * 100, 4),
            })
        
        log(f"    Val: Precision={results[-2]['Precision']:.4f}, Recall={results[-2]['Recall']:.4f}, F1={results[-2]['F1']:.4f}")
        log(f"    Test: Precision={results[-1]['Precision']:.4f}, Recall={results[-1]['Recall']:.4f}, F1={results[-1]['F1']:.4f}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "14d_isolation_forest_results.csv", index=False)
    log(f"\n  [OK] Saved: 14d_isolation_forest_results.csv")
    
    return results_df, iso

# ============================================================
# VISUALIZE
# ============================================================
def visualize_results(results_df):
    log("\n" + "=" * 70)
    log("VISUALIZING ISOLATION FOREST RESULTS")
    log("=" * 70)
    
    val_df = results_df[results_df["Dataset"] == "Val"]
    test_df = results_df[results_df["Dataset"] == "Test"]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Val F1
    axes[0].plot(val_df["Contamination"], val_df["F1"], "o-", color="steelblue", linewidth=2)
    axes[0].set_xlabel("Contamination")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Validation F1 vs Contamination")
    axes[0].grid(True, alpha=0.3)
    
    # Test F1
    axes[1].plot(test_df["Contamination"], test_df["F1"], "o-", color="coral", linewidth=2)
    axes[1].set_xlabel("Contamination")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Test F1 vs Contamination")
    axes[1].grid(True, alpha=0.3)
    
    # Precision vs Recall
    axes[2].scatter(val_df["Precision"], val_df["Recall"], s=100, c="purple", alpha=0.7, label="Val")
    axes[2].scatter(test_df["Precision"], test_df["Recall"], s=100, c="green", alpha=0.7, label="Test")
    for _, row in val_df.iterrows():
        axes[2].annotate(f"{row['Contamination']}", (row["Precision"], row["Recall"]), fontsize=8)
    axes[2].set_xlabel("Precision")
    axes[2].set_ylabel("Recall")
    axes[2].set_title("Precision vs Recall")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14d_isolation_forest.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14d_isolation_forest.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 14d: ISOLATION FOREST")
    log("=" * 70)
    
    # بارگذاری داده
    train_df, val_df, test_df, feature_df, raw_features = load_data()
    
    # استفاده از ویژگی‌های خام Anomaly
    X_train = train_df
    y_train = train_df[TARGET_ANOMALY].values
    X_val = val_df
    y_val = val_df[TARGET_ANOMALY].values
    X_test = test_df
    y_test = test_df[TARGET_ANOMALY].values
    
    log(f"\n  Anomaly Distribution:")
    log(f"    Train: {y_train.sum()} ({y_train.mean()*100:.4f}%)")
    log(f"    Val: {y_val.sum()} ({y_val.mean()*100:.4f}%)")
    log(f"    Test: {y_test.sum()} ({y_test.mean()*100:.4f}%)")
    
    # اجرای Isolation Forest
    results_df, iso = run_isolation_forest(
        X_train, y_train, X_val, y_val, X_test, y_test, raw_features
    )
    
    # Visualize
    visualize_results(results_df)
    
    # بهترین مدل
    best_val = results_df[results_df["Dataset"] == "Val"].sort_values("F1", ascending=False).iloc[0]
    
    log(f"\n  [BEST] Contamination: {best_val['Contamination']}")
    log(f"  Val F1: {best_val['F1']:.6f}")
    log(f"  Val Precision: {best_val['Precision']:.6f}")
    log(f"  Val Recall: {best_val['Recall']:.6f}")
    
    # ذخیره مدل نهایی
    best_contamination = best_val["Contamination"]
    final_iso = IsolationForest(
        contamination=best_contamination,
        random_state=SEED,
        n_estimators=200,
        n_jobs=-1,
    )
    final_iso.fit(X_train[raw_features].values)
    joblib.dump(final_iso, MODELS_DIR / "isolation_forest_final.pkl")
    log(f"  [OK] Saved: isolation_forest_final.pkl")
    
    # خلاصه
    summary = {
        "Method": "Isolation Forest",
        "Best_Contamination": best_contamination,
        "Val_F1": round(best_val["F1"], 6),
        "Val_Precision": round(best_val["Precision"], 6),
        "Val_Recall": round(best_val["Recall"], 6),
        "Features_Used": ", ".join(raw_features),
    }
    
    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "14d_isolation_forest_summary.csv", index=False
    )
    log(f"  [OK] Saved: 14d_isolation_forest_summary.csv")
    
    log("\n" + "=" * 70)
    log("PHASE 14d COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase14d_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()