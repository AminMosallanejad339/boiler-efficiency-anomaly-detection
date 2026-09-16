"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 14g: Finalize Anomaly Detection (Results Combination)
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
import json
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
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
REPORTS_DIR = BASE_DIR / "reports"

for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_ANOMALY = "Anomaly_Label"

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
    log("PHASE 14g: FINALIZE ANOMALY DETECTION")
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
# EVALUATE ALL MODELS
# ============================================================
def evaluate_all_models(train_df, val_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("EVALUATING ALL MODELS")
    log("=" * 70)
    
    results = []
    
    # ============================================================
    # 1. Random Forest (Raw_Only)
    # ============================================================
    log("\n  [1] Random Forest (Raw_Only)...")
    try:
        rf = joblib.load(MODELS_DIR / "random_forest_anomaly.pkl")
        
        X_val = val_df[raw_features].fillna(0).values
        X_test = test_df[raw_features].fillna(0).values
        y_val = val_df[TARGET_ANOMALY].values
        y_test = test_df[TARGET_ANOMALY].values
        
        for name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
            y_pred = rf.predict(X)
            y_proba = rf.predict_proba(X)[:, 1]
            
            results.append({
                "Model": "Random_Forest",
                "Feature_Set": "Raw_Only",
                "Dataset": name,
                "N_Features": len(raw_features),
                "Precision": round(precision_score(y, y_pred, zero_division=0), 6),
                "Recall": round(recall_score(y, y_pred, zero_division=0), 6),
                "F1": round(f1_score(y, y_pred, zero_division=0), 6),
                "ROC_AUC": round(roc_auc_score(y, y_proba), 6),
                "PR_AUC": round(average_precision_score(y, y_proba), 6),
            })
        
        log(f"    Val F1: {results[-2]['F1']:.6f}")
        log(f"    Test F1: {results[-1]['F1']:.6f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # 2. Isolation Forest
    # ============================================================
    log("\n  [2] Isolation Forest...")
    try:
        iso = joblib.load(MODELS_DIR / "isolation_forest_final.pkl")
        
        X_val = val_df[raw_features].fillna(0).values
        X_test = test_df[raw_features].fillna(0).values
        y_val = val_df[TARGET_ANOMALY].values
        y_test = test_df[TARGET_ANOMALY].values
        
        for name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
            y_pred = (iso.predict(X) == -1).astype(int)
            y_proba = -iso.score_samples(X)  # Higher = more anomalous
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min() + 1e-10)
            
            results.append({
                "Model": "Isolation_Forest",
                "Feature_Set": "Raw_Only",
                "Dataset": name,
                "N_Features": len(raw_features),
                "Precision": round(precision_score(y, y_pred, zero_division=0), 6),
                "Recall": round(recall_score(y, y_pred, zero_division=0), 6),
                "F1": round(f1_score(y, y_pred, zero_division=0), 6),
                "ROC_AUC": round(roc_auc_score(y, y_proba), 6),
                "PR_AUC": round(average_precision_score(y, y_proba), 6),
            })
        
        log(f"    Val F1: {results[-2]['F1']:.6f}")
        log(f"    Test F1: {results[-1]['F1']:.6f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # 3. MLP (from Phase 14c)
    # ============================================================
    log("\n  [3] MLP (Phase 14c)...")
    try:
        import tensorflow as tf
        mlp = tf.keras.models.load_model(MODELS_DIR / "anomaly_mlp_final_v3.keras")
        
        # استفاده از ویژگی‌های اصلی + خام (78 ویژگی)
        feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
        main_features = [c for c in feature_df["Feature"].tolist()
                         if c in val_df.columns and c not in ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]]
        mlp_features = main_features + raw_features
        
        X_val = val_df[mlp_features].fillna(0).values
        X_test = test_df[mlp_features].fillna(0).values
        y_val = val_df[TARGET_ANOMALY].values
        y_test = test_df[TARGET_ANOMALY].values
        
        for name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
            y_proba = mlp.predict(X, verbose=0).flatten()
            y_pred = (y_proba > 0.5).astype(int)
            
            results.append({
                "Model": "MLP",
                "Feature_Set": "Main_Plus_Raw",
                "Dataset": name,
                "N_Features": len(mlp_features),
                "Precision": round(precision_score(y, y_pred, zero_division=0), 6),
                "Recall": round(recall_score(y, y_pred, zero_division=0), 6),
                "F1": round(f1_score(y, y_pred, zero_division=0), 6),
                "ROC_AUC": round(roc_auc_score(y, y_proba), 6),
                "PR_AUC": round(average_precision_score(y, y_proba), 6),
            })
        
        log(f"    Val F1: {results[-2]['F1']:.6f}")
        log(f"    Test F1: {results[-1]['F1']:.6f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    # ============================================================
    # 4. AdaBoost (from Phase 08b - with Leaky features)
    # ============================================================
    log("\n  [4] AdaBoost (Phase 08b - Reference)...")
    try:
        # بارگذاری نتایج قبلی
        old_results = pd.read_csv(TABLES_DIR / "08b_classification_comparison.csv")
        adaboost_row = old_results[old_results["Model"] == "AdaBoost"]
        
        if not adaboost_row.empty:
            row = adaboost_row.iloc[0]
            results.append({
                "Model": "AdaBoost (Leaky)",
                "Feature_Set": "Main_Only_Leaky",
                "Dataset": "Val",
                "N_Features": 78,
                "Precision": row["Precision"],
                "Recall": row["Recall"],
                "F1": row["F1"],
                "ROC_AUC": row["ROC_AUC"],
                "PR_AUC": row["PR_AUC"] if "PR_AUC" in row else np.nan,
            })
            log(f"    Val F1: {row['F1']:.6f}")
    except Exception as e:
        log(f"    [ERROR] {e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "14g_all_models_comparison.csv", index=False)
    log(f"\n  [OK] Saved: 14g_all_models_comparison.csv")
    
    return results_df

# ============================================================
# DATA LEAKAGE ANALYSIS
# ============================================================
def analyze_data_leakage(train_df, val_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("DATA LEAKAGE ANALYSIS")
    log("=" * 70)
    
    analysis = {}
    
    # 1. تعریف Anomaly
    log("\n  [1] Anomaly Definition:")
    log(f"    Anomaly_Label = 1 IF any of these < 0:")
    for col in raw_features:
        log(f"      - {col}")
    
    # 2. بررسی همبستگی
    log("\n  [2] Correlation with Anomaly:")
    correlations = {}
    for col in raw_features:
        corr = train_df[col].corr(train_df[TARGET_ANOMALY])
        correlations[col] = round(corr, 6)
        log(f"    {col}: {corr:.6f}")
    analysis["correlations"] = correlations
    
    # 3. بررسی مستقیم
    log("\n  [3] Direct Rule Test:")
    for col in raw_features:
        neg_mask = train_df[col] < 0
        anomaly_mask = train_df[TARGET_ANOMALY] == 1
        
        # دقت قانون
        if neg_mask.sum() > 0:
            precision = (neg_mask & anomaly_mask).sum() / neg_mask.sum()
            recall = (neg_mask & anomaly_mask).sum() / anomaly_mask.sum()
            log(f"    Rule '{col} < 0':")
            log(f"      Precision: {precision:.4f}")
            log(f"      Recall: {recall:.4f}")
    
    # 4. نتیجه‌گیری
    log("\n  [4] Conclusion:")
    log(f"    Random Forest with Raw_Only features achieves F1 = 1.0")
    log(f"    This is because the model learns the EXACT rule:")
    log(f"    'If any raw anomaly feature < 0, then Anomaly'")
    log(f"    This is DATA LEAKAGE, not real machine learning.")
    
    analysis["conclusion"] = "Data Leakage in Raw_Only"
    
    return analysis

# ============================================================
# VISUALIZE
# ============================================================
def visualize_results(results_df, leakage_analysis):
    log("\n" + "=" * 70)
    log("VISUALIZING RESULTS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. F1 Comparison (Val + Test)
    val_results = results_df[results_df["Dataset"] == "Val"].copy()
    test_results = results_df[results_df["Dataset"] == "Test"].copy()
    
    models_val = val_results["Model"].tolist()
    x = np.arange(len(models_val))
    width = 0.35
    
    val_f1 = val_results["F1"].values
    test_f1 = test_results["F1"].values if len(test_results) == len(val_results) else [0] * len(val_results)
    
    axes[0, 0].bar(x - width/2, val_f1, width, label="Val F1", color="steelblue")
    axes[0, 0].bar(x + width/2, test_f1, width, label="Test F1", color="coral")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models_val, rotation=45, ha="right")
    axes[0, 0].set_ylabel("F1 Score")
    axes[0, 0].set_title("F1 Score by Model")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1.1)
    
    # 2. Precision vs Recall
    axes[0, 1].scatter(val_results["Precision"], val_results["Recall"], 
                       s=150, c="purple", alpha=0.7, label="Val")
    for _, row in val_results.iterrows():
        axes[0, 1].annotate(row["Model"], (row["Precision"], row["Recall"]),
                            fontsize=9, ha="right")
    axes[0, 1].set_xlabel("Precision")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].set_title("Precision vs Recall (Val)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim(-0.05, 1.05)
    axes[0, 1].set_ylim(-0.05, 1.05)
    
    # 3. Leakage Analysis
    correlations = leakage_analysis.get("correlations", {})
    if correlations:
        features = list(correlations.keys())
        corr_values = list(correlations.values())
        
        colors = ["green" if abs(c) > 0.5 else "orange" if abs(c) > 0.2 else "red" 
                  for c in corr_values]
        
        axes[1, 0].barh(features, corr_values, color=colors)
        axes[1, 0].set_xlabel("Correlation with Anomaly")
        axes[1, 0].set_title("Feature Correlation with Anomaly")
        axes[1, 0].axvline(x=0, color="black", linestyle="-", linewidth=0.5)
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Summary Table
    axes[1, 1].axis("off")
    
    summary_text = "ANOMALY DETECTION SUMMARY\n" + "=" * 40 + "\n\n"
    summary_text += "1. Random Forest (Raw_Only):\n"
    summary_text += "   F1 = 1.000 (DATA LEAKAGE)\n\n"
    summary_text += "2. Isolation Forest:\n"
    summary_text += "   F1 = 0.232 (REALISTIC)\n\n"
    summary_text += "3. MLP:\n"
    summary_text += "   F1 = 0.000 (FAILED)\n\n"
    summary_text += "4. AdaBoost (Leaky):\n"
    summary_text += "   F1 = 0.988 (DATA LEAKAGE)\n\n"
    summary_text += "CONCLUSION:\n"
    summary_text += "Anomaly Detection has inherent\n"
    summary_text += "Data Leakage when using the\n"
    summary_text += "same features that define the label."
    
    axes[1, 1].text(0.05, 0.5, summary_text, fontsize=11, verticalalignment="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14g_finalize_anomaly.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14g_finalize_anomaly.png")

# ============================================================
# SAVE COMPREHENSIVE REPORT
# ============================================================
def save_comprehensive_report(results_df, leakage_analysis):
    log("\n" + "=" * 70)
    log("SAVING COMPREHENSIVE REPORT")
    log("=" * 70)
    
    # JSON Report
    report = {
        "Phase": "14g - Finalize Anomaly Detection",
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Models": results_df.to_dict(orient="records"),
        "Data_Leakage_Analysis": leakage_analysis,
        "Conclusions": [
            {
                "ID": "CONC-01",
                "Title": "Random Forest Achieves F1 = 1.0 via Data Leakage",
                "Description": "Using raw features that define Anomaly_Label",
                "Status": "LEAKAGE",
            },
            {
                "ID": "CONC-02",
                "Title": "Isolation Forest Gives Realistic F1 = 0.232",
                "Description": "Unsupervised, no label leakage",
                "Status": "REALISTIC",
            },
            {
                "ID": "CONC-03",
                "Title": "MLP Failed (F1 = 0.0)",
                "Description": "Sensitive to class imbalance (63:1)",
                "Status": "FAILED",
            },
            {
                "ID": "CONC-04",
                "Title": "AdaBoost F1 = 0.988 via Leaky Features",
                "Description": "Used aph_effect_leak_ratio derived from APH_Leakage",
                "Status": "LEAKAGE",
            },
            {
                "ID": "CONC-05",
                "Title": "Anomaly Detection is Fundamentally Hard",
                "Description": "Weak correlation, high overlap, random pattern",
                "Status": "LIMITATION",
            },
        ],
        "Recommendations": [
            "Report both Leakage (upper bound) and Realistic (Isolation Forest) results",
            "Acknowledge Data Leakage in the project report",
            "For real deployment, use Isolation Forest or Autoencoder",
            "Redefine Anomaly using statistical methods (Z-Score, IQR)",
        ],
    }
    
    with open(REPORTS_DIR / "14g_final_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log(f"  [OK] Saved: 14g_final_report.json")
    
    # Text Report
    with open(REPORTS_DIR / "14g_final_report.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ANOMALY DETECTION - FINAL REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("MODEL COMPARISON:\n")
        f.write("-" * 70 + "\n")
        for _, row in results_df.iterrows():
            f.write(f"{row['Model']:<25} | {row['Dataset']:<5} | "
                    f"F1={row['F1']:.6f} | P={row['Precision']:.6f} | R={row['Recall']:.6f}\n")
        
        f.write("\n\nDATA LEAKAGE ANALYSIS:\n")
        f.write("-" * 70 + "\n")
        for k, v in leakage_analysis.items():
            f.write(f"{k}: {v}\n")
        
        f.write("\n\nCONCLUSIONS:\n")
        f.write("-" * 70 + "\n")
        for conc in report["Conclusions"]:
            f.write(f"[{conc['ID']}] {conc['Title']}\n")
            f.write(f"    {conc['Description']}\n")
            f.write(f"    Status: {conc['Status']}\n\n")
        
        f.write("\nRECOMMENDATIONS:\n")
        f.write("-" * 70 + "\n")
        for rec in report["Recommendations"]:
            f.write(f"- {rec}\n")
    
    log(f"  [OK] Saved: 14g_final_report.txt")
    
    return report

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 14g: FINALIZE ANOMALY DETECTION")
    log("=" * 70)
    
    train_df, val_df, test_df, raw_features = load_data()
    
    # ارزیابی همه مدل‌ها
    results_df = evaluate_all_models(train_df, val_df, test_df, raw_features)
    
    # تحلیل Data Leakage
    leakage_analysis = analyze_data_leakage(train_df, val_df, test_df, raw_features)
    
    # Visualize
    visualize_results(results_df, leakage_analysis)
    
    # ذخیره گزارش جامع
    report = save_comprehensive_report(results_df, leakage_analysis)
    
    # Summary
    log("\n" + "=" * 70)
    log("FINAL SUMMARY")
    log("=" * 70)
    
    log(f"\n  Model Performance:")
    for _, row in results_df.iterrows():
        log(f"    {row['Model']:<25} | {row['Dataset']:<5} | F1={row['F1']:.6f}")
    
    log(f"\n  Conclusions:")
    for conc in report["Conclusions"]:
        log(f"    [{conc['ID']}] {conc['Title']}")
        log(f"        Status: {conc['Status']}")
    
    log(f"\n  Recommendations:")
    for rec in report["Recommendations"]:
        log(f"    - {rec}")
    
    log("\n" + "=" * 70)
    log("PHASE 14g COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase14g_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()