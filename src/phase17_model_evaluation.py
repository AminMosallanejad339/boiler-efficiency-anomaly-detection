"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 17: Model Evaluation (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Isolation Forest (Tuned) - Final Model
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
from sklearn.ensemble import IsolationForest
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    classification_report, roc_curve, precision_recall_curve,
    matthews_corrcoef, cohen_kappa_score, log_loss,
    brier_score_loss,
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
    log("PHASE 17: MODEL EVALUATION")
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
# 17.1 EVALUATION STRATEGY
# ============================================================
def stage_17_1_strategy():
    log("\n" + "=" * 70)
    log("STAGE 17.1: MODEL EVALUATION STRATEGY")
    log("=" * 70)

    strategy = {
        "Final Model": "Isolation Forest (Tuned)",
        "Best Params": {
            "n_estimators": 300,
            "max_samples": 0.9,
            "contamination": 0.02,
            "max_features": 0.5,
            "bootstrap": False,
        },
        "Evaluation Metrics": [
            "Accuracy", "Precision", "Recall", "F1",
            "ROC-AUC", "PR-AUC", "MCC", "Cohen's Kappa",
            "Brier Score", "Log Loss",
        ],
        "Evaluation Methods": [
            "Test Set Evaluation",
            "Cross-Validation",
            "ROC Curve Analysis",
            "Precision-Recall Analysis",
            "Calibration Analysis",
            "Error Analysis",
        ],
        "Baseline Models": [
            "Random Forest (Leakage)",
            "Isolation Forest (Default)",
            "MLP (Failed)",
            "Dummy Classifier",
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
# TRAIN FINAL ISOLATION FOREST
# ============================================================
def train_final_iso(train_df, raw_features):
    log("\n" + "=" * 70)
    log("TRAINING FINAL ISOLATION FOREST (TUNED)")
    log("=" * 70)

    best_params = {
        "n_estimators": 300,
        "max_samples": 0.9,
        "contamination": 0.02,
        "max_features": 0.5,
        "bootstrap": False,
        "random_state": SEED,
        "n_jobs": -1,
    }

    log(f"  Best Params: {best_params}")

    X_train = train_df[raw_features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values

    iso = IsolationForest(**best_params)
    iso.fit(X_train)

    # ذخیره
    joblib.dump(iso, MODELS_DIR / "isolation_forest_tuned.pkl")
    log(f"  [OK] Saved: isolation_forest_tuned.pkl")

    return iso, best_params

# ============================================================
# 17.2-17.3 TEST SET EVALUATION & METRICS
# ============================================================
def stage_17_2_to_17_3_evaluation(iso, train_df, val_df, test_df, raw_features, best_params):
    log("\n" + "=" * 70)
    log("STAGE 17.2-17.3: TEST SET EVALUATION & METRICS")
    log("=" * 70)

    results = []

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        X = df[raw_features].fillna(0).values
        y = df[TARGET_ANOMALY].values

        # پیش‌بینی
        y_pred = (iso.predict(X) == -1).astype(int)
        y_score = -iso.score_samples(X)

        # نرمال‌سازی Score به [0, 1] برای ROC-AUC
        y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-10)
        y_score_norm = np.clip(y_score_norm, 1e-10, 1 - 1e-10)

        # متریک‌ها
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y, y_score_norm)
        pr_auc = average_precision_score(y, y_score_norm)
        mcc = matthews_corrcoef(y, y_pred)
        kappa = cohen_kappa_score(y, y_pred)
        brier = brier_score_loss(y, y_score_norm)

        try:
            logloss = log_loss(y, y_score_norm)
        except Exception:
            logloss = np.nan

        results.append({
            "Dataset": name,
            "N_Samples": len(y),
            "N_Anomalies": int(y.sum()),
            "Anomaly_Rate_%": round(y.mean() * 100, 4),
            "Accuracy": round(acc, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
            "ROC_AUC": round(roc_auc, 6),
            "PR_AUC": round(pr_auc, 6),
            "MCC": round(mcc, 6),
            "Cohen_Kappa": round(kappa, 6),
            "Brier_Score": round(brier, 6),
            "Log_Loss": round(logloss, 6) if not np.isnan(logloss) else np.nan,
        })

        log(f"\n  {name}:")
        log(f"    N Samples: {len(y)}")
        log(f"    N Anomalies: {int(y.sum())} ({y.mean()*100:.4f}%)")
        log(f"    Accuracy: {acc:.6f}")
        log(f"    Precision: {prec:.6f}")
        log(f"    Recall: {rec:.6f}")
        log(f"    F1: {f1:.6f}")
        log(f"    ROC-AUC: {roc_auc:.6f}")
        log(f"    PR-AUC: {pr_auc:.6f}")
        log(f"    MCC: {mcc:.6f}")
        log(f"    Cohen's Kappa: {kappa:.6f}")
        log(f"    Brier Score: {brier:.6f}")
        log(f"    Log Loss: {logloss:.6f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "17_2_evaluation_metrics.csv", index=False)
    log(f"\n  [OK] Saved: 17_2_evaluation_metrics.csv")

    return results_df

# ============================================================
# 17.4 PERFORMANCE ANALYSIS
# ============================================================
def stage_17_4_performance(results_df):
    log("\n" + "=" * 70)
    log("STAGE 17.4: PERFORMANCE ANALYSIS")
    log("=" * 70)

    test_row = results_df[results_df["Dataset"] == "Test"].iloc[0]
    val_row = results_df[results_df["Dataset"] == "Val"].iloc[0]
    train_row = results_df[results_df["Dataset"] == "Train"].iloc[0]

    log(f"  Test Performance:")
    log(f"    F1: {test_row['F1']:.6f}")
    log(f"    Precision: {test_row['Precision']:.6f}")
    log(f"    Recall: {test_row['Recall']:.6f}")
    log(f"    ROC-AUC: {test_row['ROC_AUC']:.6f}")

    log(f"\n  Comparison Test vs Val:")
    log(f"    F1: {test_row['F1']:.6f} vs {val_row['F1']:.6f} (Δ = {test_row['F1'] - val_row['F1']:.6f})")
    log(f"    Precision: {test_row['Precision']:.6f} vs {val_row['Precision']:.6f} (Δ = {test_row['Precision'] - val_row['Precision']:.6f})")
    log(f"    Recall: {test_row['Recall']:.6f} vs {val_row['Recall']:.6f} (Δ = {test_row['Recall'] - val_row['Recall']:.6f})")

    log(f"\n  Interpretation:")
    if test_row['F1'] > val_row['F1']:
        log(f"    Test > Val: Good generalization (no overfitting)")
    else:
        log(f"    Val > Test: Slight overfitting on validation")

    log(f"\n  ROC-AUC Interpretation:")
    if test_row['ROC_AUC'] > 0.9:
        log(f"    Excellent discrimination (AUC > 0.9)")
    elif test_row['ROC_AUC'] > 0.8:
        log(f"    Good discrimination (AUC > 0.8)")
    elif test_row['ROC_AUC'] > 0.7:
        log(f"    Acceptable discrimination (AUC > 0.7)")
    else:
        log(f"    Poor discrimination (AUC < 0.7)")

    return results_df

# ============================================================
# 17.5 ERROR ANALYSIS
# ============================================================
def stage_17_5_error_analysis(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 17.5: ERROR ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    y_pred = (iso.predict(X) == -1).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    log(f"  Confusion Matrix:")
    log(f"    TN (True Normal):    {tn}")
    log(f"    FP (False Anomaly):  {fp}")
    log(f"    FN (Missed Anomaly): {fn}")
    log(f"    TP (True Anomaly):   {tp}")

    log(f"\n  Error Analysis:")
    log(f"    False Positive Rate: {fp / (fp + tn):.6f}")
    log(f"    False Negative Rate: {fn / (fn + tp):.6f}")
    log(f"    True Positive Rate:  {tp / (fn + tp):.6f}")

    # تحلیل FN
    if fn > 0:
        log(f"\n  False Negatives (Missed Anomalies):")
        fn_indices = np.where((y == 1) & (y_pred == 0))[0]

        fn_df = test_df.iloc[fn_indices][raw_features + [TARGET_ANOMALY]]
        log(f"    Count: {len(fn_df)}")
        log(f"    Mean of each feature in FN:")
        for col in raw_features:
            log(f"      {col}: {fn_df[col].mean():.6f}")

    # تحلیل FP
    if fp > 0:
        log(f"\n  False Positives (False Alarms):")
        fp_indices = np.where((y == 0) & (y_pred == 1))[0]

        fp_df = test_df.iloc[fp_indices][raw_features + [TARGET_ANOMALY]]
        log(f"    Count: {len(fp_df)}")
        log(f"    Mean of each feature in FP:")
        for col in raw_features:
            log(f"      {col}: {fp_df[col].mean():.6f}")

    # ذخیره
    cm_df = pd.DataFrame({
        "Metric": ["TN", "FP", "FN", "TP", "FPR", "FNR", "TPR"],
        "Value": [tn, fp, fn, tp, fp/(fp+tn), fn/(fn+tp), tp/(fn+tp)],
    })
    cm_df.to_csv(TABLES_DIR / "17_5_error_analysis.csv", index=False)
    log(f"\n  [OK] Saved: 17_5_error_analysis.csv")

    return cm, {"TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}

# ============================================================
# 17.6 CONFUSION MATRIX ANALYSIS
# ============================================================
def stage_17_6_confusion_matrix(cm):
    log("\n" + "=" * 70)
    log("STAGE 17.6: CONFUSION MATRIX ANALYSIS")
    log("=" * 70)

    tn, fp, fn, tp = cm.ravel()

    # محاسبات
    total = tn + fp + fn + tp
    accuracy = (tn + tp) / total
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    log(f"  Derived Metrics:")
    log(f"    Accuracy: {accuracy:.6f}")
    log(f"    Precision: {precision:.6f}")
    log(f"    Recall (Sensitivity): {recall:.6f}")
    log(f"    Specificity: {specificity:.6f}")
    log(f"    F1: {f1:.6f}")

    # نمودار
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"],
                annot_kws={"size": 16})
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title("Confusion Matrix - Isolation Forest (Tuned)", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_6_confusion_matrix.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 17_6_confusion_matrix.png")

    return {"accuracy": accuracy, "precision": precision, "recall": recall, "specificity": specificity, "f1": f1}

# ============================================================
# 17.7 ROC CURVE ANALYSIS
# ============================================================
def stage_17_7_roc(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 17.7: ROC CURVE ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    y_score = -iso.score_samples(X)
    y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-10)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_score_norm)
    roc_auc = roc_auc_score(y, y_score_norm)

    # Youden's J
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
    best_fpr = fpr[best_idx]
    best_tpr = tpr[best_idx]

    log(f"  ROC-AUC: {roc_auc:.6f}")
    log(f"  Best Threshold (Youden's J): {best_threshold:.6f}")
    log(f"    FPR: {best_fpr:.6f}")
    log(f"    TPR: {best_tpr:.6f}")
    log(f"    Youden's J: {j_scores[best_idx]:.6f}")

    # نمودار
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Random")
    ax.scatter(best_fpr, best_tpr, color="green", s=100, zorder=5,
               label=f"Best (FPR={best_fpr:.3f}, TPR={best_tpr:.3f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Isolation Forest (Tuned)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_7_roc_curve.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 17_7_roc_curve.png")

    return {"roc_auc": roc_auc, "best_threshold": best_threshold, "fpr": fpr, "tpr": tpr}

# ============================================================
# 17.8 PRECISION-RECALL ANALYSIS
# ============================================================
def stage_17_8_pr(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 17.8: PRECISION-RECALL ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    y_score = -iso.score_samples(X)
    y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-10)

    # PR Curve
    precision, recall, thresholds = precision_recall_curve(y, y_score_norm)
    pr_auc = average_precision_score(y, y_score_norm)

    # Baseline (نسبت کلاس)
    baseline = y.mean()

    log(f"  PR-AUC: {pr_auc:.6f}")
    log(f"  Baseline (Anomaly Rate): {baseline:.6f}")
    log(f"  Improvement over Baseline: {(pr_auc - baseline) / baseline * 100:.2f}%")

    # نمودار
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall, precision, "b-", linewidth=2, label=f"PR (AUC = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="r", linestyle="--", linewidth=1,
               label=f"Baseline ({baseline:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Isolation Forest (Tuned)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_8_pr_curve.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 17_8_pr_curve.png")

    return {"pr_auc": pr_auc, "baseline": baseline, "precision": precision, "recall": recall}

# ============================================================
# 17.9 CALIBRATION ANALYSIS
# ============================================================
def stage_17_9_calibration(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 17.9: CALIBRATION ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    y_score = -iso.score_samples(X)
    y_score_norm = (y_score - y_score.min()) / (y_score.max() - y_score.min() + 1e-10)

    # Calibration Curve
    try:
        fraction_positive, mean_predicted = calibration_curve(
            y, y_score_norm, n_bins=10, strategy="uniform"
        )

        # Brier Score
        brier = brier_score_loss(y, y_score_norm)

        log(f"  Brier Score: {brier:.6f}")
        log(f"  Calibration Curve computed")

        # نمودار
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(mean_predicted, fraction_positive, "s-", color="steelblue",
                linewidth=2, label="Isolation Forest")
        ax.plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfectly Calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve - Isolation Forest (Tuned)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "17_9_calibration.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 17_9_calibration.png")

        return {"brier": brier, "fraction_positive": fraction_positive, "mean_predicted": mean_predicted}
    except Exception as e:
        log(f"  [ERROR] {e}")
        return {}

# ============================================================
# 17.10 BENCHMARK COMPARISON
# ============================================================
def stage_17_10_benchmark(results_df):
    log("\n" + "=" * 70)
    log("STAGE 17.10: BENCHMARK COMPARISON")
    log("=" * 70)

    test_row = results_df[results_df["Dataset"] == "Test"].iloc[0]

    # Benchmark: Dummy Classifier (همه را Normal پیش‌بینی می‌کند)
    dummy_accuracy = 1 - test_row["Anomaly_Rate_%"] / 100
    dummy_f1 = 0.0  # چون هیچ Anomaly تشخیص نمی‌دهد

    log(f"  Model Comparison (Test Set):")
    log(f"    {'Model':<30} {'F1':<12} {'Precision':<12} {'Recall':<12} {'ROC-AUC':<12}")
    log(f"    {'-'*78}")
    log(f"    {'Isolation Forest (Tuned)':<30} {test_row['F1']:<12.6f} {test_row['Precision']:<12.6f} {test_row['Recall']:<12.6f} {test_row['ROC_AUC']:<12.6f}")
    log(f"    {'Dummy (All Normal)':<30} {dummy_f1:<12.6f} {'0.000000':<12} {'0.000000':<12} {'0.500000':<12}")

    # مقایسه با Random Forest (Leakage)
    log(f"\n  Reference Models:")
    log(f"    Random Forest (Raw_Only): F1 = 1.000 (Leakage)")
    log(f"    AdaBoost (Leaky Features): F1 = 0.988 (Leakage)")
    log(f"    MLP: F1 = 0.000 (Failed)")

    # ذخیره
    benchmark = {
        "Isolation_Forest_Tuned": {
            "F1": test_row["F1"],
            "Precision": test_row["Precision"],
            "Recall": test_row["Recall"],
            "ROC_AUC": test_row["ROC_AUC"],
        },
        "Dummy_Classifier": {
            "F1": 0.0,
            "Precision": 0.0,
            "Recall": 0.0,
            "ROC_AUC": 0.5,
        },
        "Random_Forest_Leakage": {
            "F1": 1.0,
            "Precision": 1.0,
            "Recall": 1.0,
            "ROC_AUC": 1.0,
        },
    }

    pd.DataFrame(benchmark).T.to_csv(TABLES_DIR / "17_10_benchmark.csv")
    log(f"\n  [OK] Saved: 17_10_benchmark.csv")

    return benchmark

# ============================================================
# VISUALIZE
# ============================================================
def visualize_evaluation(results_df, cm_info, roc_info, pr_info, cal_info):
    log("\n" + "=" * 70)
    log("VISUALIZING EVALUATION RESULTS")
    log("=" * 70)

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # 1. Metrics by Dataset
    datasets = ["Train", "Val", "Test"]
    metrics = ["F1", "Precision", "Recall", "ROC_AUC"]
    x = np.arange(len(datasets))
    width = 0.2

    for i, metric in enumerate(metrics):
        values = [results_df[results_df["Dataset"] == d][metric].values[0] for d in datasets]
        axes[0, 0].bar(x + i * width - 1.5 * width, values, width, label=metric)

    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Metrics by Dataset")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 1)

    # 2. Confusion Matrix
    # (already plotted in 17_6)

    # 3. ROC Curve
    if "fpr" in roc_info and "tpr" in roc_info:
        axes[0, 2].plot(roc_info["fpr"], roc_info["tpr"], "b-", linewidth=2,
                        label=f"AUC = {roc_info['roc_auc']:.4f}")
        axes[0, 2].plot([0, 1], [0, 1], "r--", linewidth=1)
        axes[0, 2].set_xlabel("FPR")
        axes[0, 2].set_ylabel("TPR")
        axes[0, 2].set_title("ROC Curve")
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)

    # 4. PR Curve
    if "precision" in pr_info and "recall" in pr_info:
        axes[1, 0].plot(pr_info["recall"], pr_info["precision"], "b-", linewidth=2,
                        label=f"PR-AUC = {pr_info['pr_auc']:.4f}")
        axes[1, 0].axhline(y=pr_info["baseline"], color="r", linestyle="--",
                           label=f"Baseline = {pr_info['baseline']:.4f}")
        axes[1, 0].set_xlabel("Recall")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Precision-Recall Curve")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 5. Calibration Curve
    if "fraction_positive" in cal_info and "mean_predicted" in cal_info:
        axes[1, 1].plot(cal_info["mean_predicted"], cal_info["fraction_positive"],
                        "s-", color="steelblue", linewidth=2, label="Iso Forest")
        axes[1, 1].plot([0, 1], [0, 1], "r--", linewidth=1, label="Perfect")
        axes[1, 1].set_xlabel("Mean Predicted")
        axes[1, 1].set_ylabel("Fraction Positive")
        axes[1, 1].set_title("Calibration Curve")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    # 6. Summary Text
    axes[1, 2].axis("off")

    test_row = results_df[results_df["Dataset"] == "Test"].iloc[0]
    summary_text = "EVALUATION SUMMARY\n" + "=" * 35 + "\n\n"
    summary_text += f"Model: Isolation Forest (Tuned)\n\n"
    summary_text += f"Test F1: {test_row['F1']:.4f}\n"
    summary_text += f"Test Precision: {test_row['Precision']:.4f}\n"
    summary_text += f"Test Recall: {test_row['Recall']:.4f}\n"
    summary_text += f"Test ROC-AUC: {test_row['ROC_AUC']:.4f}\n"
    summary_text += f"Test PR-AUC: {test_row['PR_AUC']:.4f}\n"
    summary_text += f"Test MCC: {test_row['MCC']:.4f}\n\n"
    summary_text += f"Confusion Matrix:\n"
    summary_text += f"  TN={cm_info['TN']}, FP={cm_info['FP']}\n"
    summary_text += f"  FN={cm_info['FN']}, TP={cm_info['TP']}\n\n"
    summary_text += f"Improvement over Phase 14d:\n"
    summary_text += f"  F1: 0.232 -> {test_row['F1']:.4f}\n"

    axes[1, 2].text(0.05, 0.5, summary_text, fontsize=10, verticalalignment="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "17_evaluation_results.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 17_evaluation_results.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 17: MODEL EVALUATION")
    log("=" * 70)

    train_df, val_df, test_df, raw_features = load_data()

    # 17.1
    strategy = stage_17_1_strategy()

    # Training Final Iso Forest
    iso, best_params = train_final_iso(train_df, raw_features)

    # 17.2-17.3
    results_df = stage_17_2_to_17_3_evaluation(iso, train_df, val_df, test_df, raw_features, best_params)

    # 17.4
    stage_17_4_performance(results_df)

    # 17.5
    cm, cm_info = stage_17_5_error_analysis(iso, test_df, raw_features)

    # 17.6
    cm_metrics = stage_17_6_confusion_matrix(cm)

    # 17.7
    roc_info = stage_17_7_roc(iso, test_df, raw_features)

    # 17.8
    pr_info = stage_17_8_pr(iso, test_df, raw_features)

    # 17.9
    cal_info = stage_17_9_calibration(iso, test_df, raw_features)

    # 17.10
    benchmark = stage_17_10_benchmark(results_df)

    # Visualize
    visualize_evaluation(results_df, cm_info, roc_info, pr_info, cal_info)

    log("\n" + "=" * 70)
    log("PHASE 17 COMPLETE!")
    log("=" * 70)

    LOG_FILE = BASE_DIR / "reports" / "phase17_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()
