"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 21: Error & Fairness Analysis (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Isolation Forest (Tuned) - Error & Fairness
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
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, f1_score,
    accuracy_score, roc_auc_score, classification_report,
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
    log("PHASE 21: ERROR & FAIRNESS ANALYSIS")
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
# 21.1 ERROR ANALYSIS
# ============================================================
def stage_21_1_error_analysis(iso, train_df, val_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.1: ERROR ANALYSIS")
    log("=" * 70)

    results = []

    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        X = df[raw_features].fillna(0).values
        y = df[TARGET_ANOMALY].values
        y_pred = (iso.predict(X) == -1).astype(int)

        cm = confusion_matrix(y, y_pred)
        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        tpr = tp / (fn + tp) if (fn + tp) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

        results.append({
            "Dataset": name,
            "N_Samples": len(y),
            "N_Anomalies": int(y.sum()),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "FPR": round(fpr, 6),
            "FNR": round(fnr, 6),
            "TPR": round(tpr, 6),
            "TNR": round(tnr, 6),
            "Accuracy": round(accuracy_score(y, y_pred), 6),
            "Precision": round(precision_score(y, y_pred, zero_division=0), 6),
            "Recall": round(recall_score(y, y_pred, zero_division=0), 6),
            "F1": round(f1_score(y, y_pred, zero_division=0), 6),
        })

        log(f"\n  {name}:")
        log(f"    TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        log(f"    FPR={fpr:.6f}, FNR={fnr:.6f}")
        log(f"    F1={results[-1]['F1']:.6f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "21_1_error_analysis.csv", index=False)
    log(f"\n  [OK] Saved: 21_1_error_analysis.csv")

    # نمودار
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for i, (name, df) in enumerate([("Train", train_df), ("Val", val_df), ("Test", test_df)]):
        X = df[raw_features].fillna(0).values
        y = df[TARGET_ANOMALY].values
        y_pred = (iso.predict(X) == -1).astype(int)
        cm = confusion_matrix(y, y_pred)

        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Normal", "Anomaly"],
                    yticklabels=["Normal", "Anomaly"])
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")
        axes[i].set_title(f"{name}: Confusion Matrix")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "21_1_error_analysis.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 21_1_error_analysis.png")

    return results_df

# ============================================================
# 21.2 MISCLASSIFICATION ANALYSIS
# ============================================================
def stage_21_2_misclassification(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.2: MISCLASSIFICATION ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    y_pred = (iso.predict(X) == -1).astype(int)

    fn_indices = np.where((y == 1) & (y_pred == 0))[0]
    fp_indices = np.where((y == 0) & (y_pred == 1))[0]
    tp_indices = np.where((y == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y == 0) & (y_pred == 0))[0]

    log(f"  Misclassification Analysis:")
    log(f"    Total Samples: {len(y)}")
    log(f"    FN (Missed Anomaly): {len(fn_indices)}")
    log(f"    FP (False Alarm): {len(fp_indices)}")
    log(f"    TP (Correct Anomaly): {len(tp_indices)}")
    log(f"    TN (Correct Normal): {len(tn_indices)}")

    # تحلیل ویژگی‌ها در FN و FP
    log(f"\n  Feature Analysis in FN (Missed Anomalies):")
    fn_df = test_df.iloc[fn_indices][raw_features] if len(fn_indices) > 0 else pd.DataFrame()
    for col in raw_features:
        if len(fn_df) > 0:
            log(f"    {col}: mean={fn_df[col].mean():.6f}, std={fn_df[col].std():.6f}")

    log(f"\n  Feature Analysis in FP (False Alarms):")
    fp_df = test_df.iloc[fp_indices][raw_features] if len(fp_indices) > 0 else pd.DataFrame()
    for col in raw_features:
        if len(fp_df) > 0:
            log(f"    {col}: mean={fp_df[col].mean():.6f}, std={fp_df[col].std():.6f}")

    log(f"\n  Feature Analysis in TP (Correct Anomalies):")
    tp_df = test_df.iloc[tp_indices][raw_features] if len(tp_indices) > 0 else pd.DataFrame()
    for col in raw_features:
        if len(tp_df) > 0:
            log(f"    {col}: mean={tp_df[col].mean():.6f}, std={tp_df[col].std():.6f}")

    # ذخیره
    misclass = {
        "FN_Count": len(fn_indices),
        "FP_Count": len(fp_indices),
        "TP_Count": len(tp_indices),
        "TN_Count": len(tn_indices),
    }

    for col in raw_features:
        if len(fn_df) > 0:
            misclass[f"FN_{col}_mean"] = round(fn_df[col].mean(), 6)
        if len(fp_df) > 0:
            misclass[f"FP_{col}_mean"] = round(fp_df[col].mean(), 6)
        if len(tp_df) > 0:
            misclass[f"TP_{col}_mean"] = round(tp_df[col].mean(), 6)

    pd.DataFrame(list(misclass.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "21_2_misclassification.csv", index=False
    )
    log(f"\n  [OK] Saved: 21_2_misclassification.csv")

    # نمودار مقایسه
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(raw_features):
        data_to_plot = []
        labels = []

        for group_name, indices in [("TN", tn_indices), ("FP", fp_indices), ("FN", fn_indices), ("TP", tp_indices)]:
            if len(indices) > 0:
                data_to_plot.append(test_df.iloc[indices][col].values)
                labels.append(group_name)

        if len(data_to_plot) > 0:
            # ✅ اصلاح: استفاده از tick_labels با fallback برای نسخه‌های قدیمی
            try:
                bp = axes[i].boxplot(data_to_plot, tick_labels=labels, patch_artist=True)
            except TypeError:
                bp = axes[i].boxplot(data_to_plot, labels=labels, patch_artist=True)

            colors = ["steelblue", "coral", "orange", "green"]
            for patch, color in zip(bp["boxes"], colors[:len(bp["boxes"])]):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        axes[i].set_title(f"{col}")
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "21_2_misclassification.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 21_2_misclassification.png")

    return misclass

# ============================================================
# 21.3 FAILURE CASE ANALYSIS
# ============================================================
def stage_21_3_failure_cases(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.3: FAILURE CASE ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    y_pred = (iso.predict(X) == -1).astype(int)
    scores = -iso.score_samples(X)

    # FN با بالاترین Score (نزدیک به مرز)
    fn_indices = np.where((y == 1) & (y_pred == 0))[0]

    if len(fn_indices) > 0:
        fn_scores = scores[fn_indices]
        fn_sorted = fn_indices[np.argsort(fn_scores)[::-1]]

        log(f"  Top 5 FN (Highest Score - Almost Detected):")
        for i, idx in enumerate(fn_sorted[:5]):
            log(f"    {i+1}. Index {idx}: Score={scores[idx]:.6f}")
            for col in raw_features:
                log(f"       {col}: {test_df.iloc[idx][col]:.6f}")
    else:
        fn_sorted = []
        log(f"  No FN found")

    # FP با پایین‌ترین Score (نزدیک به مرز)
    fp_indices = np.where((y == 0) & (y_pred == 1))[0]

    if len(fp_indices) > 0:
        fp_scores = scores[fp_indices]
        fp_sorted = fp_indices[np.argsort(fp_scores)]

        log(f"\n  Top 5 FP (Lowest Score - Almost Normal):")
        for i, idx in enumerate(fp_sorted[:5]):
            log(f"    {i+1}. Index {idx}: Score={scores[idx]:.6f}")
            for col in raw_features:
                log(f"       {col}: {test_df.iloc[idx][col]:.6f}")
    else:
        fp_sorted = []
        log(f"  No FP found")

    results = {
        "Top_FN_Index": int(fn_sorted[0]) if len(fn_sorted) > 0 else None,
        "Top_FN_Score": float(scores[fn_sorted[0]]) if len(fn_sorted) > 0 else None,
        "Top_FP_Index": int(fp_sorted[0]) if len(fp_sorted) > 0 else None,
        "Top_FP_Score": float(scores[fp_sorted[0]]) if len(fp_sorted) > 0 else None,
    }

    pd.DataFrame(list(results.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "21_3_failure_cases.csv", index=False
    )
    log(f"\n  [OK] Saved: 21_3_failure_cases.csv")

    return results

# ============================================================
# 21.4 BIAS ANALYSIS
# ============================================================
def stage_21_4_bias_analysis(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.4: BIAS ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    y_pred = (iso.predict(X) == -1).astype(int)
    scores = -iso.score_samples(X)

    test_df_copy = test_df.copy()
    test_df_copy["y_pred"] = y_pred
    test_df_copy["score"] = scores

    # Bias by Month
    test_df_copy["Timestamp"] = pd.to_datetime(test_df_copy["Timestamp"])
    test_df_copy["month"] = test_df_copy["Timestamp"].dt.month

    log(f"  Bias Analysis by Month:")
    monthly_results = []
    for month in sorted(test_df_copy["month"].unique()):
        month_df = test_df_copy[test_df_copy["month"] == month]
        if len(month_df) < 10:
            continue

        f1 = f1_score(month_df[TARGET_ANOMALY], month_df["y_pred"], zero_division=0)
        prec = precision_score(month_df[TARGET_ANOMALY], month_df["y_pred"], zero_division=0)
        rec = recall_score(month_df[TARGET_ANOMALY], month_df["y_pred"], zero_division=0)

        monthly_results.append({
            "Month": int(month),
            "N_Samples": len(month_df),
            "N_Anomalies": int(month_df[TARGET_ANOMALY].sum()),
            "F1": round(f1, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
        })

        log(f"    Month {month}: N={len(month_df)}, F1={f1:.4f}, P={prec:.4f}, R={rec:.4f}")

    monthly_df = pd.DataFrame(monthly_results)
    monthly_df.to_csv(TABLES_DIR / "21_4_bias_by_month.csv", index=False)
    log(f"  [OK] Saved: 21_4_bias_by_month.csv")

    # Bias by Hour
    test_df_copy["hour"] = test_df_copy["Timestamp"].dt.hour

    log(f"\n  Bias Analysis by Hour:")
    hourly_results = []
    for hour in sorted(test_df_copy["hour"].unique()):
        hour_df = test_df_copy[test_df_copy["hour"] == hour]
        if len(hour_df) < 10:
            continue

        f1 = f1_score(hour_df[TARGET_ANOMALY], hour_df["y_pred"], zero_division=0)

        hourly_results.append({
            "Hour": int(hour),
            "N_Samples": len(hour_df),
            "F1": round(f1, 6),
        })

    hourly_df = pd.DataFrame(hourly_results)
    hourly_df.to_csv(TABLES_DIR / "21_4_bias_by_hour.csv", index=False)
    log(f"  [OK] Saved: 21_4_bias_by_hour.csv")

    # نمودار
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].bar(monthly_df["Month"], monthly_df["F1"], color="steelblue")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("F1 Score")
    axes[0].set_title("Bias Analysis by Month")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hourly_df["Hour"], hourly_df["F1"], "o-", color="coral", linewidth=2)
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("Bias Analysis by Hour")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "21_4_bias_analysis.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 21_4_bias_analysis.png")

    return monthly_df, hourly_df

# ============================================================
# 21.5 VARIANCE ANALYSIS
# ============================================================
def stage_21_5_variance(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.5: VARIANCE ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    scores = -iso.score_samples(X)

    normal_scores = scores[y == 0]
    anomaly_scores = scores[y == 1]

    log(f"  Variance Analysis:")
    log(f"    Normal: mean={normal_scores.mean():.6f}, var={normal_scores.var():.6f}, std={normal_scores.std():.6f}")
    log(f"    Anomaly: mean={anomaly_scores.mean():.6f}, var={anomaly_scores.var():.6f}, std={anomaly_scores.std():.6f}")

    # F-Test for variance equality
    f_stat = normal_scores.var() / anomaly_scores.var()
    df1 = len(normal_scores) - 1
    df2 = len(anomaly_scores) - 1
    p_value = 2 * min(stats.f.cdf(f_stat, df1, df2), 1 - stats.f.cdf(f_stat, df1, df2))

    log(f"\n  F-Test for Variance Equality:")
    log(f"    F-Statistic: {f_stat:.6f}")
    log(f"    P-Value: {p_value:.4e}")
    log(f"    Equal Variance: {p_value > 0.05}")

    results = {
        "normal_mean": float(normal_scores.mean()),
        "normal_var": float(normal_scores.var()),
        "normal_std": float(normal_scores.std()),
        "anomaly_mean": float(anomaly_scores.mean()),
        "anomaly_var": float(anomaly_scores.var()),
        "anomaly_std": float(anomaly_scores.std()),
        "f_stat": float(f_stat),
        "f_p": float(p_value),
        "equal_variance": bool(p_value > 0.05),
    }

    pd.DataFrame(list(results.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "21_5_variance.csv", index=False
    )
    log(f"  [OK] Saved: 21_5_variance.csv")

    return results

# ============================================================
# 21.6 FAIRNESS ANALYSIS
# ============================================================
def stage_21_6_fairness(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.6: FAIRNESS ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    y_pred = (iso.predict(X) == -1).astype(int)

    test_df_copy = test_df.copy()
    test_df_copy["y_pred"] = y_pred
    test_df_copy["Timestamp"] = pd.to_datetime(test_df_copy["Timestamp"])

    # Fairness by Shift
    test_df_copy["hour"] = test_df_copy["Timestamp"].dt.hour
    test_df_copy["shift"] = pd.cut(
        test_df_copy["hour"],
        bins=[-1, 6, 14, 22, 24],
        labels=["Night", "Morning", "Evening", "Late"]
    )

    log(f"  Fairness Analysis by Shift:")
    shift_results = []
    for shift in test_df_copy["shift"].unique():
        shift_df = test_df_copy[test_df_copy["shift"] == shift]
        if len(shift_df) < 10:
            continue

        f1 = f1_score(shift_df[TARGET_ANOMALY], shift_df["y_pred"], zero_division=0)
        prec = precision_score(shift_df[TARGET_ANOMALY], shift_df["y_pred"], zero_division=0)
        rec = recall_score(shift_df[TARGET_ANOMALY], shift_df["y_pred"], zero_division=0)
        acc = accuracy_score(shift_df[TARGET_ANOMALY], shift_df["y_pred"])

        shift_results.append({
            "Shift": str(shift),
            "N_Samples": len(shift_df),
            "N_Anomalies": int(shift_df[TARGET_ANOMALY].sum()),
            "Accuracy": round(acc, 6),
            "F1": round(f1, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
        })

        log(f"    {shift}: N={len(shift_df)}, F1={f1:.4f}, P={prec:.4f}, R={rec:.4f}")

    shift_df = pd.DataFrame(shift_results)
    shift_df.to_csv(TABLES_DIR / "21_6_fairness_by_shift.csv", index=False)
    log(f"  [OK] Saved: 21_6_fairness_by_shift.csv")

    # Fairness Metrics
    if len(shift_df) > 1:
        f1_range = shift_df["F1"].max() - shift_df["F1"].min()
        f1_std = shift_df["F1"].std()

        log(f"\n  Fairness Metrics:")
        log(f"    F1 Range: {f1_range:.6f}")
        log(f"    F1 Std: {f1_std:.6f}")
        log(f"    Fairness: {'FAIR' if f1_std < 0.1 else 'UNFAIR'}")

    # نمودار
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(shift_df))
    width = 0.25

    ax.bar(x - width, shift_df["Precision"], width, label="Precision", color="steelblue")
    ax.bar(x, shift_df["Recall"], width, label="Recall", color="coral")
    ax.bar(x + width, shift_df["F1"], width, label="F1", color="green")

    ax.set_xticks(x)
    ax.set_xticklabels(shift_df["Shift"])
    ax.set_ylabel("Score")
    ax.set_title("Fairness Analysis by Shift")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "21_6_fairness.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 21_6_fairness.png")

    return shift_df

# ============================================================
# 21.7 ROBUSTNESS ANALYSIS
# ============================================================
def stage_21_7_robustness(train_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.7: ROBUSTNESS ANALYSIS")
    log("=" * 70)

    X_train = train_df[raw_features].fillna(0).values
    X_test = test_df[raw_features].fillna(0).values
    y_test = test_df[TARGET_ANOMALY].values

    log(f"  Testing robustness across different random seeds...")

    results = []
    for seed in range(10):
        iso = IsolationForest(
            n_estimators=300, max_samples=0.9, contamination=0.02,
            max_features=0.5, bootstrap=False,
            random_state=seed, n_jobs=-1,
        )
        iso.fit(X_train)
        y_pred = (iso.predict(X_test) == -1).astype(int)

        f1 = f1_score(y_test, y_pred, zero_division=0)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)

        results.append({
            "Seed": seed,
            "F1": round(f1, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "21_7_robustness.csv", index=False)
    log(f"  [OK] Saved: 21_7_robustness.csv")

    log(f"\n  Robustness Analysis:")
    log(f"    F1 Mean: {results_df['F1'].mean():.6f}")
    log(f"    F1 Std: {results_df['F1'].std():.6f}")
    log(f"    F1 Min: {results_df['F1'].min():.6f}")
    log(f"    F1 Max: {results_df['F1'].max():.6f}")
    log(f"    Robust: {results_df['F1'].std() < 0.05}")

    return results_df

# ============================================================
# 21.8 ADVERSARIAL ANALYSIS
# ============================================================
def stage_21_8_adversarial(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.8: ADVERSARIAL ANALYSIS")
    log("=" * 70)

    log(f"  [INFO] Adversarial attacks are for Supervised Learning")
    log(f"  [INFO] For Unsupervised Anomaly Detection, we test:")
    log(f"    - Noise injection")
    log(f"    - Feature perturbation")

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values

    # 1. Noise Injection
    log(f"\n  [1] Noise Injection Test:")

    noise_levels = [0.01, 0.05, 0.1, 0.2]
    noise_results = []

    for noise in noise_levels:
        X_noisy = X + np.random.normal(0, noise, X.shape)
        y_pred = (iso.predict(X_noisy) == -1).astype(int)

        f1 = f1_score(y, y_pred, zero_division=0)

        noise_results.append({
            "Noise_Level": noise,
            "F1": round(f1, 6),
        })

        log(f"    Noise={noise}: F1={f1:.6f}")

    # 2. Feature Perturbation
    log(f"\n  [2] Feature Perturbation Test:")
    perturbation_results = []

    for i, col in enumerate(raw_features):
        X_pert = X.copy()
        X_pert[:, i] += np.random.normal(0, 0.1, len(X_pert))
        y_pred = (iso.predict(X_pert) == -1).astype(int)

        f1 = f1_score(y, y_pred, zero_division=0)

        perturbation_results.append({
            "Feature": col,
            "F1": round(f1, 6),
        })

        log(f"    {col}: F1={f1:.6f}")

    pd.DataFrame(noise_results).to_csv(TABLES_DIR / "21_8_noise.csv", index=False)
    pd.DataFrame(perturbation_results).to_csv(TABLES_DIR / "21_8_perturbation.csv", index=False)
    log(f"  [OK] Saved: 21_8_noise.csv")
    log(f"  [OK] Saved: 21_8_perturbation.csv")

    return {"noise_results": noise_results, "perturbation_results": perturbation_results}

# ============================================================
# 21.9 EDGE CASE ANALYSIS
# ============================================================
def stage_21_9_edge_cases(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 21.9: EDGE CASE ANALYSIS")
    log("=" * 70)

    X = test_df[raw_features].fillna(0).values
    y = test_df[TARGET_ANOMALY].values
    scores = -iso.score_samples(X)
    y_pred = (iso.predict(X) == -1).astype(int)

    log(f"  Edge Case Analysis:")

    # 1. FN with lowest score (most confident mistake)
    fn_indices = np.where((y == 1) & (y_pred == 0))[0]
    worst_fn = None
    if len(fn_indices) > 0:
        fn_scores = scores[fn_indices]
        worst_fn = fn_indices[np.argmin(fn_scores)]
        log(f"\n  Worst FN (Most Confident Mistake):")
        log(f"    Index: {worst_fn}, Score: {scores[worst_fn]:.6f}")
        for col in raw_features:
            log(f"    {col}: {test_df.iloc[worst_fn][col]:.6f}")

    # 2. FP with highest score
    fp_indices = np.where((y == 0) & (y_pred == 1))[0]
    worst_fp = None
    if len(fp_indices) > 0:
        fp_scores = scores[fp_indices]
        worst_fp = fp_indices[np.argmax(fp_scores)]
        log(f"\n  Worst FP (Most Confident Mistake):")
        log(f"    Index: {worst_fp}, Score: {scores[worst_fp]:.6f}")
        for col in raw_features:
            log(f"    {col}: {test_df.iloc[worst_fp][col]:.6f}")

    results = {
        "worst_fn_index": int(worst_fn) if worst_fn is not None else None,
        "worst_fn_score": float(scores[worst_fn]) if worst_fn is not None else None,
        "worst_fp_index": int(worst_fp) if worst_fp is not None else None,
        "worst_fp_score": float(scores[worst_fp]) if worst_fp is not None else None,
    }

    pd.DataFrame(list(results.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "21_9_edge_cases.csv", index=False
    )
    log(f"\n  [OK] Saved: 21_9_edge_cases.csv")

    return results

# ============================================================
# 21.10 LIMITATION IDENTIFICATION
# ============================================================
def stage_21_10_limitations(error_results, bias_results, fairness_results, robustness_results):
    log("\n" + "=" * 70)
    log("STAGE 21.10: LIMITATION IDENTIFICATION")
    log("=" * 70)

    limitations = []

    test_row = error_results[error_results["Dataset"] == "Test"].iloc[0]

    limitations.append({
        "ID": "LIM-01",
        "Category": "Performance",
        "Limitation": f"F1 = {test_row['F1']:.4f} (not perfect)",
        "Impact": "HIGH",
        "Mitigation": "Use ensemble or redefine Anomaly",
    })

    limitations.append({
        "ID": "LIM-02",
        "Category": "Recall",
        "Limitation": f"Recall = {test_row['Recall']:.4f} ({test_row['Recall']*100:.1f}% detected)",
        "Impact": "HIGH",
        "Mitigation": "Lower threshold or use class weights",
    })

    limitations.append({
        "ID": "LIM-03",
        "Category": "Precision",
        "Limitation": f"Precision = {test_row['Precision']:.4f} ({test_row['Precision']*100:.1f}% correct)",
        "Impact": "MEDIUM",
        "Mitigation": "Raise threshold or add features",
    })

    if len(bias_results) > 0 and "F1" in bias_results.columns:
        f1_std = bias_results["F1"].std()
        limitations.append({
            "ID": "LIM-04",
            "Category": "Bias",
            "Limitation": f"F1 varies by month (Std = {f1_std:.4f})",
            "Impact": "MEDIUM",
            "Mitigation": "Add temporal features",
        })

    if len(fairness_results) > 0:
        f1_range = fairness_results["F1"].max() - fairness_results["F1"].min()
        limitations.append({
            "ID": "LIM-05",
            "Category": "Fairness",
            "Limitation": f"F1 varies by shift (Range = {f1_range:.4f})",
            "Impact": "LOW",
            "Mitigation": "Add shift-specific features",
        })

    f1_std_robust = robustness_results["F1"].std()
    limitations.append({
        "ID": "LIM-06",
        "Category": "Robustness",
        "Limitation": f"F1 Std across seeds = {f1_std_robust:.4f}",
        "Impact": "LOW" if f1_std_robust < 0.05 else "MEDIUM",
        "Mitigation": "Use more estimators or ensemble",
    })

    limitations.append({
        "ID": "LIM-07",
        "Category": "Data Leakage",
        "Limitation": "RF with Raw_Only achieves F1=1.0 (Leakage)",
        "Impact": "HIGH",
        "Mitigation": "Report both Leakage and Realistic results",
    })

    limitations.append({
        "ID": "LIM-08",
        "Category": "Method",
        "Limitation": "Isolation Forest is Unsupervised (no labels)",
        "Impact": "MEDIUM",
        "Mitigation": "Combine with Supervised when labels available",
    })

    limitations.append({
        "ID": "LIM-09",
        "Category": "Features",
        "Limitation": "Only 4 raw anomaly features",
        "Impact": "MEDIUM",
        "Mitigation": "Add more sensor features",
    })

    limitations.append({
        "ID": "LIM-10",
        "Category": "Data",
        "Limitation": "63:1 class imbalance",
        "Impact": "HIGH",
        "Mitigation": "Use anomaly-specific methods",
    })

    limitations_df = pd.DataFrame(limitations)
    limitations_df.to_csv(TABLES_DIR / "21_10_limitations.csv", index=False)
    log(f"  [OK] Saved: 21_10_limitations.csv")

    log(f"\n  Limitations Identified:")
    for _, row in limitations_df.iterrows():
        log(f"    [{row['ID']}] {row['Category']}: {row['Limitation']}")
        log(f"         Impact: {row['Impact']}, Mitigation: {row['Mitigation']}")

    return limitations_df

# ============================================================
# VISUALIZE
# ============================================================
def visualize_error_fairness(error_results, bias_results, fairness_results, robustness_results, limitations_df):
    log("\n" + "=" * 70)
    log("VISUALIZING ERROR & FAIRNESS RESULTS")
    log("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Error Metrics by Dataset
    datasets = error_results["Dataset"].tolist()
    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(datasets))
    width = 0.25

    for i, metric in enumerate(metrics):
        values = error_results[metric].values
        axes[0, 0].bar(x + i * width - width, values, width, label=metric)

    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(datasets)
    axes[0, 0].set_ylabel("Score")
    axes[0, 0].set_title("Error Metrics by Dataset")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Fairness by Shift
    if len(fairness_results) > 0:
        axes[0, 1].bar(fairness_results["Shift"], fairness_results["F1"], color="coral")
        axes[0, 1].set_ylabel("F1 Score")
        axes[0, 1].set_title("Fairness by Shift")
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Robustness
    if len(robustness_results) > 0:
        axes[1, 0].plot(robustness_results["Seed"], robustness_results["F1"], "o-", color="green", linewidth=2)
        axes[1, 0].axhline(y=robustness_results["F1"].mean(), color="red", linestyle="--", label="Mean")
        axes[1, 0].set_xlabel("Seed")
        axes[1, 0].set_ylabel("F1 Score")
        axes[1, 0].set_title("Robustness Across Seeds")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    # 4. Limitations Summary
    axes[1, 1].axis("off")

    summary_text = "LIMITATIONS SUMMARY\n" + "=" * 35 + "\n\n"
    for _, row in limitations_df.head(6).iterrows():
        summary_text += f"[{row['ID']}] {row['Category']}\n"
        summary_text += f"  {row['Limitation'][:50]}\n"
        summary_text += f"  Impact: {row['Impact']}\n\n"

    axes[1, 1].text(0.05, 0.5, summary_text, fontsize=10, verticalalignment="center",
                    fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "21_error_fairness.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 21_error_fairness.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 21: ERROR & FAIRNESS ANALYSIS")
    log("=" * 70)

    train_df, val_df, test_df, raw_features = load_data()

    iso = joblib.load(MODELS_DIR / "isolation_forest_tuned.pkl")
    log(f"\n  Model loaded: isolation_forest_tuned.pkl")

    # 21.1
    error_results = stage_21_1_error_analysis(iso, train_df, val_df, test_df, raw_features)

    # 21.2
    misclass_results = stage_21_2_misclassification(iso, test_df, raw_features)

    # 21.3
    failure_results = stage_21_3_failure_cases(iso, test_df, raw_features)

    # 21.4
    monthly_df, hourly_df = stage_21_4_bias_analysis(iso, test_df, raw_features)

    # 21.5
    variance_results = stage_21_5_variance(iso, test_df, raw_features)

    # 21.6
    fairness_results = stage_21_6_fairness(iso, test_df, raw_features)

    # 21.7
    robustness_results = stage_21_7_robustness(train_df, test_df, raw_features)

    # 21.8
    adversarial_results = stage_21_8_adversarial(iso, test_df, raw_features)

    # 21.9
    edge_results = stage_21_9_edge_cases(iso, test_df, raw_features)

    # 21.10
    limitations_df = stage_21_10_limitations(error_results, monthly_df, fairness_results, robustness_results)

    # Visualize
    visualize_error_fairness(error_results, monthly_df, fairness_results, robustness_results, limitations_df)

    log("\n" + "=" * 70)
    log("PHASE 21 COMPLETE!")
    log("=" * 70)

    LOG_FILE = BASE_DIR / "reports" / "phase21_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()
