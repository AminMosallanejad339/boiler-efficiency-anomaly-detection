"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 04b: EDA Fix - Stages 4.8, 4.9, 4.10
Framework: ML Model Lifecycle - 23 Main Stages
Author: Amin Mosallanejad
Date: 2026

Fix: Matplotlib 3.9+ changed 'labels' to 'tick_labels' in boxplot()
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
PROCESSED = BASE_DIR / "data" / "processed" / "industrial_dataset_phase03_final.csv"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

TARGET = "Boiler_Eff_"
ANOMALY = "Anomaly_Label"

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری
# ============================================================
log("=" * 70)
log("PHASE 04b: EDA FIX (Stages 4.8, 4.9, 4.10)")
log("=" * 70)

df = pd.read_csv(PROCESSED)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in [ANOMALY] and not c.endswith("_bin")]

log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

# ============================================================
# 4.8 ANOMALY DETECTION (FIXED)
# ============================================================
def stage_4_8_anomaly_detection(df):
    log("\n" + "=" * 70)
    log("STAGE 4.8: ANOMALY DETECTION (FIXED)")
    log("=" * 70)
    
    anomaly_df = df[df[ANOMALY] == 1]
    normal_df = df[df[ANOMALY] == 0]
    
    log(f"  Anomaly records: {len(anomaly_df)} ({len(anomaly_df)/len(df)*100:.4f}%)")
    log(f"  Normal records: {len(normal_df)} ({len(normal_df)/len(df)*100:.4f}%)")
    
    # مقایسه میانگین‌ها
    comparison = pd.DataFrame({
        "Normal_Mean": normal_df[numeric_cols].mean(),
        "Anomaly_Mean": anomaly_df[numeric_cols].mean(),
        "Difference": anomaly_df[numeric_cols].mean() - normal_df[numeric_cols].mean(),
        "Diff_Pct": ((anomaly_df[numeric_cols].mean() - normal_df[numeric_cols].mean()) / normal_df[numeric_cols].mean()).abs() * 100,
    }).sort_values("Diff_Pct", ascending=False)
    
    comparison.to_csv(TABLES_DIR / "04_08_anomaly_vs_normal.csv")
    log(f"  [OK] Saved: 04_08_anomaly_vs_normal.csv")
    
    log(f"\n  Top 10 features with largest difference (Anomaly vs Normal):")
    for idx, row in comparison.head(10).iterrows():
        log(f"    {idx}: Normal={row['Normal_Mean']:.4f}, Anomaly={row['Anomaly_Mean']:.4f}, Diff={row['Diff_Pct']:.2f}%")
    
    # Boxplot مقایسه - با tick_labels (نسخه جدید Matplotlib)
    top_diff = comparison.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top_diff):
        data_to_plot = [normal_df[col].dropna(), anomaly_df[col].dropna()]
        try:
            axes[i].boxplot(data_to_plot, tick_labels=["Normal", "Anomaly"])
        except TypeError:
            # Fallback برای نسخه‌های قدیمی
            axes[i].boxplot(data_to_plot, labels=["Normal", "Anomaly"])
        axes[i].set_title(f"{col}", fontsize=10)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_08_anomaly_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 04_08_anomaly_comparison.png")
    
    return comparison

# ============================================================
# 4.9 VISUALIZATION
# ============================================================
def stage_4_9_visualization(df):
    log("\n" + "=" * 70)
    log("STAGE 4.9: VISUALIZATION SUMMARY")
    log("=" * 70)
    
    top5 = df[numeric_cols].corrwith(df[TARGET]).abs().sort_values(ascending=False).head(5).index.tolist()
    pairplot_cols = top5 + [TARGET]
    
    log(f"  Generating pairplot for: {pairplot_cols}")
    try:
        sample_df = df[pairplot_cols].sample(min(2000, len(df)), random_state=42)
        g = sns.pairplot(sample_df, diag_kind="kde", plot_kws={"alpha": 0.4, "s": 10})
        g.fig.suptitle("Pairplot - Top 5 Features + Target", y=1.02)
        plt.savefig(FIGURES_DIR / "04_09_pairplot.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 04_09_pairplot.png")
    except Exception as e:
        log(f"  [WARNING] Pairplot failed: {e}")
    
    return top5

# ============================================================
# 4.10 INSIGHT EXTRACTION
# ============================================================
def stage_4_10_insights(df, comparison):
    log("\n" + "=" * 70)
    log("STAGE 4.10: INSIGHT EXTRACTION")
    log("=" * 70)
    
    insights = []
    
    # بینش ۱: مشکل اصلی - همبستگی ضعیف
    corr_df = pd.read_csv(TABLES_DIR / "04_03_bivariate_target_correlation.csv")
    top_corr = corr_df.iloc[1]  # اولین ویژگی (بعد از خود Target)
    insights.append({
        "ID": "INS-01",
        "Category": "CRITICAL - Weak Correlation",
        "Insight": f"Strongest feature correlation with Target is only {top_corr['Correlation_with_Target']:.4f}",
        "Implication": "Regression modeling on Boiler_Eff_ will be challenging",
        "Severity": "HIGH"
    })
    
    # بینش ۲: محدوده باریک Target
    insights.append({
        "ID": "INS-02",
        "Category": "CRITICAL - Narrow Target Range",
        "Insight": f"Boiler_Eff_ range is only {df[TARGET].max() - df[TARGET].min():.4f} (Std: {df[TARGET].std():.4f})",
        "Implication": "Very small variance limits regression performance",
        "Severity": "HIGH"
    })
    
    # بینش ۳: عدم تعادل شدید
    insights.append({
        "ID": "INS-03",
        "Category": "Class Imbalance",
        "Insight": f"Anomaly rate is {df[ANOMALY].mean()*100:.4f}% (imbalance ratio 65:1)",
        "Implication": "Requires SMOTE or class weighting for classification",
        "Severity": "MEDIUM"
    })
    
    # بینش ۴: بهترین ویژگی برای Anomaly
    insights.append({
        "ID": "INS-04",
        "Category": "Anomaly Indicator",
        "Insight": f"APH_Leakage_ shows -54.22% difference between anomaly and normal records",
        "Implication": "Strongest anomaly indicator; will be key feature",
        "Severity": "INFO"
    })
    
    # بینش ۵: Multicollinearity
    insights.append({
        "ID": "INS-05",
        "Category": "Multicollinearity",
        "Insight": "No pairs with |corr| > 0.9 detected",
        "Implication": "Low multicollinearity risk; VIF check in Phase 07",
        "Severity": "INFO"
    })
    
    # بینش ۶: نرمال نبودن Target
    insights.append({
        "ID": "INS-06",
        "Category": "Distribution",
        "Insight": "Shapiro-Wilk test rejected normality (p = 0.00034)",
        "Implication": "Non-parametric tests may be more appropriate",
        "Severity": "LOW"
    })
    
    # بینش ۷: پایداری روزانه
    insights.append({
        "ID": "INS-07",
        "Category": "Temporal Stability",
        "Insight": "Average Boiler_Eff_ is nearly identical across all days (93.67)",
        "Implication": "No day-of-week effect; no shift-based patterns",
        "Severity": "LOW"
    })
    
    # بینش ۸: تفاوت Target بین Anomaly و Normal
    insights.append({
        "ID": "INS-08",
        "Category": "Target vs Anomaly",
        "Insight": f"Correlation between Boiler_Eff_ and Anomaly_Label is only {df[TARGET].corr(df[ANOMALY]):.4f}",
        "Implication": "Anomalies do not significantly affect boiler efficiency",
        "Severity": "INFO"
    })
    
    insights_df = pd.DataFrame(insights)
    insights_df.to_csv(TABLES_DIR / "04_10_insights.csv", index=False)
    log(f"  [OK] Saved: 04_10_insights.csv")
    
    log(f"\n  Extracted {len(insights)} key insights:")
    for ins in insights:
        log(f"    [{ins['ID']}] [{ins['Severity']}] {ins['Category']}: {ins['Insight']}")
    
    return insights_df

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 04b: EDA FIX")
    log("=" * 70)
    
    comparison = stage_4_8_anomaly_detection(df)
    top5 = stage_4_9_visualization(df)
    insights_df = stage_4_10_insights(df, comparison)
    
    log("\n" + "=" * 70)
    log("PHASE 04 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase04b_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()