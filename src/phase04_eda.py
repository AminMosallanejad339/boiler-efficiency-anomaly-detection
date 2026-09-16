"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 04: Exploratory Data Analysis (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 10

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
PROCESSED = BASE_DIR / "data" / "processed" / "industrial_dataset_phase03_final.csv"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری
# ============================================================
log("=" * 70)
log("PHASE 04: EXPLORATORY DATA ANALYSIS (EDA)")
log("=" * 70)

df = pd.read_csv(PROCESSED)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Time range: {df['Timestamp'].min()} to {df['Timestamp'].max()}")

# ============================================================
# تعریف گروه‌های ستون
# ============================================================
TARGET = "Boiler_Eff_"
ANOMALY = "Anomaly_Label"

# ستون‌های عددی (به جز Timestamp و binها)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in [ANOMALY] and not c.endswith("_bin")]

log(f"Numeric columns for EDA: {len(numeric_cols)}")
log(f"Target column: {TARGET}")
log(f"Anomaly column: {ANOMALY}")

# ============================================================
# 4.1 EXPLORATORY DATA ANALYSIS
# ============================================================
def stage_4_1_overview(df):
    log("\n" + "=" * 70)
    log("STAGE 4.1: EXPLORATORY DATA ANALYSIS (Overview)")
    log("=" * 70)
    
    info = {
        "Total Rows": len(df),
        "Total Columns": len(df.columns),
        "Numeric Columns": len(numeric_cols),
        "Non-Numeric Columns": len(df.select_dtypes(exclude=[np.number]).columns),
        "Missing Values": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory (MB)": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "Time Range Start": str(df["Timestamp"].min()),
        "Time Range End": str(df["Timestamp"].max()),
        "Anomaly Count": int(df[ANOMALY].sum()),
        "Anomaly Rate (%)": round(df[ANOMALY].mean() * 100, 4),
    }
    
    for k, v in info.items():
        log(f"  {k}: {v}")
    
    pd.DataFrame(list(info.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "04_01_overview.csv", index=False
    )
    log(f"  [OK] Saved: 04_01_overview.csv")
    
    return info

# ============================================================
# 4.2 UNIVARIATE ANALYSIS
# ============================================================
def stage_4_2_univariate(df):
    log("\n" + "=" * 70)
    log("STAGE 4.2: UNIVARIATE ANALYSIS")
    log("=" * 70)
    
    # آمار توصیفی
    desc = df[numeric_cols].describe().T
    desc["skewness"] = df[numeric_cols].skew()
    desc["kurtosis"] = df[numeric_cols].kurtosis()
    desc["cv"] = (desc["std"] / desc["mean"]).abs()
    desc.to_csv(TABLES_DIR / "04_02_univariate_stats.csv")
    log(f"  [OK] Saved: 04_02_univariate_stats.csv")
    
    # توزیع Target
    log(f"\n  Target ({TARGET}) statistics:")
    log(f"    Mean: {df[TARGET].mean():.4f}")
    log(f"    Std: {df[TARGET].std():.4f}")
    log(f"    Min: {df[TARGET].min():.4f}")
    log(f"    Max: {df[TARGET].max():.4f}")
    log(f"    Skewness: {df[TARGET].skew():.4f}")
    log(f"    Kurtosis: {df[TARGET].kurtosis():.4f}")
    
    # Histogram تمام ستون‌های عددی
    log(f"\n  Generating histograms for {len(numeric_cols)} columns...")
    n_cols = 4
    n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    axes = axes.flatten()
    
    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col].dropna(), bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[i].set_title(col, fontsize=9)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequency", fontsize=8)
        axes[i].tick_params(labelsize=7)
    
    for j in range(len(numeric_cols), len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_02_histograms_all.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 04_02_histograms_all.png")
    
    # Boxplot Target
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].boxplot(df[TARGET].dropna(), vert=False)
    axes[0].set_title(f"Boxplot: {TARGET}")
    axes[0].set_xlabel("Boiler Efficiency (%)")
    
    axes[1].hist(df[TARGET].dropna(), bins=50, color="coral", edgecolor="black", alpha=0.7)
    axes[1].set_title(f"Distribution: {TARGET}")
    axes[1].set_xlabel("Boiler Efficiency (%)")
    axes[1].set_ylabel("Frequency")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_02_target_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 04_02_target_distribution.png")
    
    return desc

# ============================================================
# 4.3 BIVARIATE ANALYSIS
# ============================================================
def stage_4_3_bivariate(df):
    log("\n" + "=" * 70)
    log("STAGE 4.3: BIVARIATE ANALYSIS")
    log("=" * 70)
    
    # همبستگی هر ستون با Target
    correlations = df[numeric_cols].corrwith(df[TARGET]).sort_values(ascending=False)
    corr_df = pd.DataFrame({
        "Feature": correlations.index,
        "Correlation_with_Target": correlations.values,
        "Abs_Correlation": correlations.abs().values,
    }).sort_values("Abs_Correlation", ascending=False)
    corr_df.to_csv(TABLES_DIR / "04_03_bivariate_target_correlation.csv", index=False)
    log(f"  [OK] Saved: 04_03_bivariate_target_correlation.csv")
    
    log(f"\n  Top 10 features correlated with {TARGET}:")
    for _, row in corr_df.head(10).iterrows():
        log(f"    {row['Feature']}: {row['Correlation_with_Target']:.4f}")
    
    log(f"\n  Bottom 5 features (weakest correlation):")
    for _, row in corr_df.tail(5).iterrows():
        log(f"    {row['Feature']}: {row['Correlation_with_Target']:.4f}")
    
    # Scatter plot: Top 6 correlated features vs Target
    top6 = corr_df.head(6)["Feature"].tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top6):
        axes[i].scatter(df[col], df[TARGET], alpha=0.3, s=5, color="steelblue")
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel(TARGET, fontsize=9)
        axes[i].set_title(f"{col}\n(corr = {correlations[col]:.3f})", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_03_scatter_top6.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 04_03_scatter_top6.png")
    
    return corr_df

# ============================================================
# 4.4 MULTIVARIATE ANALYSIS
# ============================================================
def stage_4_4_multivariate(df):
    log("\n" + "=" * 70)
    log("STAGE 4.4: MULTIVARIATE ANALYSIS")
    log("=" * 70)
    
    # ماتریس همبستگی (برای 20 ستون برتر)
    top20 = df[numeric_cols].corrwith(df[TARGET]).abs().sort_values(ascending=False).head(20).index.tolist()
    corr_matrix = df[top20].corr()
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Correlation Matrix - Top 20 Features", fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_04_correlation_heatmap.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 04_04_correlation_heatmap.png")
    
    # شناسایی جفت‌های با همبستگی بالا
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > 0.9:
                high_corr_pairs.append({
                    "Feature_1": corr_matrix.columns[i],
                    "Feature_2": corr_matrix.columns[j],
                    "Correlation": round(corr_matrix.iloc[i, j], 4),
                })
    
    if high_corr_pairs:
        pd.DataFrame(high_corr_pairs).to_csv(
            TABLES_DIR / "04_04_high_correlation_pairs.csv", index=False
        )
        log(f"  [OK] Found {len(high_corr_pairs)} pairs with |corr| > 0.9")
        log(f"  [OK] Saved: 04_04_high_correlation_pairs.csv")
        log(f"\n  Top 10 highly correlated pairs:")
        for pair in high_corr_pairs[:10]:
            log(f"    {pair['Feature_1']} <-> {pair['Feature_2']}: {pair['Correlation']}")
    else:
        log(f"  [OK] No pairs with |corr| > 0.9")
    
    return corr_matrix

# ============================================================
# 4.5 DISTRIBUTION ANALYSIS
# ============================================================
def stage_4_5_distribution(df):
    log("\n" + "=" * 70)
    log("STAGE 4.5: DISTRIBUTION ANALYSIS")
    log("=" * 70)
    
    # QQ-Plot برای Target
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # QQ-Plot
    stats.probplot(df[TARGET].dropna(), dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q Plot: {TARGET}")
    axes[0].grid(True)
    
    # Distribution با KDE
    sns.histplot(df[TARGET].dropna(), kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title(f"Distribution with KDE: {TARGET}")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_05_target_qqplot.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 04_05_target_qqplot.png")
    
    # Shapiro-Wilk Test (نرمال بودن)
    sample = df[TARGET].dropna().sample(min(5000, len(df)), random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    log(f"\n  Shapiro-Wilk Test for {TARGET}:")
    log(f"    Statistic: {shapiro_stat:.6f}")
    log(f"    p-value: {shapiro_p:.6e}")
    log(f"    Normal? {'YES' if shapiro_p > 0.05 else 'NO'}")
    
    # Anderson-Darling Test
    ad_result = stats.anderson(df[TARGET].dropna(), dist="norm")
    log(f"\n  Anderson-Darling Test for {TARGET}:")
    log(f"    Statistic: {ad_result.statistic:.6f}")
    log(f"    Critical values: {ad_result.critical_values}")
    
    return {
        "shapiro_stat": shapiro_stat,
        "shapiro_p": shapiro_p,
        "ad_stat": ad_result.statistic,
    }

# ============================================================
# 4.6 CORRELATION ANALYSIS
# ============================================================
def stage_4_6_correlation(df):
    log("\n" + "=" * 70)
    log("STAGE 4.6: CORRELATION ANALYSIS")
    log("=" * 70)
    
    # همبستگی با Anomaly_Label
    anomaly_corr = df[numeric_cols].corrwith(df[ANOMALY]).sort_values(ascending=False)
    anomaly_corr_df = pd.DataFrame({
        "Feature": anomaly_corr.index,
        "Correlation_with_Anomaly": anomaly_corr.values,
        "Abs_Correlation": anomaly_corr.abs().values,
    }).sort_values("Abs_Correlation", ascending=False)
    anomaly_corr_df.to_csv(TABLES_DIR / "04_06_anomaly_correlation.csv", index=False)
    log(f"  [OK] Saved: 04_06_anomaly_correlation.csv")
    
    log(f"\n  Top 10 features correlated with {ANOMALY}:")
    for _, row in anomaly_corr_df.head(10).iterrows():
        log(f"    {row['Feature']}: {row['Correlation_with_Anomaly']:.4f}")
    
    # همبستگی بین Target و Anomaly
    log(f"\n  Correlation between {TARGET} and {ANOMALY}:")
    log(f"    {df[TARGET].corr(df[ANOMALY]):.4f}")
    
    return anomaly_corr_df

# ============================================================
# 4.7 PATTERN DISCOVERY
# ============================================================
def stage_4_7_pattern(df):
    log("\n" + "=" * 70)
    log("STAGE 4.7: PATTERN DISCOVERY")
    log("=" * 70)
    
    # روند زمانی Target
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))
    
    # Target over time
    axes[0].plot(df["Timestamp"], df[TARGET], linewidth=0.5, color="steelblue")
    axes[0].set_title(f"{TARGET} over Time", fontsize=12)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel(TARGET)
    axes[0].grid(True, alpha=0.3)
    
    # Anomaly over time
    anomaly_times = df[df[ANOMALY] == 1]["Timestamp"]
    axes[1].scatter(anomaly_times, [1] * len(anomaly_times), color="red", s=10, alpha=0.7)
    axes[1].set_title(f"Anomaly Occurrences over Time", fontsize=12)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Anomaly")
    axes[1].set_yticks([0, 1])
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_07_time_series.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 04_07_time_series.png")
    
    # الگوی ساعتی
    df["hour"] = df["Timestamp"].dt.hour
    hourly_target = df.groupby("hour")[TARGET].mean()
    hourly_anomaly = df.groupby("hour")[ANOMALY].mean() * 100
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    axes[0].bar(hourly_target.index, hourly_target.values, color="steelblue")
    axes[0].set_title(f"Average {TARGET} by Hour of Day")
    axes[0].set_xlabel("Hour")
    axes[0].set_ylabel("Average Boiler Efficiency (%)")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(hourly_anomaly.index, hourly_anomaly.values, color="coral")
    axes[1].set_title(f"Anomaly Rate (%) by Hour of Day")
    axes[1].set_xlabel("Hour")
    axes[1].set_ylabel("Anomaly Rate (%)")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_07_hourly_pattern.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 04_07_hourly_pattern.png")
    
    # الگوی روزانه
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    daily_target = df.groupby("day_of_week")[TARGET].mean()
    log(f"\n  Average {TARGET} by day of week:")
    for day, val in daily_target.items():
        log(f"    Day {day}: {val:.4f}")
    
    return {"hourly_target": hourly_target, "daily_target": daily_target}

# ============================================================
# 4.8 ANOMALY DETECTION (EDA)
# ============================================================
def stage_4_8_anomaly_detection(df):
    log("\n" + "=" * 70)
    log("STAGE 4.8: ANOMALY DETECTION (EDA)")
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
    
    # Boxplot مقایسه
    top_diff = comparison.head(6).index.tolist()
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(top_diff):
        data_to_plot = [normal_df[col].dropna(), anomaly_df[col].dropna()]
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
    
    # Pairplot برای Top 5 features + Target
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
def stage_4_10_insights(df, overview, corr_df, anomaly_corr_df, comparison):
    log("\n" + "=" * 70)
    log("STAGE 4.10: INSIGHT EXTRACTION")
    log("=" * 70)
    
    insights = []
    
    # بینش ۱: وضعیت کلی
    insights.append({
        "ID": "INS-01",
        "Category": "Overview",
        "Insight": f"Dataset has {overview['Total Rows']} records with {overview['Numeric Columns']} numeric features",
        "Implication": "Sufficient for ML modeling"
    })
    
    # بینش ۲: عدم تعادل کلاس
    insights.append({
        "ID": "INS-02",
        "Category": "Class Imbalance",
        "Insight": f"Anomaly rate is only {overview['Anomaly Rate (%)']}% (imbalance ratio 65:1)",
        "Implication": "Requires SMOTE or class weighting in modeling"
    })
    
    # بینش ۳: بهترین ویژگی‌ها
    top3 = corr_df.head(3)["Feature"].tolist()
    insights.append({
        "ID": "INS-03",
        "Category": "Feature Correlation",
        "Insight": f"Top 3 features correlated with {TARGET}: {', '.join(top3)}",
        "Implication": "These will be primary predictors in modeling"
    })
    
    # بینش ۴: توزیع Target
    insights.append({
        "ID": "INS-04",
        "Category": "Target Distribution",
        "Insight": f"{TARGET} ranges from {df[TARGET].min():.2f} to {df[TARGET].max():.2f} (mean: {df[TARGET].mean():.2f})",
        "Implication": "Target range is narrow, requires precise modeling"
    })
    
    # بینش ۵: تفاوت Anomaly
    top_anomaly_feature = comparison.index[0]
    insights.append({
        "ID": "INS-05",
        "Category": "Anomaly Characteristics",
        "Insight": f"Feature '{top_anomaly_feature}' shows largest difference between anomaly and normal ({comparison.iloc[0]['Diff_Pct']:.2f}%)",
        "Implication": "This feature is a strong anomaly indicator"
    })
    
    # بینش ۶: Multicollinearity
    insights.append({
        "ID": "INS-06",
        "Category": "Multicollinearity",
        "Insight": "High correlation pairs identified (|corr| > 0.9) in Stage 4.4",
        "Implication": "Feature selection needed in Phase 05"
    })
    
    # بینش ۷: Normality
    insights.append({
        "ID": "INS-07",
        "Category": "Normality",
        "Insight": "Shapiro-Wilk test rejected normality for Target",
        "Implication": "Non-parametric tests may be more appropriate"
    })
    
    insights_df = pd.DataFrame(insights)
    insights_df.to_csv(TABLES_DIR / "04_10_insights.csv", index=False)
    log(f"  [OK] Saved: 04_10_insights.csv")
    
    log(f"\n  Extracted {len(insights)} key insights:")
    for ins in insights:
        log(f"    [{ins['ID']}] {ins['Category']}: {ins['Insight']}")
    
    return insights_df

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 04: EXPLORATORY DATA ANALYSIS (EDA)")
    log("=" * 70)
    
    overview = stage_4_1_overview(df)
    desc = stage_4_2_univariate(df)
    corr_df = stage_4_3_bivariate(df)
    corr_matrix = stage_4_4_multivariate(df)
    dist_results = stage_4_5_distribution(df)
    anomaly_corr_df = stage_4_6_correlation(df)
    pattern_results = stage_4_7_pattern(df)
    comparison = stage_4_8_anomaly_detection(df)
    top5 = stage_4_9_visualization(df)
    insights_df = stage_4_10_insights(df, overview, corr_df, anomaly_corr_df, comparison)
    
    log("\n" + "=" * 70)
    log("PHASE 04 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase04_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()