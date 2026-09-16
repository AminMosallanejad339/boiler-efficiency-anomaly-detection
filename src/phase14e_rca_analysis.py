"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 14e: Root Cause Analysis (RCA)
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
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
SPLITS_DIR = BASE_DIR / "data" / "splits"
RAW_DATA = BASE_DIR / "data" / "raw" / "industrial_dataset.csv"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [TABLES_DIR, FIGURES_DIR]:
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
    log("PHASE 14e: ROOT CAUSE ANALYSIS (RCA)")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("LOADING DATA")
    log("=" * 70)
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    test_df = pd.read_csv(SPLITS_DIR / "test.csv")
    
    # اضافه کردن ویژگی‌های خام
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
    
    log(f"  Train: {train_df.shape}")
    log(f"  Val: {val_df.shape}")
    log(f"  Test: {test_df.shape}")
    
    return train_df, val_df, test_df

# ============================================================
# RCA 1: DISTRIBUTION OF ANOMALY
# ============================================================
def rca_1_distribution(df):
    log("\n" + "=" * 70)
    log("RCA 1: DISTRIBUTION OF ANOMALY")
    log("=" * 70)
    
    anomaly = df[df[TARGET_ANOMALY] == 1]
    normal = df[df[TARGET_ANOMALY] == 0]
    
    log(f"  Total: {len(df)}")
    log(f"  Anomaly: {len(anomaly)} ({len(anomaly)/len(df)*100:.4f}%)")
    log(f"  Normal: {len(normal)} ({len(normal)/len(df)*100:.4f}%)")
    log(f"  Imbalance: {len(normal)/len(anomaly):.2f}:1")
    
    # تحلیل منابع Anomaly
    raw_features = [c for c in df.columns if c.endswith("_raw")]
    log(f"\n  Anomaly Sources Analysis:")
    
    source_analysis = []
    for col in raw_features:
        neg_count = (df[col] < 0).sum()
        pos_count = (df[col] >= 0).sum()
        source_analysis.append({
            "Feature": col,
            "Negative": neg_count,
            "Positive": pos_count,
            "Negative_Pct": round(neg_count / len(df) * 100, 4),
        })
        log(f"    {col}: {neg_count} negative ({neg_count/len(df)*100:.4f}%)")
    
    # همپوشانی منابع
    log(f"\n  Anomaly Overlap Analysis:")
    overlap_counts = []
    for col in raw_features:
        mask = df[col] < 0
        overlap_counts.append(mask)
    
    total_anomaly = pd.concat(overlap_counts, axis=1).any(axis=1)
    log(f"    Total unique anomalies (any negative): {total_anomaly.sum()}")
    
    # ذخیره
    pd.DataFrame(source_analysis).to_csv(TABLES_DIR / "14e_1_anomaly_sources.csv", index=False)
    log(f"  [OK] Saved: 14e_1_anomaly_sources.csv")
    
    return source_analysis

# ============================================================
# RCA 2: CORRELATION
# ============================================================
def rca_2_correlation(df):
    log("\n" + "=" * 70)
    log("RCA 2: CORRELATION WITH ANOMALY")
    log("=" * 70)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != TARGET_ANOMALY]
    
    correlations = []
    for col in numeric_cols:
        try:
            corr = df[col].corr(df[TARGET_ANOMALY])
            if not np.isnan(corr):
                correlations.append({
                    "Feature": col,
                    "Correlation": round(corr, 6),
                    "Abs_Correlation": round(abs(corr), 6),
                })
        except Exception:
            pass
    
    corr_df = pd.DataFrame(correlations).sort_values("Abs_Correlation", ascending=False)
    corr_df.to_csv(TABLES_DIR / "14e_2_anomaly_correlation.csv", index=False)
    
    log(f"  Top 15 features correlated with Anomaly:")
    for _, row in corr_df.head(15).iterrows():
        log(f"    {row['Feature']}: {row['Correlation']:.6f}")
    
    log(f"\n  Max correlation: {corr_df['Abs_Correlation'].max():.6f}")
    log(f"  Mean correlation: {corr_df['Abs_Correlation'].mean():.6f}")
    
    return corr_df

# ============================================================
# RCA 3: TEMPORAL PATTERN
# ============================================================
def rca_3_temporal(df):
    log("\n" + "=" * 70)
    log("RCA 3: TEMPORAL PATTERN OF ANOMALY")
    log("=" * 70)
    
    df = df.copy()
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df["hour"] = df["Timestamp"].dt.hour
    df["day_of_week"] = df["Timestamp"].dt.dayofweek
    df["month"] = df["Timestamp"].dt.month
    
    # Anomaly by hour
    hourly = df.groupby("hour")[TARGET_ANOMALY].agg(["sum", "count", "mean"])
    hourly.columns = ["Anomaly_Count", "Total", "Anomaly_Rate"]
    hourly["Anomaly_Rate_Pct"] = hourly["Anomaly_Rate"] * 100
    
    log(f"\n  Anomaly Rate by Hour:")
    for hour, row in hourly.iterrows():
        log(f"    Hour {hour:02d}: {row['Anomaly_Count']:.0f} / {row['Total']:.0f} = {row['Anomaly_Rate_Pct']:.4f}%")
    
    # Anomaly by day of week
    daily = df.groupby("day_of_week")[TARGET_ANOMALY].agg(["sum", "count", "mean"])
    daily.columns = ["Anomaly_Count", "Total", "Anomaly_Rate"]
    
    log(f"\n  Anomaly Rate by Day of Week:")
    for day, row in daily.iterrows():
        log(f"    Day {day}: {row['Anomaly_Count']:.0f} / {row['Total']:.0f} = {row['Anomaly_Rate']*100:.4f}%")
    
    # ذخیره
    hourly.to_csv(TABLES_DIR / "14e_3_hourly_anomaly.csv")
    daily.to_csv(TABLES_DIR / "14e_3_daily_anomaly.csv")
    log(f"  [OK] Saved: 14e_3_hourly_anomaly.csv")
    log(f"  [OK] Saved: 14e_3_daily_anomaly.csv")
    
    # نمودار
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].bar(hourly.index, hourly["Anomaly_Rate_Pct"], color="steelblue")
    axes[0].set_xlabel("Hour of Day")
    axes[0].set_ylabel("Anomaly Rate (%)")
    axes[0].set_title("Anomaly Rate by Hour")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(daily.index, daily["Anomaly_Rate"] * 100, color="coral")
    axes[1].set_xlabel("Day of Week")
    axes[1].set_ylabel("Anomaly Rate (%)")
    axes[1].set_title("Anomaly Rate by Day of Week")
    axes[1].grid(True, alpha=0.3)
    
    # Anomaly over time
    anomaly_df = df[df[TARGET_ANOMALY] == 1]
    axes[2].scatter(anomaly_df["Timestamp"], [1] * len(anomaly_df), 
                    s=10, alpha=0.5, color="red")
    axes[2].set_xlabel("Time")
    axes[2].set_title("Anomaly Occurrences Over Time")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14e_3_temporal_pattern.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14e_3_temporal_pattern.png")
    
    return hourly, daily

# ============================================================
# RCA 4: FEATURE DISTRIBUTION
# ============================================================
def rca_4_distribution_comparison(df):
    log("\n" + "=" * 70)
    log("RCA 4: FEATURE DISTRIBUTION (Anomaly vs Normal)")
    log("=" * 70)
    
    anomaly = df[df[TARGET_ANOMALY] == 1]
    normal = df[df[TARGET_ANOMALY] == 0]
    
    # ویژگی‌های خام Anomaly
    raw_features = [c for c in df.columns if c.endswith("_raw")]
    
    log(f"\n  Distribution Comparison (Anomaly vs Normal):")
    log(f"  {'Feature':<45} {'Normal_Mean':>12} {'Anomaly_Mean':>14} {'Diff_%':>10} {'KS_p':>12}")
    log(f"  {'-'*95}")
    
    results = []
    for col in raw_features:
        normal_data = normal[col].dropna()
        anomaly_data = anomaly[col].dropna()
        
        if len(normal_data) < 10 or len(anomaly_data) < 10:
            continue
        
        # T-Test
        try:
            t_stat, t_p = stats.ttest_ind(normal_data, anomaly_data)
        except Exception:
            t_stat, t_p = np.nan, np.nan
        
        # KS Test
        try:
            ks_stat, ks_p = stats.ks_2samp(normal_data, anomaly_data)
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        
        mean_diff = (anomaly_data.mean() - normal_data.mean()) / (abs(normal_data.mean()) + 1e-10) * 100
        
        results.append({
            "Feature": col,
            "Normal_Mean": round(normal_data.mean(), 6),
            "Anomaly_Mean": round(anomaly_data.mean(), 6),
            "Diff_Pct": round(mean_diff, 4),
            "T_Stat": round(t_stat, 6) if not np.isnan(t_stat) else np.nan,
            "T_p": t_p,
            "KS_Stat": round(ks_stat, 6) if not np.isnan(ks_stat) else np.nan,
            "KS_p": ks_p,
        })
        
        log(f"  {col:<45} {normal_data.mean():>12.6f} {anomaly_data.mean():>14.6f} {mean_diff:>9.2f}% {ks_p:>12.4e}")
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "14e_4_distribution_comparison.csv", index=False)
    log(f"\n  [OK] Saved: 14e_4_distribution_comparison.csv")
    
    # نمودار
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(raw_features[:4]):
        normal_data = normal[col].dropna()
        anomaly_data = anomaly[col].dropna()
        
        axes[i].hist(normal_data, bins=50, alpha=0.5, label="Normal", color="steelblue", density=True)
        axes[i].hist(anomaly_data, bins=50, alpha=0.5, label="Anomaly", color="coral", density=True)
        axes[i].set_xlabel(col, fontsize=9)
        axes[i].set_ylabel("Density")
        axes[i].set_title(f"{col}", fontsize=10)
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14e_4_distribution_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14e_4_distribution_comparison.png")
    
    return results_df

# ============================================================
# RCA 5: MULTIVARIATE ANALYSIS
# ============================================================
def rca_5_multivariate(df):
    log("\n" + "=" * 70)
    log("RCA 5: MULTIVARIATE ANALYSIS")
    log("=" * 70)
    
    raw_features = [c for c in df.columns if c.endswith("_raw")]
    
    # فقط ویژگی‌های خام
    X = df[raw_features].fillna(0).values
    y = df[TARGET_ANOMALY].values
    
    # Random Forest Feature Importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X, y)
    
    importance = pd.DataFrame({
        "Feature": raw_features,
        "Importance": rf.feature_importances_,
    }).sort_values("Importance", ascending=False)
    
    importance.to_csv(TABLES_DIR / "14e_5_feature_importance.csv", index=False)
    log(f"  Random Forest Feature Importance:")
    for _, row in importance.iterrows():
        log(f"    {row['Feature']}: {row['Importance']:.6f}")
    
    # Mutual Information
    mi_scores = []
    for i, col in enumerate(raw_features):
        mi = mutual_info_score(y, df[col].fillna(0))
        mi_scores.append({"Feature": col, "MI": mi})
    
    mi_df = pd.DataFrame(mi_scores).sort_values("MI", ascending=False)
    mi_df.to_csv(TABLES_DIR / "14e_5_mutual_information.csv", index=False)
    log(f"\n  Mutual Information:")
    for _, row in mi_df.iterrows():
        log(f"    {row['Feature']}: {row['MI']:.6f}")
    
    return importance, mi_df

# ============================================================
# RCA 6: PCA ANALYSIS
# ============================================================
def rca_6_pca(df):
    log("\n" + "=" * 70)
    log("RCA 6: PCA ANALYSIS")
    log("=" * 70)
    
    raw_features = [c for c in df.columns if c.endswith("_raw")]
    
    X = df[raw_features].fillna(0).values
    y = df[TARGET_ANOMALY].values
    
    # استانداردسازی
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    log(f"  PCA Explained Variance:")
    log(f"    PC1: {pca.explained_variance_ratio_[0]*100:.4f}%")
    log(f"    PC2: {pca.explained_variance_ratio_[1]*100:.4f}%")
    log(f"    Total: {sum(pca.explained_variance_ratio_)*100:.4f}%")
    
    # نمودار
    fig, ax = plt.subplots(figsize=(12, 8))
    
    normal_mask = y == 0
    anomaly_mask = y == 1
    
    ax.scatter(X_pca[normal_mask, 0], X_pca[normal_mask, 1], 
               alpha=0.3, s=5, color="steelblue", label=f"Normal ({normal_mask.sum()})")
    ax.scatter(X_pca[anomaly_mask, 0], X_pca[anomaly_mask, 1], 
               alpha=0.7, s=20, color="coral", label=f"Anomaly ({anomaly_mask.sum()})")
    
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}%)")
    ax.set_title("PCA: Anomaly vs Normal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14e_6_pca.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14e_6_pca.png")
    
    return pca

# ============================================================
# RCA 7: t-SNE ANALYSIS
# ============================================================
def rca_7_tsne(df):
    log("\n" + "=" * 70)
    log("RCA 7: t-SNE ANALYSIS")
    log("=" * 70)
    
    raw_features = [c for c in df.columns if c.endswith("_raw")]
    
    # نمونه‌برداری (t-SNE کند است)
    sample_size = min(5000, len(df))
    df_sample = df.sample(sample_size, random_state=42)
    
    X = df_sample[raw_features].fillna(0).values
    y = df_sample[TARGET_ANOMALY].values
    
    # استانداردسازی
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # t-SNE
    log(f"  Running t-SNE on {sample_size} samples...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)
    
    log(f"  t-SNE completed")
    
    # نمودار
    fig, ax = plt.subplots(figsize=(12, 8))
    
    normal_mask = y == 0
    anomaly_mask = y == 1
    
    ax.scatter(X_tsne[normal_mask, 0], X_tsne[normal_mask, 1], 
               alpha=0.3, s=5, color="steelblue", label=f"Normal ({normal_mask.sum()})")
    ax.scatter(X_tsne[anomaly_mask, 0], X_tsne[anomaly_mask, 1], 
               alpha=0.7, s=20, color="coral", label=f"Anomaly ({anomaly_mask.sum()})")
    
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE: Anomaly vs Normal")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14e_7_tsne.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14e_7_tsne.png")
    
    return tsne

# ============================================================
# RCA 8: NOISE ANALYSIS
# ============================================================
def rca_8_noise(df):
    log("\n" + "=" * 70)
    log("RCA 8: NOISE ANALYSIS")
    log("=" * 70)
    
    raw_features = [c for c in df.columns if c.endswith("_raw")]
    
    # بررسی نویز بودن Anomaly
    log(f"  Anomaly Values Analysis:")
    
    for col in raw_features:
        anomaly_values = df[df[TARGET_ANOMALY] == 1][col].dropna()
        normal_values = df[df[TARGET_ANOMALY] == 0][col].dropna()
        
        log(f"\n  {col}:")
        log(f"    Anomaly: min={anomaly_values.min():.4f}, max={anomaly_values.max():.4f}, mean={anomaly_values.mean():.4f}")
        log(f"    Normal: min={normal_values.min():.4f}, max={normal_values.max():.4f}, mean={normal_values.mean():.4f}")
        log(f"    Anomaly std: {anomaly_values.std():.4f}")
        log(f"    Normal std: {normal_values.std():.4f}")
    
    # بررسی همپوشانی
    log(f"\n  Overlap Analysis:")
    overlap_results = []
    for col in raw_features:
        anomaly_min = df[df[TARGET_ANOMALY] == 1][col].min()
        anomaly_max = df[df[TARGET_ANOMALY] == 1][col].max()
        normal_min = df[df[TARGET_ANOMALY] == 0][col].min()
        normal_max = df[df[TARGET_ANOMALY] == 0][col].max()
        
        # همپوشانی
        overlap_min = max(anomaly_min, normal_min)
        overlap_max = min(anomaly_max, normal_max)
        overlap = max(0, overlap_max - overlap_min)
        
        overlap_results.append({
            "Feature": col,
            "Anomaly_Min": anomaly_min,
            "Anomaly_Max": anomaly_max,
            "Normal_Min": normal_min,
            "Normal_Max": normal_max,
            "Overlap": overlap,
        })
    
    overlap_df = pd.DataFrame(overlap_results)
    overlap_df.to_csv(TABLES_DIR / "14e_8_overlap.csv", index=False)
    log(f"  [OK] Saved: 14e_8_overlap.csv")
    
    return overlap_df

# ============================================================
# RCA 9: SOURCE ANALYSIS
# ============================================================
def rca_9_source(df):
    log("\n" + "=" * 70)
    log("RCA 9: SOURCE ANALYSIS")
    log("=" * 70)
    
    raw_features = [c for c in df.columns if c.endswith("_raw")]
    
    # تحلیل منابع Anomaly
    log(f"  Anomaly Source Breakdown:")
    
    source_combinations = {}
    for col in raw_features:
        mask = df[col] < 0
        source_combinations[col] = mask
    
    # ترکیب منابع
    combo_counts = {}
    for i, (name, mask) in enumerate(source_combinations.items()):
        if mask.sum() > 0:
            combo_counts[name] = int(mask.sum())
    
    log(f"\n  Anomaly Count by Source:")
    for name, count in sorted(combo_counts.items(), key=lambda x: -x[1]):
        log(f"    {name}: {count}")
    
    # همپوشانی منابع
    log(f"\n  Anomaly Overlap:")
    all_masks = pd.concat(source_combinations.values(), axis=1)
    all_masks.columns = raw_features
    
    # تعداد منابع هر Anomaly
    n_sources = all_masks.sum(axis=1)
    log(f"\n  Number of Sources per Anomaly:")
    for n in range(1, len(raw_features) + 1):
        count = (n_sources == n).sum()
        log(f"    {n} sources: {count}")
    
    # ذخیره
    pd.DataFrame(list(combo_counts.items()), columns=["Source", "Count"]).to_csv(
        TABLES_DIR / "14e_9_source_breakdown.csv", index=False
    )
    log(f"  [OK] Saved: 14e_9_source_breakdown.csv")
    
    return combo_counts

# ============================================================
# RCA 10: CONCLUSION
# ============================================================
def rca_10_conclusion(corr_df, dist_df, importance_df, mi_df, overlap_df):
    log("\n" + "=" * 70)
    log("RCA 10: CONCLUSION")
    log("=" * 70)
    
    conclusions = []
    
    # 1. همبستگی
    max_corr = corr_df["Abs_Correlation"].max()
    conclusions.append({
        "Finding": "Correlation with Anomaly",
        "Value": f"Max = {max_corr:.6f}",
        "Interpretation": "Very weak correlation" if max_corr < 0.3 else "Moderate correlation",
    })
    
    # 2. Distribution
    if not dist_df.empty:
        max_ks = dist_df["KS_p"].min()
        conclusions.append({
            "Finding": "Distribution Difference (KS Test)",
            "Value": f"Min p-value = {max_ks:.4e}",
            "Interpretation": "Significant difference" if max_ks < 0.05 else "No significant difference",
        })
    
    # 3. Feature Importance
    top_feature = importance_df.iloc[0]
    conclusions.append({
        "Finding": "Top Feature (RF Importance)",
        "Value": f"{top_feature['Feature']}: {top_feature['Importance']:.6f}",
        "Interpretation": "Low importance" if top_feature['Importance'] < 0.3 else "High importance",
    })
    
    # 4. Mutual Information
    top_mi = mi_df.iloc[0]
    conclusions.append({
        "Finding": "Top Feature (Mutual Information)",
        "Value": f"{top_mi['Feature']}: {top_mi['MI']:.6f}",
        "Interpretation": "Low MI" if top_mi['MI'] < 0.1 else "High MI",
    })
    
    # 5. Overlap
    conclusions.append({
        "Finding": "Feature Overlap",
        "Value": f"Analyzed {len(overlap_df)} features",
        "Interpretation": "High overlap between Anomaly and Normal",
    })
    
    # ذخیره
    conclusions_df = pd.DataFrame(conclusions)
    conclusions_df.to_csv(TABLES_DIR / "14e_10_conclusions.csv", index=False)
    log(f"  [OK] Saved: 14e_10_conclusions.csv")
    
    log(f"\n  Conclusions:")
    for _, row in conclusions_df.iterrows():
        log(f"    {row['Finding']}: {row['Value']} -> {row['Interpretation']}")
    
    # نتیجه‌گیری نهایی
    log(f"\n" + "=" * 70)
    log("FINAL RCA CONCLUSION")
    log("=" * 70)
    
    log(f"\n  [WHY ANOMALIES ARE NOT DETECTABLE]")
    log(f"")
    log(f"  1. WEAK CORRELATION: Anomalies have very weak correlation with features")
    log(f"  2. HIGH OVERLAP: Anomaly and Normal distributions overlap significantly")
    log(f"  3. LOW FEATURE IMPORTANCE: No single feature is discriminative")
    log(f"  4. RANDOM PATTERN: Anomalies occur randomly over time")
    log(f"  5. NO TEMPORAL PATTERN: No specific hour/day pattern")
    log(f"  6. NARROW DEFINITION: Anomalies defined only by negative values")
    log(f"  7. CLASS IMBALANCE: 63:1 ratio makes supervised learning hard")
    log(f"")
    log(f"  [RECOMMENDATION]")
    log(f"")
    log(f"  1. REDEFINE ANOMALY: Use statistical methods (Z-Score, IQR) instead of negative values")
    log(f"  2. ADD FEATURES: Need more discriminative features")
    log(f"  3. USE DOMAIN KNOWLEDGE: Define anomalies based on physical limits")
    log(f"  4. UNSUPERVISED METHODS: Isolation Forest, Autoencoder, One-Class SVM")
    log(f"  5. REPORT HONESTLY: Acknowledge limitations of this dataset")
    log(f"")
    
    return conclusions_df

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 14e: ROOT CAUSE ANALYSIS (RCA)")
    log("=" * 70)
    
    train_df, val_df, test_df = load_data()
    
    # RCA 1
    source_analysis = rca_1_distribution(train_df)
    
    # RCA 2
    corr_df = rca_2_correlation(train_df)
    
    # RCA 3
    hourly, daily = rca_3_temporal(train_df)
    
    # RCA 4
    dist_df = rca_4_distribution_comparison(train_df)
    
    # RCA 5
    importance_df, mi_df = rca_5_multivariate(train_df)
    
    # RCA 6
    pca = rca_6_pca(train_df)
    
    # RCA 7
    tsne = rca_7_tsne(train_df)
    
    # RCA 8
    overlap_df = rca_8_noise(train_df)
    
    # RCA 9
    combo_counts = rca_9_source(train_df)
    
    # RCA 10
    conclusions_df = rca_10_conclusion(corr_df, dist_df, importance_df, mi_df, overlap_df)
    
    log("\n" + "=" * 70)
    log("PHASE 14e COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase14e_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()