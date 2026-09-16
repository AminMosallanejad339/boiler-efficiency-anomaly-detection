"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 18: Diagnostics & Assumption Validation (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Isolation Forest (Tuned) - Diagnostics
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
from scipy.stats import shapiro, anderson, kstest, normaltest
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

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
    log("PHASE 18: DIAGNOSTICS & ASSUMPTION VALIDATION")
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
# 18.1 RESIDUAL ANALYSIS
# ============================================================
def stage_18_1_residuals(iso, train_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 18.1: RESIDUAL ANALYSIS")
    log("=" * 70)

    X = train_df[raw_features].fillna(0).values
    y = train_df[TARGET_ANOMALY].values

    scores = -iso.score_samples(X)

    threshold = np.percentile(scores, 100 * (1 - 0.02))
    residuals = scores - threshold

    log(f"  Residual Analysis (Score-based):")
    log(f"    Score Mean: {scores.mean():.6f}")
    log(f"    Score Std: {scores.std():.6f}")
    log(f"    Score Min: {scores.min():.6f}")
    log(f"    Score Max: {scores.max():.6f}")
    log(f"    Threshold (contamination=0.02): {threshold:.6f}")
    log(f"    Residuals Mean: {residuals.mean():.6f}")
    log(f"    Residuals Std: {residuals.std():.6f}")

    normal_scores = scores[y == 0]
    anomaly_scores = scores[y == 1]

    log(f"\n  Score Distribution by Class:")
    log(f"    Normal: mean={normal_scores.mean():.6f}, std={normal_scores.std():.6f}")
    log(f"    Anomaly: mean={anomaly_scores.mean():.6f}, std={anomaly_scores.std():.6f}")
    log(f"    Separation: {anomaly_scores.mean() - normal_scores.mean():.6f}")

    t_stat, t_p = stats.ttest_ind(normal_scores, anomaly_scores)
    log(f"\n  T-Test (Normal vs Anomaly):")
    log(f"    T-Statistic: {t_stat:.6f}")
    log(f"    P-Value: {t_p:.4e}")
    log(f"    Significant: {t_p < 0.05}")

    results = {
        "scores": scores,
        "residuals": residuals,
        "threshold": threshold,
        "normal_scores": normal_scores,
        "anomaly_scores": anomaly_scores,
    }

    return results

# ============================================================
# 18.2 RESIDUAL PLOTTING
# ============================================================
def stage_18_2_residual_plots(residuals_data, train_df):
    log("\n" + "=" * 70)
    log("STAGE 18.2: RESIDUAL PLOTTING")
    log("=" * 70)

    scores = residuals_data["scores"]
    residuals = residuals_data["residuals"]
    y = train_df[TARGET_ANOMALY].values

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Score vs Index
    axes[0, 0].plot(scores, linewidth=0.5, color="steelblue")
    axes[0, 0].axhline(y=residuals_data["threshold"], color="red", linestyle="--", label="Threshold")
    axes[0, 0].set_xlabel("Index")
    axes[0, 0].set_ylabel("Anomaly Score")
    axes[0, 0].set_title("Anomaly Score vs Index")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Score Distribution by Class
    normal_scores = residuals_data["normal_scores"]
    anomaly_scores = residuals_data["anomaly_scores"]

    axes[0, 1].hist(normal_scores, bins=50, alpha=0.5, label="Normal", color="steelblue", density=True)
    axes[0, 1].hist(anomaly_scores, bins=50, alpha=0.5, label="Anomaly", color="coral", density=True)
    axes[0, 1].axvline(x=residuals_data["threshold"], color="red", linestyle="--", label="Threshold")
    axes[0, 1].set_xlabel("Anomaly Score")
    axes[0, 1].set_ylabel("Density")
    axes[0, 1].set_title("Score Distribution by Class")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Residuals vs Fitted
    colors = ["coral" if yy == 1 else "steelblue" for yy in y]
    axes[1, 0].scatter(scores, residuals, alpha=0.3, s=5, c=colors)
    axes[1, 0].axhline(y=0, color="red", linestyle="--")
    axes[1, 0].set_xlabel("Fitted (Score)")
    axes[1, 0].set_ylabel("Residuals")
    axes[1, 0].set_title("Residuals vs Fitted")
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Residuals Histogram
    axes[1, 1].hist(residuals, bins=50, color="purple", alpha=0.7, edgecolor="black")
    axes[1, 1].axvline(x=0, color="red", linestyle="--")
    axes[1, 1].set_xlabel("Residuals")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title("Residuals Distribution")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_2_residual_plots.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 18_2_residual_plots.png")

    return True

# ============================================================
# 18.3 Q-Q PLOT ANALYSIS
# ============================================================
def stage_18_3_qq_plot(residuals_data):
    log("\n" + "=" * 70)
    log("STAGE 18.3: Q-Q PLOT ANALYSIS")
    log("=" * 70)

    residuals = residuals_data["residuals"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    stats.probplot(residuals, dist="norm", plot=axes[0])
    axes[0].set_title("Q-Q Plot: Residuals")
    axes[0].grid(True, alpha=0.3)

    stats.probplot(residuals_data["scores"], dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q Plot: Anomaly Scores")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_3_qq_plot.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 18_3_qq_plot.png")

    return True

# ============================================================
# 18.4 RESIDUAL NORMALITY TEST
# ============================================================
def stage_18_4_normality(residuals_data):
    log("\n" + "=" * 70)
    log("STAGE 18.4: RESIDUAL NORMALITY TEST")
    log("=" * 70)

    residuals = np.asarray(residuals_data["residuals"])

    # نمونه‌برداری
    if len(residuals) > 5000:
        np.random.seed(SEED)
        sample = np.random.choice(residuals, 5000, replace=False)
    else:
        sample = residuals

    # Shapiro-Wilk
    shapiro_stat, shapiro_p = shapiro(sample)

    # Anderson-Darling
    ad_result = anderson(residuals, dist="norm")

    # Kolmogorov-Smirnov
    ks_stat, ks_p = kstest(residuals, "norm", args=(residuals.mean(), residuals.std()))

    # D'Agostino
    dagostino_stat, dagostino_p = normaltest(residuals)

    log(f"  Normality Tests:")
    log(f"    Shapiro-Wilk:")
    log(f"      Statistic: {shapiro_stat:.6f}")
    log(f"      P-Value: {shapiro_p:.4e}")
    log(f"      Normal: {shapiro_p > 0.05}")
    log(f"    Anderson-Darling:")
    log(f"      Statistic: {ad_result.statistic:.6f}")
    log(f"      Critical (5%): {ad_result.critical_values[2]}")
    log(f"      Normal: {ad_result.statistic < ad_result.critical_values[2]}")
    log(f"    Kolmogorov-Smirnov:")
    log(f"      Statistic: {ks_stat:.6f}")
    log(f"      P-Value: {ks_p:.4e}")
    log(f"    D'Agostino:")
    log(f"      Statistic: {dagostino_stat:.6f}")
    log(f"      P-Value: {dagostino_p:.4e}")

    # محاسبه Skewness و Kurtosis با scipy
    skewness = float(stats.skew(residuals))
    kurtosis = float(stats.kurtosis(residuals))

    log(f"\n  Skewness: {skewness:.6f}")
    log(f"  Kurtosis: {kurtosis:.6f}")

    results = {
        "shapiro_stat": float(shapiro_stat),
        "shapiro_p": float(shapiro_p),
        "ad_stat": float(ad_result.statistic),
        "ad_critical": float(ad_result.critical_values[2]),
        "ks_stat": float(ks_stat),
        "ks_p": float(ks_p),
        "dagostino_stat": float(dagostino_stat),
        "dagostino_p": float(dagostino_p),
        "skewness": skewness,
        "kurtosis": kurtosis,
    }

    pd.DataFrame(list(results.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "18_4_normality.csv", index=False
    )
    log(f"  [OK] Saved: 18_4_normality.csv")

    return results

# ============================================================
# 18.5 HETEROSCEDASTICITY TEST
# ============================================================
def stage_18_5_heteroscedasticity(residuals_data, train_df):
    log("\n" + "=" * 70)
    log("STAGE 18.5: HETEROSCEDASTICITY TEST")
    log("=" * 70)

    scores = residuals_data["scores"]
    residuals = residuals_data["residuals"]

    median_score = np.median(scores)
    high_group = residuals[scores > median_score]
    low_group = residuals[scores <= median_score]

    levene_stat, levene_p = stats.levene(high_group, low_group)
    bartlett_stat, bartlett_p = stats.bartlett(high_group, low_group)
    fligner_stat, fligner_p = stats.fligner(high_group, low_group)

    log(f"  Heteroscedasticity Tests:")
    log(f"    Levene Test:")
    log(f"      Statistic: {levene_stat:.6f}")
    log(f"      P-Value: {levene_p:.4e}")
    log(f"      Homoscedastic: {levene_p > 0.05}")
    log(f"    Bartlett Test:")
    log(f"      Statistic: {bartlett_stat:.6f}")
    log(f"      P-Value: {bartlett_p:.4e}")
    log(f"    Fligner-Killeen Test:")
    log(f"      Statistic: {fligner_stat:.6f}")
    log(f"      P-Value: {fligner_p:.4e}")

    results = {
        "levene_stat": float(levene_stat),
        "levene_p": float(levene_p),
        "bartlett_stat": float(bartlett_stat),
        "bartlett_p": float(bartlett_p),
        "fligner_stat": float(fligner_stat),
        "fligner_p": float(fligner_p),
        "homoscedastic": bool(levene_p > 0.05),
    }

    pd.DataFrame(list(results.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "18_5_heteroscedasticity.csv", index=False
    )
    log(f"  [OK] Saved: 18_5_heteroscedasticity.csv")

    return results

# ============================================================
# 18.6 AUTOCORRELATION TEST
# ============================================================
def stage_18_6_autocorrelation(iso, train_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 18.6: AUTOCORRELATION TEST")
    log("=" * 70)

    X = train_df[raw_features].fillna(0).values
    scores = -iso.score_samples(X)

    dw = durbin_watson(scores)

    log(f"  Durbin-Watson: {dw:.6f}")
    log(f"    Interpretation: {'No autocorrelation' if 1.5 < dw < 2.5 else 'Autocorrelation present'}")

    lb_result = acorr_ljungbox(scores, lags=[10, 20, 30], return_df=True)

    log(f"\n  Ljung-Box Test:")
    for lag, row in lb_result.iterrows():
        log(f"    Lag {lag}: stat={row['lb_stat']:.6f}, p={row['lb_pvalue']:.4e}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(scores, lags=40, ax=axes[0])
    axes[0].set_title("ACF: Anomaly Scores")
    plot_pacf(scores, lags=40, ax=axes[1])
    axes[1].set_title("PACF: Anomaly Scores")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_6_autocorrelation.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 18_6_autocorrelation.png")

    results = {
        "durbin_watson": float(dw),
        "lb_lag10_stat": float(lb_result.loc[10, "lb_stat"]),
        "lb_lag10_p": float(lb_result.loc[10, "lb_pvalue"]),
        "lb_lag20_stat": float(lb_result.loc[20, "lb_stat"]),
        "lb_lag20_p": float(lb_result.loc[20, "lb_pvalue"]),
        "lb_lag30_stat": float(lb_result.loc[30, "lb_stat"]),
        "lb_lag30_p": float(lb_result.loc[30, "lb_pvalue"]),
    }

    pd.DataFrame(list(results.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "18_6_autocorrelation.csv", index=False
    )
    log(f"  [OK] Saved: 18_6_autocorrelation.csv")

    return results

# ============================================================
# 18.7 MULTICOLLINEARITY TEST (VIF)
# ============================================================
def stage_18_7_vif(train_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 18.7: MULTICOLLINEARITY TEST (VIF)")
    log("=" * 70)

    X = train_df[raw_features].fillna(0).values

    log(f"  VIF for Raw Anomaly Features:")

    vif_results = []
    for i, col in enumerate(raw_features):
        try:
            vif = variance_inflation_factor(X, i)
            vif_results.append({
                "Feature": col,
                "VIF": round(vif, 6),
                "Status": "HIGH" if vif > 10 else ("MEDIUM" if vif > 5 else "OK"),
            })
            log(f"    {col}: VIF={vif:.6f}")
        except Exception as e:
            log(f"    {col}: ERROR - {e}")

    vif_df = pd.DataFrame(vif_results)
    vif_df.to_csv(TABLES_DIR / "18_7_vif.csv", index=False)
    log(f"  [OK] Saved: 18_7_vif.csv")

    high_vif = vif_df[vif_df["VIF"] > 10]
    log(f"\n  Features with VIF > 10: {len(high_vif)}")

    corr_matrix = train_df[raw_features].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".4f", cmap="coolwarm", center=0,
                square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix - Raw Anomaly Features")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_7_correlation.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 18_7_correlation.png")

    return vif_df

# ============================================================
# 18.8-18.10 INFLUENCE, LEVERAGE, COOK'S DISTANCE
# ============================================================
def stage_18_8_to_18_10_influence():
    log("\n" + "=" * 70)
    log("STAGE 18.8-18.10: INFLUENCE, LEVERAGE, COOK'S DISTANCE")
    log("=" * 70)

    log(f"  [INFO] These tests are for Regression models (OLS)")
    log(f"  [INFO] Isolation Forest is Unsupervised Anomaly Detection")
    log(f"  [INFO] Influence Analysis: Not applicable")
    log(f"  [INFO] Leverage Analysis: Not applicable")
    log(f"  [INFO] Cook's Distance: Not applicable")
    log(f"")
    log(f"  [ALTERNATIVE] For Anomaly Detection:")
    log(f"    - Anomaly Score: Direct measure of outlierness")
    log(f"    - Score Distribution: Analyzed in Stage 18.2")
    log(f"    - Threshold Analysis: Analyzed in Stage 17.7")
    log(f"    - Feature Contribution: Via Feature Importance")

    results = {
        "influence_analysis": "Not applicable (Unsupervised)",
        "leverage_analysis": "Not applicable (Unsupervised)",
        "cooks_distance": "Not applicable (Unsupervised)",
        "alternative": "Anomaly Score + Threshold Analysis",
    }

    return results

# ============================================================
# SUMMARY
# ============================================================
def generate_summary(residuals_data, normality_results, hetero_results, autocorr_results, vif_df):
    log("\n" + "=" * 70)
    log("PHASE 18 SUMMARY")
    log("=" * 70)

    summary = []

    summary.append({
        "Test": "Residual Analysis",
        "Result": f"Score Mean={residuals_data['scores'].mean():.4f}, Std={residuals_data['scores'].std():.4f}",
        "Status": "OK",
    })

    summary.append({
        "Test": "Normality (Shapiro-Wilk)",
        "Result": f"p={normality_results['shapiro_p']:.4e}",
        "Status": "NORMAL" if normality_results['shapiro_p'] > 0.05 else "NOT NORMAL",
    })

    summary.append({
        "Test": "Heteroscedasticity (Levene)",
        "Result": f"p={hetero_results['levene_p']:.4e}",
        "Status": "HOMOSCEDASTIC" if hetero_results['homoscedastic'] else "HETEROSCEDASTIC",
    })

    dw = autocorr_results['durbin_watson']
    summary.append({
        "Test": "Autocorrelation (Durbin-Watson)",
        "Result": f"DW={dw:.4f}",
        "Status": "NO AUTOCORR" if 1.5 < dw < 2.5 else "AUTOCORR",
    })

    high_vif = (vif_df["VIF"] > 10).sum() if not vif_df.empty else 0
    summary.append({
        "Test": "Multicollinearity (VIF)",
        "Result": f"High VIF: {high_vif}/{len(vif_df)}",
        "Status": "OK" if high_vif == 0 else "MULTICOLLINEARITY",
    })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(TABLES_DIR / "18_summary.csv", index=False)
    log(f"  [OK] Saved: 18_summary.csv")

    log(f"\n  Diagnostics Summary:")
    for _, row in summary_df.iterrows():
        log(f"    {row['Test']}: {row['Result']} -> {row['Status']}")

    return summary_df

# ============================================================
# VISUALIZE
# ============================================================
def visualize_diagnostics(residuals_data, normality_results, hetero_results, autocorr_results, vif_df):
    log("\n" + "=" * 70)
    log("VISUALIZING DIAGNOSTICS")
    log("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # 1. Score Distribution
    axes[0, 0].hist(residuals_data["scores"], bins=50, color="steelblue", alpha=0.7, edgecolor="black")
    axes[0, 0].axvline(x=residuals_data["threshold"], color="red", linestyle="--", label="Threshold")
    axes[0, 0].set_xlabel("Anomaly Score")
    axes[0, 0].set_ylabel("Frequency")
    axes[0, 0].set_title("Anomaly Score Distribution")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Q-Q Plot
    stats.probplot(residuals_data["residuals"], dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title("Q-Q Plot: Residuals")
    axes[0, 1].grid(True, alpha=0.3)

    # 3. ACF
    try:
        plot_acf(residuals_data["scores"], lags=40, ax=axes[1, 0])
        axes[1, 0].set_title("ACF: Anomaly Scores")
    except Exception:
        axes[1, 0].text(0.5, 0.5, "ACF Not Available", ha="center", va="center")

    # 4. VIF
    if not vif_df.empty:
        axes[1, 1].barh(vif_df["Feature"], vif_df["VIF"], color="coral")
        axes[1, 1].axvline(x=10, color="red", linestyle="--", label="VIF=10")
        axes[1, 1].set_xlabel("VIF")
        axes[1, 1].set_title("VIF - Raw Anomaly Features")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "18_diagnostics.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 18_diagnostics.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 18: DIAGNOSTICS & ASSUMPTION VALIDATION")
    log("=" * 70)

    train_df, val_df, test_df, raw_features = load_data()

    iso = joblib.load(MODELS_DIR / "isolation_forest_tuned.pkl")
    log(f"\n  Model loaded: isolation_forest_tuned.pkl")

    residuals_data = stage_18_1_residuals(iso, train_df, raw_features)
    stage_18_2_residual_plots(residuals_data, train_df)
    stage_18_3_qq_plot(residuals_data)
    normality_results = stage_18_4_normality(residuals_data)
    hetero_results = stage_18_5_heteroscedasticity(residuals_data, train_df)
    autocorr_results = stage_18_6_autocorrelation(iso, train_df, raw_features)
    vif_df = stage_18_7_vif(train_df, raw_features)
    influence_results = stage_18_8_to_18_10_influence()
    summary_df = generate_summary(residuals_data, normality_results, hetero_results, autocorr_results, vif_df)
    visualize_diagnostics(residuals_data, normality_results, hetero_results, autocorr_results, vif_df)

    log("\n" + "=" * 70)
    log("PHASE 18 COMPLETE!")
    log("=" * 70)

    LOG_FILE = BASE_DIR / "reports" / "phase18_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()
