"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 07: Assumption Checking (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Dual Target: Boiler_Eff_ (Regression) + Anomaly_Label (Classification)
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
from scipy.stats import (
    shapiro, anderson, kstest, normaltest,
    pearsonr, spearmanr, kendalltau,
    levene, bartlett, fligner
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import (
    het_breuschpagan, het_white, acorr_ljungbox
)
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.stattools import durbin_watson

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
SPLITS_DIR = BASE_DIR / "data" / "splits"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_REG = "Boiler_Eff_"
TARGET_CLF = "Anomaly_Label"

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری
# ============================================================
log("=" * 70)
log("PHASE 07: ASSUMPTION CHECKING")
log("=" * 70)

train_df = pd.read_csv(SPLITS_DIR / "train.csv")
val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
test_df = pd.read_csv(SPLITS_DIR / "test.csv")
feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")

train_df["Timestamp"] = pd.to_datetime(train_df["Timestamp"])
val_df["Timestamp"] = pd.to_datetime(val_df["Timestamp"])
test_df["Timestamp"] = pd.to_datetime(test_df["Timestamp"])

FEATURES = feature_df["Feature"].tolist()

log(f"Train: {train_df.shape}")
log(f"Validation: {val_df.shape}")
log(f"Test: {test_df.shape}")
log(f"Features: {len(FEATURES)}")

# ============================================================
# 7.1 ASSUMPTION CHECKING (Strategy)
# ============================================================
def stage_7_1_strategy():
    log("\n" + "=" * 70)
    log("STAGE 7.1: ASSUMPTION CHECKING STRATEGY")
    log("=" * 70)
    
    strategy = {
        "Regression Assumptions": [
            "Linearity: Linear relationship between X and y",
            "Normality: Residuals are normally distributed",
            "Independence: Residuals are independent",
            "Homoscedasticity: Constant variance of residuals",
            "No Multicollinearity: Features are not highly correlated",
            "Stationarity: Time series is stationary",
        ],
        "Classification Assumptions": [
            "Independence: Samples are independent",
            "Class Balance: Handled via Class Weight",
            "Feature Distribution: Checked per class",
        ],
        "Test Methods": {
            "Linearity": "Pearson, Spearman, Kendall",
            "Normality": "Shapiro-Wilk, Anderson-Darling, KS",
            "Independence": "Durbin-Watson, Ljung-Box",
            "Homoscedasticity": "Levene, Bartlett, Breusch-Pagan",
            "Multicollinearity": "VIF",
            "Stationarity": "ADF, KPSS",
        },
    }
    
    for k, v in strategy.items():
        if isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        elif isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                log(f"    {kk}: {vv}")
        else:
            log(f"  {k}: {v}")
    
    return strategy

# ============================================================
# 7.2 LINEARITY CHECK
# ============================================================
def stage_7_2_linearity(train_df):
    log("\n" + "=" * 70)
    log("STAGE 7.2: LINEARITY CHECK")
    log("=" * 70)
    
    results = []
    for col in FEATURES:
        if col not in train_df.columns:
            continue
        x = train_df[col].dropna()
        y = train_df[TARGET_REG].loc[x.index]
        
        if len(x) < 10 or x.nunique() < 2:
            continue
        
        try:
            pearson_r, pearson_p = pearsonr(x, y)
            spearman_r, spearman_p = spearmanr(x, y)
            kendall_r, kendall_p = kendalltau(x, y)
            
            results.append({
                "Feature": col,
                "Pearson_r": round(pearson_r, 6),
                "Pearson_p": pearson_p,
                "Spearman_r": round(spearman_r, 6),
                "Spearman_p": spearman_p,
                "Kendall_r": round(kendall_r, 6),
                "Kendall_p": kendall_p,
                "Abs_Pearson": abs(pearson_r),
            })
        except Exception as e:
            log(f"  [WARNING] {col}: {e}")
    
    results_df = pd.DataFrame(results).sort_values("Abs_Pearson", ascending=False)
    results_df.to_csv(TABLES_DIR / "07_02_linearity.csv", index=False)
    log(f"  [OK] Saved: 07_02_linearity.csv")
    
    log(f"\n  Top 10 linear features (by |Pearson r|):")
    for _, row in results_df.head(10).iterrows():
        log(f"    {row['Feature']}: Pearson={row['Pearson_r']:.6f}, p={row['Pearson_p']:.4e}")
    
    # نتیجه‌گیری
    strong_linear = results_df[results_df["Abs_Pearson"] > 0.5]
    log(f"\n  Features with strong linear relationship (|r| > 0.5): {len(strong_linear)}")
    
    if len(strong_linear) == 0:
        log(f"  [CONCLUSION] NO strong linear relationships detected.")
        log(f"  [IMPLICATION] Linear models may not perform well.")
        log(f"  [RECOMMENDATION] Use tree-based or non-linear models.")
    
    # Scatter plot برای Top 6
    top6 = results_df.head(6)["Feature"].tolist()
    if top6:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        for i, col in enumerate(top6):
            axes[i].scatter(train_df[col], train_df[TARGET_REG], alpha=0.3, s=5, color="steelblue")
            axes[i].set_xlabel(col, fontsize=9)
            axes[i].set_ylabel(TARGET_REG, fontsize=9)
            axes[i].set_title(f"{col}\n(r={results_df[results_df['Feature']==col]['Pearson_r'].values[0]:.4f})", fontsize=10)
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "07_02_linearity_scatter.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 07_02_linearity_scatter.png")
    
    return results_df

# ============================================================
# 7.3 NORMALITY CHECK
# ============================================================
def stage_7_3_normality(train_df):
    log("\n" + "=" * 70)
    log("STAGE 7.3: NORMALITY CHECK")
    log("=" * 70)
    
    results = []
    for col in FEATURES + [TARGET_REG]:
        if col not in train_df.columns:
            continue
        data = train_df[col].dropna()
        if len(data) < 10:
            continue
        
        # نمونه‌برداری برای Shapiro (حداکثر 5000)
        sample = data.sample(min(5000, len(data)), random_state=42)
        
        try:
            shapiro_stat, shapiro_p = shapiro(sample)
        except Exception:
            shapiro_stat, shapiro_p = np.nan, np.nan
        
        try:
            ad_result = anderson(data, dist="norm")
            ad_stat = ad_result.statistic
            ad_critical = ad_result.critical_values[2]  # 5%
        except Exception:
            ad_stat, ad_critical = np.nan, np.nan
        
        # KS Test
        try:
            ks_stat, ks_p = kstest(data, "norm", args=(data.mean(), data.std()))
        except Exception:
            ks_stat, ks_p = np.nan, np.nan
        
        # Skewness و Kurtosis
        skew = data.skew()
        kurt = data.kurtosis()
        
        # نتیجه
        is_normal = (shapiro_p > 0.05) if not np.isnan(shapiro_p) else False
        
        results.append({
            "Feature": col,
            "Shapiro_Stat": round(shapiro_stat, 6) if not np.isnan(shapiro_stat) else np.nan,
            "Shapiro_p": shapiro_p,
            "AD_Stat": round(ad_stat, 6) if not np.isnan(ad_stat) else np.nan,
            "AD_Critical_5pct": round(ad_critical, 6) if not np.isnan(ad_critical) else np.nan,
            "KS_Stat": round(ks_stat, 6) if not np.isnan(ks_stat) else np.nan,
            "KS_p": ks_p,
            "Skewness": round(skew, 6),
            "Kurtosis": round(kurt, 6),
            "Is_Normal": is_normal,
        })
    
    results_df = pd.DataFrame(results).sort_values("Shapiro_p", ascending=False)
    results_df.to_csv(TABLES_DIR / "07_03_normality.csv", index=False)
    log(f"  [OK] Saved: 07_03_normality.csv")
    
    normal_count = results_df["Is_Normal"].sum()
    total = len(results_df)
    log(f"\n  Normal features: {normal_count} / {total}")
    
    # نمایش Target
    target_row = results_df[results_df["Feature"] == TARGET_REG]
    if not target_row.empty:
        row = target_row.iloc[0]
        log(f"\n  Target ({TARGET_REG}):")
        log(f"    Shapiro p-value: {row['Shapiro_p']:.6e}")
        log(f"    Skewness: {row['Skewness']}")
        log(f"    Kurtosis: {row['Kurtosis']}")
        log(f"    Normal? {row['Is_Normal']}")
    
    # نتیجه‌گیری
    if normal_count / total < 0.5:
        log(f"\n  [CONCLUSION] Majority of features are NOT normally distributed.")
        log(f"  [IMPLICATION] Non-parametric tests are more appropriate.")
        log(f"  [RECOMMENDATION] Use tree-based models (Random Forest, XGBoost).")
    
    # QQ-Plot برای Target
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    stats.probplot(train_df[TARGET_REG].dropna(), dist="norm", plot=axes[0])
    axes[0].set_title(f"Q-Q Plot: {TARGET_REG}")
    axes[0].grid(True)
    
    sns.histplot(train_df[TARGET_REG].dropna(), kde=True, ax=axes[1], color="steelblue")
    axes[1].set_title(f"Distribution: {TARGET_REG}")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_03_normality_qqplot.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 07_03_normality_qqplot.png")
    
    return results_df

# ============================================================
# 7.4 INDEPENDENCE CHECK
# ============================================================
def stage_7_4_independence(train_df):
    log("\n" + "=" * 70)
    log("STAGE 7.4: INDEPENDENCE CHECK")
    log("=" * 70)
    
    # Durbin-Watson برای Target
    dw_target = durbin_watson(train_df[TARGET_REG].dropna())
    log(f"  Durbin-Watson for {TARGET_REG}: {dw_target:.6f}")
    log(f"    Interpretation: {'No autocorrelation' if 1.5 < dw_target < 2.5 else 'Autocorrelation present'}")
    
    # Ljung-Box Test برای چند ویژگی کلیدی
    key_features = ["Main_steam_flow_th", "Boiler_oxygen_level_", "Flue_gas_temperature_", "CO_mgm3"]
    available = [c for c in key_features if c in train_df.columns]
    
    lb_results = []
    for col in available:
        try:
            lb = acorr_ljungbox(train_df[col].dropna(), lags=[10], return_df=True)
            lb_stat = lb["lb_stat"].values[0]
            lb_p = lb["lb_pvalue"].values[0]
            lb_results.append({
                "Feature": col,
                "LB_Stat": round(lb_stat, 6),
                "LB_p": lb_p,
                "Independent": lb_p > 0.05,
            })
            log(f"  Ljung-Box {col}: stat={lb_stat:.4f}, p={lb_p:.4e}, Independent={lb_p > 0.05}")
        except Exception as e:
            log(f"  [WARNING] Ljung-Box {col}: {e}")
    
    if lb_results:
        pd.DataFrame(lb_results).to_csv(TABLES_DIR / "07_04_independence.csv", index=False)
        log(f"  [OK] Saved: 07_04_independence.csv")
    
    # ACF Plot
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        axes = axes.flatten()
        for i, col in enumerate(available[:4]):
            plot_acf(train_df[col].dropna(), lags=40, ax=axes[i])
            axes[i].set_title(f"ACF: {col}")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "07_04_acf_plots.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 07_04_acf_plots.png")
    except Exception as e:
        log(f"  [WARNING] ACF plots failed: {e}")
    
    return {"dw_target": dw_target, "lb_results": lb_results}

# ============================================================
# 7.5 HOMOSCEDASTICITY CHECK
# ============================================================
def stage_7_5_homoscedasticity(train_df):
    log("\n" + "=" * 70)
    log("STAGE 7.5: HOMOSCEDASTICITY CHECK")
    log("=" * 70)
    
    # تقسیم Target به دو گروه (بالا و پایین میانه)
    median = train_df[TARGET_REG].median()
    high_group = train_df[train_df[TARGET_REG] > median]
    low_group = train_df[train_df[TARGET_REG] <= median]
    
    results = []
    for col in FEATURES[:20]:  # فقط ۲۰ ویژگی اول
        if col not in train_df.columns:
            continue
        high_data = high_group[col].dropna()
        low_data = low_group[col].dropna()
        
        if len(high_data) < 10 or len(low_data) < 10:
            continue
        
        try:
            levene_stat, levene_p = levene(high_data, low_data)
            bartlett_stat, bartlett_p = bartlett(high_data, low_data)
            fligner_stat, fligner_p = fligner(high_data, low_data)
            
            results.append({
                "Feature": col,
                "Levene_Stat": round(levene_stat, 6),
                "Levene_p": levene_p,
                "Bartlett_Stat": round(bartlett_stat, 6),
                "Bartlett_p": bartlett_p,
                "Fligner_Stat": round(fligner_stat, 6),
                "Fligner_p": fligner_p,
                "Homoscedastic": levene_p > 0.05,
            })
        except Exception:
            pass
    
    if results:
        results_df = pd.DataFrame(results).sort_values("Levene_p", ascending=False)
        results_df.to_csv(TABLES_DIR / "07_05_homoscedasticity.csv", index=False)
        log(f"  [OK] Saved: 07_05_homoscedasticity.csv")
        
        homoscedastic_count = results_df["Homoscedastic"].sum()
        log(f"  Homoscedastic features: {homoscedastic_count} / {len(results_df)}")
        
        log(f"\n  Top 10 homoscedastic features:")
        for _, row in results_df.head(10).iterrows():
            log(f"    {row['Feature']}: Levene p={row['Levene_p']:.4e}")
    
    # Residual plot (با یک مدل خطی ساده)
    try:
        from sklearn.linear_model import LinearRegression
        X = train_df[FEATURES].fillna(0).values
        y = train_df[TARGET_REG].values
        
        model = LinearRegression()
        model.fit(X, y)
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].scatter(y_pred, residuals, alpha=0.3, s=5, color="steelblue")
        axes[0].axhline(y=0, color="red", linestyle="--")
        axes[0].set_xlabel("Predicted")
        axes[0].set_ylabel("Residuals")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)
        
        axes[1].hist(residuals, bins=50, color="coral", edgecolor="black", alpha=0.7)
        axes[1].set_xlabel("Residuals")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Residuals Distribution")
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "07_05_residuals.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 07_05_residuals.png")
    except Exception as e:
        log(f"  [WARNING] Residual plot failed: {e}")
    
    return results_df if results else pd.DataFrame()

# ============================================================
# 7.6 MULTICOLLINEARITY CHECK (VIF)
# ============================================================
def stage_7_6_multicollinearity(train_df):
    log("\n" + "=" * 70)
    log("STAGE 7.6: MULTICOLLINEARITY CHECK (VIF)")
    log("=" * 70)
    
    # انتخاب ۲۰ ویژگی برتر بر اساس همبستگی
    corr_with_target = train_df[FEATURES].corrwith(train_df[TARGET_REG]).abs().sort_values(ascending=False)
    top20 = corr_with_target.head(20).index.tolist()
    
    X = train_df[top20].fillna(0).values
    
    vif_results = []
    for i, col in enumerate(top20):
        try:
            vif = variance_inflation_factor(X, i)
            vif_results.append({
                "Feature": col,
                "VIF": round(vif, 4),
                "Status": "HIGH" if vif > 10 else ("MEDIUM" if vif > 5 else "OK"),
            })
        except Exception as e:
            log(f"  [WARNING] VIF {col}: {e}")
    
    vif_df = pd.DataFrame(vif_results).sort_values("VIF", ascending=False)
    vif_df.to_csv(TABLES_DIR / "07_06_vif.csv", index=False)
    log(f"  [OK] Saved: 07_06_vif.csv")
    
    log(f"\n  VIF Results (Top 20 features):")
    for _, row in vif_df.iterrows():
        log(f"    {row['Feature']}: VIF={row['VIF']:.4f} [{row['Status']}]")
    
    high_vif = vif_df[vif_df["VIF"] > 10]
    log(f"\n  Features with VIF > 10: {len(high_vif)}")
    
    if len(high_vif) > 0:
        log(f"  [CONCLUSION] Multicollinearity detected!")
        log(f"  [RECOMMENDATION] Remove or combine high-VIF features.")
    else:
        log(f"  [CONCLUSION] No severe multicollinearity.")
    
    return vif_df

# ============================================================
# 7.7 STATIONARITY CHECK
# ============================================================
def stage_7_7_stationarity(train_df):
    log("\n" + "=" * 70)
    log("STAGE 7.7: STATIONARITY CHECK")
    log("=" * 70)
    
    # فقط برای ویژگی‌های کلیدی
    key_features = [
        TARGET_REG, "Main_steam_flow_th", "Boiler_oxygen_level_",
        "Flue_gas_temperature_", "CO_mgm3", "Coal_Flow_th",
    ]
    available = [c for c in key_features if c in train_df.columns]
    
    results = []
    for col in available:
        data = train_df[col].dropna()
        if len(data) < 100:
            continue
        
        try:
            # ADF Test
            adf_result = adfuller(data, autolag="AIC")
            adf_stat = adf_result[0]
            adf_p = adf_result[1]
            adf_stationary = adf_p < 0.05
            
            # KPSS Test
            kpss_result = kpss(data, regression="c", nlags="auto")
            kpss_stat = kpss_result[0]
            kpss_p = kpss_result[1]
            kpss_stationary = kpss_p > 0.05
            
            results.append({
                "Feature": col,
                "ADF_Stat": round(adf_stat, 6),
                "ADF_p": adf_p,
                "ADF_Stationary": adf_stationary,
                "KPSS_Stat": round(kpss_stat, 6),
                "KPSS_p": kpss_p,
                "KPSS_Stationary": kpss_stationary,
                "Overall": "Stationary" if (adf_stationary and kpss_stationary) else "Non-Stationary",
            })
            
            log(f"  {col}:")
            log(f"    ADF: stat={adf_stat:.4f}, p={adf_p:.4e}, Stationary={adf_stationary}")
            log(f"    KPSS: stat={kpss_stat:.4f}, p={kpss_p:.4e}, Stationary={kpss_stationary}")
        except Exception as e:
            log(f"  [WARNING] {col}: {e}")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(TABLES_DIR / "07_07_stationarity.csv", index=False)
        log(f"  [OK] Saved: 07_07_stationarity.csv")
        
        stationary_count = (results_df["Overall"] == "Stationary").sum()
        log(f"\n  Stationary features: {stationary_count} / {len(results_df)}")
    
    return pd.DataFrame(results)

# ============================================================
# 7.8 DISTRIBUTION CHECK
# ============================================================
def stage_7_8_distribution(train_df):
    log("\n" + "=" * 70)
    log("STAGE 7.8: DISTRIBUTION CHECK")
    log("=" * 70)
    
    # توزیع Target
    target = train_df[TARGET_REG].dropna()
    
    log(f"  Target ({TARGET_REG}) distribution:")
    log(f"    Mean: {target.mean():.6f}")
    log(f"    Median: {target.median():.6f}")
    log(f"    Std: {target.std():.6f}")
    log(f"    Skewness: {target.skew():.6f}")
    log(f"    Kurtosis: {target.kurtosis():.6f}")
    log(f"    Range: [{target.min():.6f}, {target.max():.6f}]")
    
    # توزیع کلاس
    if TARGET_CLF in train_df.columns:
        class_dist = train_df[TARGET_CLF].value_counts()
        log(f"\n  Class distribution:")
        for label, count in class_dist.items():
            log(f"    Class {label}: {count} ({count/len(train_df)*100:.4f}%)")
        log(f"    Imbalance ratio: {class_dist.max()/class_dist.min():.2f}")
    
    # مقایسه توزیع در سه مجموعه
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (name, data) in enumerate([("Train", train_df)]):
        axes[i].hist(data[TARGET_REG], bins=50, color="steelblue", edgecolor="black", alpha=0.7)
        axes[i].set_title(f"{name}: {TARGET_REG}")
        axes[i].set_xlabel(TARGET_REG)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_08_distribution.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 07_08_distribution.png")
    
    return {"target_stats": target.describe().to_dict()}

# ============================================================
# 7.9 HYPOTHESIS TESTING (Pre-Model)
# ============================================================
def stage_7_9_hypothesis_testing(train_df):
    log("\n" + "=" * 70)
    log("STAGE 7.9: HYPOTHESIS TESTING (Pre-Model)")
    log("=" * 70)
    
    results = []
    
    # H1: همبستگی معنادار بین ویژگی‌ها و Target
    log("  H1: Significant correlation between features and target")
    for col in FEATURES[:20]:
        if col not in train_df.columns:
            continue
        try:
            r, p = pearsonr(train_df[col].dropna(), 
                            train_df[TARGET_REG].loc[train_df[col].dropna().index])
            results.append({
                "Hypothesis": f"H1: {col} vs {TARGET_REG}",
                "Test": "Pearson",
                "Statistic": round(r, 6),
                "p_value": p,
                "Reject_H0": p < 0.05,
            })
        except Exception:
            pass
    
    # H2: تفاوت معنادار بین Anomaly و Normal
    log("\n  H2: Significant difference between Anomaly and Normal")
    normal = train_df[train_df[TARGET_CLF] == 0]
    anomaly = train_df[train_df[TARGET_CLF] == 1]
    
    for col in ["APH_Leakage_", "CO_mgm3", "Boiler_oxygen_level_"]:
        if col not in train_df.columns:
            continue
        try:
            t_stat, t_p = stats.ttest_ind(normal[col].dropna(), anomaly[col].dropna())
            results.append({
                "Hypothesis": f"H2: {col} (Normal vs Anomaly)",
                "Test": "T-Test",
                "Statistic": round(t_stat, 6),
                "p_value": t_p,
                "Reject_H0": t_p < 0.05,
            })
        except Exception:
            pass
    
    # H3: تفاوت معنادار راندمان بین Anomaly و Normal
    log("\n  H3: Significant difference in Boiler_Eff_ between Normal and Anomaly")
    try:
        t_stat, t_p = stats.ttest_ind(normal[TARGET_REG].dropna(), anomaly[TARGET_REG].dropna())
        results.append({
            "Hypothesis": f"H3: {TARGET_REG} (Normal vs Anomaly)",
            "Test": "T-Test",
            "Statistic": round(t_stat, 6),
            "p_value": t_p,
            "Reject_H0": t_p < 0.05,
        })
        log(f"    T-Test: stat={t_stat:.4f}, p={t_p:.4e}")
    except Exception:
        pass
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(TABLES_DIR / "07_09_hypothesis_testing.csv", index=False)
    log(f"  [OK] Saved: 07_09_hypothesis_testing.csv")
    
    rejected = results_df["Reject_H0"].sum()
    log(f"\n  Total hypotheses tested: {len(results_df)}")
    log(f"  Rejected (p < 0.05): {rejected}")
    log(f"  Not rejected: {len(results_df) - rejected}")
    
    return results_df

# ============================================================
# 7.10 ASSUMPTION VALIDATION
# ============================================================
def stage_7_10_validation(linearity_df, normality_df, independence_res, 
                          homosc_df, vif_df, stationarity_df, hyp_df):
    log("\n" + "=" * 70)
    log("STAGE 7.10: ASSUMPTION VALIDATION")
    log("=" * 70)
    
    validation = []
    
    # 1. Linearity
    strong_linear = (linearity_df["Abs_Pearson"] > 0.5).sum() if not linearity_df.empty else 0
    validation.append({
        "Assumption": "Linearity",
        "Status": "NOT SATISFIED" if strong_linear == 0 else f"PARTIAL ({strong_linear} features)",
        "Implication": "Use non-linear models (RF, XGBoost)",
    })
    
    # 2. Normality
    normal_count = normality_df["Is_Normal"].sum() if not normality_df.empty else 0
    total_count = len(normality_df) if not normality_df.empty else 1
    validation.append({
        "Assumption": "Normality",
        "Status": "SATISFIED" if normal_count/total_count > 0.5 else "NOT SATISFIED",
        "Implication": "Use non-parametric tests",
    })
    
    # 3. Independence
    dw = independence_res.get("dw_target", np.nan)
    validation.append({
        "Assumption": "Independence",
        "Status": "SATISFIED" if 1.5 < dw < 2.5 else "NOT SATISFIED",
        "Implication": "Autocorrelation present" if not (1.5 < dw < 2.5) else "No autocorrelation",
    })
    
    # 4. Homoscedasticity
    homosc_count = homosc_df["Homoscedastic"].sum() if not homosc_df.empty else 0
    homosc_total = len(homosc_df) if not homosc_df.empty else 1
    validation.append({
        "Assumption": "Homoscedasticity",
        "Status": "SATISFIED" if homosc_count/homosc_total > 0.5 else "NOT SATISFIED",
        "Implication": "Use robust standard errors",
    })
    
    # 5. Multicollinearity
    high_vif = (vif_df["VIF"] > 10).sum() if not vif_df.empty else 0
    validation.append({
        "Assumption": "Multicollinearity",
        "Status": "SATISFIED" if high_vif == 0 else f"NOT SATISFIED ({high_vif} features)",
        "Implication": "Remove high-VIF features" if high_vif > 0 else "No action needed",
    })
    
    # 6. Stationarity
    if not stationarity_df.empty:
        stationary_count = (stationarity_df["Overall"] == "Stationary").sum()
        validation.append({
            "Assumption": "Stationarity",
            "Status": "SATISFIED" if stationary_count/len(stationarity_df) > 0.5 else "NOT SATISFIED",
            "Implication": "Apply differencing if needed",
        })
    
    validation_df = pd.DataFrame(validation)
    validation_df.to_csv(TABLES_DIR / "07_10_validation.csv", index=False)
    log(f"  [OK] Saved: 07_10_validation.csv")
    
    log(f"\n  Assumption Validation Summary:")
    for _, row in validation_df.iterrows():
        log(f"    {row['Assumption']}: {row['Status']}")
        log(f"      -> {row['Implication']}")
    
    # توصیه‌های نهایی
    log(f"\n  [RECOMMENDATIONS]")
    log(f"    1. Use tree-based models (Random Forest, XGBoost, LightGBM)")
    log(f"    2. Use non-parametric tests")
    log(f"    3. Handle multicollinearity if VIF > 10")
    log(f"    4. Use TimeSeriesSplit for CV")
    log(f"    5. Apply Class Weight for imbalanced classification")
    
    return validation_df

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 07: ASSUMPTION CHECKING")
    log("=" * 70)
    
    # 7.1
    stage_7_1_strategy()
    
    # 7.2
    linearity_df = stage_7_2_linearity(train_df)
    
    # 7.3
    normality_df = stage_7_3_normality(train_df)
    
    # 7.4
    independence_res = stage_7_4_independence(train_df)
    
    # 7.5
    homosc_df = stage_7_5_homoscedasticity(train_df)
    
    # 7.6
    vif_df = stage_7_6_multicollinearity(train_df)
    
    # 7.7
    stationarity_df = stage_7_7_stationarity(train_df)
    
    # 7.8
    dist_res = stage_7_8_distribution(train_df)
    
    # 7.9
    hyp_df = stage_7_9_hypothesis_testing(train_df)
    
    # 7.10
    validation_df = stage_7_10_validation(
        linearity_df, normality_df, independence_res,
        homosc_df, vif_df, stationarity_df, hyp_df
    )
    
    log("\n" + "=" * 70)
    log("PHASE 07 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase07_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()