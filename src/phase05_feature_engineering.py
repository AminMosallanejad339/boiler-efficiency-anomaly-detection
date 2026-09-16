"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 05: Feature Engineering (10 Sub-stages)
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
from sklearn.feature_selection import (
    SelectKBest, f_regression, f_classif,
    mutual_info_regression, mutual_info_classif,
    RFE
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

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
OUTPUT_DIR = BASE_DIR / "data" / "processed"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [OUTPUT_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# تعریف اهداف
# ============================================================
TARGET_REG = "Boiler_Eff_"          # Regression Target
TARGET_CLF = "Anomaly_Label"       # Classification Target

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری
# ============================================================
log("=" * 70)
log("PHASE 05: FEATURE ENGINEERING")
log("=" * 70)

df = pd.read_csv(PROCESSED)
df["Timestamp"] = pd.to_datetime(df["Timestamp"])
log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
log(f"Targets: {TARGET_REG} (Regression) + {TARGET_CLF} (Classification)")

# ستون‌های پایه (بدون Target و bin)
base_cols = [c for c in df.columns if c not in [TARGET_REG, TARGET_CLF, "Timestamp"] 
             and not c.endswith("_bin")]
log(f"Base feature columns: {len(base_cols)}")

# ============================================================
# 5.1 FEATURE ENGINEERING (Strategy)
# ============================================================
def stage_5_1_strategy():
    log("\n" + "=" * 70)
    log("STAGE 5.1: FEATURE ENGINEERING STRATEGY")
    log("=" * 70)
    
    strategy = {
        "Targets": f"{TARGET_REG} (Regression) + {TARGET_CLF} (Classification)",
        "Base Features": len(base_cols),
        "Strategy": "Domain-driven + Statistical feature engineering",
        "Categories": [
            "Temporal Features (from Timestamp)",
            "Domain-Specific Ratios (Steam, Fuel, Air)",
            "Thermodynamic Efficiency Metrics",
            "Rolling Window Statistics",
            "Lag Features (Time Series)",
            "Interaction Features",
        ],
        "Output": "Feature matrix ready for modeling",
    }
    
    for k, v in strategy.items():
        if isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        else:
            log(f"  {k}: {v}")
    
    return strategy

# ============================================================
# 5.2 FEATURE EXTRACTION (Temporal)
# ============================================================
def stage_5_2_feature_extraction(df):
    log("\n" + "=" * 70)
    log("STAGE 5.2: FEATURE EXTRACTION (Temporal)")
    log("=" * 70)
    
    df_feat = df.copy()
    
    # ویژگی‌های زمانی
    df_feat["hour"] = df_feat["Timestamp"].dt.hour
    df_feat["day_of_week"] = df_feat["Timestamp"].dt.dayofweek
    df_feat["day_of_month"] = df_feat["Timestamp"].dt.day
    df_feat["month"] = df_feat["Timestamp"].dt.month
    df_feat["quarter"] = df_feat["Timestamp"].dt.quarter
    df_feat["week_of_year"] = df_feat["Timestamp"].dt.isocalendar().week.astype(int)
    
    # ویژگی‌های سیکلیک (Cyclic Encoding)
    df_feat["hour_sin"] = np.sin(2 * np.pi * df_feat["hour"] / 24)
    df_feat["hour_cos"] = np.cos(2 * np.pi * df_feat["hour"] / 24)
    df_feat["dow_sin"] = np.sin(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["dow_cos"] = np.cos(2 * np.pi * df_feat["day_of_week"] / 7)
    df_feat["month_sin"] = np.sin(2 * np.pi * df_feat["month"] / 12)
    df_feat["month_cos"] = np.cos(2 * np.pi * df_feat["month"] / 12)
    
    temporal_cols = [
        "hour", "day_of_week", "day_of_month", "month", "quarter", "week_of_year",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"
    ]
    
    log(f"  [OK] Created {len(temporal_cols)} temporal features")
    for col in temporal_cols:
        log(f"    {col}: range [{df_feat[col].min():.4f}, {df_feat[col].max():.4f}]")
    
    return df_feat, temporal_cols

# ============================================================
# 5.3 FEATURE CONSTRUCTION (Domain-Specific)
# ============================================================
def stage_5_3_feature_construction(df):
    log("\n" + "=" * 70)
    log("STAGE 5.3: FEATURE CONSTRUCTION (Domain-Specific)")
    log("=" * 70)
    
    df_feat = df.copy()
    constructed = []
    
    # ۱. نسبت بخار به سوخت
    if "Main_steam_flow_th" in df_feat.columns and "Coal_Flow_th" in df_feat.columns:
        df_feat["steam_to_fuel_ratio"] = df_feat["Main_steam_flow_th"] / (df_feat["Coal_Flow_th"] + 1e-6)
        constructed.append("steam_to_fuel_ratio")
    
    # ۲. نسبت هوا به سوخت
    if "Boiler_oxygen_level_" in df_feat.columns and "Coal_Flow_th" in df_feat.columns:
        df_feat["air_to_fuel_ratio"] = df_feat["Boiler_oxygen_level_"] / (df_feat["Coal_Flow_th"] + 1e-6)
        constructed.append("air_to_fuel_ratio")
    
    # ۳. انحراف Heat Rate
    if "NTHR_KcalKwh" in df_feat.columns and "NPHR_KcalKwh" in df_feat.columns:
        df_feat["heat_rate_deviation"] = df_feat["NTHR_KcalKwh"] - df_feat["NPHR_KcalKwh"]
        constructed.append("heat_rate_deviation")
    
    # ۴. نسبت اثربخشی APH به نشتی
    if "APH_Effectiveness_" in df_feat.columns and "APH_Leakage_" in df_feat.columns:
        df_feat["aph_effect_leak_ratio"] = df_feat["APH_Effectiveness_"] / (df_feat["APH_Leakage_"].abs() + 1e-6)
        constructed.append("aph_effect_leak_ratio")
    
    # ۵. اختلاف دمای گاز دودکش
    if "Flue_Gas_in_Temperature_C" in df_feat.columns and "Corrected_Flue_Gas_Out_Temperature_C" in df_feat.columns:
        df_feat["flue_gas_temp_diff"] = df_feat["Corrected_Flue_Gas_Out_Temperature_C"] - df_feat["Flue_Gas_in_Temperature_C"]
        constructed.append("flue_gas_temp_diff")
    
    # ۶. اختلاف دمای آب تغذیه
    if "Feedwater_temperature_" in df_feat.columns and "Refference_Temperature_C" in df_feat.columns:
        df_feat["feedwater_temp_rise"] = df_feat["Feedwater_temperature_"] - df_feat["Refference_Temperature_C"]
        constructed.append("feedwater_temp_rise")
    
    # ۷. نسبت راندمان توربین به بویلر
    if "HP_Turbine_eff_" in df_feat.columns and TARGET_REG in df_feat.columns:
        df_feat["turbine_to_boiler_eff"] = df_feat["HP_Turbine_eff_"] / (df_feat[TARGET_REG] + 1e-6)
        constructed.append("turbine_to_boiler_eff")
    
    # ۸. نسبت بار خالص به ناخالص
    if "Nett_Load_MW" in df_feat.columns and "Gross_Load_MW" in df_feat.columns:
        df_feat["net_to_gross_load"] = df_feat["Nett_Load_MW"] / (df_feat["Gross_Load_MW"] + 1e-6)
        constructed.append("net_to_gross_load")
    
    # ۹. اختلاف O2 ورودی و خروجی APH
    if "O2_in_APH_" in df_feat.columns and "O2_Out_APH_" in df_feat.columns:
        df_feat["o2_aph_diff"] = df_feat["O2_Out_APH_"] - df_feat["O2_in_APH_"]
        constructed.append("o2_aph_diff")
    
    # ۱۰. نسبت انرژی ورودی به زغال‌سنگ
    if "Energy_Input_From_Boiler_Kcalh" in df_feat.columns and "Coal_Flow_th" in df_feat.columns:
        df_feat["energy_per_coal"] = df_feat["Energy_Input_From_Boiler_Kcalh"] / (df_feat["Coal_Flow_th"] + 1e-6)
        constructed.append("energy_per_coal")
    
    # ۱۱. راندمان ترمودینامیکی
    if "Entalphy_inlet_MS_kjkg" in df_feat.columns and "Entalphy_Cold_Reheat_kjkg" in df_feat.columns:
        df_feat["enthalpy_drop"] = df_feat["Entalphy_inlet_MS_kjkg"] - df_feat["Entalphy_Cold_Reheat_kjkg"]
        constructed.append("enthalpy_drop")
    
    # ۱۲. نسبت ΔP واقعی به ایزنتروپیک
    if "ΔP_aktual_KjKg" in df_feat.columns and "ΔHP_isentropis_KjKg" in df_feat.columns:
        df_feat["deltaP_ratio"] = df_feat["ΔP_aktual_KjKg"] / (df_feat["ΔHP_isentropis_KjKg"].abs() + 1e-6)
        constructed.append("deltaP_ratio")
    
    log(f"  [OK] Created {len(constructed)} domain-specific features:")
    for col in constructed:
        log(f"    {col}: range [{df_feat[col].min():.4f}, {df_feat[col].max():.4f}]")
    
    return df_feat, constructed

# ============================================================
# 5.4 FEATURE TRANSFORMATION
# ============================================================
def stage_5_4_feature_transformation(df, constructed_cols):
    log("\n" + "=" * 70)
    log("STAGE 5.4: FEATURE TRANSFORMATION")
    log("=" * 70)
    
    df_feat = df.copy()
    transformed = []
    
    # بررسی Skewness ویژگی‌های ساخته‌شده
    for col in constructed_cols:
        if col in df_feat.columns:
            sk = df_feat[col].skew()
            if abs(sk) > 2:
                # Log transform برای کاهش Skewness
                if (df_feat[col] > 0).all():
                    df_feat[f"{col}_log"] = np.log1p(df_feat[col])
                    transformed.append(f"{col}_log")
    
    log(f"  [OK] Applied Log transformation to {len(transformed)} features")
    for col in transformed:
        log(f"    {col}")
    
    return df_feat, transformed

# ============================================================
# 5.5 FEATURE SCALING
# ============================================================
def stage_5_5_feature_scaling(df, feature_cols):
    log("\n" + "=" * 70)
    log("STAGE 5.5: FEATURE SCALING")
    log("=" * 70)
    
    # ذخیره Scaler برای استفاده در Pipeline
    scaler = StandardScaler()
    
    # فقط برای نمایش - ذخیره نمی‌کنیم
    log(f"  [INFO] Scaling strategy:")
    log(f"    - StandardScaler for tree-based models (not needed)")
    log(f"    - MinMaxScaler for neural networks")
    log(f"    - RobustScaler for outlier-prone features")
    log(f"  [NOTE] Scaling will be applied in Phase 06 (Pipeline)")
    log(f"  [NOTE] Feature count for scaling: {len(feature_cols)}")
    
    return df

# ============================================================
# 5.6 FEATURE SELECTION
# ============================================================
def stage_5_6_feature_selection(df, feature_cols):
    log("\n" + "=" * 70)
    log("STAGE 5.6: FEATURE SELECTION")
    log("=" * 70)
    
    # حذف Target و ستون‌های غیرعددی
    feature_cols = [c for c in feature_cols if c not in [TARGET_REG, TARGET_CLF, "Timestamp"]
                    and df[c].dtype.kind in "biufc"]
    
    X = df[feature_cols].fillna(0)
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]
    
    log(f"  Features for selection: {len(feature_cols)}")
    
    # 1. SelectKBest برای Regression
    try:
        selector_reg = SelectKBest(score_func=f_regression, k=min(20, len(feature_cols)))
        selector_reg.fit(X, y_reg)
        reg_scores = pd.DataFrame({
            "Feature": feature_cols,
            "F_Score_Reg": selector_reg.scores_,
            "P_Value_Reg": selector_reg.pvalues_,
        }).sort_values("F_Score_Reg", ascending=False)
        reg_scores.to_csv(TABLES_DIR / "05_06_feature_selection_regression.csv", index=False)
        log(f"  [OK] Saved: 05_06_feature_selection_regression.csv")
    except Exception as e:
        log(f"  [WARNING] Regression selection failed: {e}")
        reg_scores = pd.DataFrame()
    
    # 2. SelectKBest برای Classification
    try:
        selector_clf = SelectKBest(score_func=f_classif, k=min(20, len(feature_cols)))
        selector_clf.fit(X, y_clf)
        clf_scores = pd.DataFrame({
            "Feature": feature_cols,
            "F_Score_Clf": selector_clf.scores_,
            "P_Value_Clf": selector_clf.pvalues_,
        }).sort_values("F_Score_Clf", ascending=False)
        clf_scores.to_csv(TABLES_DIR / "05_06_feature_selection_classification.csv", index=False)
        log(f"  [OK] Saved: 05_06_feature_selection_classification.csv")
    except Exception as e:
        log(f"  [WARNING] Classification selection failed: {e}")
        clf_scores = pd.DataFrame()
    
    # 3. Mutual Information
    try:
        mi_reg = mutual_info_regression(X, y_reg, random_state=42)
        mi_clf = mutual_info_classif(X, y_clf, random_state=42)
        mi_df = pd.DataFrame({
            "Feature": feature_cols,
            "MI_Regression": mi_reg,
            "MI_Classification": mi_clf,
        }).sort_values("MI_Classification", ascending=False)
        mi_df.to_csv(TABLES_DIR / "05_06_mutual_information.csv", index=False)
        log(f"  [OK] Saved: 05_06_mutual_information.csv")
    except Exception as e:
        log(f"  [WARNING] Mutual information failed: {e}")
        mi_df = pd.DataFrame()
    
    # نمایش Top 10
    if not clf_scores.empty:
        log(f"\n  Top 10 features for Classification (F-Score):")
        for _, row in clf_scores.head(10).iterrows():
            log(f"    {row['Feature']}: F={row['F_Score_Clf']:.4f}, p={row['P_Value_Clf']:.4e}")
    
    if not reg_scores.empty:
        log(f"\n  Top 10 features for Regression (F-Score):")
        for _, row in reg_scores.head(10).iterrows():
            log(f"    {row['Feature']}: F={row['F_Score_Reg']:.4f}, p={row['P_Value_Reg']:.4e}")
    
    return reg_scores, clf_scores, mi_df

# ============================================================
# 5.7 FEATURE IMPORTANCE ANALYSIS
# ============================================================
def stage_5_7_feature_importance(df, feature_cols):
    log("\n" + "=" * 70)
    log("STAGE 5.7: FEATURE IMPORTANCE ANALYSIS")
    log("=" * 70)
    
    feature_cols = [c for c in feature_cols if c not in [TARGET_REG, TARGET_CLF, "Timestamp"]
                    and df[c].dtype.kind in "biufc"]
    
    X = df[feature_cols].fillna(0)
    y_reg = df[TARGET_REG]
    y_clf = df[TARGET_CLF]
    
    # Random Forest برای Regression
    try:
        rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf_reg.fit(X, y_reg)
        imp_reg = pd.DataFrame({
            "Feature": feature_cols,
            "Importance_Reg": rf_reg.feature_importances_,
        }).sort_values("Importance_Reg", ascending=False)
        imp_reg.to_csv(TABLES_DIR / "05_07_importance_regression.csv", index=False)
        log(f"  [OK] Random Forest Regression trained")
        log(f"  [OK] Saved: 05_07_importance_regression.csv")
        
        log(f"\n  Top 15 features for Regression (RF Importance):")
        for _, row in imp_reg.head(15).iterrows():
            log(f"    {row['Feature']}: {row['Importance_Reg']:.6f}")
    except Exception as e:
        log(f"  [WARNING] RF Regression failed: {e}")
        imp_reg = pd.DataFrame()
    
    # Random Forest برای Classification
    try:
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1,
                                        class_weight="balanced")
        rf_clf.fit(X, y_clf)
        imp_clf = pd.DataFrame({
            "Feature": feature_cols,
            "Importance_Clf": rf_clf.feature_importances_,
        }).sort_values("Importance_Clf", ascending=False)
        imp_clf.to_csv(TABLES_DIR / "05_07_importance_classification.csv", index=False)
        log(f"  [OK] Random Forest Classification trained")
        log(f"  [OK] Saved: 05_07_importance_classification.csv")
        
        log(f"\n  Top 15 features for Classification (RF Importance):")
        for _, row in imp_clf.head(15).iterrows():
            log(f"    {row['Feature']}: {row['Importance_Clf']:.6f}")
    except Exception as e:
        log(f"  [WARNING] RF Classification failed: {e}")
        imp_clf = pd.DataFrame()
    
    # نمودار اهمیت
    if not imp_clf.empty:
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        top15_clf = imp_clf.head(15)
        axes[0].barh(top15_clf["Feature"][::-1], top15_clf["Importance_Clf"][::-1], color="coral")
        axes[0].set_title("Top 15 Features - Classification (Anomaly)", fontsize=12)
        axes[0].set_xlabel("Importance")
        axes[0].grid(True, alpha=0.3)
        
        if not imp_reg.empty:
            top15_reg = imp_reg.head(15)
            axes[1].barh(top15_reg["Feature"][::-1], top15_reg["Importance_Reg"][::-1], color="steelblue")
            axes[1].set_title("Top 15 Features - Regression (Boiler Efficiency)", fontsize=12)
            axes[1].set_xlabel("Importance")
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "05_07_feature_importance.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 05_07_feature_importance.png")
    
    return imp_reg, imp_clf

# ============================================================
# 5.8 DIMENSIONALITY REDUCTION
# ============================================================
def stage_5_8_dimensionality_reduction(df, feature_cols):
    log("\n" + "=" * 70)
    log("STAGE 5.8: DIMENSIONALITY REDUCTION (PCA)")
    log("=" * 70)
    
    feature_cols = [c for c in feature_cols if c not in [TARGET_REG, TARGET_CLF, "Timestamp"]
                    and df[c].dtype.kind in "biufc"]
    
    X = df[feature_cols].fillna(0)
    
    # استانداردسازی
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=min(20, len(feature_cols)))
    X_pca = pca.fit_transform(X_scaled)
    
    # واریانس توضیح‌داده‌شده
    explained_var = pca.explained_variance_ratio_
    cumulative_var = np.cumsum(explained_var)
    
    log(f"  [OK] PCA completed with {len(explained_var)} components")
    log(f"  Explained variance (first 10):")
    for i in range(min(10, len(explained_var))):
        log(f"    PC{i+1}: {explained_var[i]*100:.2f}% (cumulative: {cumulative_var[i]*100:.2f}%)")
    
    # تعداد مؤلفه برای 95% واریانس
    n_95 = np.argmax(cumulative_var >= 0.95) + 1
    log(f"  Components for 95% variance: {n_95}")
    
    # ذخیره نتایج
    pca_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(len(explained_var))],
        "Explained_Variance": explained_var,
        "Cumulative_Variance": cumulative_var,
    })
    pca_df.to_csv(TABLES_DIR / "05_08_pca_variance.csv", index=False)
    log(f"  [OK] Saved: 05_08_pca_variance.csv")
    
    # نمودار
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(range(1, len(explained_var)+1), explained_var * 100, alpha=0.6, label="Individual")
    ax.plot(range(1, len(explained_var)+1), cumulative_var * 100, "ro-", label="Cumulative")
    ax.axhline(y=95, color="green", linestyle="--", label="95% threshold")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA - Explained Variance")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_08_pca_variance.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 05_08_pca_variance.png")
    
    return pca, explained_var

# ============================================================
# 5.9 FEATURE INTERACTION
# ============================================================
def stage_5_9_feature_interaction(df):
    log("\n" + "=" * 70)
    log("STAGE 5.9: FEATURE INTERACTION")
    log("=" * 70)
    
    log(f"  [INFO] Feature interaction strategies:")
    log(f"    - Polynomial features (degree 2) for top features")
    log(f"    - Ratio features between key variables")
    log(f"    - Product features for correlated pairs")
    log(f"  [NOTE] Interactions will be created in Phase 06 (Pipeline)")
    
    return df

# ============================================================
# 5.10 FEATURE ENCODING
# ============================================================
def stage_5_10_feature_encoding(df):
    log("\n" + "=" * 70)
    log("STAGE 5.10: FEATURE ENCODING")
    log("=" * 70)
    
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
    cat_cols = [c for c in cat_cols if c != "Timestamp"]
    
    log(f"  Categorical columns: {len(cat_cols)}")
    
    if cat_cols:
        for col in cat_cols:
            log(f"    {col}: {df[col].nunique()} unique values")
    else:
        log(f"  [OK] No categorical columns to encode")
    
    log(f"  [INFO] Temporal features already encoded as cyclic (sin/cos)")
    
    return df

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 05: FEATURE ENGINEERING")
    log("=" * 70)
    
    df = pd.read_csv(PROCESSED)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    log(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    
    # 5.1
    strategy = stage_5_1_strategy()
    
    # 5.2
    df, temporal_cols = stage_5_2_feature_extraction(df)
    
    # 5.3
    df, constructed_cols = stage_5_3_feature_construction(df)
    
    # 5.4
    df, transformed_cols = stage_5_4_feature_transformation(df, constructed_cols)
    
    # همه ویژگی‌های عددی
    all_feature_cols = [c for c in df.columns if c not in [TARGET_REG, TARGET_CLF, "Timestamp"]
                        and df[c].dtype.kind in "biufc"]
    
    # 5.5
    df = stage_5_5_feature_scaling(df, all_feature_cols)
    
    # 5.6
    reg_scores, clf_scores, mi_df = stage_5_6_feature_selection(df, all_feature_cols)
    
    # 5.7
    imp_reg, imp_clf = stage_5_7_feature_importance(df, all_feature_cols)
    
    # 5.8
    pca, explained_var = stage_5_8_dimensionality_reduction(df, all_feature_cols)
    
    # 5.9
    df = stage_5_9_feature_interaction(df)
    
    # 5.10
    df = stage_5_10_feature_encoding(df)
    
    # ذخیره دیتاست نهایی
    output_path = OUTPUT_DIR / "industrial_dataset_phase05_final.csv"
    df.to_csv(output_path, index=False)
    log("\n" + "=" * 70)
    log(f"[OK] FINAL DATASET SAVED: {output_path}")
    log(f"[OK] Final shape: {df.shape}")
    log("=" * 70)
    
    # ذخیره لاگ
    LOG_FILE = BASE_DIR / "reports" / "phase05_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()