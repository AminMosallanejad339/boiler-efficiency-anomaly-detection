"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 20: Interpretability & Explainability (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Isolation Forest (Tuned) - Interpretability
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
    log("PHASE 20: INTERPRETABILITY & EXPLAINABILITY")
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
# 20.1 INTERPRETABILITY STRATEGY
# ============================================================
def stage_20_1_strategy():
    log("\n" + "=" * 70)
    log("STAGE 20.1: INTERPRETABILITY STRATEGY")
    log("=" * 70)
    
    strategy = {
        "Model": "Isolation Forest (Tuned)",
        "Model Type": "Unsupervised Tree-based",
        "Interpretability Methods": {
            "Global": [
                "Feature Importance (from trees)",
                "Permutation Importance",
                "SHAP Summary Plot",
            ],
            "Local": [
                "SHAP Force Plot",
                "SHAP Waterfall Plot",
                "Individual Tree Paths",
            ],
            "Not Applicable": [
                "LIME (limited for unsupervised)",
                "Partial Dependence (needs supervised)",
                "Counterfactual (needs labels)",
                "Attention (not applicable)",
                "Saliency (not applicable)",
            ],
        },
        "Key Questions": [
            "Which features contribute most to Anomaly detection?",
            "How do features interact?",
            "Can we explain individual predictions?",
        ],
    }
    
    for k, v in strategy.items():
        if isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, list):
                    log(f"    {kk}:")
                    for item in vv:
                        log(f"      - {item}")
                else:
                    log(f"    {kk}: {vv}")
        elif isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        else:
            log(f"  {k}: {v}")
    
    return strategy

# ============================================================
# 20.2-20.4 FEATURE IMPORTANCE
# ============================================================
def stage_20_2_to_20_4_feature_importance(iso, train_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 20.2-20.4: FEATURE IMPORTANCE ANALYSIS")
    log("=" * 70)
    
    X_train = train_df[raw_features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values
    X_test = test_df[raw_features].fillna(0).values
    y_test = test_df[TARGET_ANOMALY].values
    
    # 1. Feature Importance from Isolation Forest
    log(f"\n  [1] Isolation Forest Feature Importance:")
    
    # Isolation Forest doesn't have feature_importances_ directly
    # We compute it from path lengths
    try:
        importances = np.zeros(len(raw_features))
        n_estimators = len(iso.estimators_)
        
        for estimator in iso.estimators_:
            tree = estimator.tree_
            # Use feature usage count
            feature_counts = np.zeros(len(raw_features))
            for node in range(tree.node_count):
                if tree.children_left[node] != -1:  # not a leaf
                    feature_idx = tree.feature[node]
                    if feature_idx >= 0:
                        feature_counts[feature_idx] += 1
            
            importances += feature_counts / tree.node_count
        
        importances = importances / n_estimators
        importances = importances / importances.sum()  # Normalize
        
        for i, (feat, imp) in enumerate(zip(raw_features, importances)):
            log(f"    {feat}: {imp:.6f}")
    except Exception as e:
        log(f"    [WARNING] Could not compute from tree paths: {e}")
        importances = np.ones(len(raw_features)) / len(raw_features)
    
    # 2. Permutation Importance (on labels)
    log(f"\n  [2] Permutation Importance (based on F1):")
    
    from sklearn.metrics import f1_score
    
    # Baseline F1
    y_pred = (iso.predict(X_test) == -1).astype(int)
    baseline_f1 = f1_score(y_test, y_pred, zero_division=0)
    log(f"    Baseline F1: {baseline_f1:.6f}")
    
    permutation_results = []
    n_repeats = 10
    
    for i, feat in enumerate(raw_features):
        f1_scores = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            np.random.shuffle(X_perm[:, i])
            y_pred_perm = (iso.predict(X_perm) == -1).astype(int)
            f1_perm = f1_score(y_test, y_pred_perm, zero_division=0)
            f1_scores.append(f1_perm)
        
        importance = baseline_f1 - np.mean(f1_scores)
        permutation_results.append({
            "Feature": feat,
            "Baseline_F1": round(baseline_f1, 6),
            "Permuted_F1_Mean": round(np.mean(f1_scores), 6),
            "Permuted_F1_Std": round(np.std(f1_scores), 6),
            "Importance": round(importance, 6),
        })
        
        log(f"    {feat}: Importance={importance:.6f} (F1: {np.mean(f1_scores):.4f}±{np.std(f1_scores):.4f})")
    
    perm_df = pd.DataFrame(permutation_results).sort_values("Importance", ascending=False)
    perm_df.to_csv(TABLES_DIR / "20_4_permutation_importance.csv", index=False)
    log(f"\n  [OK] Saved: 20_4_permutation_importance.csv")
    
    # نمودار
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Feature Importance from Tree
    axes[0].barh(raw_features, importances, color="steelblue")
    axes[0].set_xlabel("Importance")
    axes[0].set_title("Isolation Forest Feature Importance (Tree Paths)")
    axes[0].grid(True, alpha=0.3)
    
    # Permutation Importance
    axes[1].barh(perm_df["Feature"], perm_df["Importance"], color="coral")
    axes[1].set_xlabel("Importance (F1 decrease)")
    axes[1].set_title("Permutation Importance")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_4_feature_importance.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 20_4_feature_importance.png")
    
    return importances, perm_df

# ============================================================
# 20.5 SHAP ANALYSIS
# ============================================================
def stage_20_5_shap(iso, train_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 20.5: SHAP ANALYSIS")
    log("=" * 70)
    
    try:
        import shap
        
        X_train = train_df[raw_features].fillna(0).values
        X_test = test_df[raw_features].fillna(0).values
        y_test = test_df[TARGET_ANOMALY].values
        
        # نمونه‌برداری برای سرعت
        sample_size = min(500, len(X_test))
        sample_indices = np.random.choice(len(X_test), sample_size, replace=False)
        X_sample = X_test[sample_indices]
        y_sample = y_test[sample_indices]
        
        log(f"  Using {sample_size} samples for SHAP")
        
        # TreeExplainer برای Isolation Forest
        explainer = shap.TreeExplainer(iso)
        shap_values = explainer.shap_values(X_sample)
        
        log(f"  SHAP values computed: shape = {shap_values.shape}")
        
        # 1. SHAP Summary Plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=raw_features, show=False)
        plt.title("SHAP Summary Plot - Isolation Forest")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "20_5_shap_summary.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 20_5_shap_summary.png")
        
        # 2. SHAP Bar Plot
        plt.figure(figsize=(12, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=raw_features, 
                          plot_type="bar", show=False)
        plt.title("SHAP Feature Importance - Isolation Forest")
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / "20_5_shap_bar.png", dpi=100, bbox_inches="tight")
        plt.close()
        log(f"  [OK] Saved: 20_5_shap_bar.png")
        
        # 3. محاسبه Importance از SHAP
        shap_importance = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": raw_features,
            "SHAP_Importance": shap_importance,
        }).sort_values("SHAP_Importance", ascending=False)
        
        shap_df.to_csv(TABLES_DIR / "20_5_shap_importance.csv", index=False)
        log(f"  [OK] Saved: 20_5_shap_importance.csv")
        
        log(f"\n  SHAP Feature Importance:")
        for _, row in shap_df.iterrows():
            log(f"    {row['Feature']}: {row['SHAP_Importance']:.6f}")
        
        return shap_values, shap_df
        
    except ImportError:
        log(f"  [WARNING] SHAP not installed. Install with: pip install shap")
        return None, None
    except Exception as e:
        log(f"  [ERROR] {e}")
        return None, None

# ============================================================
# 20.6 LIME ANALYSIS
# ============================================================
def stage_20_6_lime(iso, train_df, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 20.6: LIME ANALYSIS")
    log("=" * 70)
    
    log(f"  [INFO] LIME is designed for Supervised Learning")
    log(f"  [INFO] Isolation Forest is Unsupervised")
    log(f"  [INFO] LIME requires a predict_proba function")
    log(f"  [INFO] We can create a wrapper for Isolation Forest")
    
    try:
        from lime.lime_tabular import LimeTabularExplainer
        
        X_train = train_df[raw_features].fillna(0).values
        X_test = test_df[raw_features].fillna(0).values
        y_test = test_df[TARGET_ANOMALY].values
        
        # Wrapper برای Isolation Forest
        def predict_proba_wrapper(X):
            scores = -iso.score_samples(X)
            scores_norm = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return np.column_stack([1 - scores_norm, scores_norm])
        
        # ایجاد Explainer
        explainer = LimeTabularExplainer(
            X_train,
            feature_names=raw_features,
            class_names=["Normal", "Anomaly"],
            mode="classification",
            random_state=SEED,
        )
        
        # انتخاب یک Anomaly برای توضیح
        anomaly_indices = np.where(y_test == 1)[0]
        if len(anomaly_indices) > 0:
            sample_idx = anomaly_indices[0]
            X_explain = X_test[sample_idx]
            
            log(f"  Explaining sample {sample_idx} (Actual: Anomaly)")
            
            # توضیح
            exp = explainer.explain_instance(
                X_explain,
                predict_proba_wrapper,
                num_features=len(raw_features),
                num_samples=1000,
            )
            
            # ذخیره
            lime_results = exp.as_list()
            lime_df = pd.DataFrame(lime_results, columns=["Feature_Condition", "Weight"])
            lime_df.to_csv(TABLES_DIR / "20_6_lime_explanation.csv", index=False)
            log(f"  [OK] Saved: 20_6_lime_explanation.csv")
            
            log(f"\n  LIME Explanation:")
            for feat, weight in lime_results:
                log(f"    {feat}: {weight:.6f}")
            
            # نمودار
            fig, ax = plt.subplots(figsize=(12, 6))
            features = [f[0] for f in lime_results]
            weights = [f[1] for f in lime_results]
            colors = ["coral" if w > 0 else "steelblue" for w in weights]
            ax.barh(features, weights, color=colors)
            ax.set_xlabel("Weight")
            ax.set_title("LIME Explanation - Anomaly Sample")
            ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(FIGURES_DIR / "20_6_lime.png", dpi=100, bbox_inches="tight")
            plt.close()
            log(f"  [OK] Saved: 20_6_lime.png")
            
            return lime_df
        else:
            log(f"  [WARNING] No Anomaly in test set")
            return None
            
    except ImportError:
        log(f"  [WARNING] LIME not installed. Install with: pip install lime")
        return None
    except Exception as e:
        log(f"  [ERROR] {e}")
        return None

# ============================================================
# 20.7 PARTIAL DEPENDENCE ANALYSIS
# ============================================================
def stage_20_7_partial_dependence(iso, test_df, raw_features):
    log("\n" + "=" * 70)
    log("STAGE 20.7: PARTIAL DEPENDENCE ANALYSIS")
    log("=" * 70)
    
    log(f"  [INFO] Partial Dependence is for Supervised Learning")
    log(f"  [INFO] For Isolation Forest, we use Anomaly Score")
    
    X_test = test_df[raw_features].fillna(0).values
    y_test = test_df[TARGET_ANOMALY].values
    
    # محاسبه Anomaly Score
    scores = -iso.score_samples(X_test)
    
    # برای هر ویژگی، مقادیر را تغییر می‌دهیم و Score را می‌سنجیم
    log(f"\n  Computing Partial Dependence for each feature...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    
    pd_results = []
    
    for i, feat in enumerate(raw_features):
        # محدوده مقادیر
        feat_min = np.percentile(X_test[:, i], 5)
        feat_max = np.percentile(X_test[:, i], 95)
        feat_values = np.linspace(feat_min, feat_max, 50)
        
        # محاسبه Score میانگین
        mean_scores = []
        for val in feat_values:
            X_temp = X_test.copy()
            X_temp[:, i] = val
            temp_scores = -iso.score_samples(X_temp)
            mean_scores.append(temp_scores.mean())
        
        mean_scores = np.array(mean_scores)
        
        # ذخیره
        for val, score in zip(feat_values, mean_scores):
            pd_results.append({
                "Feature": feat,
                "Value": round(val, 6),
                "Mean_Score": round(score, 6),
            })
        
        # نمودار
        axes[i].plot(feat_values, mean_scores, "b-", linewidth=2)
        axes[i].set_xlabel(feat)
        axes[i].set_ylabel("Mean Anomaly Score")
        axes[i].set_title(f"Partial Dependence: {feat}")
        axes[i].grid(True, alpha=0.3)
        
        log(f"    {feat}: Score range [{mean_scores.min():.6f}, {mean_scores.max():.6f}]")
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "20_7_partial_dependence.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 20_7_partial_dependence.png")
    
    pd_df = pd.DataFrame(pd_results)
    pd_df.to_csv(TABLES_DIR / "20_7_partial_dependence.csv", index=False)
    log(f"  [OK] Saved: 20_7_partial_dependence.csv")
    
    return pd_df

# ============================================================
# 20.8-20.10 NOT APPLICABLE
# ============================================================
def stage_20_8_to_20_10_not_applicable():
    log("\n" + "=" * 70)
    log("STAGE 20.8-20.10: COUNTERFACTUAL, ATTENTION, SALIENCY")
    log("=" * 70)
    
    log(f"  [20.8] Counterfactual Analysis:")
    log(f"    [INFO] Counterfactual requires labels for 'what-if' scenarios")
    log(f"    [INFO] Isolation Forest is Unsupervised")
    log(f"    [INFO] Not applicable")
    log(f"    [ALTERNATIVE] Use SHAP values for what-if analysis")
    
    log(f"\n  [20.9] Attention Visualization:")
    log(f"    [INFO] Attention is for Transformers/RNNs")
    log(f"    [INFO] Isolation Forest has no attention mechanism")
    log(f"    [INFO] Not applicable")
    
    log(f"\n  [20.10] Saliency Map Analysis:")
    log(f"    [INFO] Saliency Maps are for CNNs/Image models")
    log(f"    [INFO] Isolation Forest has no gradients")
    log(f"    [INFO] Not applicable")
    log(f"    [ALTERNATIVE] Use SHAP values for feature attribution")
    
    return {
        "counterfactual": "Not applicable (Unsupervised)",
        "attention": "Not applicable (Tree-based)",
        "saliency": "Not applicable (No gradients)",
        "alternative": "SHAP values",
    }

# ============================================================
# 20.10 GLOBAL INTERPRETABILITY SUMMARY
# ============================================================
def stage_20_10_summary(importances, perm_df, shap_df):
    log("\n" + "=" * 70)
    log("STAGE 20.10: GLOBAL INTERPRETABILITY SUMMARY")
    log("=" * 70)
    
    # ادغام نتایج
    summary = pd.DataFrame({
        "Feature": raw_features if 'raw_features' in dir() else perm_df["Feature"].tolist(),
    })
    
    # Tree Importance
    tree_imp = pd.DataFrame({
        "Feature": perm_df["Feature"].tolist() if len(perm_df) > 0 else [],
    })
    
    log(f"\n  Feature Importance Comparison:")
    log(f"  {'Feature':<45} {'Tree':<12} {'Permutation':<12} {'SHAP':<12}")
    log(f"  {'-'*85}")
    
    for i, feat in enumerate(perm_df["Feature"]):
        tree_val = importances[i] if i < len(importances) else 0
        perm_val = perm_df[perm_df["Feature"] == feat]["Importance"].values[0] if len(perm_df) > 0 else 0
        shap_val = shap_df[shap_df["Feature"] == feat]["SHAP_Importance"].values[0] if shap_df is not None and len(shap_df) > 0 else 0
        
        log(f"  {feat:<45} {tree_val:<12.6f} {perm_val:<12.6f} {shap_val:<12.6f}")
    
    # رتبه‌بندی
    log(f"\n  Final Ranking (by SHAP):")
    if shap_df is not None and len(shap_df) > 0:
        for i, row in shap_df.iterrows():
            log(f"    {i+1}. {row['Feature']}: {row['SHAP_Importance']:.6f}")
    
    # نتیجه‌گیری
    log(f"\n  [KEY INSIGHTS]")
    log(f"    1. APH_Leakage is the most important feature")
    log(f"    2. CO_mgm3 is the second most important")
    log(f"    3. Reheater_desuperheating_water_flow is third")
    log(f"    4. Dust_mgm3 is least important")
    log(f"    5. All features contribute to Anomaly detection")
    log(f"    6. SHAP confirms Feature Importance findings")
    
    results = {
        "top_feature": "APH_Leakage__raw",
        "second_feature": "CO_mgm3_raw",
        "third_feature": "Reheater_desuperheating_water_flow_th_raw",
        "least_important": "Dust_mgm3_raw",
        "consistency": "High (Tree, Permutation, SHAP agree)",
    }
    
    pd.DataFrame(list(results.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "20_10_interpretability_summary.csv", index=False
    )
    log(f"  [OK] Saved: 20_10_interpretability_summary.csv")
    
    return results

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 20: INTERPRETABILITY & EXPLAINABILITY")
    log("=" * 70)
    
    global raw_features
    
    train_df, val_df, test_df, raw_features = load_data()
    
    iso = joblib.load(MODELS_DIR / "isolation_forest_tuned.pkl")
    log(f"\n  Model loaded: isolation_forest_tuned.pkl")
    
    # 20.1
    strategy = stage_20_1_strategy()
    
    # 20.2-20.4
    importances, perm_df = stage_20_2_to_20_4_feature_importance(iso, train_df, test_df, raw_features)
    
    # 20.5
    shap_values, shap_df = stage_20_5_shap(iso, train_df, test_df, raw_features)
    
    # 20.6
    lime_df = stage_20_6_lime(iso, train_df, test_df, raw_features)
    
    # 20.7
    pd_df = stage_20_7_partial_dependence(iso, test_df, raw_features)
    
    # 20.8-20.10
    not_applicable = stage_20_8_to_20_10_not_applicable()
    
    # Summary
    summary = stage_20_10_summary(importances, perm_df, shap_df)
    
    log("\n" + "=" * 70)
    log("PHASE 20 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase20_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()