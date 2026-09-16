"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 13: Hyperparameters (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Anomaly Detection (main successful task)
Strategy: Random Search + Hyperband
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from pathlib import Path
from datetime import datetime
import time
import json
import random

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
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_ANOMALY = "Anomaly_Label"
LEAKY_CLF = ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]

# برای تکرارپذیری
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# 13.1 HYPERPARAMETER INITIALIZATION
# ============================================================
def stage_13_1_initialization():
    log("=" * 70)
    log("PHASE 13: HYPERPARAMETERS")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("STAGE 13.1: HYPERPARAMETER INITIALIZATION")
    log("=" * 70)
    
    baseline = {
        "learning_rate": 0.001,
        "batch_size": 256,
        "l2": 1e-5,
        "dropout_1": 0.2,
        "dropout_2": 0.2,
        "dropout_3": 0.1,
        "hidden_1": 128,
        "hidden_2": 64,
        "hidden_3": 32,
        "patience": 15,
        "optimizer": "RMSprop",
    }
    
    log(f"  Baseline Hyperparameters (from Phase 12):")
    for k, v in baseline.items():
        log(f"    {k}: {v}")
    
    return baseline

# ============================================================
# 13.2 HYPERPARAMETER SPACE DEFINITION
# ============================================================
def stage_13_2_space_definition():
    log("\n" + "=" * 70)
    log("STAGE 13.2: HYPERPARAMETER SPACE DEFINITION")
    log("=" * 70)
    
    space = {
        "learning_rate": {
            "type": "log_uniform",
            "min": 1e-5,
            "max": 1e-2,
            "baseline": 0.001,
        },
        "batch_size": {
            "type": "choice",
            "values": [64, 128, 256, 512],
            "baseline": 256,
        },
        "l2": {
            "type": "log_uniform",
            "min": 1e-7,
            "max": 1e-3,
            "baseline": 1e-5,
        },
        "dropout_1": {
            "type": "uniform",
            "min": 0.1,
            "max": 0.5,
            "baseline": 0.2,
        },
        "dropout_2": {
            "type": "uniform",
            "min": 0.1,
            "max": 0.5,
            "baseline": 0.2,
        },
        "dropout_3": {
            "type": "uniform",
            "min": 0.0,
            "max": 0.4,
            "baseline": 0.1,
        },
        "hidden_1": {
            "type": "choice",
            "values": [64, 128, 256],
            "baseline": 128,
        },
        "hidden_2": {
            "type": "choice",
            "values": [32, 64, 128],
            "baseline": 64,
        },
        "hidden_3": {
            "type": "choice",
            "values": [16, 32, 64],
            "baseline": 32,
        },
        "patience": {
            "type": "choice",
            "values": [10, 15, 20],
            "baseline": 15,
        },
    }
    
    log(f"  Hyperparameter Space:")
    for param, config in space.items():
        log(f"    {param}:")
        for k, v in config.items():
            log(f"      {k}: {v}")
    
    return space

# ============================================================
# 13.3 HYPERPARAMETER SEARCH STRATEGY
# ============================================================
def stage_13_3_search_strategy():
    log("\n" + "=" * 70)
    log("STAGE 13.3: HYPERPARAMETER SEARCH STRATEGY")
    log("=" * 70)
    
    strategies = {
        "Grid Search": {
            "Description": "Exhaustive search over all combinations",
            "Pros": "Complete, reproducible",
            "Cons": "Exponential time, slow",
            "Selected": False,
        },
        "Random Search": {
            "Description": "Random sampling from space",
            "Pros": "Faster, good for high dimensions",
            "Cons": "May miss optimal",
            "Selected": True,
        },
        "Bayesian Optimization": {
            "Description": "Model-based optimization",
            "Pros": "Sample efficient",
            "Cons": "Complex, needs library",
            "Selected": False,
        },
        "Hyperband": {
            "Description": "Early stopping + random search",
            "Pros": "Very fast, efficient",
            "Cons": "May miss fine-tuned",
            "Selected": "As Alternative",
        },
    }
    
    for name, config in strategies.items():
        log(f"  {name}:")
        for k, v in config.items():
            log(f"    {k}: {v}")
    
    log(f"\n  [SELECTED] Random Search (main) + Hyperband (alternative)")
    log(f"  Rationale: Best balance of speed and quality")
    
    return strategies

# ============================================================
# SAMPLE HYPERPARAMETERS
# ============================================================
def sample_hyperparameters(space):
    """نمونه‌برداری تصادفی از فضای Hyperparameter"""
    params = {}
    
    for param, config in space.items():
        if config["type"] == "log_uniform":
            log_min = np.log10(config["min"])
            log_max = np.log10(config["max"])
            value = 10 ** np.random.uniform(log_min, log_max)
            params[param] = value
        elif config["type"] == "uniform":
            params[param] = np.random.uniform(config["min"], config["max"])
        elif config["type"] == "choice":
            params[param] = np.random.choice(config["values"])
        elif config["type"] == "int":
            params[param] = np.random.randint(config["min"], config["max"] + 1)
    
    return params

# ============================================================
# BUILD MODEL WITH HYPERPARAMETERS
# ============================================================
def build_model_with_params(n_features, params):
    """ساخت مدل با Hyperparameterهای داده‌شده"""
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(params["hidden_1"], activation="relu",
                     kernel_regularizer=regularizers.l2(params["l2"])),
        layers.BatchNormalization(),
        layers.Dropout(params["dropout_1"]),
        layers.Dense(params["hidden_2"], activation="relu",
                     kernel_regularizer=regularizers.l2(params["l2"])),
        layers.BatchNormalization(),
        layers.Dropout(params["dropout_2"]),
        layers.Dense(params["hidden_3"], activation="relu",
                     kernel_regularizer=regularizers.l2(params["l2"])),
        layers.BatchNormalization(),
        layers.Dropout(params["dropout_3"]),
        layers.Dense(1, activation="sigmoid"),
    ])
    
    optimizer = optimizers.RMSprop(learning_rate=params["learning_rate"])
    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# ============================================================
# 13.4-13.5 RANDOM SEARCH
# ============================================================
def run_random_search(X_train, y_train, X_val, y_val, space, class_weight, n_iter=20):
    log("\n" + "=" * 70)
    log(f"STAGE 13.4-13.5: RANDOM SEARCH ({n_iter} iterations)")
    log("=" * 70)
    
    results = []
    n_features = X_train.shape[1]
    
    for i in range(n_iter):
        log(f"\n  Iteration {i+1}/{n_iter}")
        
        params = sample_hyperparameters(space)
        log(f"    Params: LR={params['learning_rate']:.6f}, "
            f"Batch={params['batch_size']}, L2={params['l2']:.2e}")
        log(f"    Dropout={params['dropout_1']:.2f}/{params['dropout_2']:.2f}/{params['dropout_3']:.2f}, "
            f"Hidden={params['hidden_1']}/{params['hidden_2']}/{params['hidden_3']}")
        
        start = time.time()
        
        try:
            model = build_model_with_params(n_features, params)
            
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=params["patience"],
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=0,
                ),
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=params["batch_size"],
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=0,
            )
            
            train_time = time.time() - start
            epochs_run = len(history.history["loss"])
            best_val_loss = min(history.history["val_loss"])
            best_epoch = history.history["val_loss"].index(best_val_loss) + 1
            best_val_acc = history.history["val_accuracy"][history.history["val_loss"].index(best_val_loss)]
            
            result = {
                "Iteration": i + 1,
                "Learning_Rate": round(params["learning_rate"], 6),
                "Batch_Size": params["batch_size"],
                "L2": params["l2"],
                "Dropout_1": round(params["dropout_1"], 3),
                "Dropout_2": round(params["dropout_2"], 3),
                "Dropout_3": round(params["dropout_3"], 3),
                "Hidden_1": params["hidden_1"],
                "Hidden_2": params["hidden_2"],
                "Hidden_3": params["hidden_3"],
                "Patience": params["patience"],
                "Epochs_Run": epochs_run,
                "Best_Epoch": best_epoch,
                "Best_Val_Loss": round(best_val_loss, 6),
                "Best_Val_Acc": round(best_val_acc, 6),
                "Train_Time_s": round(train_time, 2),
            }
            
            results.append(result)
            
            log(f"    Best Val Loss: {best_val_loss:.6f} | Val Acc: {best_val_acc:.6f}")
            log(f"    Epochs: {epochs_run} (best: {best_epoch}) | Time: {train_time:.2f}s")
        
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    return pd.DataFrame(results)

# ============================================================
# 13.6 BAYESIAN OPTIMIZATION (SIMULATED)
# ============================================================
def stage_13_6_bayesian(results_df):
    log("\n" + "=" * 70)
    log("STAGE 13.6: BAYESIAN OPTIMIZATION (Reference)")
    log("=" * 70)
    
    log(f"  [INFO] Bayesian Optimization would use Gaussian Process")
    log(f"  [INFO] Library: scikit-optimize or optuna")
    log(f"  [INFO] Not implemented (Random Search results are sufficient)")
    
    if not results_df.empty:
        best = results_df.sort_values("Best_Val_Loss").iloc[0]
        log(f"\n  Best from Random Search:")
        log(f"    Iteration: {best['Iteration']}")
        log(f"    Best Val Loss: {best['Best_Val_Loss']}")
        log(f"    Val Acc: {best['Best_Val_Acc']}")
    
    return {}

# ============================================================
# 13.7 HYPERBAND (SIMULATED)
# ============================================================
def stage_13_7_hyperband():
    log("\n" + "=" * 70)
    log("STAGE 13.7: HYPERBAND (Reference)")
    log("=" * 70)
    
    log(f"  [INFO] Hyperband combines Random Search with Early Stopping")
    log(f"  [INFO] Successive Halving: Train many, keep best half")
    log(f"  [INFO] Not implemented (EarlyStopping already used)")
    
    return {}

# ============================================================
# 13.8 NEURAL ARCHITECTURE SEARCH (NAS)
# ============================================================
def stage_13_8_nas():
    log("\n" + "=" * 70)
    log("STAGE 13.8: NEURAL ARCHITECTURE SEARCH (NAS)")
    log("=" * 70)
    
    log(f"  [INFO] NAS searches for optimal architecture automatically")
    log(f"  [INFO] Methods: Reinforcement Learning, Evolutionary, Gradient-based")
    log(f"  [INFO] Libraries: AutoKeras, Keras Tuner")
    log(f"  [INFO] Not implemented (manual search is sufficient)")
    
    return {}

# ============================================================
# 13.9 HYPERPARAMETER VALIDATION
# ============================================================
def stage_13_9_validation(results_df):
    log("\n" + "=" * 70)
    log("STAGE 13.9: HYPERPARAMETER VALIDATION")
    log("=" * 70)
    
    if results_df.empty:
        log(f"  [WARNING] No results to validate")
        return {}
    
    # بهترین نتیجه
    best = results_df.sort_values("Best_Val_Loss").iloc[0]
    
    log(f"  Best Configuration:")
    log(f"    Iteration: {best['Iteration']}")
    log(f"    Learning Rate: {best['Learning_Rate']}")
    log(f"    Batch Size: {best['Batch_Size']}")
    log(f"    L2: {best['L2']}")
    log(f"    Dropout: {best['Dropout_1']}/{best['Dropout_2']}/{best['Dropout_3']}")
    log(f"    Hidden: {best['Hidden_1']}/{best['Hidden_2']}/{best['Hidden_3']}")
    log(f"    Patience: {best['Patience']}")
    log(f"    Best Val Loss: {best['Best_Val_Loss']}")
    log(f"    Best Val Acc: {best['Best_Val_Acc']}")
    
    # تحلیل حساسیت
    log(f"\n  Sensitivity Analysis:")
    log(f"    Learning Rate range: {results_df['Learning_Rate'].min():.6f} - {results_df['Learning_Rate'].max():.6f}")
    log(f"    Val Loss range: {results_df['Best_Val_Loss'].min():.6f} - {results_df['Best_Val_Loss'].max():.6f}")
    log(f"    Val Acc range: {results_df['Best_Val_Acc'].min():.6f} - {results_df['Best_Val_Acc'].max():.6f}")
    
    return best.to_dict()

# ============================================================
# 13.10 HYPERPARAMETER FINALIZATION
# ============================================================
def stage_13_10_finalization(best_config):
    log("\n" + "=" * 70)
    log("STAGE 13.10: HYPERPARAMETER FINALIZATION")
    log("=" * 70)
    
    log(f"  Final Hyperparameters:")
    for k, v in best_config.items():
        log(f"    {k}: {v}")
    
    # ذخیره
    with open(MODELS_DIR / "best_hyperparameters.json", "w") as f:
        json.dump(best_config, f, indent=2, default=str)
    log(f"  [OK] Saved: best_hyperparameters.json")
    
    return best_config

# ============================================================
# VISUALIZE RESULTS
# ============================================================
def visualize_results(results_df):
    log("\n" + "=" * 70)
    log("VISUALIZING HYPERPARAMETER SEARCH RESULTS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Val Loss by Iteration
    axes[0, 0].plot(results_df["Iteration"], results_df["Best_Val_Loss"], "o-", color="steelblue")
    axes[0, 0].axhline(y=results_df["Best_Val_Loss"].min(), color="red", linestyle="--", label="Best")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("Best Val Loss")
    axes[0, 0].set_title("Val Loss by Iteration")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Learning Rate vs Val Loss
    axes[0, 1].scatter(results_df["Learning_Rate"], results_df["Best_Val_Loss"], 
                       c="coral", s=60, alpha=0.7)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Learning Rate (log)")
    axes[0, 1].set_ylabel("Best Val Loss")
    axes[0, 1].set_title("Learning Rate vs Val Loss")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Batch Size vs Val Loss
    batch_loss = results_df.groupby("Batch_Size")["Best_Val_Loss"].mean()
    axes[0, 2].bar(batch_loss.index.astype(str), batch_loss.values, color="green")
    axes[0, 2].set_xlabel("Batch Size")
    axes[0, 2].set_ylabel("Mean Val Loss")
    axes[0, 2].set_title("Batch Size vs Val Loss")
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. L2 vs Val Loss
    axes[1, 0].scatter(results_df["L2"], results_df["Best_Val_Loss"], 
                       c="purple", s=60, alpha=0.7)
    axes[1, 0].set_xscale("log")
    axes[1, 0].set_xlabel("L2 (log)")
    axes[1, 0].set_ylabel("Best Val Loss")
    axes[1, 0].set_title("L2 vs Val Loss")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Hidden Layers vs Val Loss
    hidden_configs = results_df.apply(
        lambda r: f"{r['Hidden_1']}/{r['Hidden_2']}/{r['Hidden_3']}", axis=1
    )
    hidden_loss = results_df.groupby(hidden_configs)["Best_Val_Loss"].mean().sort_values()
    axes[1, 1].barh(hidden_loss.index, hidden_loss.values, color="orange")
    axes[1, 1].set_xlabel("Mean Val Loss")
    axes[1, 1].set_title("Hidden Layers vs Val Loss")
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Val Accuracy by Iteration
    axes[1, 2].plot(results_df["Iteration"], results_df["Best_Val_Acc"], "o-", color="green")
    axes[1, 2].axhline(y=results_df["Best_Val_Acc"].max(), color="red", linestyle="--", label="Best")
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Best Val Acc")
    axes[1, 2].set_title("Val Accuracy by Iteration")
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_hyperparameter_search.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 13_hyperparameter_search.png")
    
    # Feature Importance-like: بهترین Params
    best = results_df.sort_values("Best_Val_Loss").head(5)
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(best))
    width = 0.35
    ax.bar(x - width/2, best["Best_Val_Loss"], width, label="Val Loss", color="steelblue")
    ax2 = ax.twinx()
    ax2.bar(x + width/2, best["Best_Val_Acc"], width, label="Val Acc", color="coral")
    ax.set_xlabel("Top 5 Configurations")
    ax.set_ylabel("Val Loss")
    ax2.set_ylabel("Val Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Iter {i}" for i in best["Iteration"]])
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    ax.set_title("Top 5 Hyperparameter Configurations")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "13_top5_configs.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 13_top5_configs.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 13: HYPERPARAMETERS")
    log("=" * 70)
    
    # 13.1
    baseline = stage_13_1_initialization()
    
    # 13.2
    space = stage_13_2_space_definition()
    
    # 13.3
    strategies = stage_13_3_search_strategy()
    
    # بارگذاری داده
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    
    features = [c for c in feature_df["Feature"].tolist() 
                if c in train_df.columns and c not in LEAKY_CLF]
    
    class_dist = train_df[TARGET_ANOMALY].value_counts()
    imbalance_ratio = class_dist[0] / class_dist[1]
    class_weight = {0: 1.0, 1: imbalance_ratio}
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values.astype(float)
    X_val = val_df[features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values.astype(float)
    
    log(f"\n  Features: {len(features)}")
    log(f"  Train: {X_train.shape}")
    log(f"  Val: {X_val.shape}")
    log(f"  Class Weight: {class_weight}")
    
    # 13.4-13.5
    N_ITER = 15  # تعداد تکرار (برای سرعت)
    results_df = run_random_search(
        X_train, y_train, X_val, y_val, space, class_weight, n_iter=N_ITER
    )
    results_df.to_csv(TABLES_DIR / "13_random_search_results.csv", index=False)
    log(f"\n  [OK] Saved: 13_random_search_results.csv")
    
    # 13.6
    stage_13_6_bayesian(results_df)
    
    # 13.7
    stage_13_7_hyperband()
    
    # 13.8
    stage_13_8_nas()
    
    # 13.9
    best_config = stage_13_9_validation(results_df)
    
    # 13.10
    final_config = stage_13_10_finalization(best_config)
    
    # Visualize
    visualize_results(results_df)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    log("\n" + "=" * 70)
    log("HYPERPARAMETER SEARCH SUMMARY")
    log("=" * 70)
    
    log(f"\n  Top 5 Configurations:")
    top5 = results_df.sort_values("Best_Val_Loss").head(5)
    for _, row in top5.iterrows():
        log(f"    Iter {row['Iteration']}: "
            f"LR={row['Learning_Rate']:.6f}, "
            f"Batch={row['Batch_Size']}, "
            f"L2={row['L2']:.2e}, "
            f"Val Loss={row['Best_Val_Loss']:.6f}, "
            f"Val Acc={row['Best_Val_Acc']:.6f}")
    
    summary = {
        "Search_Method": "Random Search",
        "N_Iterations": N_ITER,
        "Best_Iteration": best_config.get("Iteration"),
        "Best_Learning_Rate": best_config.get("Learning_Rate"),
        "Best_Batch_Size": best_config.get("Batch_Size"),
        "Best_L2": best_config.get("L2"),
        "Best_Dropout_1": best_config.get("Dropout_1"),
        "Best_Dropout_2": best_config.get("Dropout_2"),
        "Best_Dropout_3": best_config.get("Dropout_3"),
        "Best_Hidden_1": best_config.get("Hidden_1"),
        "Best_Hidden_2": best_config.get("Hidden_2"),
        "Best_Hidden_3": best_config.get("Hidden_3"),
        "Best_Patience": best_config.get("Patience"),
        "Best_Val_Loss": best_config.get("Best_Val_Loss"),
        "Best_Val_Acc": best_config.get("Best_Val_Acc"),
    }
    
    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "13_hyperparameter_summary.csv", index=False
    )
    log(f"  [OK] Saved: 13_hyperparameter_summary.csv")
    
    log("\n" + "=" * 70)
    log("PHASE 13 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase13_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()