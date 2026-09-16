"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 12: Regularization (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Anomaly Detection (main successful task)
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

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# 12.1 REGULARIZATION SELECTION
# ============================================================
def stage_12_1_selection():
    log("=" * 70)
    log("PHASE 12: REGULARIZATION")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("STAGE 12.1: REGULARIZATION SELECTION")
    log("=" * 70)
    
    techniques = {
        "L1 Regularization": {
            "Formula": "R(W) = λ * sum(|W|)",
            "Effect": "Sparse weights, feature selection",
            "Pros": "Automatic feature selection",
            "Cons": "Non-differentiable at 0",
        },
        "L2 Regularization": {
            "Formula": "R(W) = λ * sum(W^2)",
            "Effect": "Small weights, smooth",
            "Pros": "Differentiable, stable",
            "Cons": "No feature selection",
        },
        "Elastic Net": {
            "Formula": "R(W) = λ1*sum(|W|) + λ2*sum(W^2)",
            "Effect": "Combines L1 and L2",
            "Pros": "Best of both",
            "Cons": "More hyperparameters",
        },
        "Dropout": {
            "Formula": "Randomly zero activations during training",
            "Effect": "Prevents co-adaptation",
            "Pros": "Effective, simple",
            "Cons": "Slower training",
        },
        "Early Stopping": {
            "Formula": "Stop when val_loss stops improving",
            "Effect": "Prevents overfitting",
            "Pros": "Simple, effective",
            "Cons": "Needs validation set",
        },
        "Weight Decay": {
            "Formula": "θ = θ - lr*(gradient + λ*θ)",
            "Effect": "Shrinks weights",
            "Pros": "Simple",
            "Cons": "Similar to L2",
        },
        "Label Smoothing": {
            "Formula": "y_smooth = (1-ε)*y + ε/K",
            "Effect": "Soft labels",
            "Pros": "Better calibration",
            "Cons": "For classification only",
        },
        "Data Augmentation": {
            "Formula": "Generate synthetic samples",
            "Effect": "More training data",
            "Pros": "Better generalization",
            "Cons": "Not for tabular easily",
        },
    }
    
    for tech, config in techniques.items():
        log(f"\n  {tech}:")
        for k, v in config.items():
            log(f"    {k}: {v}")
    
    log(f"\n  [SELECTED] L2 + Dropout + EarlyStopping")
    log(f"  Rationale: Proven combination for tabular MLPs")
    
    return techniques

# ============================================================
# 12.2-12.4 L1, L2, Elastic Net
# ============================================================
def stage_12_2_4_l1_l2_elasticnet():
    log("\n" + "=" * 70)
    log("STAGE 12.2-12.4: L1, L2, ELASTIC NET SETUP")
    log("=" * 70)
    
    config = {
        "L1": {
            "λ": "1e-5 to 1e-3",
            "Selected": "Not used (no feature selection needed)",
        },
        "L2": {
            "λ": "1e-4 (baseline)",
            "Range to Test": [1e-5, 1e-4, 1e-3, 1e-2],
            "Selected": "1e-4 (to be tuned)",
        },
        "Elastic Net": {
            "λ1": "Not used",
            "λ2": "Not used",
            "Rationale": "L2 alone is sufficient",
        },
    }
    
    for method, params in config.items():
        log(f"\n  {method}:")
        for k, v in params.items():
            log(f"    {k}: {v}")
    
    return config

# ============================================================
# 12.5 DROPOUT SETUP
# ============================================================
def stage_12_5_dropout():
    log("\n" + "=" * 70)
    log("STAGE 12.5: DROPOUT SETUP")
    log("=" * 70)
    
    dropout_config = {
        "Baseline": {
            "Layer 1": 0.3,
            "Layer 2": 0.3,
            "Layer 3": 0.2,
        },
        "Low Dropout": {
            "Layer 1": 0.2,
            "Layer 2": 0.2,
            "Layer 3": 0.1,
        },
        "High Dropout": {
            "Layer 1": 0.5,
            "Layer 2": 0.5,
            "Layer 3": 0.3,
        },
        "No Dropout": {
            "Layer 1": 0.0,
            "Layer 2": 0.0,
            "Layer 3": 0.0,
        },
    }
    
    for config_name, rates in dropout_config.items():
        log(f"  {config_name}: {rates}")
    
    return dropout_config

# ============================================================
# 12.6 EARLY STOPPING SETUP
# ============================================================
def stage_12_6_early_stopping():
    log("\n" + "=" * 70)
    log("STAGE 12.6: EARLY STOPPING SETUP")
    log("=" * 70)
    
    config = {
        "Monitor": "val_loss",
        "Patience": [5, 10, 15, 20],
        "Baseline": 10,
        "Restore Best Weights": True,
        "Min Delta": 1e-4,
        "Rationale": [
            "Prevents overfitting",
            "Saves training time",
            "Patience=10 is good balance",
        ],
    }
    
    for k, v in config.items():
        if isinstance(v, list):
            log(f"  {k}: {v}")
        elif isinstance(v, str):
            log(f"  {k}: {v}")
        else:
            log(f"  {k}: {v}")
    
    return config

# ============================================================
# 12.7 WEIGHT DECAY
# ============================================================
def stage_12_7_weight_decay():
    log("\n" + "=" * 70)
    log("STAGE 12.7: WEIGHT DECAY SETUP")
    log("=" * 70)
    
    log(f"  [INFO] Weight Decay is equivalent to L2 Regularization")
    log(f"  [INFO] AdamW uses decoupled weight decay")
    log(f"  [INFO] For RMSprop, use L2 regularization in layers")
    log(f"  [DECISION] Use L2 regularization in Dense layers (not AdamW)")
    
    return {"Method": "L2 in layers", "Value": "1e-4"}

# ============================================================
# 12.8 DATA AUGMENTATION
# ============================================================
def stage_12_8_data_augmentation():
    log("\n" + "=" * 70)
    log("STAGE 12.8: DATA AUGMENTATION SETUP")
    log("=" * 70)
    
    log(f"  [INFO] Data Augmentation for tabular data:")
    log(f"    - SMOTE (Synthetic Minority Oversampling)")
    log(f"    - ADASYN (Adaptive Synthetic Sampling)")
    log(f"    - Gaussian Noise Injection")
    log(f"    - Mixup")
    log(f"  [DECISION] Use Class Weight instead of SMOTE (simpler)")
    log(f"  [NOTE] SMOTE will be tested in Phase 13 (Hyperparameters)")
    
    return {"Method": "Class Weight", "Alternative": "SMOTE"}

# ============================================================
# 12.9 LABEL SMOOTHING
# ============================================================
def stage_12_9_label_smoothing():
    log("\n" + "=" * 70)
    log("STAGE 12.9: LABEL SMOOTHING SETUP")
    log("=" * 70)
    
    config = {
        "Enabled": False,
        "Epsilon": 0.1,
        "Rationale": [
            "Label smoothing for binary classification is marginal",
            "More useful for multi-class with noisy labels",
            "Anomaly labels are deterministic (from CO, APH)",
        ],
        "If Enabled": {
            "Formula": "y_smooth = (1-ε)*y + ε/2",
            "Effect": "Soft labels, better calibration",
        },
    }
    
    for k, v in config.items():
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
    
    return config

# ============================================================
# 12.10 ENSEMBLE SETUP
# ============================================================
def stage_12_10_ensemble():
    log("\n" + "=" * 70)
    log("STAGE 12.10: ENSEMBLE SETUP")
    log("=" * 70)
    
    config = {
        "Enabled": False,
        "Reason": "Single MLP is sufficient for this task",
        "Alternatives": {
            "Bagging": "Train multiple MLPs with different seeds",
            "Boosting": "Already have AdaBoost",
            "Stacking": "Combine MLP + AdaBoost",
        },
        "If Enabled": {
            "N_Models": 5,
            "Aggregation": "Average probabilities",
        },
    }
    
    for k, v in config.items():
        if isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                log(f"    {kk}: {vv}")
        else:
            log(f"  {k}: {v}")
    
    return config

# ============================================================
# BUILD MODEL WITH REGULARIZATION
# ============================================================
def build_model(n_features, l2_val, dropout_rates, optimizer, n_classes=1):
    if n_classes == 1:
        output_activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        output_activation = "softmax"
        loss = "sparse_categorical_crossentropy"
    
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(l2_val)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rates[0]),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(l2_val)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rates[1]),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(l2_val)),
        layers.BatchNormalization(),
        layers.Dropout(dropout_rates[2]),
        layers.Dense(n_classes, activation=output_activation),
    ])
    
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

# ============================================================
# RUN REGULARIZATION EXPERIMENTS
# ============================================================
def run_experiments(train_df, val_df, features, target, class_weight):
    log("\n" + "=" * 70)
    log("RUNNING REGULARIZATION EXPERIMENTS")
    log("=" * 70)
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df[target].values.astype(float)
    X_val = val_df[features].fillna(0).values
    y_val = val_df[target].values.astype(float)
    
    experiments = {
        "Baseline": {
            "l2": 1e-4,
            "dropout": (0.3, 0.3, 0.2),
            "patience": 10,
        },
        "Low_Reg": {
            "l2": 1e-5,
            "dropout": (0.2, 0.2, 0.1),
            "patience": 15,
        },
        "High_Reg": {
            "l2": 1e-3,
            "dropout": (0.5, 0.5, 0.3),
            "patience": 5,
        },
        "No_Reg": {
            "l2": 0.0,
            "dropout": (0.0, 0.0, 0.0),
            "patience": 20,
        },
        "Medium_Reg": {
            "l2": 5e-4,
            "dropout": (0.4, 0.4, 0.2),
            "patience": 10,
        },
    }
    
    results = []
    
    for exp_name, config in experiments.items():
        log(f"\n  Running {exp_name}...")
        log(f"    L2={config['l2']}, Dropout={config['dropout']}, Patience={config['patience']}")
        
        start = time.time()
        
        try:
            model = build_model(
                len(features), config["l2"], config["dropout"],
                optimizers.RMSprop(learning_rate=0.001), n_classes=1
            )
            
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=config["patience"],
                    restore_best_weights=True,
                    verbose=0,
                ),
                ReduceLROnPlateau(
                    monitor="val_loss",
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6,
                    verbose=0,
                ),
            ]
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=100,
                batch_size=256,
                class_weight=class_weight,
                callbacks=callbacks,
                verbose=0,
            )
            
            train_time = time.time() - start
            epochs_run = len(history.history["loss"])
            
            final_train_loss = history.history["loss"][-1]
            final_val_loss = history.history["val_loss"][-1]
            final_train_acc = history.history["accuracy"][-1]
            final_val_acc = history.history["val_accuracy"][-1]
            
            # بهترین val_loss
            best_val_loss = min(history.history["val_loss"])
            best_epoch = history.history["val_loss"].index(best_val_loss) + 1
            
            results.append({
                "Experiment": exp_name,
                "L2": config["l2"],
                "Dropout": str(config["dropout"]),
                "Patience": config["patience"],
                "Epochs_Run": epochs_run,
                "Best_Epoch": best_epoch,
                "Best_Val_Loss": round(best_val_loss, 6),
                "Final_Val_Loss": round(final_val_loss, 6),
                "Final_Val_Acc": round(final_val_acc, 6),
                "Final_Train_Acc": round(final_train_acc, 6),
                "Overfit": round(final_train_acc - final_val_acc, 6),
                "Train_Time_s": round(train_time, 2),
            })
            
            log(f"    Epochs: {epochs_run} (best: {best_epoch})")
            log(f"    Best Val Loss: {best_val_loss:.6f}")
            log(f"    Final Val Acc: {final_val_acc:.6f}")
            log(f"    Overfit: {final_train_acc - final_val_acc:.6f}")
            log(f"    Time: {train_time:.2f}s")
        
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    return pd.DataFrame(results)

# ============================================================
# VISUALIZE RESULTS
# ============================================================
def visualize_results(results_df):
    log("\n" + "=" * 70)
    log("VISUALIZING REGULARIZATION RESULTS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # Best Val Loss
    sorted_df = results_df.sort_values("Best_Val_Loss")
    axes[0, 0].barh(sorted_df["Experiment"], sorted_df["Best_Val_Loss"], color="steelblue")
    axes[0, 0].set_title("Best Validation Loss by Regularization")
    axes[0, 0].set_xlabel("Best Val Loss")
    axes[0, 0].grid(True, alpha=0.3)
    
    # Final Val Accuracy
    sorted_df2 = results_df.sort_values("Final_Val_Acc", ascending=False)
    axes[0, 1].barh(sorted_df2["Experiment"], sorted_df2["Final_Val_Acc"], color="coral")
    axes[0, 1].set_title("Final Validation Accuracy")
    axes[0, 1].set_xlabel("Val Accuracy")
    axes[0, 1].grid(True, alpha=0.3)
    
    # Overfit
    axes[1, 0].barh(results_df["Experiment"], results_df["Overfit"], color="green")
    axes[1, 0].set_title("Overfit (Train Acc - Val Acc)")
    axes[1, 0].set_xlabel("Overfit")
    axes[1, 0].grid(True, alpha=0.3)
    
    # Epochs Run
    axes[1, 1].barh(results_df["Experiment"], results_df["Epochs_Run"], color="purple")
    axes[1, 1].set_title("Epochs Run (with EarlyStopping)")
    axes[1, 1].set_xlabel("Epochs")
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "12_regularization_results.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 12_regularization_results.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 12: REGULARIZATION")
    log("=" * 70)
    
    # 12.1
    techniques = stage_12_1_selection()
    
    # 12.2-12.4
    l1_l2_config = stage_12_2_4_l1_l2_elasticnet()
    
    # 12.5
    dropout_config = stage_12_5_dropout()
    
    # 12.6
    early_stopping_config = stage_12_6_early_stopping()
    
    # 12.7
    weight_decay_config = stage_12_7_weight_decay()
    
    # 12.8
    augmentation_config = stage_12_8_data_augmentation()
    
    # 12.9
    label_smoothing_config = stage_12_9_label_smoothing()
    
    # 12.10
    ensemble_config = stage_12_10_ensemble()
    
    # ============================================================
    # EXPERIMENTS
    # ============================================================
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    
    features = [c for c in feature_df["Feature"].tolist() 
                if c in train_df.columns and c not in LEAKY_CLF]
    
    class_dist = train_df[TARGET_ANOMALY].value_counts()
    imbalance_ratio = class_dist[0] / class_dist[1]
    class_weight = {0: 1.0, 1: imbalance_ratio}
    
    log(f"\n  Features: {len(features)}")
    log(f"  Class Weight: {class_weight}")
    
    results_df = run_experiments(train_df, val_df, features, TARGET_ANOMALY, class_weight)
    results_df.to_csv(TABLES_DIR / "12_regularization_comparison.csv", index=False)
    log(f"  [OK] Saved: 12_regularization_comparison.csv")
    
    log(f"\n  Regularization Ranking (by Best Val Loss):")
    for _, row in results_df.sort_values("Best_Val_Loss").iterrows():
        log(f"    {row['Experiment']}: Best Val Loss={row['Best_Val_Loss']:.6f}, "
            f"Val Acc={row['Final_Val_Acc']:.6f}, Overfit={row['Overfit']:.6f}, "
            f"Epochs={row['Epochs_Run']}")
    
    visualize_results(results_df)
    
    # ============================================================
    # SUMMARY
    # ============================================================
    best = results_df.sort_values("Best_Val_Loss").iloc[0]
    
    log("\n" + "=" * 70)
    log("REGULARIZATION SELECTION SUMMARY")
    log("=" * 70)
    log(f"  Best Experiment: {best['Experiment']}")
    log(f"  L2: {best['L2']}")
    log(f"  Dropout: {best['Dropout']}")
    log(f"  Patience: {best['Patience']}")
    log(f"  Best Val Loss: {best['Best_Val_Loss']}")
    log(f"  Final Val Acc: {best['Final_Val_Acc']}")
    log(f"  Overfit: {best['Overfit']}")
    
    summary = {
        "Best_Experiment": best["Experiment"],
        "Best_L2": best["L2"],
        "Best_Dropout": best["Dropout"],
        "Best_Patience": best["Patience"],
        "Best_Val_Loss": best["Best_Val_Loss"],
        "Best_Val_Acc": best["Final_Val_Acc"],
        "Best_Overfit": best["Overfit"],
        "Optimizer": "RMSprop",
        "LR_Schedule": "ReduceLROnPlateau",
        "Batch_Size": 256,
        "Max_Epochs": 100,
    }
    
    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "12_regularization_summary.csv", index=False
    )
    log(f"  [OK] Saved: 12_regularization_summary.csv")
    
    log("\n" + "=" * 70)
    log("PHASE 12 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase12_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()