"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 11: Optimizer Selection (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Target: MLP for Anomaly Detection + Binned Classification
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
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
TARGET_BINNED = "Boiler_Eff_Class"
LEAKY_CLF = ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# 11.1 OPTIMIZER SELECTION
# ============================================================
def stage_11_1_optimizer_selection():
    log("=" * 70)
    log("PHASE 11: OPTIMIZER SELECTION")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("STAGE 11.1: OPTIMIZER SELECTION")
    log("=" * 70)
    
    optimizers_info = {
        "Adam": {
            "Formula": "θ = θ - lr * m_hat / (sqrt(v_hat) + ε)",
            "Pros": ["Fast convergence", "Adaptive learning rate", "Works well by default"],
            "Cons": ["May overfit", "Memory for m and v"],
            "Best For": "General purpose, tabular data",
        },
        "AdamW": {
            "Formula": "Adam + decoupled weight decay",
            "Pros": ["Better generalization", "Decoupled weight decay"],
            "Cons": ["More complex", "Slightly slower"],
            "Best For": "Deep networks with regularization",
        },
        "SGD": {
            "Formula": "θ = θ - lr * gradient",
            "Pros": ["Simple", "Better generalization"],
            "Cons": ["Slow", "Sensitive to LR"],
            "Best For": "Large datasets, fine-tuning",
        },
        "SGD_Momentum": {
            "Formula": "v = β*v + gradient; θ = θ - lr*v",
            "Pros": ["Faster than SGD", "Escapes local minima"],
            "Cons": ["Needs momentum tuning"],
            "Best For": "Computer vision",
        },
        "RMSprop": {
            "Formula": "θ = θ - lr * g / (sqrt(E[g^2]) + ε)",
            "Pros": ["Works well for RNNs", "Adaptive"],
            "Cons": ["No momentum by default"],
            "Best For": "Sequential data",
        },
        "Nadam": {
            "Formula": "Adam + Nesterov momentum",
            "Pros": ["Faster than Adam", "Nesterov momentum"],
            "Cons": ["Less common"],
            "Best For": "When Adam is not enough",
        },
    }
    
    for opt, config in optimizers_info.items():
        log(f"\n  {opt}:")
        log(f"    Formula: {config['Formula']}")
        log(f"    Pros: {', '.join(config['Pros'])}")
        log(f"    Cons: {', '.join(config['Cons'])}")
        log(f"    Best For: {config['Best For']}")
    
    return optimizers_info

# ============================================================
# 11.2-11.4 OPTIMIZER CONFIGURATIONS
# ============================================================
def stage_11_2_learning_rate():
    log("\n" + "=" * 70)
    log("STAGE 11.2: LEARNING RATE INITIALIZATION")
    log("=" * 70)
    
    lr_config = {
        "Default LR": 0.001,
        "Range to Test": [1e-5, 1e-4, 1e-3, 1e-2],
        "Rationale": "1e-3 is a good starting point for Adam",
        "Too High": "Divergence, NaN loss",
        "Too Low": "Slow convergence",
    }
    
    for k, v in lr_config.items():
        log(f"  {k}: {v}")
    
    return lr_config

def stage_11_3_momentum():
    log("\n" + "=" * 70)
    log("STAGE 11.3: MOMENTUM SETUP")
    log("=" * 70)
    
    momentum_config = {
        "Adam Beta1": 0.9,
        "Adam Beta2": 0.999,
        "Adam Epsilon": 1e-7,
        "SGD Momentum": 0.9,
        "Nesterov": True,
        "Rationale": [
            "Beta1: Exponential decay for 1st moment",
            "Beta2: Exponential decay for 2nd moment",
            "Epsilon: Small constant for numerical stability",
        ],
    }
    
    for k, v in momentum_config.items():
        if isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        else:
            log(f"  {k}: {v}")
    
    return momentum_config

def stage_11_4_adaptive():
    log("\n" + "=" * 70)
    log("STAGE 11.4: ADAPTIVE METHOD SETUP")
    log("=" * 70)
    
    adaptive_config = {
        "Adam": "Adaptive per-parameter learning rates",
        "AdamW": "Adam + decoupled weight decay (1e-4)",
        "RMSprop": "Adaptive with running average of squared gradients",
        "Adagrad": "Adaptive with cumulative squared gradients",
        "Adadelta": "Adaptive without learning rate",
        "Rationale": "Adaptive methods handle sparse gradients well",
    }
    
    for k, v in adaptive_config.items():
        log(f"  {k}: {v}")
    
    return adaptive_config

# ============================================================
# 11.5 LEARNING RATE SCHEDULE
# ============================================================
def stage_11_5_lr_schedule():
    log("\n" + "=" * 70)
    log("STAGE 11.5: LEARNING RATE SCHEDULE DESIGN")
    log("=" * 70)
    
    schedules = {
        "ReduceLROnPlateau": {
            "Monitor": "val_loss",
            "Factor": 0.5,
            "Patience": 5,
            "Min_LR": 1e-6,
            "Pros": "Adaptive to training progress",
        },
        "CosineAnnealing": {
            "Formula": "lr = lr_min + 0.5*(lr_max-lr_min)*(1+cos(pi*t/T))",
            "Pros": "Smooth decay, good for fine-tuning",
        },
        "StepDecay": {
            "Formula": "lr = lr_0 * factor^(epoch/step_size)",
            "Pros": "Simple, predictable",
        },
        "ExponentialDecay": {
            "Formula": "lr = lr_0 * exp(-k*t)",
            "Pros": "Smooth, fast decay",
        },
    }
    
    for name, config in schedules.items():
        log(f"  {name}:")
        for k, v in config.items():
            log(f"    {k}: {v}")
    
    log(f"\n  [SELECTED] ReduceLROnPlateau")
    log(f"  Rationale: Adaptive to validation loss, simple to implement")
    
    return schedules

# ============================================================
# 11.6 GRADIENT CLIPPING
# ============================================================
def stage_11_6_gradient_clipping():
    log("\n" + "=" * 70)
    log("STAGE 11.6: GRADIENT CLIPPING SETUP")
    log("=" * 70)
    
    clipping = {
        "Method": "Clip by Global Norm",
        "Value": 1.0,
        "Rationale": [
            "Prevents gradient explosion",
            "Stabilizes training",
            "Common in RNNs and deep networks",
        ],
        "Alternatives": {
            "Clip by Value": "0.5 (simpler, less precise)",
            "Clip by Norm per Layer": "More fine-grained",
        },
    }
    
    for k, v in clipping.items():
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
    
    return clipping

# ============================================================
# 11.7 GRADIENT ACCUMULATION
# ============================================================
def stage_11_7_gradient_accumulation():
    log("\n" + "=" * 70)
    log("STAGE 11.7: GRADIENT ACCUMULATION SETUP")
    log("=" * 70)
    
    accumulation = {
        "Enabled": False,
        "Rationale": "Dataset is small (35K), no need for accumulation",
        "If Enabled": {
            "Steps": 4,
            "Effective Batch Size": "256 * 4 = 1024",
        },
        "When to Use": "Large models, small GPU memory",
    }
    
    for k, v in accumulation.items():
        if isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                log(f"    {kk}: {vv}")
        else:
            log(f"  {k}: {v}")
    
    return accumulation

# ============================================================
# 11.8 BATCH SIZE SELECTION
# ============================================================
def stage_11_8_batch_size():
    log("\n" + "=" * 70)
    log("STAGE 11.8: BATCH SIZE SELECTION")
    log("=" * 70)
    
    batch_config = {
        "Default": 256,
        "Range to Test": [64, 128, 256, 512],
        "Rationale": [
            "256 is standard for tabular data",
            "Larger batch = faster but less generalization",
            "Smaller batch = slower but better generalization",
        ],
        "Hardware": "CPU (adjust based on memory)",
    }
    
    for k, v in batch_config.items():
        if isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        else:
            log(f"  {k}: {v}")
    
    return batch_config

# ============================================================
# 11.9 EPOCH COUNT
# ============================================================
def stage_11_9_epoch_count():
    log("\n" + "=" * 70)
    log("STAGE 11.9: EPOCH COUNT SELECTION")
    log("=" * 70)
    
    epoch_config = {
        "Max Epochs": 100,
        "EarlyStopping": {
            "Patience": 10,
            "Monitor": "val_loss",
            "Restore Best Weights": True,
        },
        "Rationale": [
            "100 epochs as upper bound",
            "EarlyStopping prevents overfitting",
            "Typical convergence: 20-50 epochs",
        ],
    }
    
    for k, v in epoch_config.items():
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
    
    return epoch_config

# ============================================================
# 11.10 CONVERGENCE CRITERIA
# ============================================================
def stage_11_10_convergence():
    log("\n" + "=" * 70)
    log("STAGE 11.10: CONVERGENCE CRITERIA DEFINITION")
    log("=" * 70)
    
    convergence = {
        "Loss Convergence": "val_loss change < 1e-5 for 5 epochs",
        "Gradient Norm": "Gradient norm < 1e-3",
        "Learning Rate": "LR < 1e-6",
        "EarlyStopping": "val_loss not improving for 10 epochs",
        "Max Epochs": "Reach 100 epochs",
        "Metric Target": {
            "Anomaly F1": "> 0.90",
            "Binned F1-Macro": "> 0.35",
        },
    }
    
    for k, v in convergence.items():
        if isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                log(f"    {kk}: {vv}")
        else:
            log(f"  {k}: {v}")
    
    return convergence

# ============================================================
# COMPARE OPTIMIZERS
# ============================================================
def build_model(n_features, n_classes, optimizer):
    """ساخت مدل MLP با optimizer مشخص"""
    if n_classes == 1:
        output_activation = "sigmoid"
        loss = "binary_crossentropy"
    else:
        output_activation = "softmax"
        loss = "sparse_categorical_crossentropy"
    
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(n_classes, activation=output_activation),
    ])
    
    model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
    return model

def compare_optimizers(train_df, val_df, features, target, n_classes, class_weight=None):
    """مقایسه Optimizerهای مختلف"""
    X_train = train_df[features].fillna(0).values
    y_train = train_df[target].values
    X_val = val_df[features].fillna(0).values
    y_val = val_df[target].values
    
    if n_classes == 1:
        y_train = y_train.astype(float)
        y_val = y_val.astype(float)
    
    optimizers_dict = {
        "Adam": optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999),
        "AdamW": optimizers.AdamW(learning_rate=0.001, weight_decay=1e-4),
        "SGD": optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True),
        "RMSprop": optimizers.RMSprop(learning_rate=0.001),
        "Nadam": optimizers.Nadam(learning_rate=0.001),
    }
    
    results = []
    
    for opt_name, opt in optimizers_dict.items():
        log(f"\n  Testing {opt_name}...")
        start = time.time()
        
        try:
            model = build_model(len(features), n_classes, opt)
            
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=20,
                batch_size=256,
                class_weight=class_weight,
                verbose=0,
            )
            
            train_time = time.time() - start
            final_train_loss = history.history["loss"][-1]
            final_val_loss = history.history["val_loss"][-1]
            final_train_acc = history.history["accuracy"][-1]
            final_val_acc = history.history["val_accuracy"][-1]
            
            results.append({
                "Optimizer": opt_name,
                "Train_Time_s": round(train_time, 2),
                "Train_Loss": round(final_train_loss, 6),
                "Val_Loss": round(final_val_loss, 6),
                "Train_Acc": round(final_train_acc, 6),
                "Val_Acc": round(final_val_acc, 6),
                "Overfit": round(final_train_acc - final_val_acc, 6),
            })
            
            log(f"    Val Loss: {final_val_loss:.6f} | Val Acc: {final_val_acc:.6f} | Time: {train_time:.2f}s")
        
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    return pd.DataFrame(results)

def visualize_optimizer_comparison(results_df, title):
    """نمودار مقایسه Optimizerها"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Val Loss
    axes[0].bar(results_df["Optimizer"], results_df["Val_Loss"], color="steelblue")
    axes[0].set_title(f"{title}: Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    
    # Val Accuracy
    axes[1].bar(results_df["Optimizer"], results_df["Val_Acc"], color="coral")
    axes[1].set_title(f"{title}: Validation Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    
    # Train Time
    axes[2].bar(results_df["Optimizer"], results_df["Train_Time_s"], color="green")
    axes[2].set_title(f"{title}: Training Time")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"11_optimizer_comparison_{title.lower().replace(' ', '_')}.png", 
                dpi=100, bbox_inches="tight")
    plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 11: OPTIMIZER SELECTION")
    log("=" * 70)
    
    # 11.1
    optimizers_info = stage_11_1_optimizer_selection()
    
    # 11.2
    lr_config = stage_11_2_learning_rate()
    
    # 11.3
    momentum_config = stage_11_3_momentum()
    
    # 11.4
    adaptive_config = stage_11_4_adaptive()
    
    # 11.5
    schedules = stage_11_5_lr_schedule()
    
    # 11.6
    clipping = stage_11_6_gradient_clipping()
    
    # 11.7
    accumulation = stage_11_7_gradient_accumulation()
    
    # 11.8
    batch_config = stage_11_8_batch_size()
    
    # 11.9
    epoch_config = stage_11_9_epoch_count()
    
    # 11.10
    convergence = stage_11_10_convergence()
    
    # ============================================================
    # COMPARE OPTIMIZERS
    # ============================================================
    log("\n" + "=" * 70)
    log("COMPARING OPTIMIZERS")
    log("=" * 70)
    
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    
    # Anomaly Detection
    features_anomaly = [c for c in feature_df["Feature"].tolist() 
                        if c in train_df.columns and c not in LEAKY_CLF]
    
    class_dist = train_df[TARGET_ANOMALY].value_counts()
    imbalance_ratio = class_dist[0] / class_dist[1]
    class_weight = {0: 1.0, 1: imbalance_ratio}
    
    log(f"\n  Anomaly Detection:")
    log(f"    Features: {len(features_anomaly)}")
    log(f"    Class Weight: {class_weight}")
    
    anomaly_results = compare_optimizers(
        train_df, val_df, features_anomaly, TARGET_ANOMALY, 1, class_weight
    )
    anomaly_results.to_csv(TABLES_DIR / "11_anomaly_optimizer_comparison.csv", index=False)
    log(f"  [OK] Saved: 11_anomaly_optimizer_comparison.csv")
    
    log(f"\n  Anomaly Optimizer Ranking:")
    for _, row in anomaly_results.sort_values("Val_Loss").iterrows():
        log(f"    {row['Optimizer']}: Val Loss={row['Val_Loss']:.6f}, Val Acc={row['Val_Acc']:.6f}")
    
    visualize_optimizer_comparison(anomaly_results, "Anomaly Detection")
    
    # Binned Classification
    features_binned = [c for c in feature_df["Feature"].tolist() 
                       if c in train_df.columns and c not in ["turbine_to_boiler_eff"]]
    
    log(f"\n  Binned Classification:")
    log(f"    Features: {len(features_binned)}")
    
    binned_results = compare_optimizers(
        train_df, val_df, features_binned, TARGET_BINNED, 3, None
    )
    binned_results.to_csv(TABLES_DIR / "11_binned_optimizer_comparison.csv", index=False)
    log(f"  [OK] Saved: 11_binned_optimizer_comparison.csv")
    
    log(f"\n  Binned Optimizer Ranking:")
    for _, row in binned_results.sort_values("Val_Loss").iterrows():
        log(f"    {row['Optimizer']}: Val Loss={row['Val_Loss']:.6f}, Val Acc={row['Val_Acc']:.6f}")
    
    visualize_optimizer_comparison(binned_results, "Binned Classification")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    best_anomaly_opt = anomaly_results.sort_values("Val_Loss").iloc[0]["Optimizer"]
    best_binned_opt = binned_results.sort_values("Val_Loss").iloc[0]["Optimizer"]
    
    log("\n" + "=" * 70)
    log("OPTIMIZER SELECTION SUMMARY")
    log("=" * 70)
    log(f"  Best Optimizer for Anomaly: {best_anomaly_opt}")
    log(f"  Best Optimizer for Binned: {best_binned_opt}")
    
    summary = {
        "Anomaly_Best_Optimizer": best_anomaly_opt,
        "Anomaly_Val_Loss": round(anomaly_results.sort_values('Val_Loss').iloc[0]['Val_Loss'], 6),
        "Binned_Best_Optimizer": best_binned_opt,
        "Binned_Val_Loss": round(binned_results.sort_values('Val_Loss').iloc[0]['Val_Loss'], 6),
        "Default_LR": 0.001,
        "LR_Schedule": "ReduceLROnPlateau",
        "Gradient_Clipping": "Clip by Norm = 1.0",
        "Batch_Size": 256,
        "Max_Epochs": 100,
        "EarlyStopping_Patience": 10,
    }
    
    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "11_optimizer_summary.csv", index=False
    )
    log(f"  [OK] Saved: 11_optimizer_summary.csv")
    
    log("\n" + "=" * 70)
    log("PHASE 11 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase11_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()