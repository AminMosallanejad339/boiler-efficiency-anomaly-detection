"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 14: Training (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Focus: Anomaly Detection (main successful task)
Best Hyperparameters from Phase 13
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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
from datetime import datetime
import time
import json

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

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# 14.1 TRAINING INITIALIZATION
# ============================================================
def stage_14_1_initialization():
    log("=" * 70)
    log("PHASE 14: TRAINING")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("STAGE 14.1: TRAINING INITIALIZATION")
    log("=" * 70)
    
    # بارگذاری بهترین Hyperparameters از Phase 13
    with open(MODELS_DIR / "best_hyperparameters.json", "r") as f:
        best_params = json.load(f)
    
    log(f"  Loaded Best Hyperparameters from Phase 13:")
    for k, v in best_params.items():
        log(f"    {k}: {v}")
    
    # تنظیمات Training
    training_config = {
        "Optimizer": "RMSprop",
        "Learning Rate": best_params.get("Learning_Rate", 0.002347),
        "Batch Size": int(best_params.get("Batch_Size", 256)),
        "Max Epochs": 100,
        "EarlyStopping Patience": int(best_params.get("Patience", 20)),
        "ReduceLROnPlateau Patience": 5,
        "ReduceLROnPlateau Factor": 0.5,
        "Min LR": 1e-7,
        "Class Weight": "Computed from data",
        "Random Seed": SEED,
    }
    
    log(f"\n  Training Configuration:")
    for k, v in training_config.items():
        log(f"    {k}: {v}")
    
    return best_params, training_config

# ============================================================
# BUILD FINAL MODEL
# ============================================================
def build_final_model(n_features, params):
    """ساخت مدل نهایی با بهترین Hyperparameters"""
    log(f"\n  Building Final Model...")
    
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        layers.Dense(int(params["Hidden_1"]), activation="relu",
                     kernel_regularizer=regularizers.l2(params["L2"])),
        layers.BatchNormalization(),
        layers.Dropout(params["Dropout_1"]),
        layers.Dense(int(params["Hidden_2"]), activation="relu",
                     kernel_regularizer=regularizers.l2(params["L2"])),
        layers.BatchNormalization(),
        layers.Dropout(params["Dropout_2"]),
        layers.Dense(int(params["Hidden_3"]), activation="relu",
                     kernel_regularizer=regularizers.l2(params["L2"])),
        layers.BatchNormalization(),
        layers.Dropout(params["Dropout_3"]),
        layers.Dense(1, activation="sigmoid"),
    ])
    
    optimizer = optimizers.RMSprop(learning_rate=params["Learning_Rate"])
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    
    log(f"    Total Parameters: {model.count_params():,}")
    log(f"    Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    log(f"    Non-trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    
    return model

# ============================================================
# 14.2-14.10 TRAINING LOOP
# ============================================================
def stage_14_2_to_14_10_training(model, X_train, y_train, X_val, y_val, 
                                    params, class_weight, config):
    log("\n" + "=" * 70)
    log("STAGE 14.2-14.10: TRAINING LOOP EXECUTION")
    log("=" * 70)
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=int(params["Patience"]),
            restore_best_weights=True,
            min_delta=1e-4,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=config["ReduceLROnPlateau Factor"],
            patience=config["ReduceLROnPlateau Patience"],
            min_lr=config["Min LR"],
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_model_during_training.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    
    log(f"\n  Starting Training...")
    log(f"    Train samples: {len(X_train)}")
    log(f"    Val samples: {len(X_val)}")
    log(f"    Batch size: {config['Batch Size']}")
    log(f"    Max epochs: {config['Max Epochs']}")
    log(f"    Class weight: {class_weight}")
    
    start_time = time.time()
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=config["Max Epochs"],
        batch_size=config["Batch Size"],
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )
    
    train_time = time.time() - start_time
    
    log(f"\n  Training Complete!")
    log(f"    Total Time: {train_time:.2f}s")
    log(f"    Epochs Run: {len(history.history['loss'])}")
    log(f"    Best Epoch: {np.argmin(history.history['val_loss']) + 1}")
    log(f"    Best Val Loss: {min(history.history['val_loss']):.6f}")
    log(f"    Best Val Accuracy: {max(history.history['val_accuracy']):.6f}")
    
    return history, train_time

# ============================================================
# 14.10 TRAINING MONITORING
# ============================================================
def stage_14_10_monitoring(history, train_time):
    log("\n" + "=" * 70)
    log("STAGE 14.10: TRAINING MONITORING")
    log("=" * 70)
    
    epochs = len(history.history["loss"])
    
    # استخراج تاریخچه
    metrics = {
        "train_loss": history.history["loss"],
        "val_loss": history.history["val_loss"],
        "train_acc": history.history["accuracy"],
        "val_acc": history.history["val_accuracy"],
    }
    
    # Precision و Recall (اگر موجود باشند)
    if "precision" in history.history:
        metrics["train_precision"] = history.history["precision"]
        metrics["val_precision"] = history.history["val_precision"]
    if "recall" in history.history:
        metrics["train_recall"] = history.history["recall"]
        metrics["val_recall"] = history.history["val_recall"]
    
    # ذخیره تاریخچه
    history_df = pd.DataFrame(metrics)
    history_df.index = range(1, epochs + 1)
    history_df.index.name = "Epoch"
    history_df.to_csv(TABLES_DIR / "14_training_history.csv")
    log(f"  [OK] Saved: 14_training_history.csv")
    
    # تحلیل
    best_epoch = np.argmin(history.history["val_loss"]) + 1
    best_val_loss = min(history.history["val_loss"])
    best_val_acc = max(history.history["val_accuracy"])
    
    log(f"\n  Training Analysis:")
    log(f"    Total Epochs: {epochs}")
    log(f"    Best Epoch: {best_epoch}")
    log(f"    Best Val Loss: {best_val_loss:.6f}")
    log(f"    Best Val Accuracy: {best_val_acc:.6f}")
    log(f"    Final Train Loss: {history.history['loss'][-1]:.6f}")
    log(f"    Final Val Loss: {history.history['val_loss'][-1]:.6f}")
    log(f"    Final Train Acc: {history.history['accuracy'][-1]:.6f}")
    log(f"    Final Val Acc: {history.history['val_accuracy'][-1]:.6f}")
    
    # تحلیل Overfitting
    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    overfit = final_train_acc - final_val_acc
    log(f"    Overfit: {overfit:.6f}")
    
    if overfit > 0.1:
        log(f"    [WARNING] Significant Overfitting detected")
    elif overfit < -0.1:
        log(f"    [INFO] Validation > Train (Dropout/BatchNorm effect)")
    else:
        log(f"    [OK] Good generalization")
    
    return history_df, best_epoch, best_val_loss, best_val_acc

# ============================================================
# VISUALIZE TRAINING
# ============================================================
def visualize_training(history_df, best_epoch):
    log("\n" + "=" * 70)
    log("VISUALIZING TRAINING RESULTS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Loss
    axes[0, 0].plot(history_df.index, history_df["train_loss"], "b-", label="Train Loss", linewidth=2)
    axes[0, 0].plot(history_df.index, history_df["val_loss"], "r-", label="Val Loss", linewidth=2)
    axes[0, 0].axvline(x=best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training and Validation Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Accuracy
    axes[0, 1].plot(history_df.index, history_df["train_acc"], "b-", label="Train Acc", linewidth=2)
    axes[0, 1].plot(history_df.index, history_df["val_acc"], "r-", label="Val Acc", linewidth=2)
    axes[0, 1].axvline(x=best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].set_title("Training and Validation Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision
    if "train_precision" in history_df.columns:
        axes[1, 0].plot(history_df.index, history_df["train_precision"], "b-", label="Train Precision", linewidth=2)
        axes[1, 0].plot(history_df.index, history_df["val_precision"], "r-", label="Val Precision", linewidth=2)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Precision")
        axes[1, 0].set_title("Training and Validation Precision")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Recall
    if "train_recall" in history_df.columns:
        axes[1, 1].plot(history_df.index, history_df["train_recall"], "b-", label="Train Recall", linewidth=2)
        axes[1, 1].plot(history_df.index, history_df["val_recall"], "r-", label="Val Recall", linewidth=2)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Recall")
        axes[1, 1].set_title("Training and Validation Recall")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_training_curves.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14_training_curves.png")
    
    # نمودار جداگانه Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(history_df.index, history_df["train_loss"], "b-", label="Train Loss", linewidth=2)
    ax.plot(history_df.index, history_df["val_loss"], "r-", label="Val Loss", linewidth=2)
    ax.axvline(x=best_epoch, color="green", linestyle="--", label=f"Best Epoch ({best_epoch})")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Loss Curve (Anomaly Detection)", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14_loss_curve.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14_loss_curve.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 14: TRAINING")
    log("=" * 70)
    
    # 14.1
    best_params, config = stage_14_1_initialization()
    
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
    
    log(f"\n  Data Loaded:")
    log(f"    Features: {len(features)}")
    log(f"    X_train: {X_train.shape}")
    log(f"    X_val: {X_val.shape}")
    log(f"    Class distribution: {class_dist.to_dict()}")
    log(f"    Imbalance ratio: {imbalance_ratio:.2f}")
    log(f"    Class weight: {class_weight}")
    
    # ساخت مدل
    model = build_final_model(len(features), best_params)
    
    # ذخیره معماری
    with open(TABLES_DIR / "14_model_architecture.txt", "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    log(f"  [OK] Saved: 14_model_architecture.txt")
    
    # 14.2-14.10 Training
    history, train_time = stage_14_2_to_14_10_training(
        model, X_train, y_train, X_val, y_val,
        best_params, class_weight, config
    )
    
    # 14.10 Monitoring
    history_df, best_epoch, best_val_loss, best_val_acc = stage_14_10_monitoring(history, train_time)
    
    # Visualize
    visualize_training(history_df, best_epoch)
    
    # ذخیره مدل نهایی
    model.save(MODELS_DIR / "anomaly_mlp_final.keras")
    log(f"  [OK] Saved: anomaly_mlp_final.keras")
    
    # ============================================================
    # SUMMARY
    # ============================================================
    log("\n" + "=" * 70)
    log("TRAINING SUMMARY")
    log("=" * 70)
    
    summary = {
        "Model": "Anomaly MLP",
        "Optimizer": "RMSprop",
        "Learning_Rate": best_params.get("Learning_Rate"),
        "Batch_Size": int(best_params.get("Batch_Size", 256)),
        "L2": best_params.get("L2"),
        "Dropout_1": best_params.get("Dropout_1"),
        "Dropout_2": best_params.get("Dropout_2"),
        "Dropout_3": best_params.get("Dropout_3"),
        "Hidden_1": int(best_params.get("Hidden_1", 128)),
        "Hidden_2": int(best_params.get("Hidden_2", 32)),
        "Hidden_3": int(best_params.get("Hidden_3", 64)),
        "Patience": int(best_params.get("Patience", 20)),
        "Total_Params": model.count_params(),
        "Epochs_Run": len(history.history["loss"]),
        "Best_Epoch": best_epoch,
        "Best_Val_Loss": round(best_val_loss, 6),
        "Best_Val_Acc": round(best_val_acc, 6),
        "Final_Train_Loss": round(history.history["loss"][-1], 6),
        "Final_Val_Loss": round(history.history["val_loss"][-1], 6),
        "Final_Train_Acc": round(history.history["accuracy"][-1], 6),
        "Final_Val_Acc": round(history.history["val_accuracy"][-1], 6),
        "Overfit": round(history.history["accuracy"][-1] - history.history["val_accuracy"][-1], 6),
        "Train_Time_s": round(train_time, 2),
        "Class_Weight": str(class_weight),
    }
    
    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "14_training_summary.csv", index=False
    )
    log(f"  [OK] Saved: 14_training_summary.csv")
    
    log(f"\n  Final Summary:")
    for k, v in summary.items():
        log(f"    {k}: {v}")
    
    log("\n" + "=" * 70)
    log("PHASE 14 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase14_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()