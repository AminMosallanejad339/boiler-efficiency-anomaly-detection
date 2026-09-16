"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 11b: Optimizer Comparison - Fixed (Binned Classification)
Framework: ML Model Lifecycle - 23 Main Stages
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
from pathlib import Path
from datetime import datetime
import time

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)   # ✅ اصلاح شد

# ============================================================
# تنظیمات مسیر
# ============================================================
BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
SPLITS_DIR = BASE_DIR / "data" / "splits"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_BINNED = "Boiler_Eff_Class"

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# بارگذاری
# ============================================================
log("=" * 70)
log("PHASE 11b: OPTIMIZER COMPARISON - BINNED CLASSIFICATION")
log("=" * 70)

train_df = pd.read_csv(SPLITS_DIR / "train_binned.csv")
val_df = pd.read_csv(SPLITS_DIR / "validation_binned.csv")
feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")

log(f"Train: {train_df.shape}")
log(f"Val: {val_df.shape}")

if TARGET_BINNED not in train_df.columns:
    log(f"[ERROR] Column '{TARGET_BINNED}' not found!")
    log(f"Available columns: {list(train_df.columns[-10:])}")
    exit(1)

log(f"Target column: {TARGET_BINNED}")
log(f"Class distribution:")
for cls, count in train_df[TARGET_BINNED].value_counts().sort_index().items():
    log(f"  Class {cls}: {count} ({count/len(train_df)*100:.2f}%)")

# ============================================================
# BUILD MODEL
# ============================================================
def build_model(n_features, n_classes, optimizer):
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
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# ============================================================
# COMPARE OPTIMIZERS
# ============================================================
def compare_optimizers(train_df, val_df, features, target, n_classes):
    X_train = train_df[features].fillna(0).values
    y_train = train_df[target].values
    X_val = val_df[features].fillna(0).values
    y_val = val_df[target].values

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

# ============================================================
# VISUALIZE
# ============================================================
def visualize_optimizer_comparison(results_df, title):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].bar(results_df["Optimizer"], results_df["Val_Loss"], color="steelblue")
    axes[0].set_title(f"{title}: Validation Loss")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(results_df["Optimizer"], results_df["Val_Acc"], color="coral")
    axes[1].set_title(f"{title}: Validation Accuracy")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)

    axes[2].bar(results_df["Optimizer"], results_df["Train_Time_s"], color="green")
    axes[2].set_title(f"{title}: Training Time")
    axes[2].set_ylabel("Seconds")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / f"11b_optimizer_comparison_{title.lower().replace(' ', '_')}.png",
                dpi=100, bbox_inches="tight")
    plt.close()

# ============================================================
# MAIN
# ============================================================
def main():
    features_binned = [c for c in feature_df["Feature"].tolist()
                       if c in train_df.columns and c not in ["turbine_to_boiler_eff"]]

    log(f"\n  Binned Classification:")
    log(f"    Features: {len(features_binned)}")

    binned_results = compare_optimizers(
        train_df, val_df, features_binned, TARGET_BINNED, 3
    )
    binned_results.to_csv(TABLES_DIR / "11b_binned_optimizer_comparison.csv", index=False)
    log(f"  [OK] Saved: 11b_binned_optimizer_comparison.csv")

    log(f"\n  Binned Optimizer Ranking:")
    for _, row in binned_results.sort_values("Val_Loss").iterrows():
        log(f"    {row['Optimizer']}: Val Loss={row['Val_Loss']:.6f}, Val Acc={row['Val_Acc']:.6f}")

    visualize_optimizer_comparison(binned_results, "Binned Classification")

    # ============================================================
    # SUMMARY
    # ============================================================
    best_binned_opt = binned_results.sort_values("Val_Loss").iloc[0]["Optimizer"]

    log("\n" + "=" * 70)
    log("OPTIMIZER SELECTION SUMMARY")
    log("=" * 70)
    log(f"  Best Optimizer for Anomaly: RMSprop (from Phase 11)")
    log(f"  Best Optimizer for Binned: {best_binned_opt}")

    summary = {
        "Anomaly_Best_Optimizer": "RMSprop",
        "Anomaly_Val_Loss": 0.375188,
        "Anomaly_Val_Acc": 0.988513,
        "Binned_Best_Optimizer": best_binned_opt,
        "Binned_Val_Loss": round(binned_results.sort_values('Val_Loss').iloc[0]['Val_Loss'], 6),
        "Binned_Val_Acc": round(binned_results.sort_values('Val_Loss').iloc[0]['Val_Acc'], 6),
        "Default_LR": 0.001,
        "LR_Schedule": "ReduceLROnPlateau",
        "Gradient_Clipping": "Clip by Norm = 1.0",
        "Batch_Size": 256,
        "Max_Epochs": 100,
        "EarlyStopping_Patience": 10,
    }

    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "11b_optimizer_summary.csv", index=False
    )
    log(f"  [OK] Saved: 11b_optimizer_summary.csv")

    log("\n" + "=" * 70)
    log("PHASE 11 COMPLETE!")
    log("=" * 70)

    LOG_FILE = BASE_DIR / "reports" / "phase11b_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()
