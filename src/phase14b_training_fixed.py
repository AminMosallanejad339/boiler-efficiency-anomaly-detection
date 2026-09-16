"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 14b: Training - Fixed (Class Weight Adjustment)
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

BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
SPLITS_DIR = BASE_DIR / "data" / "splits"
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

TARGET_ANOMALY = "Anomaly_Label"
LEAKY_CLF = ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# BUILD MODEL
# ============================================================
def build_model(n_features, params):
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
    return model

# ============================================================
# TEST DIFFERENT CLASS WEIGHTS
# ============================================================
def test_class_weights(X_train, y_train, X_val, y_val, params):
    log("=" * 70)
    log("TESTING DIFFERENT CLASS WEIGHTS")
    log("=" * 70)
    
    class_weights_to_test = {
        "No_Weight": None,
        "Weight_3": {0: 1.0, 1: 3.0},
        "Weight_5": {0: 1.0, 1: 5.0},
        "Weight_10": {0: 1.0, 1: 10.0},
        "Weight_20": {0: 1.0, 1: 20.0},
        "Weight_63": {0: 1.0, 1: 63.57},
    }
    
    results = []
    histories = {}
    
    for name, cw in class_weights_to_test.items():
        log(f"\n  Testing {name}...")
        log(f"    Class Weight: {cw}")
        
        start = time.time()
        
        try:
            model = build_model(X_train.shape[1], params)
            
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=15,
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
                epochs=50,
                batch_size=int(params["Batch_Size"]),
                class_weight=cw,
                callbacks=callbacks,
                verbose=0,
            )
            
            train_time = time.time() - start
            
            # بهترین Epoch
            best_epoch = np.argmin(history.history["val_loss"])
            best_val_loss = history.history["val_loss"][best_epoch]
            best_val_acc = history.history["val_accuracy"][best_epoch]
            best_val_precision = history.history["val_precision"][best_epoch]
            best_val_recall = history.history["val_recall"][best_epoch]
            
            # F1 محاسبه
            if best_val_precision + best_val_recall > 0:
                best_f1 = 2 * (best_val_precision * best_val_recall) / (best_val_precision + best_val_recall)
            else:
                best_f1 = 0
            
            results.append({
                "Class_Weight_Name": name,
                "Class_Weight_Value": str(cw),
                "Best_Epoch": best_epoch + 1,
                "Epochs_Run": len(history.history["loss"]),
                "Best_Val_Loss": round(best_val_loss, 6),
                "Best_Val_Acc": round(best_val_acc, 6),
                "Best_Val_Precision": round(best_val_precision, 6),
                "Best_Val_Recall": round(best_val_recall, 6),
                "Best_F1": round(best_f1, 6),
                "Train_Time_s": round(train_time, 2),
            })
            
            histories[name] = history
            
            log(f"    Best Epoch: {best_epoch + 1}")
            log(f"    Val Loss: {best_val_loss:.6f}")
            log(f"    Val Acc: {best_val_acc:.6f}")
            log(f"    Val Precision: {best_val_precision:.6f}")
            log(f"    Val Recall: {best_val_recall:.6f}")
            log(f"    F1: {best_f1:.6f}")
            log(f"    Time: {train_time:.2f}s")
        
        except Exception as e:
            log(f"    [ERROR] {e}")
    
    return pd.DataFrame(results), histories

# ============================================================
# VISUALIZE
# ============================================================
def visualize_results(results_df, histories):
    log("\n" + "=" * 70)
    log("VISUALIZING RESULTS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    # 1. Val Loss
    x = np.arange(len(results_df))
    width = 0.25
    axes[0, 0].bar(x - width, results_df["Best_Val_Loss"], width, label="Val Loss", color="steelblue")
    axes[0, 0].bar(x, results_df["Best_Val_Acc"], width, label="Val Acc", color="coral")
    axes[0, 0].bar(x + width, results_df["Best_F1"], width, label="F1", color="green")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(results_df["Class_Weight_Name"], rotation=45, ha="right")
    axes[0, 0].set_title("Metrics by Class Weight")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision vs Recall
    axes[0, 1].scatter(results_df["Best_Val_Precision"], results_df["Best_Val_Recall"], 
                       s=100, c="purple", alpha=0.7)
    for i, row in results_df.iterrows():
        axes[0, 1].annotate(row["Class_Weight_Name"], 
                            (row["Best_Val_Precision"], row["Best_Val_Recall"]),
                            fontsize=8, ha="right")
    axes[0, 1].set_xlabel("Precision")
    axes[0, 1].set_ylabel("Recall")
    axes[0, 1].set_title("Precision vs Recall")
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. F1 by Class Weight
    axes[1, 0].barh(results_df["Class_Weight_Name"], results_df["Best_F1"], color="green")
    axes[1, 0].set_xlabel("Best F1 Score")
    axes[1, 0].set_title("F1 Score by Class Weight")
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Training Curves برای بهترین
    best_name = results_df.sort_values("Best_F1", ascending=False).iloc[0]["Class_Weight_Name"]
    best_hist = histories[best_name]
    axes[1, 1].plot(best_hist.history["loss"], "b-", label="Train Loss")
    axes[1, 1].plot(best_hist.history["val_loss"], "r-", label="Val Loss")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title(f"Best Model Loss: {best_name}")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "14b_class_weight_comparison.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 14b_class_weight_comparison.png")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 14b: TRAINING - FIXED (Class Weight Adjustment)")
    log("=" * 70)
    
    # بارگذاری Hyperparameters
    with open(MODELS_DIR / "best_hyperparameters.json", "r") as f:
        params = json.load(f)
    
    log(f"\n  Hyperparameters (from Phase 13):")
    log(f"    LR: {params['Learning_Rate']}")
    log(f"    Batch: {params['Batch_Size']}")
    log(f"    Dropout: {params['Dropout_1']}/{params['Dropout_2']}/{params['Dropout_3']}")
    log(f"    Hidden: {params['Hidden_1']}/{params['Hidden_2']}/{params['Hidden_3']}")
    
    # بارگذاری داده
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    val_df = pd.read_csv(SPLITS_DIR / "validation.csv")
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    
    features = [c for c in feature_df["Feature"].tolist() 
                if c in train_df.columns and c not in LEAKY_CLF]
    
    X_train = train_df[features].fillna(0).values
    y_train = train_df[TARGET_ANOMALY].values.astype(float)
    X_val = val_df[features].fillna(0).values
    y_val = val_df[TARGET_ANOMALY].values.astype(float)
    
    log(f"\n  Data:")
    log(f"    X_train: {X_train.shape}")
    log(f"    X_val: {X_val.shape}")
    log(f"    Anomaly rate train: {y_train.mean()*100:.4f}%")
    log(f"    Anomaly rate val: {y_val.mean()*100:.4f}%")
    
    # تست Class Weightهای مختلف
    results_df, histories = test_class_weights(X_train, y_train, X_val, y_val, params)
    results_df.to_csv(TABLES_DIR / "14b_class_weight_results.csv", index=False)
    log(f"\n  [OK] Saved: 14b_class_weight_results.csv")
    
    log(f"\n  Class Weight Ranking (by F1):")
    for _, row in results_df.sort_values("Best_F1", ascending=False).iterrows():
        log(f"    {row['Class_Weight_Name']}: F1={row['Best_F1']:.6f}, "
            f"Precision={row['Best_Val_Precision']:.6f}, Recall={row['Best_Val_Recall']:.6f}")
    
    visualize_results(results_df, histories)
    
    # ============================================================
    # TRAIN FINAL MODEL WITH BEST CLASS WEIGHT
    # ============================================================
    best_row = results_df.sort_values("Best_F1", ascending=False).iloc[0]
    best_name = best_row["Class_Weight_Name"]
    best_cw_value = best_row["Class_Weight_Value"]
    
    log(f"\n  [BEST] Class Weight: {best_name}")
    log(f"  Value: {best_cw_value}")
    log(f"  F1: {best_row['Best_F1']:.6f}")
    
    # تبدیل string به dict
    if best_cw_value == "None":
        best_cw = None
    else:
        # استخراج از دیکشنری
        import ast
        best_cw = ast.literal_eval(best_cw_value)
    
    log(f"\n  Training Final Model with {best_name}...")
    
    final_model = build_model(len(features), params)
    
    final_callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=int(params["Patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=str(MODELS_DIR / "best_anomaly_model.keras"),
            monitor="val_loss",
            save_best_only=True,
            verbose=0,
        ),
    ]
    
    final_history = final_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=int(params["Batch_Size"]),
        class_weight=best_cw,
        callbacks=final_callbacks,
        verbose=1,
    )
    
    # ذخیره مدل نهایی
    final_model.save(MODELS_DIR / "anomaly_mlp_final_v2.keras")
    log(f"  [OK] Saved: anomaly_mlp_final_v2.keras")
    
    # بهترین Epoch
    best_epoch = np.argmin(final_history.history["val_loss"])
    best_val_loss = final_history.history["val_loss"][best_epoch]
    best_val_acc = final_history.history["val_accuracy"][best_epoch]
    best_val_precision = final_history.history["val_precision"][best_epoch]
    best_val_recall = final_history.history["val_recall"][best_epoch]
    best_f1 = 2 * (best_val_precision * best_val_recall) / (best_val_precision + best_val_recall + 1e-10)
    
    log(f"\n  Final Model Results:")
    log(f"    Best Epoch: {best_epoch + 1}")
    log(f"    Val Loss: {best_val_loss:.6f}")
    log(f"    Val Acc: {best_val_acc:.6f}")
    log(f"    Val Precision: {best_val_precision:.6f}")
    log(f"    Val Recall: {best_val_recall:.6f}")
    log(f"    F1: {best_f1:.6f}")
    
    # ذخیره خلاصه
    summary = {
        "Best_Class_Weight": best_name,
        "Best_Class_Weight_Value": best_cw_value,
        "Best_Epoch": best_epoch + 1,
        "Best_Val_Loss": round(best_val_loss, 6),
        "Best_Val_Acc": round(best_val_acc, 6),
        "Best_Val_Precision": round(best_val_precision, 6),
        "Best_Val_Recall": round(best_val_recall, 6),
        "Best_F1": round(best_f1, 6),
        "Total_Params": final_model.count_params(),
        "Epochs_Run": len(final_history.history["loss"]),
    }
    
    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "14b_final_training_summary.csv", index=False
    )
    log(f"  [OK] Saved: 14b_final_training_summary.csv")
    
    log("\n" + "=" * 70)
    log("PHASE 14b COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase14b_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()