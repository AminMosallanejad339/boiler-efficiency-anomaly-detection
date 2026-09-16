"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 09: Architecture Design (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Target: MLP for Anomaly Detection (Binary) + Binned Classification (3-class)
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
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGET_ANOMALY = "Anomaly_Label"
TARGET_BINNED = "Boiler_Eff_Class"
LEAKY_REG = ["turbine_to_boiler_eff"]
LEAKY_CLF = ["aph_effect_leak_ratio", "aph_effect_leak_ratio_log"]

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# 9.1 ARCHITECTURE DESIGN STRATEGY
# ============================================================
def stage_9_1_strategy():
    log("=" * 70)
    log("PHASE 09: ARCHITECTURE DESIGN")
    log("=" * 70)
    
    strategy = {
        "Architecture Type": "Multi-Layer Perceptron (MLP)",
        "Purpose": [
            "Compare with tree-based models (AdaBoost)",
            "Test if Neural Network can learn patterns",
            "Provide a Deep Learning baseline",
        ],
        "Tasks": {
            "Task 1": "Anomaly Detection (Binary Classification)",
            "Task 2": "Binned Classification (3-class)",
        },
        "Design Principles": [
            "Progressive layer reduction (128 -> 64 -> 32)",
            "Batch Normalization for stable training",
            "Dropout for regularization",
            "ReLU activation for hidden layers",
            "Sigmoid/Softmax for output",
        ],
        "Framework": "TensorFlow / Keras",
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
# 9.2 LAYER DESIGN
# ============================================================
def stage_9_2_layer_design():
    log("\n" + "=" * 70)
    log("STAGE 9.2: LAYER DESIGN")
    log("=" * 70)
    
    layers_config = {
        "Input Layer": {
            "Size": "n_features (78 or 79)",
            "Description": "One neuron per feature",
        },
        "Hidden Layer 1": {
            "Neurons": 128,
            "Activation": "ReLU",
            "Regularization": "L2 (1e-4)",
            "Rationale": "Wide layer to capture complex patterns",
        },
        "Hidden Layer 2": {
            "Neurons": 64,
            "Activation": "ReLU",
            "Regularization": "L2 (1e-4)",
            "Rationale": "Reduce dimensionality progressively",
        },
        "Hidden Layer 3": {
            "Neurons": 32,
            "Activation": "ReLU",
            "Regularization": "L2 (1e-4)",
            "Rationale": "Further compression before output",
        },
        "Output Layer (Anomaly)": {
            "Neurons": 1,
            "Activation": "Sigmoid",
            "Rationale": "Binary classification",
        },
        "Output Layer (Binned)": {
            "Neurons": 3,
            "Activation": "Softmax",
            "Rationale": "Multi-class classification",
        },
    }
    
    for layer, config in layers_config.items():
        log(f"  {layer}:")
        for k, v in config.items():
            log(f"    {k}: {v}")
    
    return layers_config

# ============================================================
# 9.3 BLOCK DESIGN
# ============================================================
def stage_9_3_block_design():
    log("\n" + "=" * 70)
    log("STAGE 9.3: BLOCK DESIGN")
    log("=" * 70)
    
    blocks = {
        "Input Block": {
            "Components": ["Input", "StandardScaler (external)"],
            "Purpose": "Feature normalization",
        },
        "Feature Extraction Block": {
            "Components": [
                "Dense(128) + ReLU",
                "BatchNormalization",
                "Dropout(0.3)",
            ],
            "Purpose": "Extract high-level features",
        },
        "Compression Block 1": {
            "Components": [
                "Dense(64) + ReLU",
                "BatchNormalization",
                "Dropout(0.3)",
            ],
            "Purpose": "Compress features",
        },
        "Compression Block 2": {
            "Components": [
                "Dense(32) + ReLU",
                "BatchNormalization",
                "Dropout(0.2)",
            ],
            "Purpose": "Further compression",
        },
        "Output Block": {
            "Components": [
                "Dense(1 or 3)",
                "Sigmoid or Softmax",
            ],
            "Purpose": "Final prediction",
        },
    }
    
    for block, config in blocks.items():
        log(f"  {block}:")
        for k, v in config.items():
            if isinstance(v, list):
                log(f"    {k}:")
                for item in v:
                    log(f"      - {item}")
            else:
                log(f"    {k}: {v}")
    
    return blocks

# ============================================================
# 9.4 CONNECTION DESIGN
# ============================================================
def stage_9_4_connection_design():
    log("\n" + "=" * 70)
    log("STAGE 9.4: CONNECTION DESIGN")
    log("=" * 70)
    
    connection = {
        "Type": "Sequential (Fully Connected)",
        "Data Flow": "Input -> Block1 -> Block2 -> Block3 -> Output",
        "Skip Connections": "None (not needed for tabular data)",
        "Rationale": [
            "Tabular data benefits from simple feedforward",
            "Skip connections add complexity without benefit",
            "Residual connections are for deep CNNs",
        ],
    }
    
    for k, v in connection.items():
        if isinstance(v, list):
            log(f"  {k}:")
            for item in v:
                log(f"    - {item}")
        else:
            log(f"  {k}: {v}")
    
    return connection

# ============================================================
# 9.5 ACTIVATION SELECTION
# ============================================================
def stage_9_5_activation_selection():
    log("\n" + "=" * 70)
    log("STAGE 9.5: ACTIVATION SELECTION")
    log("=" * 70)
    
    activations = {
        "Hidden Layers": {
            "Activation": "ReLU",
            "Formula": "f(x) = max(0, x)",
            "Rationale": [
                "Avoids vanishing gradient",
                "Computationally efficient",
                "Standard for MLPs",
            ],
        },
        "Output (Anomaly)": {
            "Activation": "Sigmoid",
            "Formula": "f(x) = 1 / (1 + exp(-x))",
            "Rationale": "Binary classification, outputs probability",
        },
        "Output (Binned)": {
            "Activation": "Softmax",
            "Formula": "f(x_i) = exp(x_i) / sum(exp(x_j))",
            "Rationale": "Multi-class, outputs probability distribution",
        },
        "Alternatives Considered": {
            "LeakyReLU": "Rejected: ReLU works well for this size",
            "ELU": "Rejected: Slower than ReLU",
            "Tanh": "Rejected: Vanishing gradient",
            "Swish": "Rejected: Marginal improvement, more computation",
        },
    }
    
    for layer, config in activations.items():
        log(f"  {layer}:")
        for k, v in config.items():
            if isinstance(v, list):
                log(f"    {k}:")
                for item in v:
                    log(f"      - {item}")
            else:
                log(f"    {k}: {v}")
    
    return activations

# ============================================================
# 9.6 NORMALIZATION SELECTION
# ============================================================
def stage_9_6_normalization_selection():
    log("\n" + "=" * 70)
    log("STAGE 9.6: NORMALIZATION SELECTION")
    log("=" * 70)
    
    normalization = {
        "External Normalization": {
            "Method": "RobustScaler",
            "Applied": "Before feeding to NN",
            "Rationale": "Handles outliers, matches Phase 06 pipeline",
        },
        "Internal Normalization": {
            "Method": "BatchNormalization",
            "Applied": "After each Dense layer",
            "Rationale": [
                "Stabilizes training",
                "Allows higher learning rates",
                "Reduces internal covariate shift",
            ],
        },
        "Alternatives Considered": {
            "LayerNormalization": "Rejected: Better for RNNs/Transformers",
            "GroupNormalization": "Rejected: Better for small batches",
            "InstanceNormalization": "Rejected: For style transfer",
        },
    }
    
    for layer, config in normalization.items():
        log(f"  {layer}:")
        for k, v in config.items():
            if isinstance(v, list):
                log(f"    {k}:")
                for item in v:
                    log(f"      - {item}")
            else:
                log(f"    {k}: {v}")
    
    return normalization

# ============================================================
# 9.7 DROPOUT DESIGN
# ============================================================
def stage_9_7_dropout_design():
    log("\n" + "=" * 70)
    log("STAGE 9.7: DROPOUT DESIGN")
    log("=" * 70)
    
    dropout = {
        "Layer 1 Dropout": {
            "Rate": 0.3,
            "Rationale": "High dropout for wide layer",
        },
        "Layer 2 Dropout": {
            "Rate": 0.3,
            "Rationale": "Maintain regularization",
        },
        "Layer 3 Dropout": {
            "Rate": 0.2,
            "Rationale": "Lower dropout before output",
        },
        "Output Layer": {
            "Rate": 0.0,
            "Rationale": "No dropout on output",
        },
        "Total Regularization": {
            "L2": "1e-4 on all Dense layers",
            "Dropout": "0.3, 0.3, 0.2",
            "EarlyStopping": "patience=10",
            "ReduceLROnPlateau": "patience=5, factor=0.5",
        },
    }
    
    for layer, config in dropout.items():
        log(f"  {layer}:")
        for k, v in config.items():
            log(f"    {k}: {v}")
    
    return dropout

# ============================================================
# 9.8 EMBEDDING DESIGN
# ============================================================
def stage_9_8_embedding_design():
    log("\n" + "=" * 70)
    log("STAGE 9.8: EMBEDDING DESIGN")
    log("=" * 70)
    
    log(f"  [INFO] No categorical features requiring embeddings")
    log(f"  [INFO] All features are numeric (continuous)")
    log(f"  [NOTE] If categorical features added later:")
    log(f"    - Use Embedding layers for high-cardinality features")
    log(f"    - Use OneHotEncoder for low-cardinality features")
    
    return {}

# ============================================================
# 9.9 ATTENTION DESIGN
# ============================================================
def stage_9_9_attention_design():
    log("\n" + "=" * 70)
    log("STAGE 9.9: ATTENTION DESIGN")
    log("=" * 70)
    
    log(f"  [INFO] Attention not applicable for MLP on tabular data")
    log(f"  [NOTE] Attention is for:")
    log(f"    - Sequential data (Transformers, RNNs)")
    log(f"    - Image data (Vision Transformers)")
    log(f"    - Graph data (GAT)")
    log(f"  [DECISION] Skip attention for this architecture")
    
    return {}

# ============================================================
# 9.10 OUTPUT HEAD DESIGN
# ============================================================
def stage_9_10_output_head_design():
    log("\n" + "=" * 70)
    log("STAGE 9.10: OUTPUT HEAD DESIGN")
    log("=" * 70)
    
    output_heads = {
        "Anomaly Detection Head": {
            "Neurons": 1,
            "Activation": "Sigmoid",
            "Loss": "Binary Crossentropy",
            "Metric": "F1-Score, Recall, Precision",
            "Class Weight": "Applied (63.57:1 imbalance)",
        },
        "Binned Classification Head": {
            "Neurons": 3,
            "Activation": "Softmax",
            "Loss": "Categorical Crossentropy",
            "Metric": "F1-Macro, Accuracy",
            "Class Weight": "Not needed (balanced)",
        },
    }
    
    for head, config in output_heads.items():
        log(f"  {head}:")
        for k, v in config.items():
            log(f"    {k}: {v}")
    
    return output_heads

# ============================================================
# BUILD MLP MODELS
# ============================================================
def build_anomaly_mlp(n_features):
    """ساخت MLP برای Anomaly Detection"""
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        
        layers.Dense(128, activation="relu", 
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(1, activation="sigmoid"),
    ], name="Anomaly_MLP")
    
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model

def build_binned_mlp(n_features, n_classes=3):
    """ساخت MLP برای Binned Classification"""
    model = models.Sequential([
        layers.Input(shape=(n_features,)),
        
        layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        
        layers.Dense(n_classes, activation="softmax"),
    ], name="Binned_MLP")
    
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ============================================================
# VISUALIZE ARCHITECTURE
# ============================================================
def visualize_architecture(n_features_anomaly, n_features_binned):
    log("\n" + "=" * 70)
    log("VISUALIZING ARCHITECTURES")
    log("=" * 70)
    
    # ساخت مدل‌ها
    anomaly_model = build_anomaly_mlp(n_features_anomaly)
    binned_model = build_binned_mlp(n_features_binned)
    
    # ذخیره خلاصه
    with open(TABLES_DIR / "09_model_summaries.txt", "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("ANOMALY DETECTION MLP\n")
        f.write("=" * 70 + "\n")
        anomaly_model.summary(print_fn=lambda x: f.write(x + "\n"))
        f.write("\n\n" + "=" * 70 + "\n")
        f.write("BINNED CLASSIFICATION MLP\n")
        f.write("=" * 70 + "\n")
        binned_model.summary(print_fn=lambda x: f.write(x + "\n"))
    
    log(f"  [OK] Saved: 09_model_summaries.txt")
    
    # چاپ خلاصه
    log(f"\n  Anomaly MLP Summary:")
    anomaly_model.summary(print_fn=lambda x: log(f"    {x}"))
    
    log(f"\n  Binned MLP Summary:")
    binned_model.summary(print_fn=lambda x: log(f"    {x}"))
    
    # نمودار معماری
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Anomaly
    anomaly_layers = [
        f"Input\n({n_features_anomaly})",
        "Dense\n128 + ReLU\n+ BN + Drop",
        "Dense\n64 + ReLU\n+ BN + Drop",
        "Dense\n32 + ReLU\n+ BN + Drop",
        "Output\n1 + Sigmoid",
    ]
    colors = ["lightblue", "steelblue", "steelblue", "steelblue", "coral"]
    for i, (layer, color) in enumerate(zip(anomaly_layers, colors)):
        axes[0].barh(i, 1, color=color, edgecolor="black")
        axes[0].text(0.5, i, layer, ha="center", va="center", fontsize=9, fontweight="bold")
    axes[0].set_yticks(range(len(anomaly_layers)))
    axes[0].set_yticklabels([])
    axes[0].set_xlim(0, 1)
    axes[0].set_xticks([])
    axes[0].set_title("Anomaly Detection MLP", fontsize=12)
    
    # Binned
    binned_layers = [
        f"Input\n({n_features_binned})",
        "Dense\n128 + ReLU\n+ BN + Drop",
        "Dense\n64 + ReLU\n+ BN + Drop",
        "Dense\n32 + ReLU\n+ BN + Drop",
        "Output\n3 + Softmax",
    ]
    for i, (layer, color) in enumerate(zip(binned_layers, colors)):
        axes[1].barh(i, 1, color=color, edgecolor="black")
        axes[1].text(0.5, i, layer, ha="center", va="center", fontsize=9, fontweight="bold")
    axes[1].set_yticks(range(len(binned_layers)))
    axes[1].set_yticklabels([])
    axes[1].set_xlim(0, 1)
    axes[1].set_xticks([])
    axes[1].set_title("Binned Classification MLP", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "09_architecture_design.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 09_architecture_design.png")
    
    # ذخیره مدل‌ها
    anomaly_model.save(MODELS_DIR / "anomaly_mlp_initial.keras")
    binned_model.save(MODELS_DIR / "binned_mlp_initial.keras")
    log(f"  [OK] Saved: anomaly_mlp_initial.keras")
    log(f"  [OK] Saved: binned_mlp_initial.keras")
    
    return anomaly_model, binned_model

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 09: ARCHITECTURE DESIGN")
    log("=" * 70)
    
    # 9.1
    strategy = stage_9_1_strategy()
    
    # 9.2
    layers_config = stage_9_2_layer_design()
    
    # 9.3
    blocks = stage_9_3_block_design()
    
    # 9.4
    connection = stage_9_4_connection_design()
    
    # 9.5
    activations = stage_9_5_activation_selection()
    
    # 9.6
    normalization = stage_9_6_normalization_selection()
    
    # 9.7
    dropout = stage_9_7_dropout_design()
    
    # 9.8
    stage_9_8_embedding_design()
    
    # 9.9
    stage_9_9_attention_design()
    
    # 9.10
    output_heads = stage_9_10_output_head_design()
    
    # تعیین تعداد ویژگی‌ها
    feature_df = pd.read_csv(SPLITS_DIR / "feature_columns.csv")
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    
    features_anomaly = [c for c in feature_df["Feature"].tolist() 
                        if c in train_df.columns and c not in LEAKY_CLF]
    features_binned = [c for c in feature_df["Feature"].tolist() 
                       if c in train_df.columns and c not in LEAKY_REG]
    
    log(f"\n  Features for Anomaly MLP: {len(features_anomaly)}")
    log(f"  Features for Binned MLP: {len(features_binned)}")
    
    # ساخت و ذخیره مدل‌ها
    anomaly_model, binned_model = visualize_architecture(
        len(features_anomaly), len(features_binned)
    )
    
    # ذخیره خلاصه
    summary = {
        "Architecture": "MLP",
        "Anomaly_Input": len(features_anomaly),
        "Anomaly_Output": 1,
        "Anomaly_Activation": "Sigmoid",
        "Binned_Input": len(features_binned),
        "Binned_Output": 3,
        "Binned_Activation": "Softmax",
        "Hidden_Layers": "128, 64, 32",
        "Dropout": "0.3, 0.3, 0.2",
        "BatchNorm": "Yes",
        "L2_Regularization": "1e-4",
        "Total_Params_Anomaly": anomaly_model.count_params(),
        "Total_Params_Binned": binned_model.count_params(),
    }
    
    pd.DataFrame(list(summary.items()), columns=["Metric", "Value"]).to_csv(
        TABLES_DIR / "09_architecture_summary.csv", index=False
    )
    log(f"  [OK] Saved: 09_architecture_summary.csv")
    
    log(f"\n  Architecture Summary:")
    for k, v in summary.items():
        log(f"    {k}: {v}")
    
    log("\n" + "=" * 70)
    log("PHASE 09 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase09_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()