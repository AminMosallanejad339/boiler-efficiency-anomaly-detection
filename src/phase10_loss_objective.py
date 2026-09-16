"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 10: Loss / Objective Function (10 Sub-stages)
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
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from pathlib import Path
from datetime import datetime

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

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

# ============================================================
# 10.1 LOSS FUNCTION SELECTION
# ============================================================
def stage_10_1_loss_selection():
    log("=" * 70)
    log("PHASE 10: LOSS / OBJECTIVE FUNCTION")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("STAGE 10.1: LOSS FUNCTION SELECTION")
    log("=" * 70)
    
    selection = {
        "Anomaly Detection (Binary)": {
            "Primary Loss": "Binary Crossentropy",
            "Formula": "L = -[y*log(p) + (1-y)*log(1-p)]",
            "Rationale": "Standard for binary classification",
            "Alternatives": [
                "Focal Loss (for extreme imbalance)",
                "Weighted BCE (with class weights)",
                "Hinge Loss (for SVM-like)",
            ],
        },
        "Binned Classification (3-class)": {
            "Primary Loss": "Categorical Crossentropy",
            "Formula": "L = -sum(y_i * log(p_i))",
            "Rationale": "Standard for multi-class classification",
            "Alternatives": [
                "Sparse Categorical Crossentropy",
                "Kullback-Leibler Divergence",
            ],
        },
        "Regression (deprecated)": {
            "Primary Loss": "MSE",
            "Formula": "L = mean((y - y_pred)^2)",
            "Rationale": "Not used (regression failed)",
        },
    }
    
    for task, config in selection.items():
        log(f"\n  {task}:")
        for k, v in config.items():
            if isinstance(v, list):
                log(f"    {k}:")
                for item in v:
                    log(f"      - {item}")
            else:
                log(f"    {k}: {v}")
    
    return selection

# ============================================================
# 10.2 COST FUNCTION DEFINITION
# ============================================================
def stage_10_2_cost_function():
    log("\n" + "=" * 70)
    log("STAGE 10.2: COST FUNCTION DEFINITION")
    log("=" * 70)
    
    cost = {
        "Cost Function": "Average loss over all training samples",
        "Formula": "J(θ) = (1/N) * sum(L(y_i, f(x_i; θ)))",
        "Components": {
            "N": "Number of training samples",
            "L": "Per-sample loss (BCE or CE)",
            "f(x_i; θ)": "Model prediction",
            "θ": "Model parameters",
        },
        "Optimization Goal": "Minimize J(θ) w.r.t. θ",
    }
    
    for k, v in cost.items():
        if isinstance(v, dict):
            log(f"  {k}:")
            for kk, vv in v.items():
                log(f"    {kk}: {vv}")
        else:
            log(f"  {k}: {v}")
    
    return cost

# ============================================================
# 10.3 OBJECTIVE FUNCTION FORMULATION
# ============================================================
def stage_10_3_objective_formulation():
    log("\n" + "=" * 70)
    log("STAGE 10.3: OBJECTIVE FUNCTION FORMULATION")
    log("=" * 70)
    
    objective = {
        "Total Objective": "J_total = J_data + λ * J_reg",
        "Data Loss": "Binary Crossentropy or Categorical Crossentropy",
        "Regularization": "L2 (weight decay) + Dropout",
        "λ (Lambda)": "1e-4 (L2 regularization strength)",
        "Formula": {
            "Anomaly": "J = BCE(y, y_pred) + λ * sum(||W||^2)",
            "Binned": "J = CE(y, y_pred) + λ * sum(||W||^2)",
        },
        "Rationale": [
            "Data loss ensures model fits training data",
            "Regularization prevents overfitting",
            "λ controls trade-off between fit and complexity",
        ],
    }
    
    for k, v in objective.items():
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
    
    return objective

# ============================================================
# 10.4 PENALTY TERM DESIGN
# ============================================================
def stage_10_4_penalty_term():
    log("\n" + "=" * 70)
    log("STAGE 10.4: PENALTY TERM DESIGN")
    log("=" * 70)
    
    penalty = {
        "Type": "L2 Regularization (Ridge)",
        "Formula": "R(W) = λ * sum(W^2)",
        "Applied To": "All Dense layers",
        "λ Value": "1e-4",
        "Rationale": [
            "Penalizes large weights",
            "Reduces model complexity",
            "Prevents overfitting",
            "Smooth optimization landscape",
        ],
        "Alternatives": {
            "L1": "Sparse weights, feature selection",
            "Elastic Net": "Combines L1 and L2",
            "No Penalty": "Higher overfitting risk",
        },
    }
    
    for k, v in penalty.items():
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
    
    return penalty

# ============================================================
# 10.5 REGULARIZATION TERM DESIGN
# ============================================================
def stage_10_5_regularization_term():
    log("\n" + "=" * 70)
    log("STAGE 10.5: REGULARIZATION TERM DESIGN")
    log("=" * 70)
    
    regularization = {
        "Weight Regularization": {
            "Type": "L2",
            "λ": "1e-4",
            "Applied": "Dense layers",
        },
        "Activation Regularization": {
            "Type": "Dropout",
            "Rates": "0.3, 0.3, 0.2",
            "Applied": "After BatchNorm",
        },
        "Early Stopping": {
            "Monitor": "val_loss",
            "Patience": 10,
            "Restore Best": True,
        },
        "Learning Rate Schedule": {
            "Type": "ReduceLROnPlateau",
            "Monitor": "val_loss",
            "Patience": 5,
            "Factor": 0.5,
        },
    }
    
    for k, v in regularization.items():
        log(f"  {k}:")
        for kk, vv in v.items():
            log(f"    {kk}: {vv}")
    
    return regularization

# ============================================================
# 10.6 DIVERGENCE SELECTION
# ============================================================
def stage_10_6_divergence_selection():
    log("\n" + "=" * 70)
    log("STAGE 10.6: DIVERGENCE SELECTION")
    log("=" * 70)
    
    divergence = {
        "Anomaly Detection": {
            "Divergence": "Binary Crossentropy",
            "Origin": "Bernoulli distribution",
            "Formula": "KL(p || q) where p=true, q=predicted",
        },
        "Binned Classification": {
            "Divergence": "Categorical Crossentropy",
            "Origin": "Categorical distribution",
            "Formula": "KL(p || q) where p=true, q=predicted",
        },
        "General": {
            "Note": "Crossentropy = KL divergence + entropy of true distribution",
            "Since entropy is constant": "Minimizing CE = minimizing KL",
        },
    }
    
    for k, v in divergence.items():
        log(f"  {k}:")
        for kk, vv in v.items():
            log(f"    {kk}: {vv}")
    
    return divergence

# ============================================================
# 10.7 RISK FORMULATION
# ============================================================
def stage_10_7_risk_formulation():
    log("\n" + "=" * 70)
    log("STAGE 10.7: RISK FORMULATION")
    log("=" * 70)
    
    risk = {
        "True Risk": "R(θ) = E[L(y, f(x; θ))] over true distribution P(x,y)",
        "Empirical Risk": "R_emp(θ) = (1/N) * sum(L(y_i, f(x_i; θ)))",
        "Regularized Risk": "R_reg(θ) = R_emp(θ) + λ * R(W)",
        "Goal": "Minimize R_reg(θ) w.r.t. θ",
        "Generalization": "True risk ≈ Empirical risk + Regularization",
        "Trade-off": "Bias-Variance Trade-off controlled by λ",
    }
    
    for k, v in risk.items():
        log(f"  {k}: {v}")
    
    return risk

# ============================================================
# 10.8 EMPIRICAL RISK DEFINITION
# ============================================================
def stage_10_8_empirical_risk():
    log("\n" + "=" * 70)
    log("STAGE 10.8: EMPIRICAL RISK DEFINITION")
    log("=" * 70)
    
    empirical = {
        "Anomaly Detection": {
            "Formula": "R_emp = -(1/N) * sum[y*log(p) + (1-y)*log(1-p)]",
            "Class Weight": "w1 = 63.57, w0 = 1.0 (for imbalance)",
            "Weighted Formula": "R_emp = -(1/N) * sum[w_y * (y*log(p) + (1-y)*log(1-p))]",
        },
        "Binned Classification": {
            "Formula": "R_emp = -(1/N) * sum[sum(y_i * log(p_i))]",
            "Class Weight": "None (balanced classes)",
        },
        "Batch Processing": {
            "Batch Size": "256 (to be tuned in Phase 13)",
            "Mini-batch Risk": "Average over batch samples",
        },
    }
    
    for task, config in empirical.items():
        log(f"  {task}:")
        for k, v in config.items():
            log(f"    {k}: {v}")
    
    return empirical

# ============================================================
# 10.9 CUSTOM LOSS DESIGN
# ============================================================
def stage_10_9_custom_loss():
    log("\n" + "=" * 70)
    log("STAGE 10.9: CUSTOM LOSS DESIGN")
    log("=" * 70)
    
    log("  Designing Focal Loss for Anomaly Detection...")
    log("  (Focal Loss helps with extreme class imbalance)")
    
    # محاسبه Class Weight
    train_df = pd.read_csv(SPLITS_DIR / "train.csv")
    class_dist = train_df[TARGET_ANOMALY].value_counts()
    n_neg = class_dist.get(0, 0)
    n_pos = class_dist.get(1, 0)
    imbalance_ratio = n_neg / n_pos
    
    log(f"\n  Class Distribution:")
    log(f"    Negative (Normal): {n_neg}")
    log(f"    Positive (Anomaly): {n_pos}")
    log(f"    Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    class_weight = {
        0: 1.0,
        1: imbalance_ratio,
    }
    log(f"\n  Class Weights: {class_weight}")
    
    # محاسبه Focal Loss parameters
    log(f"\n  Focal Loss Parameters:")
    log(f"    Alpha (balance factor): 0.25")
    log(f"    Gamma (focusing parameter): 2.0")
    log(f"    Rationale: Focus on hard examples")
    
    # ذخیره
    custom_loss_config = {
        "Binary_Crossentropy": {
            "Type": "Standard",
            "Class_Weight": f"{{0: 1.0, 1: {imbalance_ratio:.2f}}}",
            "Rationale": "Simple, effective for most cases",
        },
        "Focal_Loss": {
            "Type": "Custom",
            "Alpha": 0.25,
            "Gamma": 2.0,
            "Rationale": "Focus on hard-to-classify examples",
        },
        "Weighted_BCE": {
            "Type": "Weighted",
            "Class_Weight": f"{{0: 1.0, 1: {imbalance_ratio:.2f}}}",
            "Rationale": "Explicit imbalance handling",
        },
    }
    
    pd.DataFrame(custom_loss_config).T.to_csv(TABLES_DIR / "10_9_custom_loss.csv")
    log(f"\n  [OK] Saved: 10_9_custom_loss.csv")
    
    return custom_loss_config, class_weight

# ============================================================
# 10.10 LOSS WEIGHTING
# ============================================================
def stage_10_10_loss_weighting(class_weight):
    log("\n" + "=" * 70)
    log("STAGE 10.10: LOSS WEIGHTING")
    log("=" * 70)
    
    weighting = {
        "Anomaly Detection": {
            "Class Weight": class_weight,
            "Rationale": "Compensate for 63:1 imbalance",
            "Effect": "Model focuses on anomaly class",
            "Trade-off": "Higher Recall, Lower Precision",
        },
        "Binned Classification": {
            "Class Weight": "None (balanced)",
            "Rationale": "Classes already balanced (33/33/33)",
        },
        "Multi-Task Weighting": {
            "Note": "If training both tasks jointly",
            "Anomaly Weight": 0.7,
            "Binned Weight": 0.3,
            "Rationale": "Prioritize anomaly detection (main goal)",
        },
    }
    
    for task, config in weighting.items():
        log(f"  {task}:")
        for k, v in config.items():
            if isinstance(v, dict):
                log(f"    {k}:")
                for kk, vv in v.items():
                    log(f"      {kk}: {vv}")
            else:
                log(f"    {k}: {v}")
    
    return weighting

# ============================================================
# CUSTOM FOCAL LOSS IMPLEMENTATION
# ============================================================
class FocalLoss(Loss):
    """Focal Loss for Binary Classification"""
    def __init__(self, alpha=0.25, gamma=2.0, name="focal_loss"):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = gamma
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        focal = alpha_t * tf.math.pow(1 - p_t, self.gamma) * bce
        
        return tf.reduce_mean(focal)

class WeightedBinaryCrossentropy(Loss):
    """Weighted Binary Crossentropy for Imbalanced Data"""
    def __init__(self, pos_weight=1.0, name="weighted_bce"):
        super().__init__(name=name)
        self.pos_weight = pos_weight
    
    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        
        weights = y_true * self.pos_weight + (1 - y_true)
        bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
        
        return tf.reduce_mean(weights * bce)

def visualize_losses(class_weight):
    """نمودار مقایسه Loss Functions"""
    log("\n" + "=" * 70)
    log("VISUALIZING LOSS FUNCTIONS")
    log("=" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Binary Crossentropy
    p = np.linspace(0.01, 0.99, 100)
    bce_pos = -np.log(p)  # y=1
    bce_neg = -np.log(1 - p)  # y=0
    axes[0, 0].plot(p, bce_pos, label="y=1", color="coral", linewidth=2)
    axes[0, 0].plot(p, bce_neg, label="y=0", color="steelblue", linewidth=2)
    axes[0, 0].set_xlabel("Predicted Probability")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Binary Crossentropy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim(0, 5)
    
    # 2. Focal Loss (different gamma)
    for gamma in [0, 1, 2, 5]:
        p_t = np.linspace(0.01, 0.99, 100)
        focal = (1 - p_t) ** gamma * (-np.log(p_t))
        axes[0, 1].plot(p_t, focal, label=f"γ={gamma}", linewidth=2)
    axes[0, 1].set_xlabel("p_t (probability of true class)")
    axes[0, 1].set_ylabel("Focal Loss")
    axes[0, 1].set_title("Focal Loss (α=0.25)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_ylim(0, 5)
    
    # 3. Weighted BCE
    p = np.linspace(0.01, 0.99, 100)
    pos_weight = class_weight[1]
    wbce_pos = pos_weight * (-np.log(p))
    wbce_neg = -np.log(1 - p)
    axes[1, 0].plot(p, wbce_pos, label=f"y=1 (weight={pos_weight:.1f})", color="coral", linewidth=2)
    axes[1, 0].plot(p, wbce_neg, label="y=0 (weight=1.0)", color="steelblue", linewidth=2)
    axes[1, 0].set_xlabel("Predicted Probability")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].set_title(f"Weighted BCE (pos_weight={pos_weight:.1f})")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Categorical Crossentropy
    p_correct = np.linspace(0.01, 0.99, 100)
    ce = -np.log(p_correct)
    axes[1, 1].plot(p_correct, ce, color="green", linewidth=2)
    axes[1, 1].set_xlabel("Probability of Correct Class")
    axes[1, 1].set_ylabel("Loss")
    axes[1, 1].set_title("Categorical Crossentropy")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_ylim(0, 5)
    
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "10_loss_functions.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 10_loss_functions.png")
    
    # تست Custom Losses
    log(f"\n  Testing Custom Losses...")
    y_true_test = tf.constant([1.0, 0.0, 1.0, 0.0])
    y_pred_test = tf.constant([0.9, 0.1, 0.6, 0.4])
    
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    wbce = WeightedBinaryCrossentropy(pos_weight=class_weight[1])
    
    log(f"    Focal Loss: {focal(y_true_test, y_pred_test):.6f}")
    log(f"    Weighted BCE: {wbce(y_true_test, y_pred_test):.6f}")
    log(f"    Standard BCE: {tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true_test, y_pred_test)):.6f}")

# ============================================================
# MAIN
# ============================================================
def main():
    log("=" * 70)
    log("PHASE 10: LOSS / OBJECTIVE FUNCTION")
    log("=" * 70)
    
    # 10.1
    selection = stage_10_1_loss_selection()
    
    # 10.2
    cost = stage_10_2_cost_function()
    
    # 10.3
    objective = stage_10_3_objective_formulation()
    
    # 10.4
    penalty = stage_10_4_penalty_term()
    
    # 10.5
    regularization = stage_10_5_regularization_term()
    
    # 10.6
    divergence = stage_10_6_divergence_selection()
    
    # 10.7
    risk = stage_10_7_risk_formulation()
    
    # 10.8
    empirical = stage_10_8_empirical_risk()
    
    # 10.9
    custom_loss_config, class_weight = stage_10_9_custom_loss()
    
    # 10.10
    weighting = stage_10_10_loss_weighting(class_weight)
    
    # Visualization
    visualize_losses(class_weight)
    
    # ذخیره خلاصه
    summary = {
        "Anomaly_Loss": "Binary Crossentropy / Focal Loss",
        "Anomaly_Class_Weight": str(class_weight),
        "Binned_Loss": "Categorical Crossentropy",
        "Binned_Class_Weight": "None (balanced)",
        "L2_Regularization": "1e-4",
        "Dropout": "0.3, 0.3, 0.2",
        "EarlyStopping": "patience=10",
        "ReduceLROnPlateau": "patience=5, factor=0.5",
        "Focal_Alpha": 0.25,
        "Focal_Gamma": 2.0,
    }
    
    pd.DataFrame(list(summary.items()), columns=["Parameter", "Value"]).to_csv(
        TABLES_DIR / "10_loss_summary.csv", index=False
    )
    log(f"  [OK] Saved: 10_loss_summary.csv")
    
    log("\n" + "=" * 70)
    log("PHASE 10 COMPLETE!")
    log("=" * 70)
    
    LOG_FILE = BASE_DIR / "reports" / "phase10_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")

if __name__ == "__main__":
    main()