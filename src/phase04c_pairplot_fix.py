"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 04c: Pairplot Fix
Framework: ML Model Lifecycle - 23 Main Stages
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

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
PROCESSED = BASE_DIR / "data" / "processed" / "industrial_dataset_phase03_final.csv"
FIGURES_DIR = BASE_DIR / "reports" / "figures"

TARGET = "Boiler_Eff_"
ANOMALY = "Anomaly_Label"

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)

log("=" * 70)
log("PHASE 04c: PAIRPLOT FIX")
log("=" * 70)

df = pd.read_csv(PROCESSED)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in [ANOMALY] and not c.endswith("_bin")]

# حذف Target از لیست top5
top5 = df[numeric_cols].corrwith(df[TARGET]).abs().sort_values(ascending=False).head(6).index.tolist()
top5 = [c for c in top5 if c != TARGET][:5]  # حذف Target و انتخاب 5 ویژگی

pairplot_cols = top5 + [TARGET]
pairplot_cols = list(dict.fromkeys(pairplot_cols))  # حذف تکراری‌ها

log(f"Pairplot columns (unique): {pairplot_cols}")
log(f"Total columns: {len(pairplot_cols)}")

# Pairplot
try:
    sample_df = df[pairplot_cols].sample(min(2000, len(df)), random_state=42)
    log(f"Sample shape: {sample_df.shape}")
    
    g = sns.pairplot(
        sample_df,
        diag_kind="kde",
        plot_kws={"alpha": 0.4, "s": 10},
        corner=False,
    )
    g.fig.suptitle("Pairplot - Top 5 Features + Target", y=1.02, fontsize=14)
    plt.savefig(FIGURES_DIR / "04_09_pairplot.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"[OK] Saved: 04_09_pairplot.png")
except Exception as e:
    log(f"[WARNING] Pairplot failed: {e}")

# Pairplot جداگانه برای Target vs Top 5
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, col in enumerate(top5):
    axes[i].scatter(df[col], df[TARGET], alpha=0.2, s=5, color="steelblue")
    axes[i].set_xlabel(col, fontsize=9)
    axes[i].set_ylabel(TARGET, fontsize=9)
    corr = df[col].corr(df[TARGET])
    axes[i].set_title(f"{col}\n(corr = {corr:.4f})", fontsize=10)
    axes[i].grid(True, alpha=0.3)

# آخرین subplot: توزیع Target
axes[5].hist(df[TARGET], bins=50, color="coral", edgecolor="black", alpha=0.7)
axes[5].set_xlabel(TARGET, fontsize=9)
axes[5].set_ylabel("Frequency", fontsize=9)
axes[5].set_title(f"Distribution of {TARGET}", fontsize=10)
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / "04_09_target_vs_top5.png", dpi=100, bbox_inches="tight")
plt.close()
log(f"[OK] Saved: 04_09_target_vs_top5.png")

LOG_FILE = BASE_DIR / "reports" / "phase04c_log.txt"
with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(log_lines))
log(f"[OK] Log saved: {LOG_FILE}")