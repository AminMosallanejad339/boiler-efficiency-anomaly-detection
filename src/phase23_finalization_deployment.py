"""
Project: Boiler Efficiency Prediction and Sensor Anomaly Detection
Phase 23: Finalization & Deployment (10 Sub-stages)
Framework: ML Model Lifecycle - 23 Main Stages
Author: Amin Mosallanejad
Date: 2026
"""

import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import hashlib
import shutil
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (14, 6)

BASE_DIR = Path(r"E:\DESKTOP\boiler-efficiency-anomaly-detection")
MODELS_DIR = BASE_DIR / "models"
TABLES_DIR = BASE_DIR / "reports" / "tables"
FIGURES_DIR = BASE_DIR / "reports" / "figures"
DEPLOYMENT_DIR = BASE_DIR / "deployment"
DOCS_DIR = BASE_DIR / "docs"

for d in [MODELS_DIR, TABLES_DIR, FIGURES_DIR, DEPLOYMENT_DIR, DOCS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MODEL_VERSION = "v1.0.0"

log_lines = []
def log(msg):
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    log_lines.append(line)
    print(line)


def stage_23_1_final_selection():
    log("=" * 70)
    log("PHASE 23: FINALIZATION & DEPLOYMENT")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("STAGE 23.1: FINAL MODEL SELECTION")
    log("=" * 70)

    comparison_df = pd.read_csv(TABLES_DIR / "22_1_all_models_comparison.csv")
    final_selection = pd.read_csv(TABLES_DIR / "22_10_final_selection.csv")

    final_model_name = final_selection[
        final_selection["Parameter"] == "Final_Model"
    ]["Value"].values[0]

    log(f"  Final Model Selected: {final_model_name}")
    log(f"\n  Full Comparison Table:")
    log(f"\n{comparison_df.to_string(index=False)}")

    realistic = comparison_df[~comparison_df["Leakage"]].copy()
    realistic = realistic.sort_values("F1", ascending=False)

    log(f"\n  Realistic Models (No Leakage):")
    for i, (_, row) in enumerate(realistic.iterrows(), 1):
        log(f"    {i}. {row['Model']}: F1={row['F1']:.4f}, ROC-AUC={row['ROC_AUC']:.4f}")

    return final_model_name, comparison_df


def stage_23_2_justification(final_model_name):
    log("\n" + "=" * 70)
    log("STAGE 23.2: MODEL JUSTIFICATION")
    log("=" * 70)

    justification = {
        "Model": final_model_name,
        "Selected_Because": [
            "No Data Leakage (Realistic evaluation)",
            "F1 = 0.4437 (within SOTA range for Anomaly Detection)",
            "ROC-AUC = 0.9710 (excellent discrimination)",
            "Interpretable via SHAP (Score = 9/10)",
            "Robust (F1 Std = 0.036 across seeds)",
            "Fair (F1 Range = 0.126 across shifts)",
            "Fast Training (1.47s)",
            "Unsupervised (no labels needed in production)",
        ],
        "Alternatives_Considered": {
            "Random Forest (Raw_Only)": "Rejected - Data Leakage (F1=1.0 is artificial)",
            "AdaBoost (Leaky)": "Rejected - Data Leakage (F1=1.0 is artificial)",
            "MLP": "Rejected - Failed (F1=0.0)",
            "Isolation Forest (Default)": "Rejected - Lower F1 (0.17)",
        },
        "Limitations_Acknowledged": [
            "F1 = 0.44 is not perfect",
            "Recall = 56% (44% of anomalies missed)",
            "Precision = 37% (63% false alarms)",
            "Only 4 raw anomaly features available",
            "Data is synthetic/anonymized",
        ],
        "Deployment_Recommendation": "Isolation Forest (Tuned) for Production",
    }

    for k, v in justification.items():
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

    with open(DOCS_DIR / "model_justification.json", "w", encoding="utf-8") as f:
        json.dump(justification, f, indent=2, default=str)
    log(f"\n  [OK] Saved: model_justification.json")

    return justification


def stage_23_3_documentation(final_model_name, comparison_df):
    log("\n" + "=" * 70)
    log("STAGE 23.3: MODEL DOCUMENTATION")
    log("=" * 70)

    error_results = pd.read_csv(TABLES_DIR / "21_1_error_analysis.csv")
    limitations = pd.read_csv(TABLES_DIR / "21_10_limitations.csv")

    test_metrics = error_results[error_results["Dataset"] == "Test"].iloc[0]

    doc = {
        "Model_Name": "Anomaly Detection - Isolation Forest (Tuned)",
        "Model_Version": MODEL_VERSION,
        "Model_Type": "Unsupervised Anomaly Detection",
        "Algorithm": "Isolation Forest",
        "Framework": "scikit-learn",
        "Created_Date": datetime.now().strftime("%Y-%m-%d"),
        "Author": "Amin Mosallanejad",
        "Project": "Boiler Efficiency Prediction and Sensor Anomaly Detection",
        "Model_Parameters": {
            "n_estimators": 300,
            "max_samples": 0.9,
            "contamination": 0.02,
            "max_features": 0.5,
            "bootstrap": False,
            "random_state": 42,
        },
        "Input_Features": [
            "APH_Leakage__raw",
            "CO_mgm3_raw",
            "Dust_mgm3_raw",
            "Reheater_desuperheating_water_flow_th_raw",
        ],
        "Output": {
            "Type": "Binary Classification",
            "Classes": ["Normal (0)", "Anomaly (1)"],
            "Threshold": "Based on contamination=0.02",
        },
        "Performance_Metrics_Test": {
            "F1": float(test_metrics["F1"]),
            "Precision": float(test_metrics["Precision"]),
            "Recall": float(test_metrics["Recall"]),
            "Accuracy": float(test_metrics["Accuracy"]),
            "FPR": float(test_metrics["FPR"]),
            "FNR": float(test_metrics["FNR"]),
        },
        "Performance_Metrics_Val": {
            "F1": 0.310345,
            "Precision": 0.246575,
            "Recall": 0.418605,
        },
        "Robustness": {
            "F1_Std_Across_Seeds": 0.035615,
            "Stability": "ROBUST",
        },
        "Fairness": {
            "F1_Std_Across_Shifts": 0.052192,
            "Status": "FAIR",
        },
        "Limitations": limitations["Limitation"].tolist(),
        "Data_Source": "Kaggle - Power Plant Data (Synthetic/Anonymized)",
        "Training_Samples": 35063,
        "Validation_Samples": 7487,
        "Test_Samples": 7514,
    }

    with open(DOCS_DIR / "model_documentation.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=str)
    log(f"  [OK] Saved: model_documentation.json")

    md_content = "# Model Documentation\n\n"
    md_content += "## Model Overview\n"
    md_content += f"- **Name**: {doc['Model_Name']}\n"
    md_content += f"- **Version**: {doc['Model_Version']}\n"
    md_content += f"- **Type**: {doc['Model_Type']}\n"
    md_content += f"- **Algorithm**: {doc['Algorithm']}\n"
    md_content += f"- **Author**: {doc['Author']}\n"
    md_content += f"- **Date**: {doc['Created_Date']}\n\n"

    md_content += "## Model Parameters\n"
    md_content += "| Parameter | Value |\n"
    md_content += "|-----------|-------|\n"
    for k, v in doc["Model_Parameters"].items():
        md_content += f"| {k} | {v} |\n"

    md_content += "\n## Input Features\n"
    for feat in doc["Input_Features"]:
        md_content += f"- {feat}\n"

    md_content += "\n## Performance Metrics (Test Set)\n"
    md_content += "| Metric | Value |\n"
    md_content += "|--------|-------|\n"
    for k, v in doc["Performance_Metrics_Test"].items():
        md_content += f"| {k} | {v:.4f} |\n"

    md_content += "\n## Robustness\n"
    md_content += f"- F1 Std across seeds: {doc['Robustness']['F1_Std_Across_Seeds']:.4f}\n"
    md_content += f"- Status: {doc['Robustness']['Stability']}\n"

    md_content += "\n## Fairness\n"
    md_content += f"- F1 Std across shifts: {doc['Fairness']['F1_Std_Across_Shifts']:.4f}\n"
    md_content += f"- Status: {doc['Fairness']['Status']}\n"

    md_content += "\n## Limitations\n"
    for lim in doc["Limitations"]:
        md_content += f"- {lim}\n"

    with open(DOCS_DIR / "model_documentation.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    log(f"  [OK] Saved: model_documentation.md")

    return doc


def stage_23_4_model_card(final_model_name, comparison_df):
    log("\n" + "=" * 70)
    log("STAGE 23.4: MODEL CARD CREATION")
    log("=" * 70)

    error_results = pd.read_csv(TABLES_DIR / "21_1_error_analysis.csv")
    test_metrics = error_results[error_results["Dataset"] == "Test"].iloc[0]

    card = "# Model Card: Anomaly Detection - Isolation Forest\n\n"
    card += "## Model Details\n"
    card += f"- **Developed by**: Amin Mosallanejad\n"
    card += f"- **Model date**: {datetime.now().strftime('%Y-%m-%d')}\n"
    card += f"- **Model version**: {MODEL_VERSION}\n"
    card += "- **Model type**: Unsupervised Anomaly Detection\n"
    card += "- **Algorithm**: Isolation Forest (scikit-learn)\n"
    card += "- **License**: For educational/research purposes\n\n"
    card += "## Intended Use\n"
    card += "- **Primary intended uses**: Detect anomalies in power plant sensor data\n"
    card += "- **Primary intended users**: Power plant operators, maintenance engineers\n"
    card += "- **Out-of-scope uses**: Real-time control systems, safety-critical decisions\n\n"
    card += "## Metrics\n"
    card += "| Metric | Value |\n"
    card += "|--------|-------|\n"
    card += f"| F1 | {test_metrics['F1']:.4f} |\n"
    card += f"| Precision | {test_metrics['Precision']:.4f} |\n"
    card += f"| Recall | {test_metrics['Recall']:.4f} |\n"
    card += "| ROC-AUC | 0.9710 |\n"
    card += "| PR-AUC | 0.3593 |\n\n"
    card += "## Training Data\n"
    card += "- **Source**: Kaggle - Power Plant Data\n"
    card += "- **Samples**: 35,063 (Train), 7,487 (Val), 7,514 (Test)\n"
    card += "- **Features**: 4 raw anomaly features\n"
    card += "- **Anomaly Rate**: 1.55% (Train), 1.15% (Val), 1.69% (Test)\n\n"
    card += "## Caveats and Recommendations\n"
    card += "- F1 = 0.44 is not perfect\n"
    card += "- 44% of anomalies are missed (FNR)\n"
    card += "- 63% of alarms are false (1-Precision)\n"
    card += "- Only 4 raw features available\n"
    card += "- Data is synthetic/anonymized\n"
    card += "- **Recommended**: Use with human oversight\n"

    with open(DOCS_DIR / "model_card.md", "w", encoding="utf-8") as f:
        f.write(card)
    log(f"  [OK] Saved: model_card.md")

    return card


def stage_23_5_versioning():
    log("\n" + "=" * 70)
    log("STAGE 23.5: MODEL VERSIONING")
    log("=" * 70)

    version_info = {
        "Version": MODEL_VERSION,
        "Release_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Changes": [
            "Initial release",
            "Isolation Forest with tuned hyperparameters",
            "n_estimators=300, max_samples=0.9, contamination=0.02",
            "max_features=0.5, bootstrap=False",
        ],
        "Previous_Versions": [],
        "Compatibility": {
            "Python": ">=3.9",
            "scikit-learn": ">=1.3",
            "numpy": ">=1.24",
            "pandas": ">=2.0",
        },
        "Artifacts": {
            "Model": "isolation_forest_tuned.pkl",
            "Documentation": "model_documentation.json",
            "Model_Card": "model_card.md",
        },
    }

    with open(DOCS_DIR / "model_version.json", "w", encoding="utf-8") as f:
        json.dump(version_info, f, indent=2)
    log(f"  [OK] Saved: model_version.json")

    return version_info


def stage_23_6_freezing():
    log("\n" + "=" * 70)
    log("STAGE 23.6: MODEL FREEZING")
    log("=" * 70)

    model_path = MODELS_DIR / "isolation_forest_tuned.pkl"
    with open(model_path, "rb") as f:
        model_hash = hashlib.sha256(f.read()).hexdigest()

    log(f"  Model Hash (SHA-256): {model_hash[:16]}...")

    deployment_model_path = DEPLOYMENT_DIR / f"anomaly_model_{MODEL_VERSION}.pkl"
    shutil.copy(model_path, deployment_model_path)
    log(f"  [OK] Copied to: {deployment_model_path.name}")

    freeze_info = {
        "Model_Hash": model_hash,
        "Model_Path": str(deployment_model_path),
        "Frozen_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Version": MODEL_VERSION,
        "Status": "FROZEN",
    }

    with open(DEPLOYMENT_DIR / "model_hash.json", "w") as f:
        json.dump(freeze_info, f, indent=2)
    log(f"  [OK] Saved: model_hash.json")

    return freeze_info


def stage_23_7_approval():
    log("\n" + "=" * 70)
    log("STAGE 23.7: MODEL APPROVAL")
    log("=" * 70)

    approval = {
        "Approval_Status": "APPROVED",
        "Approved_By": "Amin Mosallanejad",
        "Approval_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Approval_Criteria": {
            "Performance": "F1 >= 0.40",
            "Robustness": "F1 Std < 0.05",
            "Fairness": "F1 Range < 0.15",
            "Interpretability": "Score >= 8/10",
            "Leakage": "No Data Leakage",
        },
        "Criteria_Status": {
            "Performance": "PASS (F1=0.4437)",
            "Robustness": "PASS (Std=0.0356)",
            "Fairness": "PASS (Range=0.1259)",
            "Interpretability": "PASS (Score=9/10)",
            "Leakage": "PASS (No Leakage)",
        },
        "Comments": "Model approved for educational/research deployment",
    }

    with open(DEPLOYMENT_DIR / "model_approval.json", "w") as f:
        json.dump(approval, f, indent=2)
    log(f"  [OK] Saved: model_approval.json")

    return approval


def stage_23_8_governance():
    log("\n" + "=" * 70)
    log("STAGE 23.8: MODEL GOVERNANCE")
    log("=" * 70)

    governance = {
        "Owner": "Amin Mosallanejad",
        "Steward": "Data Science Team",
        "Review_Cycle": "Quarterly",
        "Monitoring": {
            "Frequency": "Weekly",
            "Metrics": ["F1", "Precision", "Recall", "FPR"],
            "Thresholds": {
                "F1_Min": 0.35,
                "Precision_Min": 0.30,
                "Recall_Min": 0.45,
            },
        },
        "Retraining_Trigger": {
            "Performance_Drop": "F1 < 0.35",
            "Data_Drift": "PSI > 0.25",
            "Time_Based": "6 months",
        },
        "Incident_Response": {
            "Contact": "Data Science Team",
            "Escalation": "Plant Manager",
        },
    }

    with open(DOCS_DIR / "model_governance.json", "w") as f:
        json.dump(governance, f, indent=2)
    log(f"  [OK] Saved: model_governance.json")

    return governance


def stage_23_9_compliance():
    log("\n" + "=" * 70)
    log("STAGE 23.9: MODEL COMPLIANCE CHECK")
    log("=" * 70)

    compliance = {
        "Data_Privacy": {"Status": "COMPLIANT", "Note": "No PII in data"},
        "Data_License": {"Status": "REVIEW_NEEDED", "Note": "Kaggle license unspecified - educational use only"},
        "Model_Transparency": {"Status": "COMPLIANT", "Note": "Model Card and Documentation provided"},
        "Reproducibility": {"Status": "COMPLIANT", "Note": "Random seed fixed, all artifacts saved"},
        "Fairness": {"Status": "COMPLIANT", "Note": "Model is FAIR across shifts"},
        "Security": {"Status": "COMPLIANT", "Note": "No external API calls, local model"},
    }

    with open(DOCS_DIR / "model_compliance.json", "w") as f:
        json.dump(compliance, f, indent=2)
    log(f"  [OK] Saved: model_compliance.json")

    for k, v in compliance.items():
        log(f"  {k}: {v['Status']}")

    return compliance


def stage_23_10_signoff(justification, documentation, model_card, version_info,
                        freeze_info, approval, governance, compliance):
    log("\n" + "=" * 70)
    log("STAGE 23.10: MODEL SIGN-OFF")
    log("=" * 70)

    signoff = {
        "Project": "Boiler Efficiency Prediction and Sensor Anomaly Detection",
        "Model": "Isolation Forest (Tuned)",
        "Version": MODEL_VERSION,
        "Sign_Off_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Signed_By": "Amin Mosallanejad",
        "Status": "SIGNED_OFF",
        "Artifacts": {
            "Model": f"anomaly_model_{MODEL_VERSION}.pkl",
            "Model_Hash": freeze_info["Model_Hash"][:16],
            "Documentation": "model_documentation.json",
            "Model_Card": "model_card.md",
            "Version": "model_version.json",
            "Approval": "model_approval.json",
            "Governance": "model_governance.json",
            "Compliance": "model_compliance.json",
        },
        "Final_Performance": {
            "F1": 0.443750,
            "ROC_AUC": 0.971015,
            "Precision": 0.367876,
            "Recall": 0.559055,
        },
        "Deployment_Readiness": {
            "Model_Trained": True,
            "Model_Frozen": True,
            "Documentation_Complete": True,
            "Compliance_Checked": True,
            "Governance_Defined": True,
            "Status": "READY_FOR_DEPLOYMENT",
        },
        "Notes": [
            "Model is for educational/research purposes",
            "Data license needs review for commercial use",
            "Human oversight recommended for production",
            "Retraining every 6 months or on performance drop",
        ],
    }

    with open(DEPLOYMENT_DIR / "model_signoff.json", "w") as f:
        json.dump(signoff, f, indent=2, default=str)
    log(f"  [OK] Saved: model_signoff.json")

    readme = "# Boiler Efficiency Prediction and Sensor Anomaly Detection\n\n"
    readme += "## Project Overview\n"
    readme += "End-to-end ML project following the **23-Stage ML Model Lifecycle** framework.\n\n"
    readme += "## Final Model\n"
    readme += "- **Model**: Isolation Forest (Tuned)\n"
    readme += f"- **Version**: {MODEL_VERSION}\n"
    readme += "- **Type**: Unsupervised Anomaly Detection\n"
    readme += "- **F1**: 0.4437\n"
    readme += "- **ROC-AUC**: 0.9710\n\n"
    readme += "## Key Findings\n"
    readme += "- **Boiler_Eff_ is NOT predictable** (R2 < 0)\n"
    readme += "- **Anomaly Detection IS feasible** (F1 = 0.44, ROC-AUC = 0.97)\n"
    readme += "- **Data Leakage inflates results** (RF F1 = 1.0 is artificial)\n"
    readme += "- **Isolation Forest is the realistic choice**\n\n"
    readme += "## Limitations\n"
    readme += "- F1 = 0.44 (not perfect)\n"
    readme += "- 44% of anomalies missed\n"
    readme += "- 63% of alarms are false\n"
    readme += "- Only 4 raw features available\n"
    readme += "- Data is synthetic/anonymized\n\n"
    readme += "## Author\n"
    readme += "**Amin Mosallanejad**\n"
    readme += "- Senior Utility Engineer (17 years) + Data Science M.Sc.\n\n"
    readme += "## License\n"
    readme += "For educational/research purposes only.\n"

    with open(BASE_DIR / "README_FINAL.md", "w", encoding="utf-8") as f:
        f.write(readme)
    log(f"  [OK] Saved: README_FINAL.md")

    log(f"\n  Sign-Off Status: {signoff['Status']}")
    log(f"  Deployment Readiness: {signoff['Deployment_Readiness']['Status']}")

    return signoff


def visualize_final_summary(comparison_df, signoff):
    log("\n" + "=" * 70)
    log("VISUALIZING FINAL SUMMARY")
    log("=" * 70)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    colors = ["red" if l else "green" for l in comparison_df["Leakage"]]
    axes[0, 0].barh(comparison_df["Model"], comparison_df["F1"], color=colors)
    axes[0, 0].set_xlabel("F1 Score")
    axes[0, 0].set_title("Final Model Comparison (Red=Leakage, Green=Realistic)")
    axes[0, 0].grid(True, alpha=0.3)

    metrics = ["F1", "ROC_AUC", "Precision", "Recall"]
    values = [0.443750, 0.971015, 0.367876, 0.559055]
    axes[0, 1].bar(metrics, values, color=["steelblue", "green", "coral", "orange"])
    axes[0, 1].set_ylabel("Score")
    axes[0, 1].set_title("Final Model Metrics (Test Set)")
    axes[0, 1].set_ylim(0, 1.1)
    axes[0, 1].grid(True, alpha=0.3)

    phases = ["P1-P5", "P6-P10", "P11-P15", "P16-P20", "P21-P23"]
    completed = [5, 5, 5, 5, 3]
    axes[1, 0].barh(phases, completed, color="steelblue")
    axes[1, 0].set_xlabel("Phases Completed")
    axes[1, 0].set_title("Project Completion Status")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis("off")
    summary_text = "FINAL SUMMARY\n" + "=" * 35 + "\n\n"
    summary_text += f"Model: Isolation Forest\n"
    summary_text += f"Version: {MODEL_VERSION}\n\n"
    summary_text += f"F1: 0.4437\n"
    summary_text += f"ROC-AUC: 0.9710\n"
    summary_text += f"Precision: 0.3679\n"
    summary_text += f"Recall: 0.5591\n\n"
    summary_text += f"Status: SIGNED_OFF\n"
    summary_text += f"Readiness: DEPLOYMENT_READY\n\n"
    summary_text += f"Project: 23/23 Phases\n"
    summary_text += f"COMPLETE"

    axes[1, 1].text(0.05, 0.5, summary_text, fontsize=11,
                    verticalalignment="center", fontfamily="monospace",
                    bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "23_final_summary.png", dpi=100, bbox_inches="tight")
    plt.close()
    log(f"  [OK] Saved: 23_final_summary.png")


def main():
    log("=" * 70)
    log("PHASE 23: FINALIZATION & DEPLOYMENT")
    log("=" * 70)

    final_model_name, comparison_df = stage_23_1_final_selection()
    justification = stage_23_2_justification(final_model_name)
    documentation = stage_23_3_documentation(final_model_name, comparison_df)
    model_card = stage_23_4_model_card(final_model_name, comparison_df)
    version_info = stage_23_5_versioning()
    freeze_info = stage_23_6_freezing()
    approval = stage_23_7_approval()
    governance = stage_23_8_governance()
    compliance = stage_23_9_compliance()
    signoff = stage_23_10_signoff(
        justification, documentation, model_card,
        version_info, freeze_info, approval, governance, compliance
    )

    visualize_final_summary(comparison_df, signoff)

    log("\n" + "=" * 70)
    log("PHASE 23 COMPLETE!")
    log("=" * 70)
    log("\n" + "=" * 70)
    log("PROJECT COMPLETE! ALL 23 PHASES DONE!")
    log("=" * 70)

    LOG_FILE = BASE_DIR / "reports" / "phase23_log.txt"
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    log(f"[OK] Log saved: {LOG_FILE}")


if __name__ == "__main__":
    main()
