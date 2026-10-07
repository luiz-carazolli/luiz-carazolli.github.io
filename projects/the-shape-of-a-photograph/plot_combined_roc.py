#!/usr/bin/env python3
"""
Plot Combined ROC Curves: 1D Topological Descriptors (Tabular Baselines) vs. 2D Persistence Images (TopoCNN).
Loads predictions from models/topo_cnn_predictions.npz and the 6 joblib models.
Outputs figures/roc_curves_with_cnn.png.
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

OUTPUT_DIR = "figures"
MODELS_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def generate_combined_roc_plot():
    print("[*] Generating combined ROC plot...")
    plt.figure(figsize=(9.0, 7.5), dpi=300)

    colors = {
        "Logistic Regression (L2)": "#34495e",
        "Linear SVM": "#7f8c8d",
        "RBF SVM": "#2980b9",
        "Random Forest": "#27ae60",
        "XGBoost": "#d35400",
        "LightGBM": "#8e44ad",
        "TopoCNN (2D Persistence Images)": "#c0392b"
    }

    # 1. Load test set for tabular models
    test_split_path = "data/cifake_test_split.parquet"
    if os.path.exists(test_split_path):
        df_test = pd.read_parquet(test_split_path)
        X_test = df_test.drop(columns=["label"])
        y_test = df_test["label"].values

        model_files = {
            "Logistic Regression (L2)": "models/logistic_regression_l2.joblib",
            "Linear SVM": "models/linear_svm.joblib",
            "RBF SVM": "models/rbf_svm.joblib",
            "Random Forest": "models/random_forest.joblib",
            "XGBoost": "models/xgboost.joblib",
            "LightGBM": "models/lightgbm.joblib"
        }

        for m_name, m_path in model_files.items():
            if os.path.exists(m_path):
                clf = joblib.load(m_path)
                y_prob = clf.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_prob)
                auc_val = roc_auc_score(y_test, y_prob)
                plt.plot(fpr, tpr, label=f"{m_name} (1D Descriptors: AUC = {auc_val:.4f})",
                         color=colors.get(m_name, "#666666"), lw=1.8, alpha=0.75)

    # 2. Load TopoCNN test predictions
    cnn_pred_path = os.path.join(MODELS_DIR, "topo_cnn_predictions.npz")
    if os.path.exists(cnn_pred_path):
        data = np.load(cnn_pred_path)
        y_true_cnn = data["y_true"]
        y_prob_cnn = data["y_prob"]
        fpr_cnn, tpr_cnn, _ = roc_curve(y_true_cnn, y_prob_cnn)
        auc_cnn = roc_auc_score(y_true_cnn, y_prob_cnn)
        plt.plot(fpr_cnn, tpr_cnn,
                 label=f"TopoCNN (2D Persistence Images: AUC = {auc_cnn:.4f})",
                 color="#c0392b", lw=3.2, zorder=10)

    # Random chance baseline
    plt.plot([0, 1], [0, 1], color="#bdc3c7", linestyle="--", lw=1.5, label="Random Chance (AUC = 0.5000)")

    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel("False Positive Rate (1 - Specificity)", fontsize=12, fontweight="bold", labelpad=8)
    plt.ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=12, fontweight="bold", labelpad=8)
    plt.title("Receiver Operating Characteristic (ROC) Comparison\n"
              "1D Topological Descriptors vs. 2D Continuous Persistence Images (N=4,000)",
              fontsize=13, fontweight="bold", pad=14)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="lower right", frameon=True, framealpha=0.95, edgecolor="#bdc3c7", fontsize=10)
    plt.tight_layout()

    out_fig = os.path.join(OUTPUT_DIR, "roc_curves_with_cnn.png")
    plt.savefig(out_fig)
    plt.close()
    print(f"[+] Saved comprehensive ROC comparison figure to {out_fig}")

if __name__ == "__main__":
    generate_combined_roc_plot()
