"""
Machine Learning Classification Benchmark for Topological Forensics
Trains and benchmarks lightweight, interpretable classifiers using exclusively
the 16 extracted topological summary features from CIFAKE (20,000 images).

Models:
  1. Regularized Logistic Regression (L2)
  2. Support Vector Machine (Linear SVM)
  3. Support Vector Machine (RBF SVM)
  4. Random Forest
  5. XGBoost Classifier
  6. LightGBM Classifier
"""

import os
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    f1_score, precision_score, recall_score, brier_score_loss,
    roc_curve, confusion_matrix
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- Configure aesthetics ---
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#2c3e50'
plt.rcParams['axes.linewidth'] = 1.0

DATA_PATH = "data/cifake_topological_dataset.parquet"
OUTPUT_DIR = "figures"
MODELS_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

TOPOLOGICAL_FEATURES = [
    'h0_count',
    'h1_count',
    'total_feature_count',
    'h0_total_persistence',
    'h1_total_persistence',
    'total_persistence',
    'h0_max_persistence',
    'h1_max_persistence',
    'max_persistence',
    'h0_entropy',
    'h1_entropy',
    'total_entropy',
    'h0_mean_persistence',
    'h1_mean_persistence',
    'h0_std_persistence',
    'h1_std_persistence'
]

def load_data():
    print(f"[*] Loading dataset from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    print(f"    Loaded {len(df)} samples.")
    
    X = df[TOPOLOGICAL_FEATURES].copy()
    y = df['label'].values  # 0: AI, 1: Real
    
    print(f"    Features ({len(TOPOLOGICAL_FEATURES)}): {TOPOLOGICAL_FEATURES}")
    print(f"    Class distribution: AI (0) = {(y == 0).sum()}, Real (1) = {(y == 1).sum()}")
    return X, y

def get_models():
    models = {
        "Logistic Regression (L2)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ]),
        "Linear SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=42, max_iter=2000), cv=3))
        ]),
        "RBF SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", CalibratedClassifierCV(SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42), cv=3))
        ]),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=5,
            n_jobs=-1, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='logloss', random_state=42, n_jobs=-1
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=-1, verbose=-1
        )
    }
    return models

def run_benchmark():
    X, y = load_data()
    
    # 80/20 Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"[*] Split data: Train = {len(X_train)} (80%), Test = {len(X_test)} (20%)")
    
    # Save test set for explainability analysis
    test_export = X_test.copy()
    test_export['label'] = y_test
    test_export.to_parquet("data/cifake_test_split.parquet")
    
    # Save train set
    train_export = X_train.copy()
    train_export['label'] = y_train
    train_export.to_parquet("data/cifake_train_split.parquet")
    
    models = get_models()
    results = []
    
    # 5-fold CV on Train set
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'roc_auc', 'f1']
    
    test_predictions = {}
    fitted_models = {}
    
    print("\n" + "="*85)
    print(f"{'Model':<26} | {'CV Acc (5-f)':<15} | {'CV AUC (5-f)':<15} | {'Test Acc':<9} | {'Test AUC':<9} | {'Test F1':<9}")
    print("="*85)
    
    for name, model in models.items():
        t0 = time.time()
        
        # 5-fold Cross-Validation in parallel
        cv_res = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        cv_acc_mean = cv_res['test_accuracy'].mean()
        cv_acc_std = cv_res['test_accuracy'].std()
        cv_auc_mean = cv_res['test_roc_auc'].mean()
        cv_auc_std = cv_res['test_roc_auc'].std()
        cv_f1_mean = cv_res['test_f1'].mean()
        cv_f1_std = cv_res['test_f1'].std()
        
        # Fit on full training set
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
        # Evaluate on Test set
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        test_predictions[name] = (y_pred, y_prob)
        
        test_acc = accuracy_score(y_test, y_pred)
        test_bal_acc = balanced_accuracy_score(y_test, y_pred)
        test_auc = roc_auc_score(y_test, y_prob)
        test_f1 = f1_score(y_test, y_pred)
        test_prec = precision_score(y_test, y_pred)
        test_rec = recall_score(y_test, y_pred)
        test_brier = brier_score_loss(y_test, y_prob)
        
        elapsed = time.time() - t0
        
        print(f"{name:<26} | {cv_acc_mean*100:5.2f}% ± {cv_acc_std*100:4.2f}% | {cv_auc_mean:0.4f} ± {cv_auc_std:0.4f} | {test_acc*100:5.2f}%   | {test_auc:0.4f}   | {test_f1:0.4f}   ({elapsed:.1f}s)")
        
        results.append({
            'Model': name,
            'CV_Accuracy_Mean': cv_acc_mean,
            'CV_Accuracy_Std': cv_acc_std,
            'CV_AUC_Mean': cv_auc_mean,
            'CV_AUC_Std': cv_auc_std,
            'CV_F1_Mean': cv_f1_mean,
            'CV_F1_Std': cv_f1_std,
            'Test_Accuracy': test_acc,
            'Test_Balanced_Accuracy': test_bal_acc,
            'Test_ROC_AUC': test_auc,
            'Test_F1_Score': test_f1,
            'Test_Precision': test_prec,
            'Test_Recall': test_rec,
            'Test_Brier_Score': test_brier,
            'Training_Time_Sec': elapsed
        })
        
        # Save individual model
        clean_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        model_filename = os.path.join(MODELS_DIR, f"{clean_name}.joblib")
        joblib.dump(model, model_filename)
        
    print("="*85)
    
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(OUTPUT_DIR, "classification_metrics_benchmark.csv")
    results_df.to_csv(results_csv_path, index=False)
    print(f"[+] Saved metrics benchmark to {results_csv_path}")
    
    # --- Generate Figures ---
    plot_roc_curves(y_test, test_predictions)
    plot_confusion_matrices(y_test, test_predictions)
    
    return results_df, fitted_models

def plot_roc_curves(y_test, test_predictions):
    print("[*] Generating ROC curves comparison plot...")
    plt.figure(figsize=(8.5, 7.0), dpi=300)
    
    colors = {
        "Logistic Regression (L2)": "#34495e",
        "Linear SVM": "#7f8c8d",
        "RBF SVM": "#2980b9",
        "Random Forest": "#27ae60",
        "XGBoost": "#d35400",
        "LightGBM": "#8e44ad"
    }
    
    for name, (_, y_prob) in test_predictions.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc_val:.4f})",
                 color=colors.get(name, '#333333'), lw=2.2)
        
    plt.plot([0, 1], [0, 1], color='#bdc3c7', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.5000)')
    
    plt.xlim([-0.01, 1.01])
    plt.ylim([-0.01, 1.01])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12, fontweight='bold', labelpad=8)
    plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12, fontweight='bold', labelpad=8)
    plt.title('Receiver Operating Characteristic (ROC) Benchmark\nStand-Alone Topological Features (16 Descriptors, N=4,000 Test Set)',
              fontsize=13, fontweight='bold', pad=14)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='#bdc3c7', fontsize=10.5)
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, "roc_curves_comparison.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[+] Saved ROC curve figure to {fig_path}")

def plot_confusion_matrices(y_test, test_predictions):
    print("[*] Generating normalized confusion matrices plot...")
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5), dpi=300)
    axes = axes.flatten()
    
    class_names = ['AI (Synthetic)', 'Real (Photograph)']
    
    for i, (name, (y_pred, _)) in enumerate(test_predictions.items()):
        cm = confusion_matrix(y_test, y_pred, normalize='true')
        cm_counts = confusion_matrix(y_test, y_pred)
        
        ax = axes[i]
        sns.heatmap(cm, annot=False, cmap='Blues', cbar=(i == 2 or i == 5),
                    vmin=0.0, vmax=1.0, ax=ax)
        
        # Annotate with percentages and absolute counts
        for r in range(2):
            for c in range(2):
                pct = cm[r, c] * 100
                cnt = cm_counts[r, c]
                text_color = "white" if cm[r, c] > 0.55 else "black"
                ax.text(c + 0.5, r + 0.5, f"{pct:.1f}%\n({cnt})",
                        ha='center', va='center', color=text_color,
                        fontweight='bold', fontsize=12)
                
        ax.set_title(name, fontsize=13, fontweight='bold', pad=10)
        ax.set_xticks([0.5, 1.5])
        ax.set_yticks([0.5, 1.5])
        ax.set_xticklabels(class_names, fontsize=10)
        ax.set_yticklabels(class_names, fontsize=10, rotation=0)
        
        if i in [0, 3]:
            ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        else:
            ax.set_ylabel('')
            
        if i in [3, 4, 5]:
            ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        else:
            ax.set_xlabel('')
            
    plt.suptitle('Normalized Confusion Matrices on Stand-Alone Topological Test Set (N=4,000)',
                 fontsize=15, fontweight='bold', y=0.99)
    plt.tight_layout()
    
    fig_path = os.path.join(OUTPUT_DIR, "confusion_matrices.png")
    plt.savefig(fig_path)
    plt.close()
    print(f"[+] Saved confusion matrices figure to {fig_path}")

if __name__ == "__main__":
    run_benchmark()
