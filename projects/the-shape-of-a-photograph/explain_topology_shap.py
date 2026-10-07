"""
Explainable AI (SHAP) Analysis for Topological Forensics
Quantifies individual and interaction contributions of topological features,
specifically focusing on H1 mean persistence and H0 persistence entropy in the decision boundary.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configure aesthetics
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#2c3e50'
plt.rcParams['axes.linewidth'] = 1.0

OUTPUT_DIR = "figures"
TEST_DATA_PATH = "data/cifake_test_split.parquet"
MODEL_PATH = "models/xgboost.joblib"
os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_NAME_MAP = {
    'h0_entropy': r'$H_0$ Persistence Entropy',
    'h1_mean_persistence': r'$H_1$ Mean Persistence',
    'h1_total_persistence': r'$H_1$ Total Persistence',
    'h1_entropy': r'$H_1$ Persistence Entropy',
    'h1_std_persistence': r'$H_1$ Std Persistence',
    'h0_count': r'$H_0$ Feature Count',
    'h0_total_persistence': r'$H_0$ Total Persistence',
    'total_feature_count': r'Total Feature Count',
    'h0_mean_persistence': r'$H_0$ Mean Persistence',
    'h1_count': r'$H_1$ Feature Count',
    'total_entropy': r'Total Persistence Entropy',
    'h1_max_persistence': r'$H_1$ Max Persistence',
    'h0_max_persistence': r'$H_0$ Max Persistence',
    'h0_std_persistence': r'$H_0$ Std Persistence',
    'max_persistence': r'Max Persistence',
    'total_persistence': r'Total Persistence'
}

def run_shap_analysis():
    print(f"[*] Loading model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)
    
    print(f"[*] Loading test split from {TEST_DATA_PATH}...")
    df_test = pd.read_parquet(TEST_DATA_PATH)
    y_test = df_test['label'].values
    X_test = df_test.drop(columns=['label'])
    
    print(f"    Evaluating SHAP on {len(X_test)} test instances...")
    explainer = shap.TreeExplainer(model)
    
    # Compute SHAP values
    shap_explanation = explainer(X_test)
    shap_values = shap_explanation.values  # shape: (4000, 16)
    
    # Compute SHAP interaction values
    print("    Computing pairwise SHAP interaction tensor...")
    shap_interaction = explainer.shap_interaction_values(X_test)  # shape: (4000, 16, 16)
    
    # --- 1. Save quantitative summary tables ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feat_imp_df = pd.DataFrame({
        'feature': X_test.columns,
        'feature_display': [FEATURE_NAME_MAP.get(f, f) for f in X_test.columns],
        'mean_abs_shap': mean_abs_shap,
        'relative_importance_pct': (mean_abs_shap / mean_abs_shap.sum()) * 100
    }).sort_values(by='mean_abs_shap', ascending=False)
    
    feat_imp_path = os.path.join(OUTPUT_DIR, "shap_feature_importance.csv")
    feat_imp_df.to_csv(feat_imp_path, index=False)
    print(f"[+] Saved SHAP feature importance table to {feat_imp_path}")
    
    # Pairwise interaction table
    mean_abs_inter = np.abs(shap_interaction).mean(axis=0)
    pairs = []
    n_feats = len(X_test.columns)
    for i in range(n_feats):
        for j in range(i + 1, n_feats):
            pairs.append({
                'feature_1': X_test.columns[i],
                'feature_2': X_test.columns[j],
                'feature_1_display': FEATURE_NAME_MAP.get(X_test.columns[i], X_test.columns[i]),
                'feature_2_display': FEATURE_NAME_MAP.get(X_test.columns[j], X_test.columns[j]),
                'interaction_strength': mean_abs_inter[i, j] * 2.0  # symmetric
            })
    inter_df = pd.DataFrame(pairs).sort_values(by='interaction_strength', ascending=False)
    inter_path = os.path.join(OUTPUT_DIR, "shap_interaction_pairs.csv")
    inter_df.to_csv(inter_path, index=False)
    print(f"[+] Saved SHAP interaction table to {inter_path}")
    
    # --- 2. Generate Plots ---
    plot_shap_bar(feat_imp_df)
    plot_shap_beeswarm(shap_explanation, X_test)
    plot_shap_dependence(X_test, shap_values, shap_interaction)
    plot_interaction_heatmap(X_test, mean_abs_inter)
    plot_interaction_scatter_phase_space(X_test, shap_interaction, y_test)
    
    print("[+] All SHAP analysis and figures completed successfully.")

def plot_shap_bar(feat_imp_df):
    print("[*] Plotting SHAP feature importance bar chart...")
    plt.figure(figsize=(9.0, 6.5), dpi=300)
    
    df_sorted = feat_imp_df.iloc[::-1]  # reverse for bottom-to-top barh
    
    colors = ['#2980b9' if 'H_0' in d else '#d35400' if 'H_1' in d else '#7f8c8d'
              for d in df_sorted['feature_display']]
    
    bars = plt.barh(df_sorted['feature_display'], df_sorted['mean_abs_shap'],
                    color=colors, edgecolor='#2c3e50', alpha=0.88, height=0.7)
    
    for bar, pct in zip(bars, df_sorted['relative_importance_pct']):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                 f"{width:.3f} ({pct:.1f}%)",
                 va='center', ha='left', fontsize=9.5, fontweight='bold', color='#2c3e50')
        
    plt.xlabel('Mean Absolute SHAP Value: ' + r'$\mathbb{E}[|\Phi_i|]$' + ' (Impact on Log-Odds Output)',
               fontsize=11.5, fontweight='bold', labelpad=8)
    plt.title('Global Topological Feature Importance (SHAP TreeExplainer, XGBoost)\nStand-Alone Evaluation on N=4,000 CIFAKE Test Set',
              fontsize=12.5, fontweight='bold', pad=12)
    plt.xlim([0, max(df_sorted['mean_abs_shap']) * 1.25])
    plt.grid(axis='x', linestyle=':', alpha=0.6)
    
    # Legend for feature families
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2980b9', edgecolor='#2c3e50', label=r'$H_0$ Descriptors (Valleys / Minima)'),
        Patch(facecolor='#d35400', edgecolor='#2c3e50', label=r'$H_1$ Descriptors (Cycles / Loops)'),
        Patch(facecolor='#7f8c8d', edgecolor='#2c3e50', label='Combined / Global Descriptors')
    ]
    plt.legend(handles=legend_elements, loc='lower right', frameon=True, framealpha=0.95, fontsize=10)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "shap_feature_importance_bar.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved SHAP bar chart to {out_path}")

def plot_shap_beeswarm(shap_explanation, X_test):
    print("[*] Plotting SHAP summary beeswarm plot...")
    plt.figure(figsize=(9.5, 7.5), dpi=300)
    
    # Rename features in explanation for publication display
    display_names = [FEATURE_NAME_MAP.get(c, c) for c in X_test.columns]
    
    # Create copy of explanation with updated feature names
    exp_display = shap.Explanation(
        values=shap_explanation.values,
        base_values=shap_explanation.base_values,
        data=shap_explanation.data,
        feature_names=display_names
    )
    
    shap.plots.beeswarm(exp_display, max_display=16, show=False)
    plt.title('SHAP Summary Beeswarm: Direction of Topological Impact on Classification\n' +
              r'($\Phi > 0 \rightarrow$ Real Photograph, $\Phi < 0 \rightarrow$ AI Synthetic)',
              fontsize=12.5, fontweight='bold', pad=14)
    plt.xlabel('SHAP Value (Impact on Log-Odds toward Real Photograph)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "shap_summary_beeswarm.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved SHAP beeswarm plot to {out_path}")

def plot_shap_dependence(X_test, shap_values, shap_interaction):
    print("[*] Plotting SHAP dependence plots for H1 Mean Persistence and H0 Entropy...")
    
    idx_h1_mean = list(X_test.columns).index('h1_mean_persistence')
    idx_h0_ent = list(X_test.columns).index('h0_entropy')
    
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), dpi=300)
    
    # 1. H1 Mean Persistence Dependence (colored by H0 Entropy)
    ax1 = axes[0]
    sc1 = ax1.scatter(
        X_test['h1_mean_persistence'],
        shap_values[:, idx_h1_mean],
        c=X_test['h0_entropy'],
        cmap='coolwarm',
        alpha=0.65,
        s=18,
        edgecolor='none'
    )
    cbar1 = plt.colorbar(sc1, ax=ax1)
    cbar1.set_label(r'$H_0$ Persistence Entropy', fontsize=11, fontweight='bold')
    
    ax1.axhline(0, color='black', linestyle='--', lw=1.2, alpha=0.7)
    ax1.set_xlabel(r'$H_1$ Mean Lifetime / Persistence', fontsize=12, fontweight='bold')
    ax1.set_ylabel(r'SHAP Value for $H_1$ Mean Persistence ($\Phi_{H_1}$)', fontsize=12, fontweight='bold')
    ax1.set_title(r'(a) Decision Impact of $H_1$ Mean Persistence' + '\n' +
                  r'Elevated $H_1$ strongly pushes prediction to AI ($\Phi < 0$)',
                  fontsize=12.5, fontweight='bold', pad=10)
    ax1.grid(True, linestyle=':', alpha=0.5)
    
    # 2. H0 Entropy Dependence (colored by H1 Mean Persistence)
    ax2 = axes[1]
    sc2 = ax2.scatter(
        X_test['h0_entropy'],
        shap_values[:, idx_h0_ent],
        c=X_test['h1_mean_persistence'],
        cmap='viridis_r',
        alpha=0.65,
        s=18,
        edgecolor='none'
    )
    cbar2 = plt.colorbar(sc2, ax=ax2)
    cbar2.set_label(r'$H_1$ Mean Persistence', fontsize=11, fontweight='bold')
    
    ax2.axhline(0, color='black', linestyle='--', lw=1.2, alpha=0.7)
    ax2.set_xlabel(r'$H_0$ Persistence Entropy', fontsize=12, fontweight='bold')
    ax2.set_ylabel(r'SHAP Value for $H_0$ Entropy ($\Phi_{H_0}$)', fontsize=12, fontweight='bold')
    ax2.set_title(r'(b) Decision Impact of $H_0$ Persistence Entropy' + '\n' +
                  r'High $H_0$ entropy decisively flags Real Photographs ($\Phi > 0$)',
                  fontsize=12.5, fontweight='bold', pad=10)
    ax2.grid(True, linestyle=':', alpha=0.5)
    
    plt.suptitle('SHAP Dependence & Cross-Feature Interaction Analysis (N=4,000)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "shap_dependence_h1_mean_h0_entropy.png")
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f"[+] Saved SHAP dependence plots to {out_path}")

def plot_interaction_heatmap(X_test, mean_abs_inter):
    print("[*] Plotting SHAP pairwise interaction heatmap...")
    
    # Select top 8 features for readability
    top_indices = np.argsort(mean_abs_inter.sum(axis=0))[::-1][:8]
    top_features = [X_test.columns[i] for i in top_indices]
    top_display = [FEATURE_NAME_MAP.get(f, f) for f in top_features]
    
    sub_matrix = mean_abs_inter[np.ix_(top_indices, top_indices)] * 2.0
    # Zero out diagonal for pure interaction focus
    np.fill_diagonal(sub_matrix, 0.0)
    
    plt.figure(figsize=(9.0, 7.5), dpi=300)
    sns.heatmap(
        sub_matrix,
        xticklabels=top_display,
        yticklabels=top_display,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        cbar_kws={'label': r'Mean Absolute SHAP Interaction Strength $\mathbb{E}[2|\Phi_{i,j}|]$'},
        linewidths=0.5,
        linecolor='#ecf0f1'
    )
    plt.title('Pairwise SHAP Interaction Matrix (Top 8 Topological Features)\n' +
              r'Revealing Non-Linear Synergies Between $H_0$ Entropy and $H_1$ Persistence',
              fontsize=12.5, fontweight='bold', pad=12)
    plt.xticks(rotation=35, ha='right', fontsize=9.5)
    plt.yticks(rotation=0, fontsize=9.5)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "shap_interaction_heatmap.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved SHAP interaction heatmap to {out_path}")

def plot_interaction_scatter_phase_space(X_test, shap_interaction, y_test):
    print("[*] Plotting pure SHAP interaction value across 2D Topological Phase Space...")
    idx_h1 = list(X_test.columns).index('h1_mean_persistence')
    idx_h0 = list(X_test.columns).index('h0_entropy')
    
    # Pure symmetric interaction: Phi_{h0, h1} + Phi_{h1, h0}
    pure_inter = shap_interaction[:, idx_h0, idx_h1] + shap_interaction[:, idx_h1, idx_h0]
    
    plt.figure(figsize=(9.0, 7.0), dpi=300)
    sc = plt.scatter(
        X_test['h0_entropy'],
        X_test['h1_mean_persistence'],
        c=pure_inter,
        cmap='Spectral_r',
        s=22,
        alpha=0.75,
        edgecolor='none'
    )
    cbar = plt.colorbar(sc)
    cbar.set_label(r'SHAP Interaction Value $\Phi_{H_0, H_1} + \Phi_{H_1, H_0}$ (Log-Odds Contribution)',
                   fontsize=11, fontweight='bold')
    
    plt.xlabel(r'$H_0$ Persistence Entropy (Micro-Basin Complexity)', fontsize=12, fontweight='bold')
    plt.ylabel(r'$H_1$ Mean Lifetime / Persistence (Meso-Scale Loop Salience)', fontsize=12, fontweight='bold')
    plt.title('Non-Linear Decision Synergy: Pure Interaction ' + r'$\Phi(H_0, H_1)$' + '\n' +
              'Quantifying Where Topological Concurrence Boosts Decision Confidence',
              fontsize=13, fontweight='bold', pad=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()
    
    out_path = os.path.join(OUTPUT_DIR, "shap_interaction_phase_space.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[+] Saved SHAP interaction phase space plot to {out_path}")

if __name__ == "__main__":
    run_shap_analysis()
