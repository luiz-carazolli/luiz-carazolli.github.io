#!/usr/bin/env python3
"""
Statistical distribution analysis & visualization for CIFAKE Persistent Homology.
Performs:
1. Non-parametric hypothesis testing (Mann-Whitney U and Kolmogorov-Smirnov tests)
2. Medians, IQRs, and non-parametric effect sizes (Rank-Biserial r, KS statistic D)
3. High-resolution publication-quality KDE and Violin plots for H0 entropy and H1 mean persistence
4. 2D Joint Topological Landscape visualization
"""

import argparse
import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Publication plot styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.titleweight": "bold",
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

PALETTE = {"real": "#1f77b4", "ai": "#ff7f0e"}
LABEL_NAMES = {"real": "Real (CIFAR-10)", "ai": "AI (Synthetic)"}


def run_nonparametric_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Mann-Whitney U and Kolmogorov-Smirnov tests across topological features."""
    real_df = df[df["label_name"] == "real"]
    ai_df = df[df["label_name"] == "ai"]
    n_real = len(real_df)
    n_ai = len(ai_df)

    topo_features = [
        ("h0_entropy", "$H_0$ Persistence Entropy (bits)"),
        ("h1_mean_persistence", "$H_1$ Mean Persistence"),
        ("h0_count", "$H_0$ Feature Count"),
        ("h1_count", "$H_1$ Feature Count"),
        ("total_feature_count", "Total Feature Count"),
        ("h0_total_persistence", "$H_0$ Total Persistence"),
        ("h1_total_persistence", "$H_1$ Total Persistence"),
        ("total_persistence", "Total Persistence"),
        ("h0_max_persistence", "$H_0$ Max Persistence"),
        ("h1_max_persistence", "$H_1$ Max Persistence"),
        ("max_persistence", "Max Persistence"),
        ("h1_entropy", "$H_1$ Persistence Entropy (bits)"),
        ("total_entropy", "Total Persistence Entropy (bits)"),
        ("h0_mean_persistence", "$H_0$ Mean Persistence"),
        ("pixel_mean", "Pixel Mean Intensity"),
        ("pixel_std", "Pixel Std Intensity"),
    ]

    print(f"\n{'='*110}")
    print(f"{'NON-PARAMETRIC STATISTICAL HYPOTHESIS TESTS (N_real = ' + str(n_real) + ', N_ai = ' + str(n_ai) + ')' :^110}")
    print(f"{'='*110}")

    header = (
        f"{'Feature':<24} | {'Real Median [IQR]':<22} | {'AI Median [IQR]':<22} | "
        f"{'MW-U p-val':<12} | {'Rank-Biserial r':<16} | {'KS Stat (D)':<12} | {'KS p-val':<12}"
    )
    print(header)
    print("-" * len(header))

    test_results = []

    for col, desc in topo_features:
        if col not in df.columns:
            continue
        r_vals = real_df[col].to_numpy()
        a_vals = ai_df[col].to_numpy()

        # Medians and IQR
        r_med = np.median(r_vals)
        r_iqr = stats.iqr(r_vals)
        a_med = np.median(a_vals)
        a_iqr = stats.iqr(a_vals)

        # Mann-Whitney U test (two-sided)
        mwu_res = stats.mannwhitneyu(r_vals, a_vals, alternative="two-sided")
        u_stat = mwu_res.statistic
        mwu_pval = mwu_res.pvalue

        # Rank-biserial correlation: r = 1 - (2*U / (n1*n2))
        rank_biserial = 1.0 - (2.0 * u_stat) / (n_real * n_ai)

        # Kolmogorov-Smirnov 2-sample test
        ks_res = stats.ks_2samp(r_vals, a_vals, alternative="two-sided")
        ks_stat = ks_res.statistic
        ks_pval = ks_res.pvalue

        mwu_p_str = "< 1e-15" if mwu_pval < 1e-15 else f"{mwu_pval:.4e}"
        ks_p_str = "< 1e-15" if ks_pval < 1e-15 else f"{ks_pval:.4e}"

        r_str = f"{r_med:.3f} [{r_iqr:.3f}]"
        a_str = f"{a_med:.3f} [{a_iqr:.3f}]"

        print(
            f"{col:<24} | {r_str:<22} | {a_str:<22} | "
            f"{mwu_p_str:<12} | {rank_biserial:+15.4f} | {ks_stat:11.4f} | {ks_p_str:<12}"
        )

        test_results.append({
            "feature": col,
            "description": desc,
            "real_median": r_med,
            "real_iqr": r_iqr,
            "ai_median": a_med,
            "ai_iqr": a_iqr,
            "mann_whitney_u": u_stat,
            "mwu_p_value": mwu_pval,
            "rank_biserial_r": rank_biserial,
            "ks_statistic_d": ks_stat,
            "ks_p_value": ks_pval,
        })

    print(f"{'='*110}\n")
    return pd.DataFrame(test_results)


def plot_kde_distributions(df: pd.DataFrame, output_path: str, stat_df: pd.DataFrame):
    """Create Kernel Density Estimation (KDE) plots for H0 entropy and H1 mean persistence."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))

    real_data = df[df["label_name"] == "real"]
    ai_data = df[df["label_name"] == "ai"]

    # 1. Panel A: H0 Persistence Entropy
    ax0 = axes[0]
    sns.kdeplot(
        data=real_data["h0_entropy"],
        ax=ax0,
        color=PALETTE["real"],
        fill=True,
        alpha=0.35,
        linewidth=2.2,
        label=f"{LABEL_NAMES['real']} (Med = {np.median(real_data['h0_entropy']):.2f})",
    )
    sns.kdeplot(
        data=ai_data["h0_entropy"],
        ax=ax0,
        color=PALETTE["ai"],
        fill=True,
        alpha=0.35,
        linewidth=2.2,
        label=f"{LABEL_NAMES['ai']} (Med = {np.median(ai_data['h0_entropy']):.2f})",
    )

    # Medians vertical lines
    ax0.axvline(np.median(real_data["h0_entropy"]), color=PALETTE["real"], linestyle="--", alpha=0.8)
    ax0.axvline(np.median(ai_data["h0_entropy"]), color=PALETTE["ai"], linestyle="--", alpha=0.8)

    # Annotation box for statistics
    h0_row = stat_df[stat_df["feature"] == "h0_entropy"].iloc[0]
    stats_text = (
        f"KS $D$ = {h0_row['ks_statistic_d']:.3f} ($p < 10^{{-15}}$)\n"
        f"Rank-Biserial $r$ = {h0_row['rank_biserial_r']:+.3f}\n"
        r"$\Delta$ Median = " f"{h0_row['real_median'] - h0_row['ai_median']:+.3f} bits"
    )
    ax0.text(
        0.05,
        0.80,
        stats_text,
        transform=ax0.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax0.set_title("A. $H_0$ Persistence Entropy (Micro-extrema Diversity)")
    ax0.set_xlabel("$H_0$ Shannon Persistence Entropy (bits)")
    ax0.set_ylabel("Probability Density")
    ax0.legend(loc="upper right", frameon=True)
    ax0.grid(axis="y", linestyle=":", alpha=0.6)

    # 2. Panel B: H1 Mean Persistence
    ax1 = axes[1]
    sns.kdeplot(
        data=real_data["h1_mean_persistence"],
        ax=ax1,
        color=PALETTE["real"],
        fill=True,
        alpha=0.35,
        linewidth=2.2,
        label=f"{LABEL_NAMES['real']} (Med = {np.median(real_data['h1_mean_persistence']):.3f})",
    )
    sns.kdeplot(
        data=ai_data["h1_mean_persistence"],
        ax=ax1,
        color=PALETTE["ai"],
        fill=True,
        alpha=0.35,
        linewidth=2.2,
        label=f"{LABEL_NAMES['ai']} (Med = {np.median(ai_data['h1_mean_persistence']):.3f})",
    )

    ax1.axvline(np.median(real_data["h1_mean_persistence"]), color=PALETTE["real"], linestyle="--", alpha=0.8)
    ax1.axvline(np.median(ai_data["h1_mean_persistence"]), color=PALETTE["ai"], linestyle="--", alpha=0.8)

    h1_row = stat_df[stat_df["feature"] == "h1_mean_persistence"].iloc[0]
    stats_text_h1 = (
        f"KS $D$ = {h1_row['ks_statistic_d']:.3f} ($p < 10^{{-15}}$)\n"
        f"Rank-Biserial $r$ = {h1_row['rank_biserial_r']:+.3f}\n"
        r"$\Delta$ Median = " f"{h1_row['real_median'] - h1_row['ai_median']:+.4f}"
    )
    ax1.text(
        0.58,
        0.80,
        stats_text_h1,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9),
    )

    ax1.set_title("B. $H_1$ Mean Persistence (1-Cycle Loop Robustness)")
    ax1.set_xlabel("$H_1$ Mean Lifetime / Persistence $(d - b)$")
    ax1.set_ylabel("Probability Density")
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Persistent Homology Density Shifts: Real Photos vs. AI Synthetics", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved KDE figure to: {output_path}")


def plot_violin_distributions(df: pd.DataFrame, output_path: str):
    """Create Violin plots with embedded boxplots and quartile markers."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

    df_plot = df.copy()
    df_plot["Class"] = df_plot["label_name"].map({"real": "Real (CIFAR-10)", "ai": "AI (Synthetic)"})
    plot_palette = {"Real (CIFAR-10)": PALETTE["real"], "AI (Synthetic)": PALETTE["ai"]}

    # Panel A: H0 Entropy Violin
    sns.violinplot(
        data=df_plot,
        x="Class",
        y="h0_entropy",
        palette=plot_palette,
        inner="quartile",
        cut=0,
        linewidth=1.4,
        ax=axes[0],
    )
    axes[0].set_title("A. Distribution of $H_0$ Persistence Entropy")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Shannon Entropy (bits)")
    axes[0].grid(axis="y", linestyle=":", alpha=0.6)

    # Panel B: H1 Mean Persistence Violin
    sns.violinplot(
        data=df_plot,
        x="Class",
        y="h1_mean_persistence",
        palette=plot_palette,
        inner="quartile",
        cut=0,
        linewidth=1.4,
        ax=axes[1],
    )
    axes[1].set_title("B. Distribution of $H_1$ Mean Persistence")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean Lifetime $(d - b)$")
    axes[1].grid(axis="y", linestyle=":", alpha=0.6)

    plt.suptitle("Violin Distributions of Distinguishing Topological Descriptors", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved Violin figure to: {output_path}")


def plot_joint_landscape(df: pd.DataFrame, output_path: str):
    """Create 2D joint topological phase space: H0 Entropy vs H1 Mean Persistence."""
    fig, ax = plt.subplots(figsize=(8.5, 6.5))

    # Downsample slightly for crisp scatter underlay if dataset is 20k
    df_sample = df.sample(n=min(6000, len(df)), random_state=42)

    sns.kdeplot(
        data=df_sample,
        x="h0_entropy",
        y="h1_mean_persistence",
        hue="label_name",
        palette=PALETTE,
        levels=6,
        alpha=0.8,
        linewidths=1.8,
        ax=ax,
    )

    # Faint scatter underlay
    sns.scatterplot(
        data=df_sample,
        x="h0_entropy",
        y="h1_mean_persistence",
        hue="label_name",
        palette=PALETTE,
        alpha=0.15,
        s=12,
        edgecolor=None,
        ax=ax,
        legend=False,
    )

    ax.set_title("2D Topological Phase Space: The Shape of a Photograph", pad=12)
    ax.set_xlabel("$H_0$ Persistence Entropy (Micro-texture complexity / grain)")
    ax.set_ylabel("$H_1$ Mean Persistence (Loop prominence / contrast depth)")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Custom legend
    handles = [
        plt.Line2D([0], [0], color=PALETTE["real"], lw=3, label="Real (High $H_0$ entropy, Low $H_1$ loop persistence)"),
        plt.Line2D([0], [0], color=PALETTE["ai"], lw=3, label="AI Synthetic (Low $H_0$ entropy, High $H_1$ loop persistence)"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=True)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print(f"Saved 2D Landscape figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Topological Distributions for Real vs AI Images")
    parser.add_argument("--dataset", type=str, default="data/cifake_topological_dataset.parquet")
    parser.add_argument("--output-dir", type=str, default="figures")
    parser.add_argument("--artifact-dir", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if args.artifact_dir:
        os.makedirs(args.artifact_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset} ...")
    df = pd.read_parquet(args.dataset) if args.dataset.endswith(".parquet") else pd.read_csv(args.dataset)
    print(f"Loaded {len(df):,} samples.")

    # 1. Non-parametric hypothesis testing
    stat_df = run_nonparametric_tests(df)
    stat_csv_path = os.path.join(args.output_dir, "nonparametric_tests_summary.csv")
    stat_df.to_csv(stat_csv_path, index=False)
    print(f"Saved statistical summary to: {stat_csv_path}")

    # 2. KDE plots
    kde_fig_path = os.path.join(args.output_dir, "topological_kde_distributions.png")
    plot_kde_distributions(df, kde_fig_path, stat_df)

    # 3. Violin plots
    violin_fig_path = os.path.join(args.output_dir, "topological_violin_distributions.png")
    plot_violin_distributions(df, violin_fig_path)

    # 4. Joint Phase Space
    landscape_fig_path = os.path.join(args.output_dir, "topological_2d_landscape.png")
    plot_joint_landscape(df, landscape_fig_path)

    # Copy to artifact directory if provided
    if args.artifact_dir:
        for fpath in [kde_fig_path, violin_fig_path, landscape_fig_path, stat_csv_path]:
            dest = os.path.join(args.artifact_dir, os.path.basename(fpath))
            shutil.copy2(fpath, dest)
            print(f"Copied figure to artifact dir: {dest}")


if __name__ == "__main__":
    main()
