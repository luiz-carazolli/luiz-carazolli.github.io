#!/usr/bin/env python3
"""
Validation and exploratory statistical analysis of the CIFAKE topological dataset.
Verifies dataset integrity, column distributions, and computes Real vs. AI comparisons.
"""

import argparse
import sys
import pandas as pd
import numpy as np
from scipy import stats


def validate_and_summarize(dataset_path: str):
    print(f"Loading dataset: {dataset_path} ...")
    if dataset_path.endswith(".csv"):
        df = pd.read_csv(dataset_path)
    else:
        df = pd.read_parquet(dataset_path)

    print(f"\n--- Dataset Overview ---")
    print(f"Rows: {len(df):,}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    # Check missing values
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print("WARNING: Null values found:")
        print(null_counts[null_counts > 0])
    else:
        print("Integrity Check: No missing (null) values found.")

    # Class balance
    print("\n--- Class Balance ---")
    counts = df["label_name"].value_counts()
    for name, cnt in counts.items():
        print(f"  {name.upper()}: {cnt:,} ({cnt/len(df)*100:.1f}%)")

    # Topological comparison
    topo_features = [
        "h0_count",
        "h1_count",
        "total_feature_count",
        "h0_total_persistence",
        "h1_total_persistence",
        "total_persistence",
        "h0_max_persistence",
        "h1_max_persistence",
        "max_persistence",
        "h0_entropy",
        "h1_entropy",
        "total_entropy",
        "h0_mean_persistence",
        "h1_mean_persistence",
        "pixel_mean",
        "pixel_std",
    ]

    print("\n--- Real vs. AI Topological Summary Statistics (Mean ± Std) ---")
    real_df = df[df["label_name"] == "real"]
    ai_df = df[df["label_name"] == "ai"]

    header = f"{'Feature':<25} | {'Real (CIFAR-10)':<22} | {'AI (Synthetic)':<22} | {'p-value (t-test)':<18} | {'Effect (Cohen d)':<16}"
    print(header)
    print("-" * len(header))

    summary_rows = []
    for feat in topo_features:
        if feat not in df.columns:
            continue
        r_vals = real_df[feat].to_numpy()
        a_vals = ai_df[feat].to_numpy()

        r_mean, r_std = np.mean(r_vals), np.std(r_vals)
        a_mean, a_std = np.mean(a_vals), np.std(a_vals)

        # Welch's t-test
        t_stat, p_val = stats.ttest_ind(r_vals, a_vals, equal_var=False)

        # Cohen's d effect size
        pooled_std = np.sqrt((r_std**2 + a_std**2) / 2)
        cohen_d = (r_mean - a_mean) / pooled_std if pooled_std > 0 else 0.0

        p_str = f"< 1e-10" if p_val < 1e-10 else f"{p_val:.4e}"
        print(
            f"{feat:<25} | {r_mean:8.4f} ± {r_std:7.4f}   | {a_mean:8.4f} ± {a_std:7.4f}   | {p_str:<18} | {cohen_d:+.4f}"
        )
        summary_rows.append({
            "feature": feat,
            "real_mean": r_mean,
            "real_std": r_std,
            "ai_mean": a_mean,
            "ai_std": a_std,
            "p_value": p_val,
            "cohens_d": cohen_d,
        })

    print("\nValidation completed successfully.")
    return pd.DataFrame(summary_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate CIFAKE Topological Dataset")
    parser.add_argument("--dataset", type=str, default="data/cifake_topological_dataset.parquet")
    args = parser.parse_args()
    validate_and_summarize(args.dataset)
