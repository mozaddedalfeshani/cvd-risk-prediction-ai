#!/usr/bin/env python3
"""
Dataset Visualization Utility
-----------------------------
Generates charts/graphs for the CVD datasets used in this project.
Outputs figures to docs/research/figures/ for inclusion in reports.

Usage:
  python ml-models/evaluation/visualize_dataset.py \
    --dataset ml-models/data/raw/MymensingUniversity_ML_Ready.csv \
    --target "CVD Risk Level"

Dependencies: pandas, numpy, matplotlib, seaborn
"""

import argparse
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_dataset(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing": df.isna().sum(),
        "unique": df.nunique(),
    })
    return summary


def plot_distributions(df: pd.DataFrame, out_dir: Path, target_col: Optional[str]) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)

    cols_to_plot = numeric_cols[:12]  # cap for readability; adjust as needed
    if not cols_to_plot:
        return

    n = len(cols_to_plot)
    rows = int(np.ceil(n / 3))
    fig, axes = plt.subplots(rows, 3, figsize=(16, 5 * rows))
    axes = axes.flatten()

    for i, col in enumerate(cols_to_plot):
        ax = axes[i]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2563eb")
        ax.set_title(f"Distribution: {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Count")

    # Hide any unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "distributions.png", dpi=200)
    plt.close(fig)


def plot_correlations(df: pd.DataFrame, out_dir: Path, target_col: Optional[str]) -> None:
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        return

    corr = numeric_df.corr(numeric_only=True)
    plt.figure(figsize=(14, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(out_dir / "correlation_heatmap.png", dpi=220)
    plt.close()


def plot_target_distribution(df: pd.DataFrame, out_dir: Path, target_col: Optional[str]) -> None:
    if not target_col or target_col not in df.columns:
        return
    plt.figure(figsize=(8, 5))
    order = None
    if df[target_col].dtype == object:
        order = sorted(df[target_col].dropna().unique())
    sns.countplot(x=target_col, data=df, order=order, palette="Set2")
    plt.title(f"Target Distribution: {target_col}")
    plt.tight_layout()
    plt.savefig(out_dir / "target_distribution.png", dpi=200)
    plt.close()


def plot_pair_relationships(df: pd.DataFrame, out_dir: Path, target_col: Optional[str]) -> None:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return
    subset = numeric_cols[:4]
    plot_df = df[subset + ([target_col] if target_col and target_col in df.columns else [])].dropna()
    sns.pairplot(plot_df, hue=target_col if target_col in plot_df.columns else None, diag_kind="kde")
    plt.suptitle("Pairwise Relationships (subset)", y=1.02)
    plt.savefig(out_dir / "pairplot_subset.png", dpi=180, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Generate dataset charts/graphs for appendix")
    parser.add_argument("--dataset", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, default="CVD Risk Level", help="Target column name")
    parser.add_argument("--output", type=str, default="docs/research/figures", help="Output directory for figures")
    args = parser.parse_args()

    csv_path = Path(args.dataset)
    out_dir = Path(args.output)
    ensure_output_dir(out_dir)

    print(f"Loading dataset: {csv_path}")
    df = load_dataset(csv_path)
    print(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")

    # Save a quick summary CSV
    summary = summarize_dataframe(df)
    summary.to_csv(out_dir / "dataset_summary.csv")
    print(f"Saved dataset summary → {out_dir / 'dataset_summary.csv'}")

    # Generate plots
    plot_target_distribution(df, out_dir, args.target)
    print("Saved target distribution plot")

    plot_distributions(df, out_dir, args.target)
    print("Saved feature distributions")

    plot_correlations(df, out_dir, args.target)
    print("Saved correlation heatmap")

    plot_pair_relationships(df, out_dir, args.target)
    print("Saved pairplot (subset)")

    print(f"All figures saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


