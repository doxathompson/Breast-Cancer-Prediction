"""
src/data/eda.py
───────────────
Exploratory Data Analysis for the Wisconsin Breast Cancer Diagnostic dataset.

Produces:
  • Class-distribution bar chart
  • Feature histograms (all 30 features)
  • Pairplot on top 6 correlated features
  • Full correlation heatmap
  • Summary statistics table

All plots are saved to *reports_dir* and also returned so that the
Jupyter notebook can render them inline.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

log = logging.getLogger(__name__)

# Shared palette: 0=malignant (red family), 1=benign (teal family)
PALETTE = {0: "#E05C5C", 1: "#5CC8C8"}
CLASS_LABELS = {0: "Malignant", 1: "Benign"}

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─── Public API ───────────────────────────────────────────────────────────────

def run_full_eda(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    reports_dir: str | Path = "reports",
) -> dict[str, plt.Figure]:
    """
    Execute all EDA plots and return a dict mapping name → Figure.
    """
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    figs: dict[str, plt.Figure] = {}

    figs["class_distribution"] = plot_class_distribution(df, target_col, reports_dir)
    figs["feature_histograms"] = plot_feature_histograms(df, target_col, reports_dir)
    figs["pairplot"]           = plot_pairplot(df, target_col, reports_dir)
    figs["correlation_heatmap"]= plot_correlation_heatmap(df, target_col, reports_dir)

    log.info("EDA complete — %d plots saved to %s", len(figs), reports_dir)
    return figs


def plot_class_distribution(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    save_dir: Path | None = None,
) -> plt.Figure:
    counts = df[target_col].map(CLASS_LABELS).value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(
        counts.index,
        counts.values,
        color=[PALETTE[k] for k in df[target_col].map(CLASS_LABELS).map(
            {v: k for k, v in CLASS_LABELS.items()}).unique()
               if k in PALETTE],
        edgecolor="none",
        width=0.5,
        alpha=0.88,
    )
    # simpler colour assignment
    colors = ["#5CC8C8", "#E05C5C"] if counts.index[0] == "Benign" else ["#E05C5C", "#5CC8C8"]
    for bar, col in zip(bars, colors):
        bar.set_color(col)

    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 4,
            f"{bar.get_height():,}",
            ha="center", va="bottom", fontsize=12, fontweight="bold",
        )

    ax.set_title("Class Distribution — WBCD", fontsize=14, fontweight="bold", pad=14)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()

    if save_dir:
        fig.savefig(save_dir / "class_distribution.png", bbox_inches="tight")
    return fig


def plot_feature_histograms(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    save_dir: Path | None = None,
) -> plt.Figure:
    features = [c for c in df.columns if c != target_col]
    ncols = 5
    nrows = int(np.ceil(len(features) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))
    axes = axes.flatten()

    for ax, feat in zip(axes, features):
        for cls_id, cls_label in CLASS_LABELS.items():
            subset = df.loc[df[target_col] == cls_id, feat]
            ax.hist(subset, bins=25, alpha=0.65, color=PALETTE[cls_id],
                    label=cls_label, density=True)
        ax.set_title(feat, fontsize=7.5, fontweight="bold")
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for ax in axes[len(features):]:
        ax.set_visible(False)

    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Feature Distributions by Class — WBCD", fontsize=14,
                 fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_dir:
        fig.savefig(save_dir / "feature_histograms.png", bbox_inches="tight")
    return fig


def plot_pairplot(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    save_dir: Path | None = None,
    n_top: int = 6,
) -> plt.Figure:
    """Pairplot of the *n_top* features most correlated with the target."""
    corr_with_target = (
        df.drop(columns=[target_col])
          .corrwith(df[target_col])
          .abs()
          .sort_values(ascending=False)
    )
    top_feats = corr_with_target.head(n_top).index.tolist()
    plot_df = df[top_feats + [target_col]].copy()
    plot_df[target_col] = plot_df[target_col].map(CLASS_LABELS)

    g = sns.pairplot(
        plot_df,
        hue=target_col,
        palette={"Malignant": "#E05C5C", "Benign": "#5CC8C8"},
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 18, "edgecolor": "none"},
        diag_kws={"alpha": 0.6, "linewidth": 1.5},
    )
    g.figure.suptitle(
        f"Pairplot — Top {n_top} Correlated Features",
        fontsize=13, fontweight="bold", y=1.01,
    )
    g.figure.tight_layout()

    if save_dir:
        g.figure.savefig(save_dir / "pairplot.png", bbox_inches="tight")
    return g.figure


def plot_correlation_heatmap(
    df: pd.DataFrame,
    target_col: str = "diagnosis",
    save_dir: Path | None = None,
) -> plt.Figure:
    corr = df.drop(columns=[target_col]).corr()

    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        ax=ax,
        cmap="coolwarm",
        center=0,
        vmin=-1, vmax=1,
        annot=False,
        linewidths=0.3,
        linecolor="white",
        cbar_kws={"shrink": 0.6},
    )
    ax.set_title("Feature Correlation Heatmap — WBCD", fontsize=14,
                 fontweight="bold", pad=14)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0, labelsize=7)
    fig.tight_layout()

    if save_dir:
        fig.savefig(save_dir / "correlation_heatmap.png", bbox_inches="tight")
    return fig


def print_summary(df: pd.DataFrame, target_col: str = "diagnosis") -> pd.DataFrame:
    """Return per-class descriptive statistics."""
    summary = df.groupby(target_col).describe().T
    log.info("Summary statistics computed.")
    return summary
