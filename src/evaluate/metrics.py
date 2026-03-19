"""
src/evaluate/metrics.py
────────────────────────
Clinical-grade evaluation of binary classifiers.

Metrics reported
----------------
  • AUC-ROC
  • Precision-Recall AUC
  • Sensitivity  (recall on the *malignant* class, label 0)
  • Specificity
  • F1-score  (macro + per-class)
  • Confusion Matrix
  • Youden Index → optimal threshold

Plots
-----
  • ROC curves (all models together + individual)
  • Calibration curves
  • Confusion matrix heatmaps
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

log = logging.getLogger(__name__)

# Malignant = class 0 in sklearn convention
MALIGNANT_LABEL: int = 0

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ─── Threshold Optimisation ───────────────────────────────────────────────────

def youden_threshold(
    y_true: np.ndarray,
    y_prob_malignant: np.ndarray,
) -> tuple[float, float]:
    """
    Return (optimal_threshold, youden_J) that maximises Sensitivity + Specificity.

    Probabilities should be P(malignant | x) = P(class_0 | x).
    """
    # Flip labels so malignant (0) → 1 (positive class) for roc_curve
    fpr, tpr, thresholds = roc_curve(1 - y_true, y_prob_malignant)
    j_scores = tpr + (1 - fpr) - 1          # Youden J = Sens + Spec − 1
    best_idx = int(np.argmax(j_scores))
    best_thresh = float(thresholds[best_idx])
    best_j      = float(j_scores[best_idx])
    log.info("Youden threshold = %.4f  (J = %.4f)", best_thresh, best_j)
    return best_thresh, best_j


# ─── Core Metrics ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float | None = None,
    model_name: str = "model",
) -> dict[str, float]:
    """
    Compute all clinical evaluation metrics.

    Parameters
    ----------
    y_true     : ground truth (0=malignant, 1=benign)
    y_prob     : predicted probability array of shape (n, 2) or (n,)
                 If 2-D, column 0 is P(malignant).
    threshold  : classification threshold; if None, Youden Index is used.
    model_name : used for logging only.

    Returns
    -------
    dict with keys: auc_roc, auc_pr, sensitivity, specificity,
                    f1_macro, f1_malignant, f1_benign, threshold
    """
    if y_prob.ndim == 2:
        prob_malignant = y_prob[:, 0]
    else:
        prob_malignant = 1.0 - y_prob      # assume y_prob = P(benign)

    # Threshold selection
    if threshold is None:
        threshold, _ = youden_threshold(y_true, prob_malignant)

    y_pred = (prob_malignant >= threshold).astype(int) * 0 + \
             (prob_malignant < threshold).astype(int) * 1
    # simpler: predict 0 (malignant) when prob_malignant >= threshold
    y_pred = np.where(prob_malignant >= threshold, 0, 1)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()   # TN=benign correct, TP=malignant correct

    sensitivity  = tp / (tp + fn + 1e-9)  # recall for malignant
    specificity  = tn / (tn + fp + 1e-9)

    # sklearn 1.8+ removed pos_label from roc_auc_score.
    # We flip labels: treat malignant (0→1) as positive class.
    y_true_flipped = 1 - y_true
    auc_roc = roc_auc_score(y_true_flipped, prob_malignant)
    auc_pr  = average_precision_score(y_true_flipped, prob_malignant)
    f1_macro     = f1_score(y_true, y_pred, average="macro")
    f1_malignant = f1_score(y_true, y_pred, pos_label=0)
    f1_benign    = f1_score(y_true, y_pred, pos_label=1)

    metrics = {
        "model":         model_name,
        "auc_roc":       round(auc_roc, 4),
        "auc_pr":        round(auc_pr, 4),
        "sensitivity":   round(sensitivity, 4),
        "specificity":   round(specificity, 4),
        "f1_macro":      round(f1_macro, 4),
        "f1_malignant":  round(f1_malignant, 4),
        "f1_benign":     round(f1_benign, 4),
        "threshold":     round(threshold, 4),
        "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
    }

    log.info(
        "[%s] AUC-ROC=%.4f | AUC-PR=%.4f | Sensitivity=%.4f | "
        "Specificity=%.4f | F1-macro=%.4f | FN=%d",
        model_name, auc_roc, auc_pr, sensitivity, specificity, f1_macro, fn,
    )
    return metrics


def evaluate_all(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    use_youden: bool = True,
) -> pd.DataFrame:
    """
    Evaluate all models and return a summary DataFrame sorted by AUC-ROC.
    """
    rows = []
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)
        thresh = None if use_youden else 0.5
        row = compute_metrics(y_test, y_prob, threshold=thresh, model_name=name)
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("auc_roc", ascending=False).reset_index(drop=True)
    return df


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_roc_curves(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path | None = None,
) -> plt.Figure:
    """Combined ROC curve plot for all models."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#5CC8C8", "#E05C5C", "#6B7FD4", "#F0A500", "#8B5CF6"]

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)
        prob_mal = y_prob[:, 0]
        fpr, tpr, _ = roc_curve(1 - y_test, prob_mal)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2.2, color=color,
                label=f"{name}  (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random classifier")
    ax.set_xlabel("False Positive Rate  (1 − Specificity)", fontsize=11)
    ax.set_ylabel("True Positive Rate  (Sensitivity)", fontsize=11)
    ax.set_title("ROC Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower right")
    fig.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / "roc_curves.png", bbox_inches="tight")
    return fig


def plot_precision_recall_curves(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#5CC8C8", "#E05C5C", "#6B7FD4", "#F0A500", "#8B5CF6"]
    no_skill = (y_test == MALIGNANT_LABEL).mean()

    for (name, model), color in zip(models.items(), colors):
        y_prob = model.predict_proba(X_test)[:, 0]
        prec, rec, _ = precision_recall_curve(1 - y_test, y_prob)
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, lw=2.2, color=color,
                label=f"{name}  (AUC-PR = {pr_auc:.3f})")

    ax.axhline(no_skill, linestyle="--", color="gray", lw=1.2,
               label=f"No-skill baseline ({no_skill:.2f})")
    ax.set_xlabel("Recall  (Sensitivity)", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False, loc="lower left")
    fig.tight_layout()

    if save_dir:
        fig.savefig(Path(save_dir) / "pr_curves.png", bbox_inches="tight")
    return fig


def plot_calibration_curves(
    models: dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    save_dir: Path | None = None,
) -> plt.Figure:
    n_bins = 10
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#5CC8C8", "#E05C5C", "#6B7FD4", "#F0A500", "#8B5CF6"]

    for (name, model), color in zip(models.items(), colors):
        CalibrationDisplay.from_estimator(
            model, X_test, y_test,
            n_bins=n_bins, ax=ax, name=name,
            color=color, strategy="uniform",
        )

    ax.set_title("Calibration Curves — All Models", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, frameon=False)
    fig.tight_layout()

    if save_dir:
        fig.savefig(Path(save_dir) / "calibration_curves.png", bbox_inches="tight")
    return fig


def plot_confusion_matrix(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "model",
    threshold: float | None = None,
    save_dir: Path | None = None,
) -> plt.Figure:
    y_prob = model.predict_proba(X_test)
    prob_mal = y_prob[:, 0]
    if threshold is None:
        threshold, _ = youden_threshold(y_test, prob_mal)
    y_pred = np.where(prob_mal >= threshold, 0, 1)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Malignant", "Benign"],
    )
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {model_name}\n(threshold = {threshold:.3f})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()

    if save_dir:
        fig.savefig(
            Path(save_dir) / f"confusion_matrix_{model_name}.png",
            bbox_inches="tight",
        )
    return fig


def select_best_model(
    eval_df: pd.DataFrame,
    target_sensitivity: float = 0.97,
) -> str:
    """
    Select the model with:
      1. sensitivity >= target_sensitivity
      2. highest AUC-ROC among those that pass the sensitivity gate

    Falls back to pure AUC-ROC ranking if no model meets the gate.
    """
    qualified = eval_df[eval_df["sensitivity"] >= target_sensitivity]
    if qualified.empty:
        log.warning(
            "No model achieved sensitivity ≥ %.2f. Falling back to AUC-ROC ranking.",
            target_sensitivity,
        )
        best = eval_df.iloc[0]["model"]
    else:
        best = qualified.iloc[0]["model"]

    log.info("Selected best model: %s", best)
    return best
