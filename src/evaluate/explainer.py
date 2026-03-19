"""
src/evaluate/explainer.py
──────────────────────────
SHAP-based model explainability for clinician-facing outputs.

Produces
--------
  • Global feature importance bar chart
  • Beeswarm (summary) plot
  • Individual patient waterfall plot
  • Top-10 feature ranking (for API response)

LIME secondary explainer is also included for cross-validation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

log = logging.getLogger(__name__)

plt.rcParams.update({
    "figure.dpi": 120,
    "font.family": "sans-serif",
})


# ─── SHAP Explainer Factory ───────────────────────────────────────────────────

def build_shap_explainer(
    model: Any,
    X_background: np.ndarray,
    model_name: str = "model",
) -> shap.Explainer:
    """
    Choose the most appropriate SHAP explainer for the given model type.

    • Tree models  → TreeExplainer  (fast & exact)
    • Others       → KernelExplainer (model-agnostic, slower)
    """
    model_class = type(model).__name__.lower()

    if any(k in model_class for k in ("forest", "xgb", "gradient", "tree", "boost")):
        log.info("[%s] Using shap.TreeExplainer", model_name)
        try:
            explainer = shap.TreeExplainer(model)
        except Exception:
            log.warning("TreeExplainer failed, falling back to KernelExplainer")
            background = shap.sample(X_background, 100)
            explainer = shap.KernelExplainer(model.predict_proba, background)
    else:
        log.info("[%s] Using shap.KernelExplainer (background=%d samples)",
                 model_name, min(100, len(X_background)))
        background = shap.sample(X_background, min(100, len(X_background)))
        explainer = shap.KernelExplainer(model.predict_proba, background)

    return explainer


def compute_shap_values(
    explainer: shap.Explainer,
    X: np.ndarray,
    model_name: str = "model",
) -> np.ndarray | list:
    """Compute SHAP values and return as array."""
    log.info("[%s] Computing SHAP values for %d samples …", model_name, len(X))
    shap_values = explainer(X)
    return shap_values


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_feature_importance_bar(
    shap_values: Any,
    feature_names: list[str],
    model_name: str = "model",
    n_top: int = 15,
    save_dir: Path | None = None,
) -> plt.Figure:
    """Global importance bar chart (mean |SHAP|)."""
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = np.array(shap_values)

    # For multi-output (2-class), take malignant class (index 0)
    if vals.ndim == 3:
        vals = vals[:, :, 0]

    mean_abs = np.abs(vals).mean(axis=0)
    idx_sorted = np.argsort(mean_abs)[::-1][:n_top]

    feat_top   = [feature_names[i] for i in idx_sorted]
    import_top = mean_abs[idx_sorted]

    fig, ax = plt.subplots(figsize=(9, 5))
    bar_colors = [
        f"#{int(255 - 180 * v / import_top.max()):02x}"
        f"8080"
        for v in import_top
    ]
    ax.barh(
        range(len(feat_top))[::-1],
        import_top,
        color="#5CC8C8",
        edgecolor="none",
        alpha=0.85,
    )
    ax.set_yticks(range(len(feat_top)))
    ax.set_yticklabels(feat_top[::-1], fontsize=9)
    ax.set_xlabel("Mean |SHAP value|  (impact on model output)", fontsize=10)
    ax.set_title(
        f"Global Feature Importance — {model_name}\n"
        "(Higher = stronger influence on malignant prediction)",
        fontsize=11, fontweight="bold",
    )
    fig.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(
            Path(save_dir) / f"shap_bar_{model_name}.png",
            bbox_inches="tight",
        )
    log.info("SHAP bar chart saved.")
    return fig


def plot_beeswarm(
    shap_values: Any,
    feature_names: list[str],
    model_name: str = "model",
    max_display: int = 15,
    save_dir: Path | None = None,
) -> plt.Figure:
    """SHAP beeswarm (summary) plot — shows direction of impact."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if hasattr(shap_values, "values"):
        vals = shap_values.values
        data = shap_values.data
    else:
        vals = np.array(shap_values)
        data = None

    if vals.ndim == 3:
        vals = vals[:, :, 0]
        if data is not None and isinstance(data, np.ndarray) and data.ndim == 3:
            data = data[:, :, 0]

    shap.summary_plot(
        vals,
        features=data,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=(10, 6),
    )
    plt.title(
        f"SHAP Beeswarm — {model_name}\n"
        "(Red = high feature value  |  Blue = low feature value)",
        fontsize=11, fontweight="bold",
    )
    plt.tight_layout()
    fig = plt.gcf()

    if save_dir:
        fig.savefig(
            Path(save_dir) / f"shap_beeswarm_{model_name}.png",
            bbox_inches="tight",
        )
    log.info("SHAP beeswarm plot saved.")
    return fig


def plot_waterfall(
    explainer: shap.Explainer,
    X_single: np.ndarray,
    feature_names: list[str],
    sample_idx: int = 0,
    model_name: str = "model",
    save_dir: Path | None = None,
) -> plt.Figure:
    """
    Waterfall plot explaining a single patient's prediction.

    This is the clinician-facing output: each bar shows how much and in
    which direction a feature pushed the prediction away from the
    base rate (E[f(x)]).
    """
    sv = explainer(X_single[sample_idx : sample_idx + 1])

    # Handle multi-output: take malignant class
    if hasattr(sv, "values") and sv.values.ndim == 3:
        sv_plot = shap.Explanation(
            values=sv.values[0, :, 0],
            base_values=sv.base_values[0, 0],
            data=sv.data[0],
            feature_names=feature_names,
        )
    elif hasattr(sv, "values") and sv.values.ndim == 2:
        sv_plot = sv[0]
    else:
        sv_plot = sv[0]

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(sv_plot, max_display=15, show=False)
    plt.title(
        f"Patient #{sample_idx} — Individual Prediction Explanation\n"
        f"({model_name})  |  Features pushing ← towards Benign / → towards Malignant",
        fontsize=10, fontweight="bold",
    )
    plt.tight_layout()
    fig = plt.gcf()

    if save_dir:
        fig.savefig(
            Path(save_dir) / f"shap_waterfall_sample{sample_idx}_{model_name}.png",
            bbox_inches="tight",
        )
    log.info("SHAP waterfall plot saved.")
    return fig


def top_features_by_shap(
    shap_values: Any,
    feature_names: list[str],
    n: int = 10,
) -> list[dict[str, float]]:
    """
    Return top-n features ranked by mean |SHAP| importance.

    Used in the API response payload.
    """
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = np.array(shap_values)

    if vals.ndim == 3:
        vals = vals[:, :, 0]

    mean_abs = np.abs(vals).mean(axis=0)
    idx_sorted = np.argsort(mean_abs)[::-1][:n]

    return [
        {"feature": feature_names[i], "importance": round(float(mean_abs[i]), 5)}
        for i in idx_sorted
    ]


# ─── LIME Secondary Explainer ─────────────────────────────────────────────────

def lime_explanation(
    model: Any,
    X_train: np.ndarray,
    X_single: np.ndarray,
    feature_names: list[str],
    class_names: list[str] | None = None,
    sample_idx: int = 0,
    save_dir: Path | None = None,
) -> Any:
    """
    Generate a LIME explanation for a single patient.

    Returns the LimeTabularExplainer explanation object so the notebook
    can call `exp.show_in_notebook()` interactively.
    """
    try:
        import lime
        import lime.lime_tabular
    except ImportError:
        log.warning("LIME not installed. Run: pip install lime")
        return None

    class_names = class_names or ["Malignant", "Benign"]

    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True,
        random_state=42,
    )

    exp = explainer.explain_instance(
        X_single[sample_idx],
        model.predict_proba,
        num_features=10,
        labels=(0,),   # explain malignant class
    )

    if save_dir:
        html_path = Path(save_dir) / f"lime_explanation_sample{sample_idx}.html"
        exp.save_to_file(str(html_path))
        log.info("LIME explanation saved to %s", html_path)

    return exp
