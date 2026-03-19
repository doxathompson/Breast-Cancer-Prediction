"""
src/data/preprocessor.py
─────────────────────────
sklearn Pipeline that handles:
  1. SimpleImputer (median strategy) — handles any missing values
  2. HighCorrelationDropper (custom transformer, threshold > 0.95)
  3. StandardScaler

Also exposes two stand-alone feature-selection approaches:
  • SelectKBest with f_classif
  • PCA (retaining 95% explained variance)

Both are kept separate from the main pipeline so they can be compared
fairly during the feature-engineering exploration in the notebook.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

RANDOM_STATE: int = 42


# ─── Custom Transformer ───────────────────────────────────────────────────────

class HighCorrelationDropper(BaseEstimator, TransformerMixin):
    """
    Drop features whose absolute pairwise correlation exceeds *threshold*.

    Keeps the first feature encountered in each highly-correlated pair
    (consistent with the order columns appear in the training data).
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X: np.ndarray, y=None) -> "HighCorrelationDropper":
        if isinstance(X, pd.DataFrame):
            df = X
        else:
            df = pd.DataFrame(X)

        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape, dtype=bool), k=1)
        )
        self.drop_cols_: list[int] = [
            i for i, col in enumerate(upper.columns)
            if (upper[col] > self.threshold).any()
        ]
        log.info(
            "HighCorrelationDropper: dropping %d / %d features (threshold=%.2f)",
            len(self.drop_cols_), df.shape[1], self.threshold,
        )
        return self

    def transform(self, X: np.ndarray, y=None) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            X = X.values
        mask = np.ones(X.shape[1], dtype=bool)
        for idx in self.drop_cols_:
            mask[idx] = False
        return X[:, mask]

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return None
        return [f for i, f in enumerate(input_features) if i not in self.drop_cols_]


# ─── Main Preprocessing Pipeline ──────────────────────────────────────────────

def build_preprocessing_pipeline(
    correlation_threshold: float = 0.95,
) -> Pipeline:
    """
    Return a fitted-ready sklearn Pipeline:
      Imputer → HighCorrelationDropper → StandardScaler
    """
    return Pipeline(
        steps=[
            ("imputer",      SimpleImputer(strategy="median")),
            ("corr_dropper", HighCorrelationDropper(threshold=correlation_threshold)),
            ("scaler",       StandardScaler()),
        ],
        verbose=False,
    )


# ─── Feature Selection Helpers ────────────────────────────────────────────────

def select_k_best(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    k: int = 15,
    feature_names: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, SelectKBest, list[str]]:
    """
    Fit SelectKBest(f_classif) on training data and transform both sets.

    Returns
    -------
    X_train_sel, X_test_sel, selector, selected_feature_names
    """
    selector = SelectKBest(score_func=f_classif, k=k)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel  = selector.transform(X_test)

    if feature_names:
        selected = [feature_names[i] for i in selector.get_support(indices=True)]
    else:
        selected = [str(i) for i in selector.get_support(indices=True)]

    log.info("SelectKBest kept features: %s", selected)
    return X_train_sel, X_test_sel, selector, selected


def apply_pca(
    X_train: np.ndarray,
    X_test: np.ndarray,
    variance_threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """
    Fit PCA (retaining *variance_threshold* explained variance) on training
    data and transform both sets.

    Returns
    -------
    X_train_pca, X_test_pca, pca
    """
    pca = PCA(n_components=variance_threshold, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca  = pca.transform(X_test)

    log.info(
        "PCA: %d components explain %.1f%% of variance",
        pca.n_components_, sum(pca.explained_variance_ratio_) * 100,
    )
    return X_train_pca, X_test_pca, pca
