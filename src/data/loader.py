"""
src/data/loader.py
──────────────────
Data ingestion and stratified train / val / test splitting for the
Wisconsin Breast Cancer Diagnostic (WBCD) dataset.

The sklearn fetch_openml loader is used as the primary source so that
no external file is required; the raw CSV is also saved to data/raw/
for reproducibility.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

RANDOM_STATE: int = 42


# ─── Public API ───────────────────────────────────────────────────────────────

def load_wbcd(
    raw_dir: str | Path = "data/raw",
    *,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Load the Wisconsin Breast Cancer Diagnostic dataset.

    Parameters
    ----------
    raw_dir : path-like
        Directory to persist the raw CSV cache.
    save_csv : bool
        Whether to write the CSV to *raw_dir*.

    Returns
    -------
    pd.DataFrame
        569 rows × 31 columns  (30 features + ``diagnosis`` target).
        ``diagnosis``: 0 = malignant, 1 = benign  (sklearn convention).
    """
    raw_dir = Path(raw_dir)
    csv_path = raw_dir / "breast_cancer.csv"

    if csv_path.exists():
        log.info("Loading cached dataset from %s", csv_path)
        df = pd.read_csv(csv_path)
        return df

    log.info("Fetching Wisconsin Breast Cancer dataset via sklearn …")
    bunch = load_breast_cancer(as_frame=True)

    df: pd.DataFrame = bunch.frame.copy()
    # sklearn encodes: 0=malignant, 1=benign — keep that convention
    # Rename for clarity
    df = df.rename(columns={"target": "diagnosis"})

    if save_csv:
        raw_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        log.info("Raw data saved to %s  (%d rows)", csv_path, len(df))

    return df


def split_data(
    df: pd.DataFrame,
    *,
    target_col: str = "diagnosis",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,
    pd.Series,   pd.Series,   pd.Series,
]:
    """
    Stratified 70 / 15 / 15 train / validation / test split.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # First carve out the holdout test set
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # From the remaining data, carve out validation
    relative_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp,
        test_size=relative_val,
        stratify=y_tmp,
        random_state=random_state,
    )

    log.info(
        "Split sizes → train=%d  val=%d  test=%d  (target 70/15/15)",
        len(X_train), len(X_val), len(X_test),
    )
    _log_class_distribution("train", y_train)
    _log_class_distribution("val",   y_val)
    _log_class_distribution("test",  y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _log_class_distribution(split_name: str, y: pd.Series) -> None:
    counts = y.value_counts().sort_index()
    pct = (counts / len(y) * 100).round(1)
    log.info(
        "  %s class dist: malignant=%d (%.1f%%)  benign=%d (%.1f%%)",
        split_name,
        counts.get(0, 0), pct.get(0, 0.0),
        counts.get(1, 0), pct.get(1, 0.0),
    )


def feature_names() -> list[str]:
    """Return the canonical 30 feature names from sklearn."""
    return list(load_breast_cancer().feature_names)
