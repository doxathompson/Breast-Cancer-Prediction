"""
src/api/monitoring.py
──────────────────────
Data-drift detection and retraining strategy using Evidently.

Strategy
--------
  1. Rolling 30-day prediction log is maintained (CSV append).
  2. On each evaluation window, run Evidently DataDriftPreset
     comparing the current window vs the training reference data.
  3. If Kolmogorov–Smirnov drift p-value < 0.05 in > 30% of features
     OR AUC drops below 0.95 → trigger retraining alert.
  4. Clinical feedback loop: radiologist confirmations consumed from
     a verified_labels.csv file to update ground truth.

Classes
-------
  DriftMonitor   — wraps Evidently reports
  RetrainingGate — decides whether to trigger retraining
"""

from __future__ import annotations

import csv
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ─── Constants ────────────────────────────────────────────────────────────────
AUC_THRESHOLD: float = 0.95
DRIFT_FEATURE_RATIO: float = 0.30     # >30% features drifted → alert
ROLLING_WINDOW_DAYS: int = 30


# ─── Prediction Logger ────────────────────────────────────────────────────────

class PredictionLogger:
    """Append-only CSV log of every prediction for monitoring."""

    def __init__(self, log_path: str | Path = "data/prediction_log.csv"):
        self.path = Path(log_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write_header()

    def _write_header(self):
        with open(self.path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "p_malignant", "prediction", "threshold", "verified_label"])

    def log(
        self,
        p_malignant: float,
        prediction: str,
        threshold: float,
        verified_label: str | None = None,
    ) -> None:
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.utcnow().isoformat(),
                round(p_malignant, 6),
                prediction,
                round(threshold, 4),
                verified_label or "",
            ])

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.path, parse_dates=["timestamp"])
        return df

    def rolling_window(self, days: int = ROLLING_WINDOW_DAYS) -> pd.DataFrame:
        df = self.load()
        if df.empty:
            return df
        cutoff = datetime.utcnow() - timedelta(days=days)
        return df[df["timestamp"] >= cutoff]


# ─── Drift Monitor ────────────────────────────────────────────────────────────

class DriftMonitor:
    """
    Wraps Evidently AI DataDriftPreset for feature-level drift detection.

    Usage
    -----
    >>> monitor = DriftMonitor(reference_df, feature_names)
    >>> report = monitor.run(current_df)
    >>> drifted = monitor.is_drift_detected(report)
    """

    def __init__(
        self,
        reference_data: pd.DataFrame,
        feature_names: list[str],
        drift_threshold: float = DRIFT_FEATURE_RATIO,
    ):
        self.reference = reference_data[feature_names].copy()
        self.feature_names = feature_names
        self.drift_threshold = drift_threshold

    def run(self, current_data: pd.DataFrame) -> dict:
        """
        Run Evidently data drift report.

        Returns a dict summary; full HTML report is saved if save_dir given.
        """
        try:
            from evidently.report import Report
            from evidently.metric_preset import DataDriftPreset

            report = Report(metrics=[DataDriftPreset()])
            report.run(
                reference_data=self.reference,
                current_data=current_data[self.feature_names],
            )
            result = report.as_dict()
            return result
        except ImportError:
            log.warning("Evidently not installed. Falling back to KS-test drift detection.")
            return self._ks_drift(current_data)

    def _ks_drift(self, current_data: pd.DataFrame) -> dict:
        """Fallback: scipy KS-test per feature."""
        from scipy import stats

        drifted = []
        for feat in self.feature_names:
            if feat not in current_data.columns:
                continue
            ref_vals  = self.reference[feat].dropna().values
            curr_vals = current_data[feat].dropna().values
            if len(curr_vals) == 0:
                continue
            ks_stat, p_val = stats.ks_2samp(ref_vals, curr_vals)
            if p_val < 0.05:
                drifted.append({"feature": feat, "ks_stat": ks_stat, "p_value": p_val})

        return {
            "n_drifted": len(drifted),
            "n_total": len(self.feature_names),
            "drift_ratio": len(drifted) / max(len(self.feature_names), 1),
            "drifted_features": drifted,
        }

    def is_drift_detected(self, report: dict) -> bool:
        """Return True if >drift_threshold fraction of features are drifted."""
        drift_ratio = report.get("drift_ratio", 0.0)
        if "metrics" in report:
            # Evidently full report — parse summary
            try:
                summary = report["metrics"][0]["result"]["share_of_drifted_columns"]
                drift_ratio = summary
            except (KeyError, IndexError):
                pass
        detected = drift_ratio > self.drift_threshold
        if detected:
            log.warning(
                "DATA DRIFT DETECTED — %.1f%% of features drifted (threshold %.0f%%)",
                drift_ratio * 100, self.drift_threshold * 100,
            )
        return detected


# ─── Retraining Gate ──────────────────────────────────────────────────────────

class RetrainingGate:
    """
    Decides whether retraining should be triggered.

    Trigger conditions (either one sufficient):
      1. AUC on rolling window < AUC_THRESHOLD
      2. Data drift detected on > drift_feature_ratio of features
    """

    def __init__(
        self,
        auc_threshold: float = AUC_THRESHOLD,
        drift_feature_ratio: float = DRIFT_FEATURE_RATIO,
    ):
        self.auc_threshold      = auc_threshold
        self.drift_feature_ratio = drift_feature_ratio

    def should_retrain(
        self,
        rolling_auc: float | None = None,
        drift_detected: bool = False,
    ) -> tuple[bool, str]:
        """
        Return (should_retrain, reason).
        """
        if drift_detected:
            reason = f"Data drift detected in >{self.drift_feature_ratio*100:.0f}% of features."
            log.warning("RETRAINING TRIGGER: %s", reason)
            return True, reason

        if rolling_auc is not None and rolling_auc < self.auc_threshold:
            reason = (
                f"Rolling AUC ({rolling_auc:.4f}) dropped below threshold "
                f"({self.auc_threshold:.2f})."
            )
            log.warning("RETRAINING TRIGGER: %s", reason)
            return True, reason

        log.info(
            "Retraining gate: no trigger  (AUC=%.4f, drift=%s)",
            rolling_auc or -1.0, drift_detected,
        )
        return False, "No retraining required."


# ─── Clinical Feedback Loop ───────────────────────────────────────────────────

class ClinicalFeedbackLoop:
    """
    Consumes radiologist-verified labels and appends them to the
    prediction log, enabling ground-truth refresh.

    Expected CSV schema for verified_labels.csv:
        timestamp, p_malignant, radiologist_label
        (radiologist_label: 'malignant' | 'benign')
    """

    def __init__(
        self,
        prediction_logger: PredictionLogger,
        verified_labels_path: str | Path = "data/verified_labels.csv",
    ):
        self.logger = prediction_logger
        self.verified_path = Path(verified_labels_path)

    def ingest_verified_labels(self) -> int:
        """
        Read verified_labels.csv and update prediction log.
        Returns number of labels ingested.
        """
        if not self.verified_path.exists():
            log.info("No verified labels file found at %s", self.verified_path)
            return 0

        new_labels = pd.read_csv(self.verified_path)
        ingested = 0
        for _, row in new_labels.iterrows():
            self.logger.log(
                p_malignant=row.get("p_malignant", 0.0),
                prediction=row.get("radiologist_label", ""),
                threshold=0.0,
                verified_label=row.get("radiologist_label", ""),
            )
            ingested += 1

        log.info("Ingested %d verified labels from %s", ingested, self.verified_path)
        return ingested

    def compute_rolling_auc(self, days: int = ROLLING_WINDOW_DAYS) -> float | None:
        """
        Compute AUC on rolling window of verified predictions.
        Returns None if insufficient labeled data is available.
        """
        from sklearn.metrics import roc_auc_score

        df = self.logger.rolling_window(days=days)
        labeled = df[df["verified_label"].astype(str).str.strip() != ""]
        if len(labeled) < 30:
            log.info("Insufficient verified labels (%d < 30) for AUC computation.", len(labeled))
            return None

        y_true = (labeled["verified_label"] == "malignant").astype(int)
        y_prob = labeled["p_malignant"].values
        try:
            auc = roc_auc_score(1 - y_true, 1 - y_prob)   # keep malignant=positive
            log.info("Rolling %d-day AUC: %.4f", days, auc)
            return auc
        except ValueError as e:
            log.warning("AUC computation failed: %s", e)
            return None
