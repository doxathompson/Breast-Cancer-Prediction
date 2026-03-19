"""
src/models/trainer.py
──────────────────────
Train and tune all five classifiers with GridSearchCV (stratified 5-fold CV).

Supported models
----------------
  • Logistic Regression (L2)
  • Random Forest
  • XGBoost
  • SVM  (RBF kernel, probability=True)
  • MLP  (shallow neural net)

Class-imbalance handling
------------------------
  • Tree-based models & LR → class_weight="balanced"
  • XGBoost              → scale_pos_weight = n_benign / n_malignant
  • Optional SMOTE       → pass smote=True to train_all_models()
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

log = logging.getLogger(__name__)

RANDOM_STATE: int = 42


# ─── Model Definitions ────────────────────────────────────────────────────────

def _build_model_configs(
    scale_pos_weight: float = 1.68,
) -> dict[str, dict[str, Any]]:
    """
    Return a dictionary mapping model_name → {estimator, param_grid}.
    """
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(
                class_weight="balanced",
                solver="lbfgs",
                max_iter=500,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
            },
        },
        "xgboost": {
            "estimator": XGBClassifier(
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            "param_grid": {
                "learning_rate": [0.01, 0.05, 0.1, 0.2],
                "max_depth": [3, 5, 7],
                "n_estimators": [100, 200, 300],
                "subsample": [0.8, 1.0],
            },
        },
        "svm": {
            "estimator": SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "C": [0.1, 1, 10, 100],
                "gamma": ["scale", "auto"],
            },
        },
        "mlp": {
            "estimator": MLPClassifier(
                activation="relu",
                solver="adam",
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=RANDOM_STATE,
            ),
            "param_grid": {
                "hidden_layer_sizes": [(64,), (128,), (64, 32), (128, 64)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01],
            },
        },
    }


# ─── Public API ───────────────────────────────────────────────────────────────

def train_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    scale_pos_weight: float = 1.68,
    use_smote: bool = False,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> GridSearchCV:
    """
    Train a single model with GridSearchCV and return the CV object.

    Parameters
    ----------
    name            : key from _build_model_configs
    X_train         : pre-processed training features
    y_train         : training labels (0/1)
    scale_pos_weight: XGBoost class-weight parameter
    use_smote       : wrap estimator in an imblearn SMOTE pipeline
    cv_splits       : number of stratified folds
    scoring         : GridSearchCV metric
    n_jobs          : parallel jobs
    """
    configs = _build_model_configs(scale_pos_weight=scale_pos_weight)
    if name not in configs:
        raise ValueError(f"Unknown model '{name}'. Choices: {list(configs)}")

    cfg = configs[name]
    estimator = cfg["estimator"]
    param_grid = cfg["param_grid"]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)

    if use_smote:
        log.info("[%s] Applying SMOTE upsampling inside CV pipeline.", name)
        pipeline = ImbPipeline([
            ("smote", SMOTE(k_neighbors=5, random_state=RANDOM_STATE)),
            ("clf",   estimator),
        ])
        param_grid = {f"clf__{k}": v for k, v in param_grid.items()}
    else:
        pipeline = estimator

    search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
        return_train_score=True,
    )

    t0 = time.perf_counter()
    search.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    log.info(
        "[%s] Best %s = %.4f  |  params = %s  |  time = %.1fs",
        name, scoring, search.best_score_, search.best_params_, elapsed,
    )
    return search


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    scale_pos_weight: float = 1.68,
    use_smote: bool = False,
    cv_splits: int = 5,
    scoring: str = "roc_auc",
    n_jobs: int = -1,
) -> dict[str, GridSearchCV]:
    """
    Train and return all five models.
    """
    results: dict[str, GridSearchCV] = {}
    model_names = list(_build_model_configs().keys())

    for name in model_names:
        log.info("═" * 55)
        log.info("Training: %s", name.upper())
        results[name] = train_model(
            name, X_train, y_train,
            scale_pos_weight=scale_pos_weight,
            use_smote=use_smote,
            cv_splits=cv_splits,
            scoring=scoring,
            n_jobs=n_jobs,
        )

    return results


def save_model(
    model: Any,
    path: str | Path,
    *,
    compress: int = 3,
) -> None:
    """Serialise a trained estimator / pipeline to disk with joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=compress)
    size_kb = path.stat().st_size / 1024
    log.info("Model saved to %s  (%.1f KB)", path, size_kb)


def load_model(path: str | Path) -> Any:
    """Load a serialised estimator from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    model = joblib.load(path)
    log.info("Model loaded from %s", path)
    return model


def summarise_cv_results(results: dict[str, GridSearchCV]) -> pd.DataFrame:
    """Return a DataFrame comparing best CV AUC across all models."""
    rows = []
    for name, gs in results.items():
        rows.append({
            "model": name,
            "best_cv_auc": gs.best_score_,
            "best_params": gs.best_params_,
        })
    df = pd.DataFrame(rows).sort_values("best_cv_auc", ascending=False).reset_index(drop=True)
    return df
