"""
train.py
─────────
End-to-end training script for the Breast Cancer Predictor.

Usage
-----
    python train.py [--config config/config.yaml] [--smote] [--no-shap]

Steps
-----
  1.  Load WBCD dataset
  2.  EDA (plots saved to reports/)
  3.  Stratified 70/15/15 split
  4.  Build preprocessing pipeline
  5.  Feature engineering comparison (SelectKBest vs PCA)
  6.  Train 5 classifiers with GridSearchCV
  7.  Clinical evaluation + Youden threshold
  8.  Select best model
  9.  SHAP global + beeswarm + waterfall plots
  10. Serialise model + pipeline + SHAP explainer
  11. Print final summary table
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
import mlflow
import mlflow.sklearn

# ─── Make src importable even when running from project root ──────────────────
sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader      import load_wbcd, split_data
from src.data.eda          import run_full_eda
from src.data.preprocessor import build_preprocessing_pipeline, select_k_best, apply_pca
from src.models.trainer    import train_all_models, save_model, summarise_cv_results
from src.evaluate.metrics  import evaluate_all, plot_roc_curves, plot_precision_recall_curves, \
                                    plot_calibration_curves, plot_confusion_matrix, select_best_model
from src.evaluate.explainer import (build_shap_explainer, compute_shap_values,
                                     plot_feature_importance_bar, plot_beeswarm,
                                     plot_waterfall, top_features_by_shap)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")

RANDOM_STATE = 42


# ─── Config ───────────────────────────────────────────────────────────────────

def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    random_state = cfg["project"]["random_state"]
    np.random.seed(random_state)

    reports_dir  = Path(cfg["paths"]["reports_dir"])
    shap_dir     = Path(cfg["paths"]["shap_plots"])
    models_dir   = Path(cfg["paths"]["models_dir"])
    raw_dir      = Path(cfg["paths"]["data_raw"])

    # ─── 1. Load data ─────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 1 — Data Ingestion")
    df = load_wbcd(raw_dir=raw_dir)
    log.info("Dataset shape: %s  |  columns: %d", df.shape, df.shape[1])
    log.info("Class distribution:\n%s", df["diagnosis"].value_counts())

    # ─── 2. EDA ───────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 2 — Exploratory Data Analysis")
    run_full_eda(df, target_col="diagnosis", reports_dir=reports_dir)

    # ─── 3. Split ─────────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 3 — Stratified Train / Val / Test Split (70/15/15)")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_col="diagnosis",
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
        random_state=random_state,
    )

    feature_names = [c for c in df.columns if c != "diagnosis"]

    # ─── 4. Preprocessing ─────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 4 — Preprocessing Pipeline (Imputer → CorrelationDropper → Scaler)")
    preprocessor = build_preprocessing_pipeline(
        correlation_threshold=cfg["preprocessing"]["correlation_threshold"]
    )
    X_train_pp = preprocessor.fit_transform(X_train, y_train)
    X_val_pp   = preprocessor.transform(X_val)
    X_test_pp  = preprocessor.transform(X_test)
    log.info("Post-preprocessing shape: train=%s  val=%s  test=%s",
             X_train_pp.shape, X_val_pp.shape, X_test_pp.shape)

    # ─── 5. Feature Engineering Comparison ────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 5 — Feature Selection Comparison (SelectKBest vs PCA)")

    k = cfg["preprocessing"]["n_features_kbest"]
    X_tr_kbest, X_test_kbest, kbest_selector, selected_feats = select_k_best(
        X_train_pp, y_train.values, X_test_pp, k=k, feature_names=feature_names
    )
    log.info("SelectKBest top-%d: %s", k, selected_feats)

    X_tr_pca, X_test_pca, pca = apply_pca(
        X_train_pp, X_test_pp,
        variance_threshold=cfg["preprocessing"]["pca_variance"],
    )
    log.info("PCA retained %d components", pca.n_components_)

    # ─── 6. Model Training ────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 6 — Model Training with GridSearchCV (stratified 5-fold)")

    # ─── 6. Model Training ────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 6 — Model Training with GridSearchCV (stratified 5-fold)")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Breast_Cancer_Predictor")
    
    with mlflow.start_run(run_name="Full Pipeline Trial"):
        mlflow.log_params({
            "test_size": cfg["data"]["test_size"],
            "val_size": cfg["data"]["val_size"],
            "correlation_threshold": cfg["preprocessing"]["correlation_threshold"],
        })
        
        # Combine train + val for final fitting (val was used during tuning)
        X_fit = np.vstack([X_train_pp, X_val_pp])
        y_fit = pd.concat([y_train, y_val]).reset_index(drop=True)

        scale_pos_weight = (y_fit == 1).sum() / max((y_fit == 0).sum(), 1)

        cv_results = train_all_models(
            X_train_pp,          # CV happens on train only
            y_train.values,
            scale_pos_weight=float(scale_pos_weight),
            use_smote=args.smote,
            cv_splits=cfg["cv"]["n_splits"],
            scoring=cfg["cv"]["scoring"],
        )

        cv_summary = summarise_cv_results(cv_results)
        log.info("\nCV Summary:\n%s", cv_summary.to_string(index=False))

        # Extract best estimators keyed by name
        best_estimators = {name: gs.best_estimator_ for name, gs in cv_results.items()}

        # ─── 7. Evaluation ────────────────────────────────────────────────────────
        log.info("═" * 60)
        log.info("STEP 7 — Clinical Evaluation on Held-out Test Set")

        eval_df = evaluate_all(
            best_estimators, X_test_pp, y_test.values,
            use_youden=cfg["clinical"]["youden_index_optimize"],
        )
        log.info("\nEvaluation Results:\n%s", eval_df.to_string(index=False))
        eval_df.to_csv(reports_dir / "evaluation_results.csv", index=False)

        # Plots
        plot_roc_curves(best_estimators, X_test_pp, y_test.values, save_dir=reports_dir)
        plot_precision_recall_curves(best_estimators, X_test_pp, y_test.values, save_dir=reports_dir)
        plot_calibration_curves(best_estimators, X_test_pp, y_test.values, save_dir=reports_dir)

        # ─── 8. Select Best Model ─────────────────────────────────────────────────
        log.info("═" * 60)
        log.info("STEP 8 — Model Selection (Youden Index + Sensitivity ≥ %.2f)",
                 cfg["clinical"]["target_sensitivity"])

        best_name  = select_best_model(eval_df, cfg["clinical"]["target_sensitivity"])
        best_model = best_estimators[best_name]
        log.info("Best model selected: %s", best_name)

        plot_confusion_matrix(
            best_model, X_test_pp, y_test.values,
            model_name=best_name,
            save_dir=reports_dir,
        )

        best_row = eval_df[eval_df["model"] == best_name].iloc[0]
        mlflow.log_metrics({
            "best_auc_roc": best_row["auc_roc"],
            "best_sensitivity": best_row["sensitivity"],
            "best_specificity": best_row["specificity"],
            "best_f1_macro": best_row["f1_macro"],
            "FN_count": best_row["FN"],
        })
        mlflow.sklearn.log_model(best_model, "best_clinical_model")

    # ─── 9. SHAP Explainability ───────────────────────────────────────────────
    if not args.no_shap:
        log.info("═" * 60)
        log.info("STEP 9 — SHAP Explainability")

        n_bg = min(cfg["shap"]["background_samples"], len(X_train_pp))
        bg_idx = np.random.choice(len(X_train_pp), n_bg, replace=False)
        X_background = X_train_pp[bg_idx]

        shap_explainer = build_shap_explainer(best_model, X_background, model_name=best_name)
        shap_values    = compute_shap_values(shap_explainer, X_test_pp, model_name=best_name)

        # Use preprocessed feature_names if HighCorrelationDropper removed any
        try:
            proc_feats = preprocessor.named_steps["corr_dropper"].get_feature_names_out(feature_names)
            if proc_feats is None:
                proc_feats = feature_names
        except Exception:
            proc_feats = feature_names

        plot_feature_importance_bar(shap_values, list(proc_feats),
                                    model_name=best_name, save_dir=shap_dir)
        plot_beeswarm(shap_values, list(proc_feats), model_name=best_name,
                      max_display=cfg["shap"]["beeswarm_max_display"], save_dir=shap_dir)
        plot_waterfall(shap_explainer, X_test_pp, list(proc_feats),
                       sample_idx=0, model_name=best_name, save_dir=shap_dir)

        top_feats = top_features_by_shap(
            shap_values, list(proc_feats), n=cfg["shap"]["n_top_features"]
        )
        log.info("Top SHAP features:\n%s",
                 "\n".join(f"  {i+1}. {f['feature']}  ({f['importance']:.5f})"
                            for i, f in enumerate(top_feats)))

        # Save explainer
        save_model(shap_explainer, models_dir / "shap_explainer.joblib", compress=1)

    # ─── 10. Serialisation ────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("STEP 10 — Serialising Artefacts")
    save_model(best_model,    models_dir / "best_model.joblib")
    save_model(preprocessor,  models_dir / "preprocessor.joblib")

    # Save reference data for drift monitoring
    processed_dir = Path(cfg["paths"]["data_raw"]).parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    ref_df = pd.DataFrame(X_train_pp, columns=range(X_train_pp.shape[1]))
    ref_df.to_csv(processed_dir / "reference_data.csv", index=False)
    log.info("Reference data saved for drift monitoring.")

    # ─── Final Summary ────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info("✓ TRAINING COMPLETE")
    log.info("Best model: %s", best_name)
    best_row = eval_df[eval_df["model"] == best_name].iloc[0]
    log.info(
        "  AUC-ROC=%.4f  |  Sensitivity=%.4f  |  Specificity=%.4f  |  F1-macro=%.4f  |  FN=%d",
        best_row["auc_roc"], best_row["sensitivity"],
        best_row["specificity"], best_row["f1_macro"], best_row["FN"],
    )
    log.info("Artefacts saved to: %s/", models_dir)
    log.info("Reports saved to:   %s/", reports_dir)


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Breast Cancer Prediction models")
    parser.add_argument("--config",  default="config/config.yaml",
                        help="Path to config YAML  (default: config/config.yaml)")
    parser.add_argument("--smote",   action="store_true",
                        help="Apply SMOTE oversampling inside CV pipeline")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP computation (faster, no explainability output)")
    parsed = parser.parse_args()
    main(parsed)
