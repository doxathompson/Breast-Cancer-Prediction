# 🔬 Breast Cancer Predictor

> **Clinical-grade binary classification** for the Wisconsin Breast Cancer Diagnostic (WBCD) dataset.  
> Prioritises **minimising false negatives** (missed malignancies) above all other metrics.

---

## Architecture

```
Data Ingestion → EDA → Preprocessing Pipeline → Feature Engineering → Class Balancing
       ↓
Model Training (LR · RF · XGBoost · SVM · MLP) [GridSearchCV, Strat 5-Fold]
       ↓
Clinical Evaluation (AUC-ROC · Sensitivity ≥ 0.97 · Youden Threshold)
       ↓
SHAP Explainability (Bar · Beeswarm · Waterfall)
       ↓
FastAPI Deployment → Monitoring & Retraining Strategy
```

See also: [`breast_cancer_ml_pipeline.svg`](breast_cancer_ml_pipeline.svg) for the full visual diagram.

---

## Project Structure

```
Breast Cancer Prediction/
├── config/
│   └── config.yaml            # All hyperparameters & thresholds
├── data/
│   ├── raw/                   # Auto-downloaded dataset CSV
│   └── processed/             # Reference data for drift monitoring
├── models/
│   └── registry/              # Serialised model artefacts (joblib)
├── notebooks/
│   └── breast_cancer_pipeline.ipynb  # Full interactive walkthrough
├── reports/                   # EDA & evaluation plots
│   └── shap_plots/            # SHAP explainability plots
├── src/
│   ├── data/
│   │   ├── loader.py          # Data ingestion & stratified splits
│   │   ├── eda.py             # EDA plots (class dist, histograms, pairplot, heatmap)
│   │   └── preprocessor.py   # Pipeline: Imputer → CorrelationDropper → Scaler
│   ├── models/
│   │   └── trainer.py         # GridSearchCV training for all 5 classifiers
│   ├── evaluate/
│   │   ├── metrics.py         # AUC-ROC, sensitivity, specificity, Youden Index
│   │   └── explainer.py       # SHAP + LIME explainability
│   └── api/
│       ├── main.py            # FastAPI REST service
│       └── monitoring.py      # Drift detection & retraining strategy
├── train.py                   # CLI end-to-end training script
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train all models

```bash
python train.py
```

Optional flags:
```bash
python train.py --smote        # Enable SMOTE oversampling
python train.py --no-shap      # Skip SHAP (faster run)
python train.py --config config/config.yaml
```

### 3. Launch the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Make a prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
    "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
    "mean_concavity": 0.3001, "mean_concave_points": 0.1471, "mean_symmetry": 0.2419,
    "mean_fractal_dimension": 0.07871, "radius_error": 1.095, "texture_error": 0.9053,
    "perimeter_error": 8.589, "area_error": 153.4, "smoothness_error": 0.006399,
    "compactness_error": 0.04904, "concavity_error": 0.05373, "concave_points_error": 0.01587,
    "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193, "worst_radius": 25.38,
    "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
    "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119,
    "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
  }'
```

Expected response:
```json
{
  "prediction": "malignant",
  "confidence": 0.9872,
  "p_malignant": 0.9872,
  "threshold": 0.4312,
  "top_features": [
    {"feature": "worst_concave_points", "shap_value": 0.82, "importance": 0.82},
    ...
  ]
}
```

### 5. Check API health

```bash
curl http://localhost:8000/health
```

### 6. Open the Jupyter notebook

```bash
jupyter notebook notebooks/breast_cancer_pipeline.ipynb
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/health` | Service liveness + model readiness |
| `GET`  | `/model-info` | Loaded model metadata & features |
| `POST` | `/predict` | Predict malignant/benign + confidence + SHAP features |
| `GET`  | `/docs` | Interactive OpenAPI documentation |

---

## Clinical Design Decisions

### Minimising False Negatives
- **Youden Index** threshold selection maximises `Sensitivity + Specificity − 1`
- Final model selected only if **Sensitivity ≥ 0.97** on held-out test set
- Class imbalance handled via `class_weight='balanced'` (preferred for n=569 over SMOTE)

### Model Selection Logic
1. All models evaluated by AUC-ROC on the held-out test set
2. Sensitivity gate: any model not achieving ≥0.97 sensitivity is disqualified
3. Among qualified models → highest AUC-ROC wins

### Explainability for Clinicians
- **Global SHAP bar chart**: which features matter overall
- **SHAP beeswarm**: direction and magnitude of each feature's effect
- **Waterfall plot**: per-patient explanation showing exactly which features pushed the decision
- **LIME**: secondary model-agnostic explanation for cross-validation

---

## Monitoring & Retraining

### Drift Detection
- **Primary**: Evidently AI `DataDriftPreset` (KS test per feature)
- **Fallback**: SciPy 2-sample KS test
- **Trigger**: Drift in >30% of features OR p-value < 0.05

### Retraining Gate
- Rolling 30-day AUC monitored from `data/prediction_log.csv`
- Retraining triggered if AUC drops below **0.95**

### Clinical Feedback Loop
1. Predictions logged to `data/prediction_log.csv`
2. Radiologist confirmations uploaded to `data/verified_labels.csv`
3. `ClinicalFeedbackLoop.ingest_verified_labels()` merges ground truth
4. Rolling AUC recomputed → `RetrainingGate.should_retrain()` fires if needed

---

## Configuration

All tunable parameters live in `config/config.yaml`:

```yaml
project:
  random_state: 42

clinical:
  target_sensitivity: 0.97
  youden_index_optimize: true

preprocessing:
  correlation_threshold: 0.95
  n_features_kbest: 15

monitoring:
  rolling_window_days: 30
  auc_retraining_trigger: 0.95
```

---

## Dataset Citation

> Wolberg, W.H., Street, W.N., & Mangasarian, O.L. (1995).  
> *Breast Cancer Wisconsin (Diagnostic) Data Set.*  
> UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B

---

## Regulatory Note (FDA/CE)

This system is a **clinical decision support tool** and must:
- Be validated prospectively before clinical use
- Not replace physician judgement
- Be retrained and re-validated on institutional data
- Maintain audit logs of all predictions and radiologist overrides
- Be governed by relevant IEC 62304 / ISO 14971 frameworks
