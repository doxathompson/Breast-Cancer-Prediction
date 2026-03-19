"""
src/api/main.py
────────────────
FastAPI REST service for the Breast Cancer Predictor.

Endpoints
---------
  GET  /health          → service health check
  POST /predict         → predict malignant | benign from 30 features
  GET  /model-info      → loaded model metadata

Input validation is handled by Pydantic v2.
Response includes prediction, confidence, and top SHAP features.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import hashlib
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from src.api.auth import get_current_user

log = logging.getLogger("breast_cancer_api")
logging.basicConfig(level=logging.INFO)

# ─── Model paths (can be overridden via env-vars) ─────────────────────────────
MODEL_PATH     = Path(os.getenv("MODEL_PATH",    "models/registry/best_model.joblib"))
PIPELINE_PATH  = Path(os.getenv("PIPELINE_PATH", "models/registry/preprocessor.joblib"))
EXPLAINER_PATH = Path(os.getenv("EXPLAINER_PATH","models/registry/shap_explainer.joblib"))

# Global state
_model: Any = None
_pipeline: Any = None
_explainer: Any = None
_feature_names: list[str] = []
_load_time: str = ""

FEATURE_NAMES_30 = [
    "mean_radius","mean_texture","mean_perimeter","mean_area","mean_smoothness",
    "mean_compactness","mean_concavity","mean_concave_points","mean_symmetry",
    "mean_fractal_dimension","radius_error","texture_error","perimeter_error",
    "area_error","smoothness_error","compactness_error","concavity_error",
    "concave_points_error","symmetry_error","fractal_dimension_error",
    "worst_radius","worst_texture","worst_perimeter","worst_area","worst_smoothness",
    "worst_compactness","worst_concavity","worst_concave_points","worst_symmetry",
    "worst_fractal_dimension",
]


# ─── Lifespan ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artefacts on startup."""
    global _model, _pipeline, _explainer, _feature_names, _load_time
    t0 = time.perf_counter()

    if MODEL_PATH.exists():
        _model = joblib.load(MODEL_PATH)
        log.info("Model loaded from %s", MODEL_PATH)
    else:
        log.warning("Model file not found at %s — /predict will return 503", MODEL_PATH)

    if PIPELINE_PATH.exists():
        _pipeline = joblib.load(PIPELINE_PATH)
        log.info("Preprocessing pipeline loaded from %s", PIPELINE_PATH)

    if EXPLAINER_PATH.exists():
        _explainer = joblib.load(EXPLAINER_PATH)
        log.info("SHAP explainer loaded from %s", EXPLAINER_PATH)

    _feature_names = FEATURE_NAMES_30
    _load_time = f"{time.perf_counter() - t0:.2f}s"

    yield   # Application runs here

    log.info("Shutting down Breast Cancer Predictor API.")


# ─── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Breast Cancer Predictor API",
    description=(
        "Clinical-grade binary classification endpoint for the Wisconsin "
        "Breast Cancer Diagnostic dataset. "
        "Minimises false negatives (missed malignancies) using a Youden-optimal threshold."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ─── Schemas ──────────────────────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    """30 fine needle aspirate features (mean, SE, worst for 10 cell nuclei properties)."""

    mean_radius:             float = Field(..., ge=0, description="Mean of distances from center to perimeter points")
    mean_texture:            float = Field(..., ge=0, description="Standard deviation of gray-scale values")
    mean_perimeter:          float = Field(..., ge=0)
    mean_area:               float = Field(..., ge=0)
    mean_smoothness:         float = Field(..., ge=0, le=1)
    mean_compactness:        float = Field(..., ge=0)
    mean_concavity:          float = Field(..., ge=0)
    mean_concave_points:     float = Field(..., ge=0)
    mean_symmetry:           float = Field(..., ge=0, le=1)
    mean_fractal_dimension:  float = Field(..., ge=0, le=1)

    radius_error:            float = Field(..., ge=0)
    texture_error:           float = Field(..., ge=0)
    perimeter_error:         float = Field(..., ge=0)
    area_error:              float = Field(..., ge=0)
    smoothness_error:        float = Field(..., ge=0)
    compactness_error:       float = Field(..., ge=0)
    concavity_error:         float = Field(..., ge=0)
    concave_points_error:    float = Field(..., ge=0)
    symmetry_error:          float = Field(..., ge=0)
    fractal_dimension_error: float = Field(..., ge=0)

    worst_radius:            float = Field(..., ge=0)
    worst_texture:           float = Field(..., ge=0)
    worst_perimeter:         float = Field(..., ge=0)
    worst_area:              float = Field(..., ge=0)
    worst_smoothness:        float = Field(..., ge=0, le=1)
    worst_compactness:       float = Field(..., ge=0)
    worst_concavity:         float = Field(..., ge=0)
    worst_concave_points:    float = Field(..., ge=0)
    worst_symmetry:          float = Field(..., ge=0, le=1)
    worst_fractal_dimension: float = Field(..., ge=0, le=1)

    model_config = {"json_schema_extra": {
        "example": {
            "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8,
            "mean_area": 1001.0, "mean_smoothness": 0.1184, "mean_compactness": 0.2776,
            "mean_concavity": 0.3001, "mean_concave_points": 0.1471, "mean_symmetry": 0.2419,
            "mean_fractal_dimension": 0.07871, "radius_error": 1.095, "texture_error": 0.9053,
            "perimeter_error": 8.589, "area_error": 153.4, "smoothness_error": 0.006399,
            "compactness_error": 0.04904, "concavity_error": 0.05373, "concave_points_error": 0.01587,
            "symmetry_error": 0.03003, "fractal_dimension_error": 0.006193, "worst_radius": 25.38,
            "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
            "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119,
            "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189,
        }
    }}


class PredictionResponse(BaseModel):
    prediction:   str   = Field(..., description="'malignant' or 'benign'")
    confidence:   float = Field(..., ge=0, le=1, description="Probability of predicted class")
    p_malignant:  float = Field(..., ge=0, le=1, description="Probability of malignant class")
    threshold:    float = Field(..., description="Youden-optimal classification threshold")
    top_features: list[dict] = Field(default_factory=list,
                                     description="Top SHAP features driving this prediction")


class HealthResponse(BaseModel):
    status:      str
    model_ready: bool
    load_time:   str
    version:     str


# ─── Helpers ──────────────────────────────────────────────────────────────────

YOUDEN_THRESHOLD: float = 0.45   # set after training; overrideable via env
_THRESHOLD = float(os.getenv("YOUDEN_THRESHOLD", YOUDEN_THRESHOLD))

def _features_to_array(features: PatientFeatures) -> np.ndarray:
    return np.array([getattr(features, f) for f in FEATURE_NAMES_30]).reshape(1, -1)


def _get_top_shap_features(X: np.ndarray, n: int = 5) -> list[dict]:
    if _explainer is None:
        return []
    try:
        sv = _explainer(X)
        vals = sv.values if hasattr(sv, "values") else np.array(sv)
        if vals.ndim == 3:
            vals = vals[:, :, 0]
        importance = np.abs(vals[0])
        idx_sorted = np.argsort(importance)[::-1][:n]
        return [
            {
                "feature": _feature_names[i],
                "shap_value": round(float(sv.values[0, i, 0] if hasattr(sv, "values") and sv.values.ndim == 3
                                         else (sv.values[0, i] if hasattr(sv, "values") else vals[0, i])), 5),
                "importance": round(float(importance[i]), 5),
            }
            for i in idx_sorted
        ]
    except Exception as exc:
        log.warning("SHAP computation failed: %s", exc)
        return []


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Service liveness + readiness check."""
    return HealthResponse(
        status="healthy" if _model is not None else "degraded",
        model_ready=_model is not None,
        load_time=_load_time,
        version="1.0.0",
    )


@app.get("/model-info", tags=["Monitoring"])
async def model_info():
    """Return metadata about the loaded model."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "model_type": type(_model).__name__,
        "youden_threshold": _THRESHOLD,
        "n_features": len(_feature_names),
        "features": _feature_names,
    }


@app.post("/extract-features", tags=["Computer Vision"])
async def extract_features_from_image(file: UploadFile = File(...), username: str = Depends(get_current_user)):
    """
    Simulated Computer Vision (CNN) pipeline. 
    In a real hospital, this would pass the FNA histopathology slide image through a ResNet/EfficientNet 
    to extract the 30 geometric cell nucleus parameters.
    For this demo, we consistently mock the extracted features based on the image hash.
    """
    contents = await file.read()
    img_hash = int(hashlib.md5(contents).hexdigest(), 16)
    np.random.seed(img_hash % (2**32 - 1))
    
    # Generate realistic values based on typical WBCD bounds
    features = {
        "mean_radius": round(np.random.uniform(10.0, 25.0), 3),
        "mean_texture": round(np.random.uniform(12.0, 30.0), 3),
        "mean_perimeter": round(np.random.uniform(60.0, 160.0), 3),
        "mean_area": round(np.random.uniform(300.0, 1500.0), 1),
        "mean_smoothness": round(np.random.uniform(0.08, 0.12), 4),
        "mean_compactness": round(np.random.uniform(0.05, 0.25), 4),
        "mean_concavity": round(np.random.uniform(0.05, 0.35), 4),
        "mean_concave_points": round(np.random.uniform(0.02, 0.15), 4),
        "mean_symmetry": round(np.random.uniform(0.13, 0.25), 4),
        "mean_fractal_dimension": round(np.random.uniform(0.05, 0.08), 5),
        
        "radius_error": round(np.random.uniform(0.2, 1.5), 4),
        "texture_error": round(np.random.uniform(0.5, 3.0), 4),
        "perimeter_error": round(np.random.uniform(1.0, 10.0), 4),
        "area_error": round(np.random.uniform(10.0, 150.0), 4),
        "smoothness_error": round(np.random.uniform(0.003, 0.01), 6),
        "compactness_error": round(np.random.uniform(0.01, 0.06), 6),
        "concavity_error": round(np.random.uniform(0.01, 0.08), 6),
        "concave_points_error": round(np.random.uniform(0.005, 0.02), 6),
        "symmetry_error": round(np.random.uniform(0.01, 0.04), 6),
        "fractal_dimension_error": round(np.random.uniform(0.002, 0.01), 6),
        
        "worst_radius": round(np.random.uniform(12.0, 30.0), 3),
        "worst_texture": round(np.random.uniform(15.0, 40.0), 3),
        "worst_perimeter": round(np.random.uniform(80.0, 200.0), 3),
        "worst_area": round(np.random.uniform(400.0, 2000.0), 1),
        "worst_smoothness": round(np.random.uniform(0.1, 0.18), 4),
        "worst_compactness": round(np.random.uniform(0.1, 0.7), 4),
        "worst_concavity": round(np.random.uniform(0.1, 0.8), 4),
        "worst_concave_points": round(np.random.uniform(0.05, 0.25), 4),
        "worst_symmetry": round(np.random.uniform(0.2, 0.5), 4),
        "worst_fractal_dimension": round(np.random.uniform(0.06, 0.12), 4),
    }
    return {"status": "success", "extracted_features": features}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(patient: PatientFeatures, username: str = Depends(get_current_user)):
    """
    Predict whether a tumour is **malignant** or **benign**.

    Accepts a JSON body with 30 FNA features.
    Returns prediction, confidence, raw P(malignant), Youden threshold,
    and top SHAP contributing features.
    """
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model not yet loaded. Check /health and ensure model artefacts are present.",
        )

    X = _features_to_array(patient)

    # Apply preprocessing pipeline if available
    if _pipeline is not None:
        try:
            X = _pipeline.transform(X)
        except Exception as exc:
            log.error("Preprocessing failed: %s", exc)
            raise HTTPException(status_code=422, detail=f"Preprocessing error: {exc}")

    # Predict
    try:
        y_prob = _model.predict_proba(X)      # shape (1, 2)
        p_malignant = float(y_prob[0, 0])
        p_benign    = float(y_prob[0, 1])
    except Exception as exc:
        log.error("Model inference failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Inference error: {exc}")

    # Apply Youden threshold
    is_malignant = p_malignant >= _THRESHOLD
    label        = "malignant" if is_malignant else "benign"
    confidence   = p_malignant if is_malignant else p_benign

    # SHAP features
    top_features = _get_top_shap_features(X, n=5)

    log.info(
        "Prediction: %s  |  P(malignant)=%.4f  |  threshold=%.4f",
        label, p_malignant, _THRESHOLD,
    )

    return PredictionResponse(
        prediction=label,
        confidence=round(confidence, 4),
        p_malignant=round(p_malignant, 4),
        threshold=_THRESHOLD,
        top_features=top_features,
    )
