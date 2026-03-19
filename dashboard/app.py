"""
dashboard/app.py
────────────────
Streamlit dashboard for the Breast Cancer Predictor.
Interacts with the FastAPI backend running at http://localhost:8000.
"""

import streamlit as st
import requests
import json
import pandas as pd

API_URL = "http://api:8000" if "docker" in __name__ else "http://localhost:8000"

st.set_page_config(
    page_title="Breast Cancer Clinician Dashboard",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Breast Cancer Clinician Support Dashboard")
st.markdown("""
This dashboard provides a clinical interface to evaluate fine needle aspirate (FNA) 
features predicting whether a breast mass is **Malignant** or **Benign**.
""")

AUTH = ("clinician", "secure_password_123")

# Health check
try:
    health = requests.get(f"{API_URL}/health", timeout=2, auth=AUTH)
    if health.status_code == 200:
        st.sidebar.success("✅ Backend API is online")
    else:
        st.sidebar.warning("⚠️ Backend API returning non-200 status")
except requests.RequestException:
    st.sidebar.error("❌ Backend API is offline. Ensure `uvicorn src.api.main:app` is running.")

st.sidebar.header("Patient FNA Input Parameters")
st.sidebar.markdown("Provide the 30 cell nucleus features below:")

def create_sliders():
    features = {}
    
    st.sidebar.subheader("Mean Values")
    features["mean_radius"] = st.sidebar.slider("Mean Radius", 6.0, 30.0, 17.99)
    features["mean_texture"] = st.sidebar.slider("Mean Texture", 9.0, 40.0, 10.38)
    features["mean_perimeter"] = st.sidebar.slider("Mean Perimeter", 40.0, 190.0, 122.8)
    features["mean_area"] = st.sidebar.slider("Mean Area", 140.0, 2500.0, 1001.0)
    features["mean_smoothness"] = st.sidebar.slider("Mean Smoothness", 0.05, 0.2, 0.1184)
    features["mean_compactness"] = st.sidebar.slider("Mean Compactness", 0.01, 0.4, 0.2776)
    features["mean_concavity"] = st.sidebar.slider("Mean Concavity", 0.0, 0.5, 0.3001)
    features["mean_concave_points"] = st.sidebar.slider("Mean Concave Points", 0.0, 0.2, 0.1471)
    features["mean_symmetry"] = st.sidebar.slider("Mean Symmetry", 0.1, 0.3, 0.2419)
    features["mean_fractal_dimension"] = st.sidebar.slider("Mean Fractal Dimension", 0.04, 0.1, 0.0787)
    
    st.sidebar.subheader("Standard Error (SE) Values")
    features["radius_error"] = st.sidebar.number_input("Radius SE", value=1.095, step=0.1)
    features["texture_error"] = st.sidebar.number_input("Texture SE", value=0.9053, step=0.1)
    features["perimeter_error"] = st.sidebar.number_input("Perimeter SE", value=8.589, step=1.0)
    features["area_error"] = st.sidebar.number_input("Area SE", value=153.4, step=10.0)
    features["smoothness_error"] = st.sidebar.number_input("Smoothness SE", value=0.006399, format="%f", step=0.001)
    features["compactness_error"] = st.sidebar.number_input("Compactness SE", value=0.04904, format="%f", step=0.01)
    features["concavity_error"] = st.sidebar.number_input("Concavity SE", value=0.05373, format="%f", step=0.01)
    features["concave_points_error"] = st.sidebar.number_input("Concave Points SE", value=0.01587, format="%f", step=0.001)
    features["symmetry_error"] = st.sidebar.number_input("Symmetry SE", value=0.03003, format="%f", step=0.01)
    features["fractal_dimension_error"] = st.sidebar.number_input("Fractal Dimension SE", value=0.006193, format="%f", step=0.001)

    st.sidebar.subheader("Worst (Maximum) Values")
    features["worst_radius"] = st.sidebar.number_input("Worst Radius", value=25.38, step=1.0)
    features["worst_texture"] = st.sidebar.number_input("Worst Texture", value=17.33, step=1.0)
    features["worst_perimeter"] = st.sidebar.number_input("Worst Perimeter", value=184.6, step=10.0)
    features["worst_area"] = st.sidebar.number_input("Worst Area", value=2019.0, step=100.0)
    features["worst_smoothness"] = st.sidebar.number_input("Worst Smoothness", value=0.1622, format="%f", step=0.01)
    features["worst_compactness"] = st.sidebar.number_input("Worst Compactness", value=0.6656, format="%f", step=0.1)
    features["worst_concavity"] = st.sidebar.number_input("Worst Concavity", value=0.7119, format="%f", step=0.1)
    features["worst_concave_points"] = st.sidebar.number_input("Worst Concave Points", value=0.2654, format="%f", step=0.01)
    features["worst_symmetry"] = st.sidebar.number_input("Worst Symmetry", value=0.4601, format="%f", step=0.01)
    features["worst_fractal_dimension"] = st.sidebar.number_input("Worst Fractal Dimension", value=0.1189, format="%f", step=0.01)

    return features

patient_features = create_sliders()

if st.sidebar.button("Run Prediction", type="primary"):
    with st.spinner("Analyzing patient features..."):
        try:
            res = requests.post(f"{API_URL}/predict", json=patient_features, auth=AUTH)
            if res.status_code == 200:
                data = res.json()
                
                col1, col2, col3 = st.columns(3)
                is_malignant = data['prediction'] == 'malignant'
                
                with col1:
                    if is_malignant:
                        st.error("### ⚠️ Result: MALIGNANT")
                    else:
                        st.success("### ✅ Result: BENIGN")
                
                with col2:
                    st.metric("Confidence", f"{data['confidence'] * 100:.1f}%")
                
                with col3:
                    st.metric("P(Malignant)", f"{data['p_malignant']:.4f}", 
                              help=f"Classification threshold: {data['threshold']}")
                
                st.divider()
                st.subheader("Decision Explainability (SHAP Top Features)")
                st.markdown("Features pushing toward **Malignant** (Positive) vs **Benign** (Negative).")
                
                df_shap = pd.DataFrame(data['top_features'])
                if not df_shap.empty:
                    st.bar_chart(df_shap.set_index('feature')['shap_value'])
                    st.table(df_shap)
                else:
                    st.info("SHAP explainability not available for this prediction.")
            else:
                st.error(f"Prediction failed with status code {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Failed to connect to backend: {str(e)}")
