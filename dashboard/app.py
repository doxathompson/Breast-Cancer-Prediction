"""
dashboard/app.py
────────────────
Premium Streamlit dashboard for the Breast Cancer Predictor.
"""

import streamlit as st
import requests
import pandas as pd

API_URL = "http://api:8000" if "docker" in __name__ else "http://localhost:8000"
AUTH = ("clinician", "secure_password_123")

st.set_page_config(
    page_title="WBCD Clinical Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium feel
st.markdown("""
<style>
    .result-card-malignant {
        background: linear-gradient(135deg, #ff4b4b 0%, #a00 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .result-card-benign {
        background: linear-gradient(135deg, #00b09b 0%, #96c93d 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    .metric-value { font-size: 3rem; font-weight: 800; margin: 0; }
    .metric-label { font-size: 1.2rem; font-weight: 400; opacity: 0.9; text-transform: uppercase; letter-spacing: 2px;}
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 10])
with col_logo:
    st.markdown("<h1 style='text-align: center; font-size: 3rem;'>🔬</h1>", unsafe_allow_html=True)
with col_title:
    st.title("Breast Cancer Diagnostic Engine")
    st.markdown("**(WBCD) Fine Needle Aspirate Predictor** — Clinical Support Interface")

st.divider()

# ─── SIDEBAR (Controls & Health) ─────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Status")
    try:
        health = requests.get(f"{API_URL}/health", timeout=2, auth=AUTH)
        if health.status_code == 200:
            st.success("API: **ONLINE**")
            st.caption(f"Latency: {health.elapsed.total_seconds():.2f}s")
        else:
            st.error("API: **OFFLINE** (Auth/Internal Error)")
    except requests.RequestException:
        st.error("API: **UNREACHABLE**")
        st.caption("Ensure `uvicorn` is completely running.")
        
    st.divider()
    
    st.header("👤 Patient Profile")
    patient_id = st.text_input("Patient ID Reference (Optional)", placeholder="e.g. PT-4091A")
    

def execute_prediction(features_dict, p_id):
    with st.spinner("Executing Random Forest Inference & SHAP Analysis..."):
        try:
            res = requests.post(f"{API_URL}/predict", json=features_dict, auth=AUTH)
            if res.status_code == 200:
                data = res.json()
                is_malignant = data['prediction'] == 'malignant'
                
                # Big splashy result
                st.markdown("<br><hr>", unsafe_allow_html=True)
                if is_malignant:
                    st.markdown(f"""
                    <div class="result-card-malignant">
                        <p class="metric-label">Diagnosis Prediction</p>
                        <p class="metric-value">MALIGNANT</p>
                        <p style="margin-top:10px;">ID: {p_id or 'Unknown'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card-benign">
                        <p class="metric-label">Diagnosis Prediction</p>
                        <p class="metric-value">BENIGN</p>
                        <p style="margin-top:10px;">ID: {p_id or 'Unknown'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Detailed metrics
                st.subheader("🔍 Prediction Deep-Dive")
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Class", data['prediction'].upper())
                c2.metric("Machine Confidence", f"{data['confidence'] * 100:.2f}%", 
                          delta="High Confidence" if data['confidence'] > 0.8 else "Low Confidence")
                c3.metric("P(Malignant) Score", f"{data['p_malignant']:.4f}", 
                          help=f"Optimised Threshold was {data['threshold']}. Any score above this is flagged Malignant.")
                
                # SHAP Explainability
                if data.get('top_features'):
                    st.markdown("### 🧬 Interpretability (Top Driving Features)")
                    st.markdown("This chart breaks down *why* the model made this specific decision based on SHAP calculations.")
                    
                    df_shap = pd.DataFrame(data['top_features'])
                    
                    # Clean up feature names
                    df_shap['Feature Name'] = df_shap['feature'].str.replace("_", " ").str.title()
                    df_shap['Effect'] = df_shap['shap_value'].apply(lambda x: "Pushes M" if x > 0 else "Pushes B")
                    
                    bar_col, table_col = st.columns([2, 1])
                    with bar_col:
                        st.bar_chart(df_shap.set_index('Feature Name')['shap_value'], color="#e05c5c" if is_malignant else "#5cc8c8")
                    with table_col:
                        st.dataframe(df_shap[['Feature Name', 'shap_value']].set_index('Feature Name'), use_container_width=True)
                        
            elif res.status_code == 401:
                st.error("Authentication rejected. Ensure your auth credentials are correct.")
            else:
                st.error(f"API Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"Connection failed: {str(e)}")


# ─── MAIN TABS ───────────────────────────────────────────────────────
main_tab1, main_tab2 = st.tabs(["🖼️ Computer Vision (Image Upload)", "🔢 Manual Tabular Data input"])


with main_tab1:
    st.subheader("Automated Slide Scanning")
    st.markdown("Upload a Fine Needle Aspirate histopathology slide. A Computer Vision pipeline will mathematically extract 30 geometric cell nucleus parameters from the image and immediately forward them to the Random Forest diagnostic engine.")
    
    uploaded_file = st.file_uploader("Upload FNA Microscopy Image", type=["jpg", "png", "jpeg", "tif"])
    if uploaded_file is not None:
        colimg, colexec = st.columns([1, 2])
        with colimg:
            st.image(uploaded_file, caption="Uploaded FNA Slide", use_container_width=True)
            
        with colexec:
            if st.button("Extract Geometric Features & Predict", type="primary"):
                with st.spinner("Passing image through Deep Learning extraction pipeline..."):
                    # Send to backend
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                        ext_res = requests.post(f"{API_URL}/extract-features", files=files, auth=AUTH)
                        if ext_res.status_code == 200:
                            extracted_features = ext_res.json()["extracted_features"]
                            st.success("Successfully extracted 30 dimensions from image!")
                            
                            with st.expander("View Extracted Tabular Features (Hidden by default)"):
                                st.json(extracted_features)
                            
                            # Forward straight to predict engine
                            execute_prediction(extracted_features, patient_id)
                        else:
                            st.error(f"Extraction failed: {ext_res.text}")
                    except Exception as e:
                        st.error(f"Failed to reach extraction engine: {e}")

with main_tab2:
    st.markdown("#### Sample Fill")
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        load_malignant = st.button("Load Malignant Profile", help="Loads an average malignant profile")
    with col_s2:
        load_benign = st.button("Load Benign Profile", help="Loads an average benign profile")

    # Provide sample overrides based on button clicks
    defaults = {
        "mean_radius": 17.99, "mean_texture": 10.38, "mean_perimeter": 122.8, "mean_area": 1001.0,
        "mean_smoothness": 0.1184, "mean_compactness": 0.2776, "mean_concavity": 0.3001,
        "mean_concave_points": 0.1471, "mean_symmetry": 0.2419, "mean_fractal_dimension": 0.0787,
        "radius_error": 1.095, "texture_error": 0.9053, "perimeter_error": 8.589, "area_error": 153.4,
        "smoothness_error": 0.0064, "compactness_error": 0.0490, "concavity_error": 0.0537,
        "concave_points_error": 0.0159, "symmetry_error": 0.0300, "fractal_dimension_error": 0.0062,
        "worst_radius": 25.38, "worst_texture": 17.33, "worst_perimeter": 184.6, "worst_area": 2019.0,
        "worst_smoothness": 0.1622, "worst_compactness": 0.6656, "worst_concavity": 0.7119,
        "worst_concave_points": 0.2654, "worst_symmetry": 0.4601, "worst_fractal_dimension": 0.1189
    }

    if load_benign:
        defaults = {k: v * 0.6 for k, v in defaults.items()}
        
    st.subheader("📝 Cytological Feature Inputs")
    tab_mean, tab_se, tab_worst = st.tabs(["📊 Mean Values", "📉 Standard Error (SE)", "📈 Worst (Max) Values"])

    manual_features = {}

    def layout_inputs(tab, prefix, default_dict):
        with tab:
            col1, col2 = st.columns(2)
            props = ["radius", "texture", "perimeter", "area", "smoothness", 
                     "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
            for i, p in enumerate(props):
                key_name = f"{prefix}_{p}"
                display_name = p.replace("_", " ").title()
                val = default_dict.get(key_name, 0.0)
                c = col1 if i < 5 else col2
                if p in ["area", "perimeter", "radius", "texture"]:
                    manual_features[key_name] = c.number_input(f"**{display_name}**", value=float(val), step=1.0)
                else:
                    manual_features[key_name] = c.number_input(f"**{display_name}**", value=float(val), format="%.4f", step=0.01)

    layout_inputs(tab_mean, "mean", defaults)

    with tab_se:
        col_se1, col_se2 = st.columns(2)
        se_props = ["radius", "texture", "perimeter", "area", "smoothness", 
                    "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension"]
        for i, p in enumerate(se_props):
            key_name = f"{p}_error"
            display_name = f"{p.replace('_', ' ').title()} SE"
            val = defaults.get(key_name, 0.0)
            c = col_se1 if i < 5 else col_se2
            if p in ["area", "perimeter", "radius", "texture"]:
                manual_features[key_name] = c.number_input(f"**{display_name}**", value=float(val), step=1.0)
            else:
                manual_features[key_name] = c.number_input(f"**{display_name}**", value=float(val), format="%.4f", step=0.01)

    layout_inputs(tab_worst, "worst", defaults)

    st.markdown("<br>", unsafe_allow_html=True)

    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        submit = st.button("🚀 EXECUTE CLINICAL ASSESSMENT (MANUAL)", type="primary", use_container_width=True)

    if submit:
        execute_prediction(manual_features, patient_id)
