import streamlit as st
import numpy as np
import pandas as pd
import joblib
from io import StringIO

# ----------------------------
# Load Model + Scaler
# ----------------------------
model = joblib.load("breast_cancer_model.pkl")
scaler = joblib.load("scaler.pkl")  # remove if you didn't use scaling

# ----------------------------
# Page config + CSS
# ----------------------------
st.set_page_config(page_title="Breast Cancer Diagnostic AI", layout="wide")

# Simple medical-themed CSS
st.markdown(
    """
        <style>
        body { background: #f4fbff; }
        .stApp { background: #f4fbff; }
        .block-container { padding: 1.5rem 2rem; }
        .stButton>button { background: #0b6efd; color: white; border-radius: 8px; }
        .stExpander { background: white; border-radius: 8px; box-shadow: 0 1px 6px rgba(0,0,0,0.05); }
        .metric-label { color: #0b6efd; font-weight: 600; }
        .hero {
            background-size: cover;
            background-position: center;
            border-radius: 12px;
            padding: 60px 30px;
            color: white;
            margin-bottom: 12px;
            position: relative;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        }
        .hero-overlay {
            background: rgba(6, 30, 63, 0.45);
            padding: 40px;
            border-radius: 12px;
            max-width: 900px;
        }
        .hero h1 { margin: 0; font-size: 34px; }
        .hero p { margin: 6px 0 0; font-size: 16px; opacity: .95; }
        </style>
        """,
        unsafe_allow_html=True,
)

hero_img = "https://images.unsplash.com/photo-1603398938378-e54eab446dde?w=500&auto=format&fit=crop&q=60&ixlib=rb-4.1.0&ixid=M3wxMjA3fDB8MHxzZWFyY2h8Nnx8bWVkaWNhbHxlbnwwfHwwfHx8MA%3D%3D"
st.markdown(f"""
<div class="hero" style="background-image: url('{hero_img}');">
    <div class="hero-overlay">
        <h1>Breast Cancer Diagnostic System</h1>
        <p>AI-powered tumor classification — educational use only.</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.warning("This system is for educational purposes only and not a medical diagnosis tool.")
st.write("")  # spacer

# ----------------------------
# Helper / configuration
# ----------------------------
FEATURE_ORDER = [
    # mean (10)
    "radius_mean","texture_mean","perimeter_mean","area_mean","smoothness_mean",
    "compactness_mean","concavity_mean","concave_points_mean","symmetry_mean","fractal_dimension_mean",
    # se (10)
    "radius_se","texture_se","perimeter_se","area_se","smoothness_se",
    "compactness_se","concavity_se","concave_points_se","symmetry_se","fractal_dimension_se",
    # worst (10)
    "radius_worst","texture_worst","perimeter_worst","area_worst","smoothness_worst",
    "compactness_worst","concavity_worst","concave_points_worst","symmetry_worst","fractal_dimension_worst",
]

def build_full_features_from_quick(quick_vals):
    # Create base values from scaler means if available, otherwise zeros
    try:
        base = scaler.mean_.copy()
    except Exception:
        base = np.zeros(len(FEATURE_ORDER))
    full = base.astype(float)
    # Only replace the mean-feature positions (first 10)
    mapping = {
        "radius_mean": 0, "texture_mean": 1, "perimeter_mean": 2, "area_mean": 3, "smoothness_mean": 4
    }
    for k, v in quick_vals.items():
        if k in mapping:
            full[mapping[k]] = v
    return full.reshape(1, -1)

def predict_from_features(raw_features):
    try:
        features_scaled = scaler.transform(raw_features)
    except Exception:
        features_scaled = raw_features  # if scaler not appropriate, try raw
    prediction = model.predict(features_scaled)
    probability = model.predict_proba(features_scaled)
    return prediction, probability

# ----------------------------
# Main UI: Modes
# ----------------------------
tab1, tab2, tab3 = st.tabs(["Quick Mode", "Advanced Mode", "Batch Upload"])

with tab1:
    st.header("Quick Mode — fast estimate")
    st.write("Provide a few core measurements; remaining features will be filled with training means.")
    # Quick sliders for a handful of mean features
    qcol1, qcol2, qcol3 = st.columns(3)
    with qcol1:
        radius_mean = st.slider("Radius (mean)", min_value=5.0, max_value=30.0, value=14.0, step=0.1)
        texture_mean = st.slider("Texture (mean)", min_value=5.0, max_value=40.0, value=19.0, step=0.1)
    with qcol2:
        perimeter_mean = st.slider("Perimeter (mean)", min_value=30.0, max_value=200.0, value=91.0, step=0.1)
        area_mean = st.slider("Area (mean)", min_value=200.0, max_value=3000.0, value=654.0, step=1.0)
    with qcol3:
        smoothness_mean = st.slider("Smoothness (mean)", min_value=0.02, max_value=0.2, value=0.1, step=0.001)

    if st.button("Run Quick Diagnosis"):
        quick_vals = {
            "radius_mean": radius_mean,
            "texture_mean": texture_mean,
            "perimeter_mean": perimeter_mean,
            "area_mean": area_mean,
            "smoothness_mean": smoothness_mean,
        }
        features = build_full_features_from_quick(quick_vals)
        pred, prob = predict_from_features(features)
        confidence = round(np.max(prob) * 100, 2)
        if pred[0] == 1:
            st.error("Malignant Tumor Detected")
        else:
            st.success("Benign Tumor Detected")
        st.metric(label="Model Confidence", value=f"{confidence}%")

with tab2:
    st.header("Advanced Mode — full input")
    st.write("Enter all 30 features. Use expanders to keep the form tidy.")
    colA, colB = st.columns(2)

    with colA:
        with st.expander("Mean Features", expanded=False):
            radius_mean = st.number_input("Radius Mean", format="%.4f")
            texture_mean = st.number_input("Texture Mean", format="%.4f")
            perimeter_mean = st.number_input("Perimeter Mean", format="%.4f")
            area_mean = st.number_input("Area Mean", format="%.4f")
            smoothness_mean = st.number_input("Smoothness Mean", format="%.6f")
            compactness_mean = st.number_input("Compactness Mean", format="%.6f")
            concavity_mean = st.number_input("Concavity Mean", format="%.6f")
            concave_points_mean = st.number_input("Concave Points Mean", format="%.6f")
            symmetry_mean = st.number_input("Symmetry Mean", format="%.6f")
            fractal_dimension_mean = st.number_input("Fractal Dimension Mean", format="%.6f")

        with st.expander("Standard Error (SE) Features", expanded=False):
            radius_se = st.number_input("Radius SE", format="%.6f")
            texture_se = st.number_input("Texture SE", format="%.6f")
            perimeter_se = st.number_input("Perimeter SE", format="%.6f")
            area_se = st.number_input("Area SE", format="%.6f")
            smoothness_se = st.number_input("Smoothness SE", format="%.6f")
            compactness_se = st.number_input("Compactness SE", format="%.6f")
            concavity_se = st.number_input("Concavity SE", format="%.6f")
            concave_points_se = st.number_input("Concave Points SE", format="%.6f")
            symmetry_se = st.number_input("Symmetry SE", format="%.6f")
            fractal_dimension_se = st.number_input("Fractal Dimension SE", format="%.6f")

    with colB:
        with st.expander("Worst Features", expanded=False):
            radius_worst = st.number_input("Radius Worst", format="%.4f")
            texture_worst = st.number_input("Texture Worst", format="%.4f")
            perimeter_worst = st.number_input("Perimeter Worst", format="%.4f")
            area_worst = st.number_input("Area Worst", format="%.4f")
            smoothness_worst = st.number_input("Smoothness Worst", format="%.6f")
            compactness_worst = st.number_input("Compactness Worst", format="%.6f")
            concavity_worst = st.number_input("Concavity Worst", format="%.6f")
            concave_points_worst = st.number_input("Concave Points Worst", format="%.6f")
            symmetry_worst = st.number_input("Symmetry Worst", format="%.6f")
            fractal_dimension_worst = st.number_input("Fractal Dimension Worst", format="%.6f")

    if st.button("Run Advanced Diagnosis"):
        features = np.array([[
            radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean,
            compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean,
            radius_se, texture_se, perimeter_se, area_se, smoothness_se,
            compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se,
            radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst,
            compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst
        ]])
        pred, prob = predict_from_features(features)
        confidence = round(np.max(prob) * 100, 2)
        if pred[0] == 1:
            st.error("Malignant Tumor Detected")
        else:
            st.success("Benign Tumor Detected")
        st.metric(label="Model Confidence", value=f"{confidence}%")

with tab3:
    st.header("Batch Upload (CSV)")
    st.write("Upload a CSV with 30 columns in the same order as the model features.")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:")
        st.dataframe(df.head())
        if st.button("Run Batch Predictions"):
            try:
                X = df.values
                preds, probs = predict_from_features(X)
                df["prediction"] = preds
                df["confidence"] = np.max(probs, axis=1)
                st.write(df.head())
                st.success("Batch predictions completed.")
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button("Download results CSV", data=csv, file_name="predictions.csv")
            except Exception as e:
                st.error(f"Batch prediction failed: {e}")
