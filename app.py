import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def p(filename):
    return os.path.join(BASE_DIR, filename)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🧠 Stroke Prediction",
    page_icon="🧠",
    layout="centered",
)

# ── Load Artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open(p("svm_stroke_model.pkl"), "rb") as f:
        model = pickle.load(f)
    with open(p("stroke_scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(p("stroke_encoders.pkl"), "rb") as f:
        encoders = pickle.load(f)
    with open(p("stroke_feature_cols.pkl"), "rb") as f:
        feature_cols = pickle.load(f)
    return model, scaler, encoders, feature_cols

model, scaler, encoders, feature_cols = load_artifacts()

# ── Styles ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stButton > button {
        background: linear-gradient(135deg, #e74c3c, #c0392b);
        color: white; border: none; border-radius: 10px;
        font-size: 16px; font-weight: bold; padding: 12px;
        width: 100%; cursor: pointer;
    }
    .stButton > button:hover { opacity: 0.9; }
    .metric-card {
        background: white; border-radius: 12px; padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07); margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='color:#c0392b; font-size:2.4rem; margin-bottom:4px;'>🧠 Stroke Risk Predictor</h1>
    <p style='color:gray; font-size:1rem;'>Powered by Support Vector Machine (SVM) · 82.92% Accuracy</p>
</div>
<hr style='border:1px solid #f0f0f0; margin-bottom:24px;'>
""", unsafe_allow_html=True)

# ── Info Tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Predict", "📊 About the Model", "📋 Dataset Info"])

with tab2:
    st.markdown("### How SVM Works for Stroke Prediction")
    st.markdown("""
    **Support Vector Machine (SVM)** finds the optimal hyperplane that best separates stroke vs. no-stroke patients in high-dimensional feature space.

    - **Kernel:** RBF (Radial Basis Function) — handles non-linear boundaries
    - **C = 1.0** — balances margin width vs. misclassification
    - **Class Imbalance Fix:** Minority class (stroke=1) upsampled to match majority
    - **Preprocessing:** StandardScaler normalizes all numeric features

    | Metric | Value |
    |--------|-------|
    | Accuracy | 82.92% |
    | Precision (Stroke) | 78% |
    | Recall (Stroke) | 90% |
    | F1-Score (Stroke) | 84% |
    """)
    st.info("⚠️ High **recall** for stroke cases means the model catches 90% of actual stroke patients — critical for medical screening.")

with tab3:
    st.markdown("### Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", "5,110")
    col2.metric("Stroke Cases", "249 (4.9%)")
    col3.metric("Features Used", "10")
    
    st.markdown("**Features in this dataset:**")
    feature_desc = {
        "gender": "Male / Female",
        "age": "Patient age in years",
        "hypertension": "0 = No, 1 = Yes",
        "heart_disease": "0 = No, 1 = Yes",
        "ever_married": "Yes / No",
        "work_type": "Private, Self-employed, Govt_job, children, Never_worked",
        "Residence_type": "Urban / Rural",
        "avg_glucose_level": "Average blood glucose level",
        "bmi": "Body Mass Index",
        "smoking_status": "formerly smoked, never smoked, smokes, Unknown",
    }
    fdf = pd.DataFrame({"Feature": feature_desc.keys(), "Description": feature_desc.values()})
    st.dataframe(fdf, use_container_width=True, hide_index=True)

with tab1:
    st.markdown("### 📝 Enter Patient Details")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**👤 Demographics**")
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.slider("Age (years)", 1, 100, 50)
        ever_married = st.selectbox("Ever Married", ["Yes", "No"])
        residence = st.selectbox("Residence Type", ["Urban", "Rural"])
        work_type = st.selectbox("Work Type", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])

    with col2:
        st.markdown("**🏥 Medical History**")
        hypertension = st.radio("Hypertension", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
        heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x else "No", horizontal=True)
        avg_glucose = st.number_input("Avg Glucose Level (mg/dL)", min_value=50.0, max_value=300.0, value=100.0, step=0.5)
        bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
        smoking = st.selectbox("Smoking Status", ["never smoked", "formerly smoked", "smokes", "Unknown"])

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Predict Stroke Risk", use_container_width=True)

    if predict_btn:
        # Encode inputs using fitted label encoders
        def encode(col, val):
            le = encoders[col]
            if val in le.classes_:
                return le.transform([val])[0]
            else:
                return le.transform([le.classes_[0]])[0]

        input_dict = {
            "gender": encode("gender", gender),
            "age": age,
            "hypertension": hypertension,
            "heart_disease": heart_disease,
            "ever_married": encode("ever_married", ever_married),
            "work_type": encode("work_type", work_type),
            "Residence_type": encode("Residence_type", residence),
            "avg_glucose_level": avg_glucose,
            "bmi": bmi,
            "smoking_status": encode("smoking_status", smoking),
        }

        input_df = pd.DataFrame([input_dict])[feature_cols]
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        stroke_prob = probability[1] * 100
        no_stroke_prob = probability[0] * 100

        st.markdown("---")
        st.markdown("### 🧾 Prediction Result")

        if prediction == 1:
            st.markdown(f"""
            <div style='text-align:center; padding:28px; border-radius:16px;
                        background:linear-gradient(135deg, #fdecea, #fbc9c4);
                        border: 2px solid #e74c3c;'>
                <p style='font-size:16px; color:#888; margin-bottom:4px;'>Risk Assessment</p>
                <h1 style='color:#c0392b; font-size:3rem; margin:0;'>⚠️ HIGH STROKE RISK</h1>
                <p style='color:#c0392b; font-size:1.2rem; margin-top:8px;'>
                    Stroke Probability: <b>{stroke_prob:.1f}%</b>
                </p>
                <p style='color:#888; font-size:0.9rem;'>Please consult a medical professional immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='text-align:center; padding:28px; border-radius:16px;
                        background:linear-gradient(135deg, #e9f7ef, #c8f0d8);
                        border: 2px solid #2ecc71;'>
                <p style='font-size:16px; color:#888; margin-bottom:4px;'>Risk Assessment</p>
                <h1 style='color:#27ae60; font-size:3rem; margin:0;'>✅ LOW STROKE RISK</h1>
                <p style='color:#27ae60; font-size:1.2rem; margin-top:8px;'>
                    No-Stroke Probability: <b>{no_stroke_prob:.1f}%</b>
                </p>
                <p style='color:#888; font-size:0.9rem;'>Maintain a healthy lifestyle for continued wellbeing.</p>
            </div>
            """, unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### 📊 Probability Breakdown")
        prob_col1, prob_col2 = st.columns(2)
        prob_col1.metric("🟢 No Stroke", f"{no_stroke_prob:.1f}%")
        prob_col2.metric("🔴 Stroke", f"{stroke_prob:.1f}%")

        prog_val = stroke_prob / 100
        bar_color = "#e74c3c" if prediction == 1 else "#2ecc71"
        st.markdown(f"""
        <div style='background:#eee; border-radius:10px; height:22px; margin-top:8px;'>
            <div style='background:{bar_color}; width:{stroke_prob:.1f}%; height:22px;
                        border-radius:10px; transition: width 0.5s;
                        display:flex; align-items:center; justify-content:center;
                        color:white; font-size:12px; font-weight:bold;'>
                {stroke_prob:.1f}%
            </div>
        </div>
        <p style='color:gray; font-size:12px; margin-top:4px;'>Stroke Risk Meter</p>
        """, unsafe_allow_html=True)

        # Key risk factors
        st.markdown("#### 🔑 Key Risk Factors Entered")
        risk_data = {
            "Factor": ["Age", "Hypertension", "Heart Disease", "Avg Glucose", "BMI", "Smoking"],
            "Value": [age, "Yes" if hypertension else "No", "Yes" if heart_disease else "No",
                      f"{avg_glucose} mg/dL", bmi, smoking],
            "Risk Level": [
                "🔴 High" if age > 60 else "🟡 Medium" if age > 40 else "🟢 Low",
                "🔴 High" if hypertension else "🟢 Low",
                "🔴 High" if heart_disease else "🟢 Low",
                "🔴 High" if avg_glucose > 200 else "🟡 Medium" if avg_glucose > 125 else "🟢 Low",
                "🔴 High" if bmi > 30 else "🟡 Medium" if bmi > 25 else "🟢 Low",
                "🔴 High" if smoking == "smokes" else "🟡 Medium" if smoking == "formerly smoked" else "🟢 Low",
            ]
        }
        st.dataframe(pd.DataFrame(risk_data), use_container_width=True, hide_index=True)

        st.info("⚕️ **Disclaimer:** This tool is for educational purposes only. Always consult a qualified healthcare professional for medical advice.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:gray; font-size:13px;'>
    Built with ❤️ using Streamlit & Scikit-learn | SVM Classifier (RBF Kernel) | Dataset: Kaggle Stroke Prediction
</p>
""", unsafe_allow_html=True)
