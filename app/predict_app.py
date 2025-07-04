# predict_app.py
import streamlit as st
import pandas as pd
import joblib

# ==== Load Saved Artifacts ====
MODEL_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/model.pkl"
SCALER_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/scaler.pkl"
FEATURES_PATH = "D:/Unified Mentors/lung_cancer/Lung Cancer/saved_models/feature_names.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)

# ==== UI ====
st.set_page_config(page_title="Lung Cancer Survival Predictor")
st.title("🩺 Lung Cancer Survival Prediction")
st.markdown("Enter patient details to estimate survival probability.")

# ==== Input Form ====
with st.form("patient_form"):
    age = st.number_input("Age", min_value=1, max_value=120, value=60)
    gender = st.selectbox("Gender", ["Male", "Female"])
    cancer_stage = st.selectbox("Cancer Stage", ["Stage I", "Stage II", "Stage III", "Stage IV"])
    family_history = st.selectbox("Family History", ["Yes", "No"])
    smoking_status = st.selectbox("Smoking Status", ["Never Smoked", "Passive Smoker", "Former Smoker", "Current Smoker"])
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
    cholesterol = st.number_input("Cholesterol Level", min_value=100, max_value=400, value=200)
    hypertension = st.selectbox("Hypertension", ["Yes", "No"])
    asthma = st.selectbox("Asthma", ["Yes", "No"])
    cirrhosis = st.selectbox("Cirrhosis", ["Yes", "No"])
    other_cancer = st.selectbox("Other Cancer History", ["Yes", "No"])
    treatment_type = st.selectbox("Treatment Type", ["Chemotherapy", "Radiation", "Surgery", "Combined"])
    treatment_duration = st.slider("Treatment Duration (days)", min_value=0, max_value=1000, value=300)
    threshold = st.slider("Prediction Threshold", 0.0, 1.0, 0.5)
    submit = st.form_submit_button("Predict Survival")

# ==== Predict ====
if submit:
    # Convert to DataFrame
    input_dict = {
        'age': age,
        'bmi': bmi,
        'cholesterol_level': cholesterol,
        'treatment_duration': treatment_duration,
        'gender_' + gender: 1,
        'cancer_stage_' + cancer_stage: 1,
        'family_history_' + family_history: 1,
        'smoking_status_' + smoking_status: 1,
        'hypertension_' + hypertension: 1,
        'asthma_' + asthma: 1,
        'cirrhosis_' + cirrhosis: 1,
        'other_cancer_' + other_cancer: 1,
        'treatment_type_' + treatment_type: 1
    }

    df_input = pd.DataFrame([input_dict])
    df_input = df_input.reindex(columns=feature_names, fill_value=0)
    X_scaled = scaler.transform(df_input)
    prob = model.predict_proba(X_scaled)[0][1]

    st.markdown(f"### 🧪 Model estimated survival probability: `{prob*100:.2f}%`")

    if prob >= threshold:
        st.success(f"✅ Predicted: SURVIVED (confidence ≥ {threshold:.2f})")
    else:
        st.error(f"⚠️ Predicted: DID NOT SURVIVE (confidence < {threshold:.2f})")
