import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========== Load Artifacts ==========
model = joblib.load("lgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
selected_features = joblib.load("selected_features.pkl")
threshold = joblib.load("threshold.pkl")

# ========== App UI ==========
st.set_page_config(page_title="Diabetes Readmission Predictor", layout="centered")
st.title("🩺 Diabetes Readmission Predictor")
st.markdown("Upload a patient dataset **OR** manually enter patient details to predict the likelihood of hospital readmission within 30 days.")

# ========== Sidebar ==========
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
This tool helps clinicians and researchers estimate the risk of diabetes-related hospital readmission.

**Features**:
- Upload `.csv` file of patients
- Enter individual patient details
- Download predictions

Developed by **Saumya Tiwari**  
UC Davis – Health Informatics
""")

# ========== Prediction Logic ==========
def make_prediction(df_input):
    df_input = df_input[selected_features]
    df_scaled = pd.DataFrame(scaler.transform(df_input), columns=df_input.columns)
    probabilities = model.predict_proba(df_scaled)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    df_input["Readmission Probability"] = probabilities
    df_input["Readmission Prediction (<30 days)"] = ["Yes" if p == 1 else "No" for p in predictions]
    return df_input[["Readmission Probability", "Readmission Prediction (<30 days)"]]

# ========== 1. Upload ==========
st.subheader("📂 Upload Patient CSV File")
uploaded_file = st.file_uploader("Upload a pre-processed .csv file matching model structure", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        result = make_prediction(df)
        st.success("✅ Prediction complete.")
        st.dataframe(result)
        st.download_button("📥 Download Results", result.to_csv(index=False), "predictions.csv", "text/csv")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# ========== Divider ==========
st.markdown("<h4 style='text-align:center;'>──────────────  OR  ──────────────</h4>", unsafe_allow_html=True)

# ========== 2. Manual Entry ==========
with st.expander("📝 Manually Enter Patient Data", expanded=False):
    with st.form("manual_entry_form"):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.radio("Gender", ["Male", "Female"], help="Select the patient's biological sex.")
            age_numeric = st.slider("Age (years)", 0, 100, 55, step=1)
            time_in_hospital = st.slider("Hospital stay duration (days)", 1, 30, 5)
            num_comorbidities = st.slider("Number of comorbidities", 0, 10, 2)
            medication_count = st.slider("Number of medications", 0, 20, 5)
            high_risk_discharge = st.checkbox("Discharged to hospice or similar high-risk", help="Check if discharge disposition was hospice-related")

        with c2:
            num_outpatient = st.number_input("Outpatient visits", 0, 100, 1)
            num_emergency = st.number_input("Emergency visits", 0, 100, 0)
            num_inpatient = st.number_input("Inpatient visits", 0, 100, 0)
            has_insulin = st.selectbox("Insulin administered", [0, 1], format_func=lambda x: "Yes" if x else "No")
            abnormal_glucose = st.selectbox("Abnormal glucose (>200 mg/dL)", [0, 1], format_func=lambda x: "Yes" if x else "No")
            abnormal_A1C = st.selectbox("Abnormal A1C (>7%)", [0, 1], format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("Predict")

        if submitted:
            total_visits = num_outpatient + num_emergency + num_inpatient
            long_stay = int(time_in_hospital > 7)
            ageXvisits = age_numeric * total_visits
            comorbXmed = num_comorbidities * medication_count

            manual_data = {
                "gender": [1 if gender == "Male" else 0],
                "age_numeric": [age_numeric],
                "time_in_hospital": [time_in_hospital],
                "number_outpatient": [num_outpatient],
                "number_emergency": [num_emergency],
                "number_inpatient": [num_inpatient],
                "total_visits": [total_visits],
                "long_stay": [long_stay],
                "num_comorbidities": [num_comorbidities],
                "medication_count": [medication_count],
                "has_insulin": [has_insulin],
                "high_risk_discharge": [int(high_risk_discharge)],
                "abnormal_glucose": [abnormal_glucose],
                "abnormal_A1C": [abnormal_A1C],
                "ageXvisits": [ageXvisits],
                "comorbXmed": [comorbXmed]
            }

            df_manual = pd.DataFrame(manual_data)
            for col in selected_features:
                if col not in df_manual.columns:
                    df_manual[col] = 0
            df_manual = df_manual[selected_features]

            result = make_prediction(df_manual)
            st.success("✅ Prediction complete.")
            st.write("### 📊 Result")
            st.dataframe(result)
