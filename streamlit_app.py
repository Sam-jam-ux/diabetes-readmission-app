# ========== Imports ==========
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ========== Load Artifacts ==========
model = joblib.load("lgbm_model.pkl")
scaler = joblib.load("scaler.pkl")
scaler_features = joblib.load("scaler_features.pkl")
best_threshold = joblib.load("threshold.pkl")

# ========== Constants for Processing ==========
med_cols = ['metformin', 'glipizide', 'glyburide', 'pioglitazone', 'insulin']
age_map = {'[0-10)': 5, '[10-20)': 15, '[20-30)': 25, '[30-40)': 35, '[40-50)': 45,
           '[50-60)': 55, '[60-70)': 65, '[70-80)': 75, '[80-90)': 85, '[90-100)': 95}
diag_map = {
    'Circulatory': lambda x: (390 <= x < 460) or (x == 785),
    'Respiratory': lambda x: (460 <= x < 520) or (x == 786),
    'Digestive': lambda x: (520 <= x < 580) or (x == 787),
    'Diabetes': lambda x: 250 <= x < 251,
    'Injury': lambda x: 800 <= x < 1000,
    'Musculoskeletal': lambda x: 710 <= x < 740,
    'Genitourinary': lambda x: (580 <= x < 630) or (x == 788),
    'Neoplasms': lambda x: 140 <= x < 240
}

# ========== Preprocessing Function ==========
def group_diag(code):
    try:
        code = float(code)
        for label, cond in diag_map.items():
            if cond(code): return label
        return 'Other'
    except:
        return 'Other'

def preprocess_input(df):
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['age_numeric'] = df['age'].map(age_map)
    df.drop('age', axis=1, inplace=True, errors='ignore')

    for col in ['diag_1', 'diag_2', 'diag_3']:
        if col in df.columns:
            df[col] = df[col].apply(group_diag)
    df = pd.get_dummies(df, columns=['diag_1', 'diag_2', 'diag_3'], prefix=['D1', 'D2', 'D3'], drop_first=True)

    df['total_visits'] = df.get('number_outpatient', 0) + df.get('number_emergency', 0) + df.get('number_inpatient', 0)
    df['long_stay'] = (df['time_in_hospital'] > 7).astype(int)
    diag_cats = [col for col in df.columns if col.startswith('D1_')]
    df['num_comorbidities'] = df[diag_cats].sum(axis=1) if diag_cats else 0
    df['has_insulin'] = df['insulin']
    df['medication_count'] = df[med_cols].astype(int).sum(axis=1)
    df['high_risk_discharge'] = df['discharge_disposition_id'].isin([11, 13, 14, 19, 20, 21]).astype(int)
    df['abnormal_glucose'] = df['max_glu_serum'].isin(['>200', '>300']).astype(int)
    df['abnormal_A1C'] = df['A1Cresult'].isin(['>7', '>8']).astype(int)

    return df

# ========== Streamlit UI ==========
st.set_page_config(page_title="Diabetes Readmission Predictor", layout="centered")

st.title("🩺 Diabetes Readmission Predictor")
st.markdown("Upload patient data below to predict likelihood of readmission within 30 days.")

with st.sidebar:
    st.markdown("### 📄 About")
    st.write(
        """
        This tool predicts hospital readmission risk in diabetic patients based on their medical data.
        - Upload a properly formatted `.csv` file.
        - Predictions will be shown for each row.
        """
    )
    st.markdown("---")
    st.markdown("**Developed by Saumya Tiwari**  \nUniversity of California, Davis  \nHealth Informatics")

st.header("📤 Upload Patient CSV File")
uploaded_file = st.file_uploader("Choose a CSV file with patient data", type=["csv"])

if uploaded_file:
    try:
        input_df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
        st.subheader("📋 Data Preview")
        st.dataframe(input_df.head(), use_container_width=True)

        input_df_processed = preprocess_input(input_df)

        # Align with training features
        for col in scaler_features:
            if col not in input_df_processed.columns:
                input_df_processed[col] = 0
        input_df_processed = input_df_processed[scaler_features]

        # Scale and Predict
        input_scaled = scaler.transform(input_df_processed)
        input_scaled_df = pd.DataFrame(input_scaled, columns=scaler_features)

        probs = model.predict_proba(input_scaled_df)[:, 1]
        preds = (probs >= best_threshold).astype(int)

        # Show results
        st.subheader("📊 Prediction Results")
        input_df['Readmission Probability'] = probs.round(3)
        input_df['Readmission Risk (<30 days)'] = np.where(preds == 1, "Yes", "No")
        st.dataframe(input_df[['Readmission Probability', 'Readmission Risk (<30 days)']], use_container_width=True)

    except Exception as e:
        st.error(f"⚠️ Error processing the file: {e}")
