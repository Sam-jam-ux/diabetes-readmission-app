import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load artifacts
scaler = joblib.load("scaler.pkl")
model = joblib.load("lgbm_model.pkl")
selected_features = joblib.load("selected_features.pkl")
scaler_features = joblib.load("scaler_features.pkl")
best_threshold = joblib.load("threshold.pkl")

# Constants
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

# App title
st.title("Diabetes Readmission Predictor")

# Footer - Developer info
st.sidebar.markdown("""
**Developed by Saumya Tiwari**  
University of California, Davis  
Health Informatics  
Department of Public Health
""")
