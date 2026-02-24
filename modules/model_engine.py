import pandas as pd
import joblib
import streamlit as st


# Load once at startup
@st.cache_resource
def load_model_components():
    model = joblib.load("models/bp_model.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")
    return model, imputer, scaler


model, imputer, scaler = load_model_components()


def predict_bp_risk_with_explanation(input_data: dict):

    feature_order = [
        "Age",
        "BMI",
        "Level_of_Hemoglobin",
        "Genetic_Pedigree_Coefficient",
        "Smoking",
        "salt_content_in_the_diet",
        "alcohol_consumption_per_day",
        "Level_of_Stress",
        "avg_steps"
    ]

    raw_df = pd.DataFrame([input_data])[feature_order]

    imputed = imputer.transform(raw_df)
    scaled = scaler.transform(imputed)

    probability = model.predict_proba(scaled)[0][1]
    risk_percentage = round(probability * 100, 2)

    # Precomputed feature importance (very fast)
    contributions = dict(zip(feature_order, model.feature_importances_))

    return risk_percentage, contributions