import pandas as pd
import joblib
import streamlit as st


# ------------------------------------------------------
# Load Model Components (Cached)
# ------------------------------------------------------

@st.cache_resource
def load_model_components():
    """
    Load trained model and preprocessing components.
    Cached to avoid reloading on every Streamlit rerun.
    """

    model = joblib.load("models/bp_model.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")

    return model, imputer, scaler


# ------------------------------------------------------
# Risk Prediction Function
# ------------------------------------------------------

def predict_bp_risk_with_explanation(input_data: dict):

    model, imputer, scaler = load_model_components()

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

    # Convert input into DataFrame with correct order
    raw_df = pd.DataFrame([input_data])[feature_order]

    # Apply preprocessing
    imputed = imputer.transform(raw_df)
    scaled = scaler.transform(imputed)

    # Predict probability
    probability = model.predict_proba(scaled)[0][1]
    risk_percentage = round(probability * 100, 2)

    # Use feature importance for explanation (fast alternative to SHAP)
    importances = model.feature_importances_
    contributions = dict(zip(feature_order, importances))

    return risk_percentage, contributions