import pandas as pd
import joblib
import shap


def predict_bp_risk_with_explanation(input_data: dict):

    # Load saved components
    model = joblib.load("models/bp_model.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")

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

    # Create DataFrame in correct order
    raw_df = pd.DataFrame([input_data])[feature_order]

    # Apply preprocessing
    imputed = imputer.transform(raw_df)
    scaled = scaler.transform(imputed)

    # Predict probability
    probability = model.predict_proba(scaled)[0][1]
    risk_percentage = round(probability * 100, 2)

    # SHAP explanation (Tree-based model)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled)

    contributions = dict(zip(feature_order, shap_values[1][0]))

    return risk_percentage, contributionsimport pandas as pd
import joblib
import streamlit as st


# ------------------------------------------------------
# Cache Model Components (Load Only Once)
# ------------------------------------------------------

@st.cache_resource
def load_model_components():
    """
    Load model, imputer, scaler only once.
    Prevents reloading on every rerun.
    """

    model = joblib.load("models/bp_model.pkl")
    imputer = joblib.load("models/imputer.pkl")
    scaler = joblib.load("models/scaler.pkl")

    return model, imputer, scaler


# ------------------------------------------------------
# Prediction Function
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

    # Convert input into correct DataFrame format
    raw_df = pd.DataFrame([input_data])[feature_order]

    # Apply preprocessing
    imputed = imputer.transform(raw_df)
    scaled = scaler.transform(imputed)

    # Predict probability
    probability = model.predict_proba(scaled)[0][1]
    risk_percentage = round(probability * 100, 2)

    # Use Feature Importance instead of SHAP (much faster)
    importances = model.feature_importances_
    contributions = dict(zip(feature_order, importances))

    return risk_percentage, contributions