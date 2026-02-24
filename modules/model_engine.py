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

    return risk_percentage, contributions