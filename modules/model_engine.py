import pandas as pd
import joblib
import shap


def predict_bp_risk_with_explanation(input_data: dict):

    # Load trained pipeline
    pipeline = joblib.load("models/bp_model.pkl")

    # Expected feature order (MUST match training script)
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

    # Convert input dictionary into DataFrame with correct order
    raw_features = pd.DataFrame([input_data])[feature_order]

    # Predict probability
    probability = pipeline.predict_proba(raw_features)[0][1]
    risk_percentage = round(probability * 100, 2)

    # SHAP explanation (tree-based model safe)
    model = pipeline.named_steps["model"]
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(raw_features)

    contributions = dict(zip(feature_order, shap_values[1][0]))

    return risk_percentage, contributions