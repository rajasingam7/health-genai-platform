import joblib
import shap
import numpy as np

MODEL_PATH = "models/bp_model.pkl"


def load_model():
    return joblib.load(MODEL_PATH)


def predict_bp_risk_with_explanation(input_data: dict):

    pipeline = load_model()

    feature_names = [
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

    # Prepare raw input
    raw_features = np.array([[ 
        input_data["Age"],
        input_data["BMI"],
        input_data["Level_of_Hemoglobin"],
        input_data["Genetic_Pedigree_Coefficient"],
        input_data["Smoking"],
        input_data["salt_content_in_the_diet"],
        input_data["alcohol_consumption_per_day"],
        input_data["Level_of_Stress"],
        input_data["avg_steps"]
    ]])

    # Step 1: Predict using pipeline
    probability = pipeline.predict_proba(raw_features)[0][1]

    # Step 2: Extract preprocessing steps
    imputer = pipeline.named_steps["imputer"]
    scaler = pipeline.named_steps["scaler"]
    model = pipeline.named_steps["model"]

    # Step 3: Transform features manually
    processed = imputer.transform(raw_features)
    processed = scaler.transform(processed)

    # Step 4: SHAP with final model
    explainer = shap.LinearExplainer(model, processed)
    shap_values = explainer(processed)

    contributions = dict(zip(feature_names, shap_values.values[0]))

    return round(probability * 100, 2), contributions