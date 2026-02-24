import streamlit as st

# ----------------------------
# Import Modules
# ----------------------------

from modules.data_loader import load_dataset_1, load_dataset_2, validate_schema
from modules.feature_engineering import (
    aggregate_activity_data,
    engineer_profile_features
)
from modules.temp_join_engine import create_temp_join
from modules.analytics_engine import (
    bp_by_stress,
    bp_by_salt,
    bp_by_activity,
    ckd_by_obesity,
    overall_summary
)
from modules.llm_engine import generate_explanation
from modules.security import validate_user_query
from modules.confidence_engine import calculate_confidence
from modules.model_engine import predict_bp_risk_with_explanation
from modules.monitoring import log_event, load_logs
from modules.model_metrics import load_model_metrics


# ----------------------------
# Page Configuration
# ----------------------------

st.set_page_config(page_title="Enterprise Health GenAI Platform", layout="wide")
st.title("🏥 Enterprise Health GenAI Platform")


# ----------------------------
# Load and Prepare Data
# ----------------------------

with st.spinner("Loading and preparing data..."):
    df1 = load_dataset_1()
    df2 = load_dataset_2()
    validate_schema(df1, df2)

    df1_feat = engineer_profile_features(df1)
    df2_agg = aggregate_activity_data(df2)
    df_temp = create_temp_join(df1_feat, df2_agg)

st.success("Data Loaded Successfully ✅")


# ----------------------------
# Sidebar Navigation
# ----------------------------

mode = st.sidebar.radio(
    "Select Mode",
    [
        "Population Analytics",
        "Personalized Risk Prediction",
        "System Overview",
        "Monitoring Dashboard",
        "Model Performance"
    ]
)


# ======================================================
# MODE 1 — Population Analytics
# ======================================================

if mode == "Population Analytics":

    st.subheader("📊 Population-Level Health Analytics")

    user_question = st.text_input(
        "Enter your question:",
        "How does high salt intake affect blood pressure?"
    )

    if user_question:

        try:
            validate_user_query(user_question)
        except ValueError as e:
            log_event("SECURITY_BLOCK", {"query": user_question})
            st.error(str(e))
            st.stop()

        question_lower = user_question.lower()

        if "salt" in question_lower:
            analytics_result = bp_by_salt(df_temp)
        elif "stress" in question_lower:
            analytics_result = bp_by_stress(df_temp)
        elif "activity" in question_lower or "steps" in question_lower:
            analytics_result = bp_by_activity(df_temp)
        elif "obesity" in question_lower or "kidney" in question_lower:
            analytics_result = ckd_by_obesity(df_temp)
        else:
            analytics_result = overall_summary(df_temp)

        log_event("POPULATION_QUERY", {
            "query": user_question,
            "analytics_result": analytics_result
        })

        st.write("### 🔎 Structured Analytics Output")
        st.json(analytics_result)

        with st.spinner("Generating explanation..."):
            explanation = generate_explanation(
                analytics_result,
                user_question
            )

        st.write("### 🤖 GenAI Explanation")
        st.write(explanation)

        confidence_score = calculate_confidence(df_temp, analytics_result)

        st.write("### 📊 Confidence Score")
        st.progress(confidence_score / 100)

        if confidence_score > 75:
            st.success(f"High Confidence ({confidence_score}%)")
        elif confidence_score > 50:
            st.warning(f"Moderate Confidence ({confidence_score}%)")
        else:
            st.error(f"Low Confidence ({confidence_score}%)")


# ======================================================
# MODE 2 — Personalized Risk Prediction
# ======================================================

elif mode == "Personalized Risk Prediction":

    st.subheader("🧬 Personalized Blood Pressure Risk Prediction")

    age = st.number_input("Age", 18, 100, 45)
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0)
    hemoglobin = st.number_input("Hemoglobin Level", 5.0, 20.0, 13.0)
    gpc = st.slider("Genetic Pedigree Coefficient", 0.0, 1.0, 0.5)
    smoking = st.selectbox("Smoking (0=No, 1=Yes)", [0, 1])
    salt = st.number_input("Salt Intake (mg/day)", 1000, 10000, 4000)
    alcohol = st.number_input("Alcohol Consumption (ml/day)", 0, 500, 50)
    stress = st.selectbox("Stress Level (1=Low, 2=Normal, 3=High)", [1, 2, 3])
    steps = st.number_input("Average Daily Steps", 0, 20000, 6000)

    if st.button("Predict Risk"):

        input_data = {
            "Age": age,
            "BMI": bmi,
            "Level_of_Hemoglobin": hemoglobin,
            "Genetic_Pedigree_Coefficient": gpc,
            "Smoking": smoking,
            "salt_content_in_the_diet": salt,
            "alcohol_consumption_per_day": alcohol,
            "Level_of_Stress": stress,
            "avg_steps": steps
        }
    with st.spinner("Calculating risk..."):
        risk, contributions = predict_bp_risk_with_explanation(input_data)

        log_event("PREDICTION", {
            "risk_percentage": risk,
            "inputs": input_data
        })

        st.write(f"### 🔎 Predicted BP Abnormality Risk: {risk}%")

        st.write("### 📊 Feature Contributions (SHAP Values)")
        st.json(contributions)

        explanation = generate_explanation(
            {
                "Predicted_BP_Risk_Percentage": risk,
                "Feature_Contributions": contributions
            },
            "Explain this predicted health risk and key contributing factors."
        )

        st.write("### 🤖 GenAI Explanation")
        st.write(explanation)


# ======================================================
# MODE 3 — System Overview
# ======================================================

elif mode == "System Overview":

    st.subheader("📈 Dataset Overview")

    st.write("### Dataset 1 Shape:", df1.shape)
    st.write("### Dataset 2 Shape:", df2.shape)
    st.write("### Temporary Joined Dataset Shape:", df_temp.shape)

    st.write("### Overall Summary Metrics")
    st.json(overall_summary(df_temp))


# ======================================================
# MODE 4 — Monitoring Dashboard
# ======================================================

elif mode == "Monitoring Dashboard":

    st.subheader("📊 System Monitoring & Audit Logs")

    logs = load_logs()

    st.write(f"Total Events Logged: {len(logs)}")

    if logs:
        st.write("### Last 20 Events")
        st.json(logs[-20:])

        event_counts = {}
        for log in logs:
            event_counts[log["event_type"]] = (
                event_counts.get(log["event_type"], 0) + 1
            )

        st.write("### Event Type Distribution")
        st.json(event_counts)


# ======================================================
# MODE 5 — Model Performance
# ======================================================

elif mode == "Model Performance":

    st.subheader("📈 Model Performance Metrics")

    metrics = load_model_metrics()

    if metrics is None:
        st.warning("No model metrics found. Please retrain the model.")
    else:
        st.write("### Accuracy:", metrics["accuracy"])
        st.write("### AUC:", metrics["auc"])

        if metrics["auc"] > 0.75:
            st.success("Model performance is strong.")
        elif metrics["auc"] > 0.65:
            st.warning("Model performance is moderate.")
        else:
            st.error("Model performance needs improvement.")