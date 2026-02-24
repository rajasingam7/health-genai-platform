import pandas as pd
import numpy as np


# ----------------------------
# Activity Aggregation
# ----------------------------

def aggregate_activity_data(df_activity):
    """
    Aggregate time-series activity data per patient.
    """
    df_agg = df_activity.groupby("Patient_Number").agg(
        avg_steps=("Physical_activity", "mean"),
        min_steps=("Physical_activity", "min"),
        max_steps=("Physical_activity", "max"),
        std_steps=("Physical_activity", "std")
    ).reset_index()

    # Sedentary threshold (can move to config later)
    df_agg["sedentary_flag"] = np.where(df_agg["avg_steps"] < 4000, 1, 0)

    return df_agg


# ----------------------------
# Profile Feature Engineering
# ----------------------------

def engineer_profile_features(df_profile):
    """
    Create derived features from patient profile dataset.
    """

    df = df_profile.copy()

    # Obesity flag
    df["obesity_flag"] = np.where(df["BMI"] >= 30, 1, 0)

    # High salt intake flag
    df["high_salt_flag"] = np.where(df["salt_content_in_the_diet"] >= 5000, 1, 0)

    # High stress flag (Stress level 3 = High)
    df["high_stress_flag"] = np.where(df["Level_of_Stress"] == 3, 1, 0)

    # Age bucket
    df["age_bucket"] = pd.cut(
        df["Age"],
        bins=[0, 30, 45, 60, 100],
        labels=["Young", "Mid-Age", "Senior", "Elderly"]
    )

    # Genetic risk category
    df["genetic_risk_category"] = pd.cut(
        df["Genetic_Pedigree_Coefficient"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Moderate", "High"]
    )

    return df