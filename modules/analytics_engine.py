import pandas as pd
import numpy as np


MIN_SAMPLE_THRESHOLD = 5  # Privacy protection


# ------------------------------------------
# 1. BP Rate by Stress Level
# ------------------------------------------

def bp_by_stress(df):
    grouped = df.groupby("Level_of_Stress")["Blood_Pressure_Abnormality"].agg(
        ["mean", "count"]
    ).reset_index()

    # Privacy rule: enforce minimum sample size
    grouped = grouped[grouped["count"] >= MIN_SAMPLE_THRESHOLD]

    result = {}

    for _, row in grouped.iterrows():
        stress_level = int(row["Level_of_Stress"])
        result[f"Stress_{stress_level}_BP_Rate"] = round(row["mean"], 3)
        result[f"Stress_{stress_level}_Sample_Size"] = int(row["count"])

    return result


# ------------------------------------------
# 2. BP Rate by High Salt Intake
# ------------------------------------------

def bp_by_salt(df):
    grouped = df.groupby("high_salt_flag")["Blood_Pressure_Abnormality"].agg(
        ["mean", "count"]
    ).reset_index()

    grouped = grouped[grouped["count"] >= MIN_SAMPLE_THRESHOLD]

    result = {}

    for _, row in grouped.iterrows():
        salt_flag = int(row["high_salt_flag"])
        label = "High_Salt" if salt_flag == 1 else "Normal_Salt"

        result[f"{label}_BP_Rate"] = round(row["mean"], 3)
        result[f"{label}_Sample_Size"] = int(row["count"])

    return result


# ------------------------------------------
# 3. BP Rate by Activity Level
# ------------------------------------------

def bp_by_activity(df):
    grouped = df.groupby("sedentary_flag")["Blood_Pressure_Abnormality"].agg(
        ["mean", "count"]
    ).reset_index()

    grouped = grouped[grouped["count"] >= MIN_SAMPLE_THRESHOLD]

    result = {}

    for _, row in grouped.iterrows():
        activity_flag = int(row["sedentary_flag"])
        label = "Sedentary" if activity_flag == 1 else "Active"

        result[f"{label}_BP_Rate"] = round(row["mean"], 3)
        result[f"{label}_Sample_Size"] = int(row["count"])

    return result


# ------------------------------------------
# 4. Obesity vs CKD
# ------------------------------------------

def ckd_by_obesity(df):
    grouped = df.groupby("obesity_flag")["Chronic_kidney_disease"].agg(
        ["mean", "count"]
    ).reset_index()

    grouped = grouped[grouped["count"] >= MIN_SAMPLE_THRESHOLD]

    result = {}

    for _, row in grouped.iterrows():
        obesity_flag = int(row["obesity_flag"])
        label = "Obese" if obesity_flag == 1 else "Non_Obese"

        result[f"{label}_CKD_Rate"] = round(row["mean"], 3)
        result[f"{label}_Sample_Size"] = int(row["count"])

    return result


# ------------------------------------------
# 5. Overall Summary
# ------------------------------------------

def overall_summary(df):
    return {
        "Total_Patients": int(df["Patient_Number"].nunique()),
        "Overall_BP_Rate": round(df["Blood_Pressure_Abnormality"].mean(), 3),
        "Overall_CKD_Rate": round(df["Chronic_kidney_disease"].mean(), 3),
        "Overall_Thyroid_Rate": round(df["Adrenal_and_thyroid_disorders"].mean(), 3),
    }