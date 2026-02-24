import pandas as pd
import streamlit as st
import os

DATASET_1_PATH = "data/Health Dataset 1.xlsm"
DATASET_2_PATH = "data/Health Dataset 2.xlsm"


@st.cache_data
def load_dataset_1():
    """
    Load patient health profile dataset.
    """
    if not os.path.exists(DATASET_1_PATH):
        raise FileNotFoundError("Dataset 1 not found in data folder.")

    df = pd.read_excel(DATASET_1_PATH, engine="openpyxl")
    return df


@st.cache_data
def load_dataset_2():
    """
    Load physical activity dataset.
    """
    if not os.path.exists(DATASET_2_PATH):
        raise FileNotFoundError("Dataset 2 not found in data folder.")

    df = pd.read_excel(DATASET_2_PATH, engine="openpyxl")
    return df


def validate_schema(df1, df2):
    required_df1_cols = [
        "Patient_Number",
        "Blood_Pressure_Abnormality",
        "Level_of_Hemoglobin",
        "Genetic_Pedigree_Coefficient",
        "Age",
        "BMI",
        "Sex",
        "Pregnancy",
        "Smoking",
        "salt_content_in_the_diet",
        "alcohol_consumption_per_day",
        "Level_of_Stress",
        "Chronic_kidney_disease",
        "Adrenal_and_thyroid_disorders",
    ]

    required_df2_cols = [
        "Patient_Number",
        "Day_Number",
        "Physical_activity"
    ]

    missing_df1 = [col for col in required_df1_cols if col not in df1.columns]
    missing_df2 = [col for col in required_df2_cols if col not in df2.columns]

    if missing_df1:
        raise ValueError(f"Missing columns in Dataset 1: {missing_df1}")

    if missing_df2:
        raise ValueError(f"Missing columns in Dataset 2: {missing_df2}")

    return True