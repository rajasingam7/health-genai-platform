import streamlit as st
import pandas as pd


@st.cache_data
def load_dataset_1():
    """
    Load Health Dataset 1.
    Cached to prevent reloading on every Streamlit rerun.
    """
    df1 = pd.read_excel("data/Health Dataset 1.xlsm")
    return df1


@st.cache_data
def load_dataset_2():
    """
    Load Health Dataset 2.
    Cached to prevent reloading on every Streamlit rerun.
    """
    df2 = pd.read_excel("data/Health Dataset 2.xlsm")
    return df2


def validate_schema(df1, df2):
    """
    Basic schema validation to ensure required columns exist.
    """

    required_columns_df1 = [
        "Patient_Number",
        "Age",
        "BMI",
        "Level_of_Hemoglobin",
        "Genetic_Pedigree_Coefficient",
        "Smoking",
        "salt_content_in_the_diet",
        "alcohol_consumption_per_day",
        "Level_of_Stress",
        "Blood_Pressure_Abnormality"
    ]

    required_columns_df2 = [
        "Patient_Number",
        "Number_of_Steps"
    ]

    for col in required_columns_df1:
        if col not in df1.columns:
            raise ValueError(f"Missing column in Dataset 1: {col}")

    for col in required_columns_df2:
        if col not in df2.columns:
            raise ValueError(f"Missing column in Dataset 2: {col}")

    return True