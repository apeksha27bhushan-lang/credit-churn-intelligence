"""
feature_engineering.py
-----------------------
Creates meaningful business-oriented derived features.
All engineered features have a clear business rationale.
"""

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the DataFrame.
    Operates on the clean DataFrame (leakage cols already removed).
    Returns a new DataFrame with additional columns.
    """
    df = df.copy()

    # ── 1. Transaction Intensity ─────────────────────────────────────────────
    # Average transaction value: higher values may indicate premium usage
    df["Avg_Trans_Value"] = np.where(
        df["Total_Trans_Ct"] > 0,
        df["Total_Trans_Amt"] / df["Total_Trans_Ct"],
        0.0,
    )

    # ── 2. Engagement Indicators ─────────────────────────────────────────────
    # Customers inactive for 3+ months in a year are flagged
    df["Is_Inactive"] = (df["Months_Inactive_12_mon"] >= 3).astype(int)

    # High contact frequency: 4+ contacts may signal friction/dissatisfaction
    df["Is_High_Contact"] = (df["Contacts_Count_12_mon"] >= 4).astype(int)

    # Low transaction count: fewer than 40 transactions in a year
    df["Is_Low_Trans"] = (df["Total_Trans_Ct"] < 40).astype(int)

    # ── 3. Revolving Utilization Category ────────────────────────────────────
    # Already captured by Avg_Utilization_Ratio but add a zero-balance flag
    # Customers carrying zero revolving balance may be less engaged
    df["Zero_Revolving_Bal"] = (df["Total_Revolving_Bal"] == 0).astype(int)

    # ── 4. Transaction Change Score ───────────────────────────────────────────
    # Combined signal: both amount and count change from Q4 to Q1
    # A large drop in both is a strong attrition signal
    df["Trans_Change_Score"] = (
        df["Total_Amt_Chng_Q4_Q1"] * df["Total_Ct_Chng_Q4_Q1"]
    )

    # ── 5. Relationship Depth ─────────────────────────────────────────────────
    # Customers with only 1 product are less embedded
    df["Low_Relationship"] = (df["Total_Relationship_Count"] <= 2).astype(int)

    # ── 6. Age Group ─────────────────────────────────────────────────────────
    # Coarse age segments for interpretability
    df["Age_Group"] = pd.cut(
        df["Customer_Age"],
        bins=[0, 35, 45, 55, 120],
        labels=["Under 35", "35-44", "45-54", "55+"],
    ).astype(str)

    return df


def get_engineered_feature_names() -> list:
    """Return the list of engineered feature column names."""
    return [
        "Avg_Trans_Value",
        "Is_Inactive",
        "Is_High_Contact",
        "Is_Low_Trans",
        "Zero_Revolving_Bal",
        "Trans_Change_Score",
        "Low_Relationship",
        "Age_Group",
    ]


def get_all_model_features(df: pd.DataFrame) -> tuple:
    """
    Return (categorical_features, numerical_features) for the full
    feature set including engineered features.
    Excludes CLIENTNUM, Attrition_Flag, Churn, and any leakage columns.
    """
    exclude = {"CLIENTNUM", "Attrition_Flag", "Churn"}

    categorical_features = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
        "Age_Group",
    ]

    numerical_features = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        # Engineered
        "Avg_Trans_Value",
        "Is_Inactive",
        "Is_High_Contact",
        "Is_Low_Trans",
        "Zero_Revolving_Bal",
        "Trans_Change_Score",
        "Low_Relationship",
    ]

    # Only return columns that actually exist in df
    categorical_features = [c for c in categorical_features if c in df.columns]
    numerical_features = [c for c in numerical_features if c in df.columns]

    return categorical_features, numerical_features
