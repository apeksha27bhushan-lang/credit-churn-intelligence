"""
data_processing.py
------------------
Loads, audits, and cleans the credit card customer dataset.
Leakage-safe: CLIENTNUM and Naive_Bayes columns are identified and excluded here.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path

# ── Column constants ─────────────────────────────────────────────────────────
TARGET_COL = "Attrition_Flag"
ID_COL = "CLIENTNUM"

# Auto-detected at import time and used everywhere
LEAKAGE_PREFIX = "Naive_Bayes_Classifier"


def load_data(filepath: str) -> pd.DataFrame:
    """Load the raw CSV and return a DataFrame."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filepath}")
    df = pd.read_csv(filepath)
    return df


def audit_data(df: pd.DataFrame) -> dict:
    """
    Perform a full data-quality audit and return a summary dict.
    Does NOT modify the DataFrame.
    """
    leakage_cols = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    summary = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_clientnum": int(df[ID_COL].duplicated().sum()) if ID_COL in df.columns else "N/A",
        "target_distribution": df[TARGET_COL].value_counts().to_dict(),
        "leakage_columns_detected": leakage_cols,
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "categorical_columns": list(df.select_dtypes(include="object").columns),
        "numerical_columns": list(df.select_dtypes(include=[np.number]).columns),
    }
    # Unique values for categoricals
    cat_unique = {}
    for col in summary["categorical_columns"]:
        cat_unique[col] = df[col].value_counts().to_dict()
    summary["categorical_distributions"] = cat_unique
    # Numerical summary
    summary["numerical_summary"] = df.select_dtypes(include=[np.number]).describe().to_dict()
    return summary


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the DataFrame:
    - Drop leakage columns.
    - Encode target.
    - Standardize categorical values where needed.
    Returns a clean copy.
    """
    df = df.copy()

    # 1. Drop leakage columns
    leakage_cols = [c for c in df.columns if c.startswith(LEAKAGE_PREFIX)]
    if leakage_cols:
        df.drop(columns=leakage_cols, inplace=True)

    # 2. Encode target: Attrited Customer = 1, Existing Customer = 0
    df["Churn"] = (df[TARGET_COL] == "Attrited Customer").astype(int)

    # 3. Strip whitespace from string columns
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].str.strip()

    return df


def get_feature_lists(df: pd.DataFrame):
    """
    Return (categorical_features, numerical_features) excluding
    ID, raw target, and encoded target.
    """
    exclude = {ID_COL, TARGET_COL, "Churn"}
    # Leakage cols should already be dropped but guard anyway
    leakage_cols = {c for c in df.columns if c.startswith(LEAKAGE_PREFIX)}
    exclude |= leakage_cols

    categorical_features = [
        c for c in df.select_dtypes(include="object").columns if c not in exclude
    ]
    numerical_features = [
        c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude
    ]
    return categorical_features, numerical_features


def print_audit_report(summary: dict) -> None:
    """Pretty-print the data audit summary."""
    print("=" * 60)
    print("DATA QUALITY AUDIT REPORT")
    print("=" * 60)
    print(f"Rows            : {summary['n_rows']:,}")
    print(f"Columns         : {summary['n_cols']}")
    print(f"Total Missing   : {summary['total_missing']}")
    print(f"Duplicate Rows  : {summary['duplicate_rows']}")
    print(f"Duplicate IDs   : {summary['duplicate_clientnum']}")
    print()
    print("Target Distribution:")
    for k, v in summary["target_distribution"].items():
        pct = v / summary["n_rows"] * 100
        print(f"  {k:<22}: {v:>5,}  ({pct:.1f}%)")
    print()
    print(f"Leakage columns detected ({len(summary['leakage_columns_detected'])}):")
    for c in summary["leakage_columns_detected"]:
        print(f"  - {c}")
    print()
    print("Categorical columns:", summary["categorical_columns"])
    print("Numerical columns :", summary["numerical_columns"])
    print("=" * 60)
