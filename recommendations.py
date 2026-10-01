"""
recommendations.py
------------------
Rule-based retention recommendation engine.
Rules are derived from EDA findings and model feature importance.
All recommendations are framed as decision-support suggestions,
NOT causal guarantees.
"""

import pandas as pd


# ── Risk Category Mapping ────────────────────────────────────────────────────
def assign_risk_category(prob: float, threshold: float) -> str:
    """
    Convert churn probability to a risk tier.
    - High Risk  : prob >= threshold + 0.15
    - Medium Risk: threshold <= prob < threshold + 0.15
    - Low Risk   : prob < threshold
    Using a buffer above threshold to reduce false High-Risk labelling.
    """
    high_cutoff = min(threshold + 0.15, 0.85)
    if prob >= high_cutoff:
        return "High Risk"
    elif prob >= threshold:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_retention_recommendation(row: pd.Series, threshold: float) -> str:
    """
    Generate a plain-language suggested retention action for a single customer.
    Based on observable risk factors in the row.
    Returns a string recommendation.
    """
    prob = row.get("Churn_Probability", 0.0)
    risk = assign_risk_category(prob, threshold)

    if risk == "Low Risk":
        return "No immediate action required. Monitor engagement quarterly."

    # Collect risk signals
    signals = []

    months_inactive = row.get("Months_Inactive_12_mon", 0)
    contacts = row.get("Contacts_Count_12_mon", 0)
    trans_ct = row.get("Total_Trans_Ct", 100)
    revolving_bal = row.get("Total_Revolving_Bal", 1)
    relationship_count = row.get("Total_Relationship_Count", 3)
    trans_change = row.get("Total_Ct_Chng_Q4_Q1", 1.0)
    utilization = row.get("Avg_Utilization_Ratio", 0.3)

    if months_inactive >= 3:
        signals.append("high_inactivity")
    if contacts >= 4:
        signals.append("high_contact")
    if trans_ct < 40:
        signals.append("low_transactions")
    if revolving_bal == 0:
        signals.append("zero_revolving")
    if relationship_count <= 2:
        signals.append("low_relationship")
    if trans_change < 0.6:
        signals.append("declining_transactions")
    if utilization < 0.1:
        signals.append("low_utilization")

    # Priority-order rule matching
    if "high_inactivity" in signals and "low_transactions" in signals:
        return (
            "Suggested action: Re-engagement campaign — "
            "offer personalised reward bonus for card usage within 30 days."
        )
    if "high_contact" in signals:
        return (
            "Suggested action: Proactive service review — "
            "contact customer to resolve outstanding concerns and improve experience."
        )
    if "declining_transactions" in signals and risk == "High Risk":
        return (
            "Suggested action: Loyalty incentive — "
            "offer cashback or reward points to encourage increased card activity."
        )
    if "low_relationship" in signals:
        return (
            "Suggested action: Cross-sell engagement — "
            "introduce relevant additional products to deepen the banking relationship."
        )
    if "zero_revolving" in signals or "low_utilization" in signals:
        return (
            "Suggested action: Usage stimulus — "
            "communicate card benefits and consider a targeted promotional offer."
        )
    if "low_transactions" in signals:
        return (
            "Suggested action: Targeted communication — "
            "highlight rewards program and card usage benefits."
        )

    # Generic high/medium risk fallback
    if risk == "High Risk":
        return (
            "Suggested action: Priority retention outreach — "
            "assign to relationship manager for personalised retention conversation."
        )
    return (
        "Suggested action: Preventive engagement — "
        "include in next loyalty/reward communication campaign."
    )


def score_customers(
    df: pd.DataFrame,
    pipeline,
    threshold: float,
    feature_cols: list,
) -> pd.DataFrame:
    """
    Produce a customer-level risk scoring table.
    df must contain CLIENTNUM and all feature_cols.
    Returns a DataFrame with CLIENTNUM, Churn_Probability, Risk_Category,
    and Recommended_Action.
    """
    X = df[feature_cols]
    probs = pipeline.predict_proba(X)[:, 1]

    result = df[["CLIENTNUM"]].copy()
    result["Churn_Probability"] = probs.round(4)
    result["Risk_Category"] = [
        assign_risk_category(p, threshold) for p in probs
    ]

    # Merge needed columns for recommendation engine
    cols_needed = [
        "Months_Inactive_12_mon", "Contacts_Count_12_mon", "Total_Trans_Ct",
        "Total_Revolving_Bal", "Total_Relationship_Count",
        "Total_Ct_Chng_Q4_Q1", "Avg_Utilization_Ratio",
    ]
    existing_cols = [c for c in cols_needed if c in df.columns]
    for col in existing_cols:
        result[col] = df[col].values

    result["Recommended_Action"] = result.apply(
        lambda row: get_retention_recommendation(row, threshold), axis=1
    )

    # Drop helper columns from result to keep output clean
    result.drop(columns=existing_cols, inplace=True, errors="ignore")

    return result
