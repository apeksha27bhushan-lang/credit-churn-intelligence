"""
app.py — Credit Churn Intelligence
====================================
Professional 5-page Streamlit dashboard for credit card churn analysis.
Loads pre-trained model artifacts; does NOT retrain on startup.
"""

import sys
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Path setup ─────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR

SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
PRED_DIR = ROOT / "outputs" / "predictions"

# Root-level modules such as model_training.py
sys.path.insert(0, str(ROOT))

# Optional source directory for modules stored in src/
if SRC_DIR.is_dir():
    sys.path.insert(0, str(SRC_DIR))
# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Base */
[data-testid="stAppViewContainer"] { background: #f0f2f6; }
[data-testid="stSidebar"] { background: #0f172a; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stRadio label { font-size: 0.95rem; }

/* Metric cards */
.metric-card {
    background: #ffffff;
    border-radius: 10px;
    padding: 18px 22px;
    border-left: 4px solid #3b82f6;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    margin-bottom: 8px;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #64748b;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 4px;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #94a3b8;
    margin-top: 3px;
}
.metric-card.red { border-left-color: #ef4444; }
.metric-card.amber { border-left-color: #f59e0b; }
.metric-card.green { border-left-color: #22c55e; }
.metric-card.purple { border-left-color: #8b5cf6; }

/* Risk badges */
.badge-high { background:#fee2e2; color:#dc2626; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.82rem; }
.badge-medium { background:#fef3c7; color:#d97706; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.82rem; }
.badge-low { background:#dcfce7; color:#16a34a; padding:3px 10px; border-radius:20px; font-weight:600; font-size:0.82rem; }

/* Section headings */
.section-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #0f172a;
    border-bottom: 2px solid #e2e8f0;
    padding-bottom: 6px;
    margin-bottom: 14px;
}
/* Prediction box */
.pred-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 24px 28px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    text-align: center;
    margin-bottom: 16px;
}
.pred-box .prob { font-size: 3rem; font-weight: 800; }
.pred-box .risk-label { font-size: 1.1rem; font-weight: 600; margin-top: 8px; }
.insight-box {
    background: #f8fafc;
    border-left: 3px solid #3b82f6;
    padding: 14px 18px;
    border-radius: 6px;
    margin-top: 12px;
}
</style>
""", unsafe_allow_html=True)


# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model artifacts...")
def load_model_artifacts():
    from model_training import load_artifacts
    pipeline, metadata = load_artifacts()
    return pipeline, metadata


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    path = DATA_DIR / "credit_card_customers.csv"
    from data_processing import load_data, clean_data
    from feature_engineering import engineer_features
    df_raw  = load_data(str(path))
    df_clean = clean_data(df_raw)
    df_feat  = engineer_features(df_clean)
    return df_feat


@st.cache_data(show_spinner="Loading risk scores...")
def load_risk_scores():
    path = PRED_DIR / "customer_risk_scores.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_resource(show_spinner="Computing SHAP values...")
def load_shap_data(_pipeline, _df, feature_cols):
    """Compute SHAP values once and cache."""
    try:
        from explainability import get_explainer, get_shap_values, global_feature_importance
        X = _df[feature_cols]
        explainer, X_transformed, feature_names = get_explainer(_pipeline, X)
        shap_values = get_shap_values(explainer, X_transformed)
        importance_df = global_feature_importance(shap_values, feature_names)
        return explainer, shap_values, feature_names, importance_df
    except Exception as e:
        return None, None, None, None


# ── Helpers ───────────────────────────────────────────────────────────────────
def badge(risk: str) -> str:
    cls = {"High Risk": "badge-high", "Medium Risk": "badge-medium", "Low Risk": "badge-low"}.get(risk, "badge-low")
    return f'<span class="{cls}">{risk}</span>'


def metric_card(label, value, sub="", color="blue"):
    color_cls = {"red": "red", "amber": "amber", "green": "green", "purple": "purple"}.get(color, "")
    st.markdown(f"""
    <div class="metric-card {color_cls}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="sub">{sub}</div>
    </div>""", unsafe_allow_html=True)


def prob_color(p):
    if p >= 0.6:
        return "#dc2626"
    elif p >= 0.4:
        return "#f59e0b"
    return "#16a34a"


# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Credit Churn Intelligence")
    st.markdown("*AI-Powered Customer Analytics*")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Customer Risk Analyzer",
            "Customer Risk Explorer",
            "Explainable AI",
            "Retention Strategy",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("**Dataset**: BankChurners (Kaggle)")
    st.markdown("**Model**: XGBoost")
    st.markdown("**Threshold**: 0.554")
    st.markdown("**Test ROC-AUC**: 0.9932")
    st.markdown("---")
    st.markdown("<small style='color:#475569'>Credit Churn Intelligence v1.0<br>Apeksha Bhushan — IBM SkillsBuild Internship</small>", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
try:
    pipeline, metadata = load_model_artifacts()
    threshold = metadata["threshold"]
    all_features = metadata["all_features"]
except Exception as e:
    st.error(f"Model artifacts not found. Please run `python train_and_save.py` first.\n\nDetails: {e}")
    st.stop()

try:
    df = load_dataset()
except Exception as e:
    st.error(f"Dataset not found: {e}")
    st.stop()

risk_df = load_risk_scores()

# Merge risk scores into main df if available
if risk_df is not None:
    df = df.merge(risk_df[["CLIENTNUM", "Churn_Probability", "Risk_Category", "Recommended_Action"]], on="CLIENTNUM", how="left")
else:
    from recommendations import score_customers, assign_risk_category
    rs = score_customers(df, pipeline, threshold, all_features)
    df = df.merge(rs[["CLIENTNUM", "Churn_Probability", "Risk_Category", "Recommended_Action"]], on="CLIENTNUM", how="left")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Executive Overview
# ═══════════════════════════════════════════════════════════════════════════════
if page == "Executive Overview":
    st.markdown("# Executive Overview")
    st.markdown("Key churn metrics, risk distribution, and business intelligence at a glance.")
    st.markdown("---")

    # KPI row
    total = len(df)
    attrited = df["Churn"].sum()
    attrition_rate = attrited / total * 100
    high_risk = (df["Risk_Category"] == "High Risk").sum()
    avg_prob = df["Churn_Probability"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: metric_card("Total Customers", f"{total:,}", "In dataset", "blue")
    with c2: metric_card("Attrited Customers", f"{attrited:,}", "Historical churn", "red")
    with c3: metric_card("Attrition Rate", f"{attrition_rate:.1f}%", "Of all customers", "amber")
    with c4: metric_card("High-Risk Customers", f"{high_risk:,}", "Model prediction", "red")
    with c5: metric_card("Avg Churn Probability", f"{avg_prob:.1f}%", "Across all customers", "purple")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-title">Attrition Distribution</div>', unsafe_allow_html=True)
        fig_pie = px.pie(
            names=["Existing Customer", "Attrited Customer"],
            values=[total - attrited, attrited],
            color=["Existing Customer", "Attrited Customer"],
            color_discrete_map={"Existing Customer": "#3b82f6", "Attrited Customer": "#ef4444"},
            hole=0.45,
        )
        fig_pie.update_layout(showlegend=True, margin=dict(t=10, b=10, l=0, r=0), height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown('<div class="section-title">Risk Category Distribution</div>', unsafe_allow_html=True)
        risk_counts = df["Risk_Category"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        color_map = {"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#22c55e"}
        fig_risk = px.bar(risk_counts, x="Risk", y="Count", color="Risk",
                          color_discrete_map=color_map,
                          text="Count")
        fig_risk.update_traces(textposition="outside")
        fig_risk.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300,
                                xaxis_title="", yaxis_title="Customers")
        st.plotly_chart(fig_risk, use_container_width=True)

    # Churn by Card Category and Income Category
    col_l2, col_r2 = st.columns(2)

    with col_l2:
        st.markdown('<div class="section-title">Churn Rate by Card Category</div>', unsafe_allow_html=True)
        card_churn = df.groupby("Card_Category")["Churn"].agg(["sum", "count"]).reset_index()
        card_churn["Churn_Rate"] = card_churn["sum"] / card_churn["count"] * 100
        card_churn.columns = ["Card_Category", "Churned", "Total", "Churn_Rate"]
        fig_card = px.bar(card_churn, x="Card_Category", y="Churn_Rate",
                          color="Churn_Rate", color_continuous_scale="Reds",
                          text=card_churn["Churn_Rate"].apply(lambda x: f"{x:.1f}%"))
        fig_card.update_traces(textposition="outside")
        fig_card.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300,
                                xaxis_title="", yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig_card, use_container_width=True)

    with col_r2:
        st.markdown('<div class="section-title">Churn Rate by Income Category</div>', unsafe_allow_html=True)
        income_order = ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"]
        income_churn = df.groupby("Income_Category")["Churn"].agg(["sum", "count"]).reset_index()
        income_churn["Churn_Rate"] = income_churn["sum"] / income_churn["count"] * 100
        income_churn["Income_Category"] = pd.Categorical(income_churn["Income_Category"], categories=income_order, ordered=True)
        income_churn = income_churn.sort_values("Income_Category")
        fig_inc = px.bar(income_churn, x="Income_Category", y="Churn_Rate",
                         color="Churn_Rate", color_continuous_scale="Oranges",
                         text=income_churn["Churn_Rate"].apply(lambda x: f"{x:.1f}%"))
        fig_inc.update_traces(textposition="outside")
        fig_inc.update_layout(showlegend=False, margin=dict(t=10, b=10), height=300,
                               xaxis_title="", yaxis_title="Churn Rate (%)", coloraxis_showscale=False)
        st.plotly_chart(fig_inc, use_container_width=True)

    # Transaction behaviour
    st.markdown('<div class="section-title">Transaction Count vs Attrition</div>', unsafe_allow_html=True)
    fig_trans = px.histogram(df, x="Total_Trans_Ct", color="Attrition_Flag",
                              nbins=40, barmode="overlay", opacity=0.7,
                              color_discrete_map={"Existing Customer": "#3b82f6", "Attrited Customer": "#ef4444"},
                              labels={"Total_Trans_Ct": "Total Transactions (12 months)", "count": "Customers"})
    fig_trans.update_layout(margin=dict(t=10, b=10), height=280, legend_title="")
    st.plotly_chart(fig_trans, use_container_width=True)

    # Business Insights
    st.markdown("---")
    st.markdown("### Business Insights")
    cols = st.columns(3)
    insights = [
        ("Transaction Activity", "Attrited customers show significantly lower transaction counts (median ~34 vs ~68 for retained), making transaction frequency the strongest observable attrition signal."),
        ("Inactivity Risk", "Customers with 3+ months of inactivity are substantially more likely to churn. Proactive engagement before the 3-month mark may reduce attrition risk."),
        ("Card Category", "Gold and Platinum cardholders show higher churn rates than Blue cardholders, suggesting that higher-tier customers may have unmet expectations or stronger competitive alternatives."),
    ]
    for col, (title, text) in zip(cols, insights):
        with col:
            st.markdown(f'<div class="insight-box"><strong>{title}</strong><br><small>{text}</small></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Customer Risk Analyzer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Risk Analyzer":
    st.markdown("# Customer Risk Analyzer")
    st.markdown("Select an existing customer or enter attributes to receive a churn risk prediction with explanation.")
    st.markdown("---")

    mode = st.radio("Input Mode", ["Select Existing Customer", "Manual Entry"], horizontal=True)

    from feature_engineering import engineer_features
    from recommendations import get_retention_recommendation, assign_risk_category

    if mode == "Select Existing Customer":
        client_ids = df["CLIENTNUM"].astype(str).tolist()
        selected_id = st.selectbox("Select Customer ID", client_ids)
        row = df[df["CLIENTNUM"] == int(selected_id)].iloc[0]

        # Build input dataframe
        input_data = pd.DataFrame([row[all_features]])

    else:
        st.markdown("#### Enter Customer Attributes")
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.slider("Customer Age", 18, 80, 45)
            gender = st.selectbox("Gender", ["M", "F"])
            dep_count = st.number_input("Dependent Count", 0, 10, 2)
            edu_level = st.selectbox("Education Level", ["Graduate", "High School", "Uneducated", "College", "Post-Graduate", "Doctorate", "Unknown"])
            marital = st.selectbox("Marital Status", ["Married", "Single", "Divorced", "Unknown"])
        with c2:
            income_cat = st.selectbox("Income Category", ["Less than $40K", "$40K - $60K", "$60K - $80K", "$80K - $120K", "$120K +", "Unknown"])
            card_cat = st.selectbox("Card Category", ["Blue", "Silver", "Gold", "Platinum"])
            months_book = st.slider("Months on Book", 12, 56, 36)
            rel_count = st.slider("Total Relationship Count", 1, 6, 3)
            months_inactive = st.slider("Months Inactive (12 mon)", 0, 6, 1)
        with c3:
            contacts_count = st.slider("Contacts Count (12 mon)", 0, 6, 2)
            credit_limit = st.number_input("Credit Limit ($)", 1000, 35000, 5000)
            revolving_bal = st.number_input("Total Revolving Balance ($)", 0, 3000, 500)
            avg_open_buy = credit_limit - revolving_bal
            total_trans_amt = st.number_input("Total Transaction Amount ($)", 500, 20000, 4000)
            total_trans_ct = st.number_input("Total Transaction Count", 10, 150, 60)
            total_amt_chng = st.number_input("Total Amt Change Q4/Q1", 0.0, 3.5, 0.8, format="%.3f")
            total_ct_chng = st.number_input("Total Count Change Q4/Q1", 0.0, 3.5, 0.7, format="%.3f")
            avg_util = revolving_bal / credit_limit if credit_limit > 0 else 0.0

        raw_input = pd.DataFrame([{
            "Customer_Age": age, "Gender": gender, "Dependent_count": dep_count,
            "Education_Level": edu_level, "Marital_Status": marital,
            "Income_Category": income_cat, "Card_Category": card_cat,
            "Months_on_book": months_book, "Total_Relationship_Count": rel_count,
            "Months_Inactive_12_mon": months_inactive, "Contacts_Count_12_mon": contacts_count,
            "Credit_Limit": float(credit_limit), "Total_Revolving_Bal": revolving_bal,
            "Avg_Open_To_Buy": float(avg_open_buy),
            "Total_Amt_Chng_Q4_Q1": total_amt_chng,
            "Total_Trans_Amt": total_trans_amt, "Total_Trans_Ct": total_trans_ct,
            "Total_Ct_Chng_Q4_Q1": total_ct_chng,
            "Avg_Utilization_Ratio": float(avg_util),
            "Attrition_Flag": "Existing Customer", "Churn": 0,
        }])
        raw_feat = engineer_features(raw_input)
        input_data = raw_feat[all_features]
        row = raw_feat.iloc[0]

    # Prediction
    st.markdown("---")
    prob = float(pipeline.predict_proba(input_data)[:, 1][0])
    risk = assign_risk_category(prob, threshold)

    pc = "#dc2626" if risk == "High Risk" else "#f59e0b" if risk == "Medium Risk" else "#16a34a"

    # Columns: profile | prediction
    pcol, predcol = st.columns([1, 1])

    with pcol:
        st.markdown("#### Customer Profile")
        profile_data = {
            "Customer Age": int(row.get("Customer_Age", 0)),
            "Gender": row.get("Gender", ""),
            "Income Category": row.get("Income_Category", ""),
            "Card Category": row.get("Card_Category", ""),
            "Months on Book": int(row.get("Months_on_book", 0)),
            "Relationship Count": int(row.get("Total_Relationship_Count", 0)),
            "Months Inactive": int(row.get("Months_Inactive_12_mon", 0)),
            "Contacts (12 mon)": int(row.get("Contacts_Count_12_mon", 0)),
            "Total Transactions": int(row.get("Total_Trans_Ct", 0)),
            "Transaction Amount": f"${int(row.get('Total_Trans_Amt', 0)):,}",
            "Revolving Balance": f"${int(row.get('Total_Revolving_Bal', 0)):,}",
            "Utilization Ratio": f"{float(row.get('Avg_Utilization_Ratio', 0)):.2f}",
        }
        for k, v in profile_data.items():
            st.markdown(f"**{k}:** {v}")

    with predcol:
        st.markdown("#### Churn Prediction")
        st.markdown(f"""
        <div class="pred-box">
            <div class="prob" style="color:{pc}">{prob*100:.1f}%</div>
            <div class="risk-label" style="color:{pc}">{risk}</div>
            <div style="font-size:0.82rem;color:#64748b;margin-top:8px;">Churn Probability</div>
        </div>""", unsafe_allow_html=True)

        # SHAP explanation
        st.markdown("#### Why?")
        try:
            shap_data = load_shap_data(pipeline, df, all_features)
            if shap_data[0] is not None:
                explainer, _, feature_names, _ = shap_data
                from explainability import local_explanation
                explanation = local_explanation(explainer, pipeline, input_data, feature_names, top_n=6)
                for _, exrow in explanation.iterrows():
                    direction_color = "#dc2626" if "Increases" in exrow["Direction"] else "#16a34a"
                    st.markdown(
                        f"<div style='padding:4px 0;font-size:0.88rem;'>"
                        f"<span style='color:{direction_color};font-weight:600;'>{exrow['Direction']}</span> — "
                        f"<strong>{exrow['Feature']}</strong>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
            else:
                st.info("SHAP explanation not available for this session.")
        except Exception as ex:
            st.info(f"SHAP explanation could not be computed: {ex}")

        # Recommendation
        st.markdown("#### Suggested Retention Action")
        row_for_rec = row.copy()
        row_for_rec["Churn_Probability"] = prob
        rec = get_retention_recommendation(row_for_rec, threshold)
        st.markdown(f'<div class="insight-box">{rec}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Customer Risk Explorer
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Customer Risk Explorer":
    st.markdown("# Customer Risk Explorer")
    st.markdown("Search, filter, and sort customers by predicted churn risk.")
    st.markdown("---")

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        risk_filter = st.multiselect("Risk Category", ["High Risk", "Medium Risk", "Low Risk"],
                                      default=["High Risk", "Medium Risk"])
    with fc2:
        card_filter = st.multiselect("Card Category",
                                      sorted(df["Card_Category"].unique()),
                                      default=sorted(df["Card_Category"].unique()))
    with fc3:
        income_filter = st.multiselect("Income Category",
                                        sorted(df["Income_Category"].unique()),
                                        default=sorted(df["Income_Category"].unique()))
    with fc4:
        prob_range = st.slider("Churn Probability Range", 0.0, 1.0, (0.0, 1.0), 0.01)

    filtered = df[
        df["Risk_Category"].isin(risk_filter) &
        df["Card_Category"].isin(card_filter) &
        df["Income_Category"].isin(income_filter) &
        (df["Churn_Probability"] >= prob_range[0]) &
        (df["Churn_Probability"] <= prob_range[1])
    ].copy()

    st.markdown(f"**{len(filtered):,}** customers match the current filters.")

    display_cols = ["CLIENTNUM", "Churn_Probability", "Risk_Category",
                    "Customer_Age", "Card_Category", "Income_Category",
                    "Months_Inactive_12_mon", "Total_Trans_Ct",
                    "Total_Trans_Amt", "Recommended_Action"]
    available = [c for c in display_cols if c in filtered.columns]
    show_df = filtered[available].sort_values("Churn_Probability", ascending=False)

    # Format
    show_df["Churn_Probability"] = show_df["Churn_Probability"].apply(lambda x: f"{x*100:.1f}%")
    show_df = show_df.rename(columns={
        "CLIENTNUM": "Customer ID",
        "Churn_Probability": "Churn Prob",
        "Risk_Category": "Risk",
        "Customer_Age": "Age",
        "Card_Category": "Card",
        "Income_Category": "Income",
        "Months_Inactive_12_mon": "Months Inactive",
        "Total_Trans_Ct": "Trans Count",
        "Total_Trans_Amt": "Trans Amount ($)",
        "Recommended_Action": "Suggested Action",
    })

    st.dataframe(show_df, use_container_width=True, height=480)

    # Download
    csv_data = filtered[available].sort_values("Churn_Probability", ascending=False).to_csv(index=False)
    st.download_button("Download Filtered Data (CSV)", csv_data, "filtered_customers.csv", "text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Explainable AI
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Explainable AI":
    st.markdown("# Explainable AI")
    st.markdown("""
    SHAP (SHapley Additive exPlanations) shows the contribution of each feature to the model's prediction.
    A positive SHAP value increases the predicted churn probability; a negative value decreases it.
    
    > **Important**: SHAP values explain the model's output — they reflect statistical association, not causation.
    """)
    st.markdown("---")

    shap_data = load_shap_data(pipeline, df, all_features)

    if shap_data[0] is None:
        st.warning("SHAP computation is unavailable. Please ensure the model is loaded correctly.")
    else:
        explainer, shap_values, feature_names, importance_df = shap_data

        # Global importance
        st.markdown("### Global Feature Importance (Mean |SHAP|)")
        st.markdown("Features ranked by their average absolute contribution to the model prediction across all customers.")

        top_n = st.slider("Show top N features", 10, min(30, len(importance_df)), 15)
        top_imp = importance_df.head(top_n)

        fig_imp = px.bar(
            top_imp.iloc[::-1],
            x="Mean_SHAP",
            y="Feature",
            orientation="h",
            color="Mean_SHAP",
            color_continuous_scale="Blues",
            labels={"Mean_SHAP": "Mean |SHAP| Value", "Feature": ""},
        )
        fig_imp.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=180, r=20),
            height=max(350, top_n * 22),
            yaxis=dict(tickfont=dict(size=11)),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        st.markdown("---")
        st.markdown("### Top Churn Drivers")
        st.markdown("The five most influential features in the model, with business interpretation.")

        drivers = importance_df.head(5)
        driver_interpretations = {
            "Total_Trans_Ct": "Customers with fewer transactions are more strongly associated with predicted attrition. Low card usage is the model's strongest signal.",
            "Total_Trans_Amt": "Lower total spending is closely linked with attrition prediction — disengaged customers tend to use the card less frequently and for smaller amounts.",
            "Total_Ct_Chng_Q4_Q1": "A declining transaction count ratio between Q4 and Q1 signals waning engagement and is strongly associated with model-predicted churn.",
            "Total_Amt_Chng_Q4_Q1": "A drop in transaction amount over time is a key behavioural indicator of disengagement.",
            "Months_Inactive_12_mon": "Extended inactivity periods are strongly associated with higher predicted churn probability.",
        }
        for _, drow in drivers.iterrows():
            interp = driver_interpretations.get(drow["Feature"], "Contributing factor to predicted churn risk.")
            st.markdown(f"""
            <div class="insight-box" style="margin-bottom:10px;">
                <strong>{drow['Feature']}</strong> — Mean |SHAP| = {drow['Mean_SHAP']:.4f}<br>
                <small style="color:#475569">{interp}</small>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Individual Customer Explanation")
        client_ids = df["CLIENTNUM"].astype(str).tolist()
        sel_id = st.selectbox("Select Customer ID for Explanation", client_ids, key="shap_customer")
        sel_row = df[df["CLIENTNUM"] == int(sel_id)].iloc[0]
        sel_input = pd.DataFrame([sel_row[all_features]])

        try:
            from explainability import local_explanation
            local_exp = local_explanation(explainer, pipeline, sel_input, feature_names, top_n=10)
            sel_prob = float(pipeline.predict_proba(sel_input)[:, 1][0])

            from recommendations import assign_risk_category
            sel_risk = assign_risk_category(sel_prob, threshold)
            pc = "#dc2626" if sel_risk == "High Risk" else "#f59e0b" if sel_risk == "Medium Risk" else "#16a34a"

            st.markdown(f"**Customer {sel_id}** — Predicted Churn Probability: "
                        f"<span style='color:{pc};font-weight:700'>{sel_prob*100:.1f}%</span> "
                        f"(<span style='color:{pc}'>{sel_risk}</span>)", unsafe_allow_html=True)

            fig_local = go.Figure(go.Bar(
                x=local_exp["SHAP_Value"],
                y=local_exp["Feature"],
                orientation="h",
                marker_color=["#ef4444" if v > 0 else "#22c55e" for v in local_exp["SHAP_Value"]],
                text=[f"{v:+.4f}" for v in local_exp["SHAP_Value"]],
                textposition="outside",
            ))
            fig_local.update_layout(
                xaxis_title="SHAP Value (positive = increases risk)",
                margin=dict(t=10, b=10, l=200, r=60),
                height=360,
                yaxis=dict(tickfont=dict(size=11)),
            )
            st.plotly_chart(fig_local, use_container_width=True)
            st.caption("Red bars increase predicted churn risk; green bars decrease it. Values represent SHAP contributions, not causal effects.")
        except Exception as ex:
            st.warning(f"Individual explanation unavailable: {ex}")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — Retention Strategy
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "Retention Strategy":
    st.markdown("# Retention Strategy")
    st.markdown("Data-driven suggested retention interventions based on predicted risk patterns.")
    st.markdown("> All recommendations are decision-support suggestions based on model predictions and observable patterns. They do not guarantee churn prevention.")
    st.markdown("---")

    from recommendations import assign_risk_category

    high_risk_df = df[df["Risk_Category"] == "High Risk"]
    med_risk_df  = df[df["Risk_Category"] == "Medium Risk"]
    low_risk_df  = df[df["Risk_Category"] == "Low Risk"]

    # KPIs
    k1, k2, k3 = st.columns(3)
    with k1: metric_card("High-Risk Customers", f"{len(high_risk_df):,}", f"{len(high_risk_df)/len(df)*100:.1f}% of customers", "red")
    with k2: metric_card("Medium-Risk Customers", f"{len(med_risk_df):,}", f"{len(med_risk_df)/len(df)*100:.1f}% of customers", "amber")
    with k3: metric_card("Low-Risk Customers", f"{len(low_risk_df):,}", f"{len(low_risk_df)/len(df)*100:.1f}% of customers", "green")

    st.markdown("---")

    # Risk probability distribution
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="section-title">Churn Probability Distribution by Risk Tier</div>', unsafe_allow_html=True)
        fig_dist = px.histogram(
            df, x="Churn_Probability", color="Risk_Category",
            nbins=50, barmode="overlay", opacity=0.75,
            color_discrete_map={"High Risk": "#ef4444", "Medium Risk": "#f59e0b", "Low Risk": "#22c55e"},
        )
        fig_dist.update_layout(margin=dict(t=10, b=10), height=300, legend_title="", xaxis_title="Churn Probability", yaxis_title="Customers")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Recommended Actions Distribution</div>', unsafe_allow_html=True)
        # Shorten action labels for chart
        def short_action(a):
            if pd.isna(a): return "Other"
            if "Re-engagement" in a: return "Re-engagement Campaign"
            if "Cross-sell" in a: return "Cross-sell Engagement"
            if "service review" in a: return "Service Review"
            if "Loyalty" in a or "cashback" in a: return "Loyalty Incentive"
            if "Usage stimulus" in a: return "Usage Stimulus"
            if "Priority retention" in a: return "Priority Outreach"
            if "communication" in a: return "Targeted Communication"
            if "Preventive" in a: return "Preventive Engagement"
            return "No Action Needed"

        high_risk_df = high_risk_df.copy()
        high_risk_df["Short_Action"] = high_risk_df["Recommended_Action"].apply(short_action)
        action_counts = high_risk_df["Short_Action"].value_counts().reset_index()
        action_counts.columns = ["Action", "Count"]
        fig_act = px.pie(action_counts, values="Count", names="Action", hole=0.4)
        fig_act.update_layout(margin=dict(t=10, b=10), height=300)
        st.plotly_chart(fig_act, use_container_width=True)

    st.markdown("---")
    st.markdown("### Retention Intervention Segments")

    segments = [
        {
            "Segment": "Re-engagement Required",
            "Risk Pattern": "High inactivity (3+ months) + Low transaction count",
            "Size": int((df["Months_Inactive_12_mon"] >= 3).sum()),
            "Suggested Action": "Personalised reward bonus for card usage within 30 days",
        },
        {
            "Segment": "Loyalty Incentive Needed",
            "Risk Pattern": "Declining transaction count and amount trends",
            "Size": int((df["Total_Ct_Chng_Q4_Q1"] < 0.6).sum()),
            "Suggested Action": "Cashback or reward points to stimulate card activity",
        },
        {
            "Segment": "Service Review Priority",
            "Risk Pattern": "High contact frequency (4+ calls/year)",
            "Size": int((df["Contacts_Count_12_mon"] >= 4).sum()),
            "Suggested Action": "Proactive service review; assign relationship manager",
        },
        {
            "Segment": "Cross-sell Opportunity",
            "Risk Pattern": "Low relationship count (1-2 products)",
            "Size": int((df["Total_Relationship_Count"] <= 2).sum()),
            "Suggested Action": "Introduce relevant additional banking products",
        },
        {
            "Segment": "Usage Stimulus",
            "Risk Pattern": "Low revolving balance / low utilization",
            "Size": int((df["Total_Revolving_Bal"] == 0).sum()),
            "Suggested Action": "Highlight card benefits; targeted promotional offer",
        },
    ]
    seg_df = pd.DataFrame(segments)
    st.dataframe(seg_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### High-Risk Customers — Detailed View")
    disp = high_risk_df[["CLIENTNUM", "Churn_Probability", "Customer_Age",
                           "Card_Category", "Income_Category",
                           "Months_Inactive_12_mon", "Total_Trans_Ct",
                           "Recommended_Action"]].sort_values("Churn_Probability", ascending=False)
    disp["Churn_Probability"] = disp["Churn_Probability"].apply(lambda x: f"{x*100:.1f}%")
    disp = disp.rename(columns={
        "CLIENTNUM": "Customer ID", "Churn_Probability": "Churn Prob",
        "Customer_Age": "Age", "Card_Category": "Card",
        "Income_Category": "Income", "Months_Inactive_12_mon": "Months Inactive",
        "Total_Trans_Ct": "Trans Count", "Recommended_Action": "Suggested Action"
    })
    st.dataframe(disp, use_container_width=True, height=380)

    csv_hr = high_risk_df[["CLIENTNUM", "Churn_Probability", "Risk_Category", "Recommended_Action"]].to_csv(index=False)
    st.download_button("Download High-Risk Customer List (CSV)", csv_hr, "high_risk_customers.csv", "text/csv")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#94a3b8;font-size:0.78rem;'>"
    "Credit Churn Intelligence &nbsp;|&nbsp; Apeksha Bhushan &nbsp;|&nbsp; IBM SkillsBuild Data Analytics with AI Internship"
    "</div>",
    unsafe_allow_html=True,
)
