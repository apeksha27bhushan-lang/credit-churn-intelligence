# Credit Churn Intelligence

> **AI-Powered Credit Card Customer Churn Prediction and Data-Driven Retention Strategies Using Machine Learning and Explainable AI**

[![IBM SkillsBuild](https://img.shields.io/badge/IBM%20SkillsBuild-Data%20Analytics%20with%20AI-blue)](https://skillsbuild.org)
[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)

---

## Project Overview

Credit Churn Intelligence is an end-to-end AI-powered customer analytics product that predicts which credit card customers are at risk of attriting, explains the factors associated with that risk, and generates data-driven retention recommendations.

This project was developed as part of the **IBM SkillsBuild Data Analytics with AI Academic Internship** and is suitable as a portfolio-grade, interview-ready data science project.

---

## Business Problem

Customer attrition (churn) is a critical challenge in the financial services industry. Acquiring a new customer costs significantly more than retaining an existing one. For credit card businesses, identifying at-risk customers early — before they close their account — enables targeted, cost-effective retention interventions.

This project addresses the following business questions:
- Which customers are most likely to churn in the near term?
- What behavioural and demographic factors are associated with attrition risk?
- How can the business prioritise and personalise retention efforts?

---

## Objectives

1. Build a reliable, leakage-free machine learning pipeline for churn prediction.
2. Achieve strong discriminative performance (ROC-AUC > 0.95).
3. Provide transparent, explainable predictions using SHAP.
4. Segment customers by risk tier for targeted action.
5. Generate data-driven retention recommendations per customer.
6. Deliver an interactive, professional Streamlit dashboard for business users.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | [Kaggle — Credit Card Customers](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers) |
| Total Customers | 10,127 |
| Features | 23 (raw) |
| Target | `Attrition_Flag` |
| Attrited Customers | 1,627 (16.1%) |
| Existing Customers | 8,500 (83.9%) |
| Missing Values | None |
| Duplicate Rows | None |

---

## Dataset Features

### Demographic
- `Customer_Age`, `Gender`, `Dependent_count`, `Education_Level`, `Marital_Status`, `Income_Category`

### Product / Relationship
- `Card_Category`, `Months_on_book`, `Total_Relationship_Count`

### Engagement
- `Months_Inactive_12_mon`, `Contacts_Count_12_mon`

### Financial / Credit
- `Credit_Limit`, `Total_Revolving_Bal`, `Avg_Open_To_Buy`, `Avg_Utilization_Ratio`

### Transaction Behaviour
- `Total_Trans_Amt`, `Total_Trans_Ct`, `Total_Amt_Chng_Q4_Q1`, `Total_Ct_Chng_Q4_Q1`

### Excluded — Data Leakage
The following columns are **pre-computed Naive Bayes classifier probabilities directly derived from the target variable** and were excluded from all modelling:
- `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1`
- `Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2`

`CLIENTNUM` was also excluded as a model feature (used only for customer identification).

---

## Data Quality

| Check | Result |
|-------|--------|
| Missing Values | 0 |
| Duplicate Rows | 0 |
| Duplicate Customer IDs | 0 |
| Class Imbalance | Yes — 83.9% / 16.1% |
| Leakage Columns | 2 detected and excluded |

---

## Methodology

```
Raw Customer Data
     ↓
Data Quality Audit
     ↓
Data Cleaning + Target Encoding
     ↓
Feature Engineering (8 derived features)
     ↓
Stratified Train/Test Split (80/20, seed=42)
     ↓
ColumnTransformer Preprocessing Pipeline
     ↓
Cross-Validated Model Comparison (5-Fold CV)
     ↓
Final Model Training + Threshold Optimisation
     ↓
SHAP Explainability
     ↓
Customer Risk Scoring
     ↓
Retention Recommendation Engine
     ↓
Streamlit Interactive Dashboard
```

---

## Machine Learning Models

| Model | Purpose |
|-------|---------|
| Logistic Regression | Interpretable linear baseline |
| Random Forest | Bagging / ensemble tree model |
| **XGBoost** | **Selected final model** |

All models used `class_weight='balanced'` or `scale_pos_weight` to handle class imbalance. Preprocessing was fit only on training data to prevent leakage.

---

## Evaluation Metrics

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|-------|:--------:|:---------:|:------:|:--:|:-------:|:------:|
| **XGBoost** | **0.9724** | **0.9164** | **0.9108** | **0.9136** | **0.9932** | **0.9699** |
| Random Forest | 0.9418 | 0.7867 | 0.8738 | 0.8280 | 0.9803 | 0.9204 |
| Logistic Regression | 0.9166 | 0.7167 | 0.7938 | 0.7533 | 0.9533 | 0.8239 |

*Evaluated on the held-out test set (20% of data). Threshold optimised on validation data only.*

**Final Model**: XGBoost  
**Operating Threshold**: 0.554 (optimised for F1 on validation set)

---

## Explainable AI

SHAP (SHapley Additive exPlanations) is used via `TreeExplainer` for the XGBoost model. 

- **Global explanations**: Mean |SHAP| across all customers to identify the most influential features.
- **Local explanations**: Per-customer feature contributions, showing which factors increase or decrease the predicted risk.

Top contributing features (by Mean |SHAP|):
1. `Total_Trans_Ct` — Transaction count
2. `Total_Trans_Amt` — Transaction amount
3. `Total_Ct_Chng_Q4_Q1` — Declining transaction count trend
4. `Total_Amt_Chng_Q4_Q1` — Declining transaction amount trend
5. `Months_Inactive_12_mon` — Inactivity duration

> SHAP values represent model contributions and reflect statistical associations, not causal relationships.

---

## Customer Risk Scoring

Each customer receives:
- **Churn Probability** (0–100%)
- **Risk Category**: High Risk / Medium Risk / Low Risk

Thresholds:
- **High Risk**: probability ≥ 0.704 (threshold + 0.15)
- **Medium Risk**: 0.554 ≤ probability < 0.704
- **Low Risk**: probability < 0.554

---

## Retention Recommendation Engine

A transparent rule-based system that maps observable risk patterns to suggested retention actions:

| Risk Pattern | Suggested Action |
|-------------|-----------------|
| High inactivity + low transactions | Re-engagement campaign with reward bonus |
| High contact frequency | Proactive service review |
| Declining transaction trends | Loyalty incentive (cashback/rewards) |
| Low relationship count | Cross-sell engagement |
| Low utilization / zero revolving balance | Usage stimulus communication |

All recommendations are framed as decision-support suggestions, not causal guarantees.

---

## Streamlit Dashboard

The interactive dashboard provides five views:

| Page | Description |
|------|-------------|
| Executive Overview | KPIs, attrition distribution, churn by category, business insights |
| Customer Risk Analyzer | Individual customer prediction with SHAP explanation and retention recommendation |
| Customer Risk Explorer | Filterable, sortable customer risk table with export |
| Explainable AI | Global feature importance, top churn drivers, individual SHAP waterfall |
| Retention Strategy | Risk tier summary, intervention segments, high-risk customer table |

---

## Project Structure

```
credit-churn-intelligence/
│
├── data/
│   └── credit_card_customers.csv
│
├── notebooks/
│   └── Apeksha Bhushan_CreditCardCustomerChurn.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   ├── evaluation.py
│   ├── explainability.py
│   └── recommendations.py
│
├── models/
│   ├── final_model.joblib
│   └── model_metadata.json
│
├── app/
│   └── app.py
│
├── outputs/
│   ├── figures/
│   ├── tables/
│   │   └── model_comparison.csv
│   └── predictions/
│       └── customer_risk_scores.csv
│
├── train_and_save.py
├── requirements.txt
├── README.md
└── Apeksha Bhushan_CreditCardCustomerChurnProjectReport.docx
```

---

## Installation

```bash
git clone <repository-url>
cd credit-churn-intelligence
pip install -r requirements.txt
```

---

## Train the Model

If model artifacts are not yet present, train the models:

```bash
python train_and_save.py
```

This will:
- Load and audit the data
- Engineer features
- Train and compare Logistic Regression, Random Forest, and XGBoost
- Select the best model (XGBoost)
- Save `models/final_model.joblib` and `models/model_metadata.json`
- Generate `outputs/predictions/customer_risk_scores.csv`

---

## Run Application

```bash
streamlit run app/app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Results

| Metric | Value |
|--------|-------|
| Final Model | XGBoost |
| Test ROC-AUC | 0.9932 |
| Test PR-AUC | 0.9699 |
| Test F1 Score | 0.9136 |
| Test Recall | 0.9108 |
| Test Precision | 0.9164 |
| Test Accuracy | 0.9724 |
| Operating Threshold | 0.554 |
| High-Risk Customers | 1,615 |

---

## Business Insights

1. **Transaction activity is the strongest model signal**: Customers with fewer than 40 annual transactions are substantially more likely to be predicted as attrition-risk.
2. **Declining trends matter more than levels**: The Q4/Q1 change ratios for transaction count and amount are among the top predictors, indicating that *direction of change* is highly informative.
3. **Inactivity compounds risk**: Three or more months of card inactivity is a high-priority early warning indicator.
4. **High-contact customers need service review**: Customers who contacted the bank 4+ times are more likely to be at risk, suggesting unresolved friction.
5. **Gold and Platinum cards show higher churn rates**, which may reflect unmet expectations among premium cardholders.

---

## Limitations

- The model is trained on a static, cross-sectional snapshot. It does not capture real-time behavioural sequences.
- The dataset does not include external factors (competitive offers, economic conditions, life events).
- Causal inference is not supported — SHAP values reflect model associations, not causal effects.
- The recommendation engine is rule-based and has not been experimentally validated for effectiveness.
- Model performance may degrade over time as customer behaviour evolves; periodic retraining is recommended.

---

## Future Improvements

- Implement temporal validation (time-based train/test split) as longitudinal data becomes available.
- Add survival analysis to estimate *when* a customer is likely to churn.
- Integrate real-time scoring via an API endpoint.
- A/B test retention recommendations to measure actual effectiveness.
- Explore deep learning approaches (e.g., tabular transformers) for potential performance gains.

---

## Author

**Apeksha Bhushan**  
IBM SkillsBuild Data Analytics with AI Academic Internship  
Dataset: [Kaggle — Credit Card Customers by Sakshi Goyal](https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers)
