"""
train_and_save.py
-----------------
Standalone script that:
1. Loads and cleans data
2. Engineers features
3. Trains all models with CV
4. Selects the best model
5. Fits the final model on the full training set
6. Saves artifacts to models/
7. Saves predictions to outputs/predictions/

Run from the project root:
    python train_and_save.py
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Allow imports from src/
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from data_processing import load_data, audit_data, clean_data, print_audit_report
from feature_engineering import engineer_features, get_all_model_features
from model_training import (
    build_preprocessor,
    split_data,
    build_model_pipelines,
    evaluate_pipeline_cv,
    find_optimal_threshold,
    save_artifacts,
    RANDOM_SEED,
)
from evaluation import compute_metrics, compute_confusion, compare_models, print_model_comparison
from recommendations import score_customers

DATA_PATH = ROOT / "data" / "credit_card_customers.csv"
OUTPUTS_DIR = ROOT / "outputs"

# ── 1. Load & Audit ──────────────────────────────────────────────────────────
print("Loading data...")
df_raw = load_data(str(DATA_PATH))
audit = audit_data(df_raw)
print_audit_report(audit)

# ── 2. Clean ─────────────────────────────────────────────────────────────────
print("Cleaning data...")
df_clean = clean_data(df_raw)

# ── 3. Feature Engineering ────────────────────────────────────────────────────
print("Engineering features...")
df_feat = engineer_features(df_clean)

categorical_features, numerical_features = get_all_model_features(df_feat)
all_features = numerical_features + categorical_features

print(f"Features: {len(numerical_features)} numerical, {len(categorical_features)} categorical")

# ── 4. Train/Test Split ───────────────────────────────────────────────────────
print("Splitting data...")
X_train, X_test, y_train, y_test = split_data(df_feat, all_features, "Churn")
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Train churn rate: {y_train.mean():.3f}  |  Test churn rate: {y_test.mean():.3f}")

# ── 5. CV Model Comparison ────────────────────────────────────────────────────
print("\nRunning cross-validated model comparison...")
preprocessor = build_preprocessor(categorical_features, numerical_features)
pipelines = build_model_pipelines(preprocessor)

cv_results = {}
for name, pipe in pipelines.items():
    print(f"  CV: {name} ...", end=" ")
    roc, pr = evaluate_pipeline_cv(pipe, X_train, y_train, cv=5)
    cv_results[name] = {"CV_ROC_AUC": round(roc, 4), "CV_PR_AUC": round(pr, 4)}
    print(f"ROC-AUC={roc:.4f}  PR-AUC={pr:.4f}")

# ── 6. Train Final Models on Full Training Set & Evaluate on Test ─────────────
print("\nTraining final models on full training set...")
test_results = {}
trained_pipelines = {}

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    # Find optimal threshold on validation portion (20% of train)
    from sklearn.model_selection import train_test_split as tts
    X_tr2, X_val, y_tr2, y_val = tts(
        X_train, y_train, test_size=0.20, stratify=y_train, random_state=RANDOM_SEED
    )
    # Re-fit on reduced training to get threshold
    pipe_thresh = build_model_pipelines(
        build_preprocessor(categorical_features, numerical_features)
    )[name]
    pipe_thresh.fit(X_tr2, y_tr2)
    threshold, _ = find_optimal_threshold(pipe_thresh, X_val, y_val)

    # Evaluate on test set using full-train pipeline
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_pred, y_prob)
    metrics["Threshold"] = round(threshold, 4)
    test_results[name] = metrics
    print(f"  {name}: ROC-AUC={metrics['ROC-AUC']:.4f}  F1={metrics['F1']:.4f}  Threshold={threshold:.3f}")

comparison_df = compare_models({k: {kk: vv for kk, vv in v.items() if kk != "Threshold"}
                                  for k, v in test_results.items()})
print_model_comparison(comparison_df)

# ── 7. Select Final Model ─────────────────────────────────────────────────────
# Select by highest ROC-AUC
best_model_name = comparison_df.index[0]
print(f"\nSelected final model: {best_model_name}")
final_pipeline = trained_pipelines[best_model_name]
final_threshold = test_results[best_model_name]["Threshold"]
final_metrics = test_results[best_model_name]

print(f"Final threshold: {final_threshold:.4f}")
print(f"Test ROC-AUC:    {final_metrics['ROC-AUC']:.4f}")
print(f"Test PR-AUC:     {final_metrics['PR-AUC']:.4f}")
print(f"Test F1:         {final_metrics['F1']:.4f}")
print(f"Test Recall:     {final_metrics['Recall']:.4f}")
print(f"Test Precision:  {final_metrics['Precision']:.4f}")

# ── 8. Save Artifacts ─────────────────────────────────────────────────────────
save_artifacts(
    pipeline=final_pipeline,
    threshold=final_threshold,
    categorical_features=categorical_features,
    numerical_features=numerical_features,
    model_name=best_model_name,
    metrics={**final_metrics, **cv_results.get(best_model_name, {})},
)

# Also save all model comparison results
comparison_path = OUTPUTS_DIR / "tables" / "model_comparison.csv"
comparison_df.to_csv(comparison_path)
print("[OK] Saved model_comparison.csv")

# ── 9. Generate Customer Risk Scores for All Customers ───────────────────────
print("\nGenerating customer risk scores for full dataset...")
risk_scores = score_customers(
    df=df_feat,
    pipeline=final_pipeline,
    threshold=final_threshold,
    feature_cols=all_features,
)

# Merge additional display columns
display_cols = [
    "CLIENTNUM", "Attrition_Flag", "Customer_Age", "Gender",
    "Income_Category", "Card_Category", "Months_Inactive_12_mon",
    "Total_Trans_Ct", "Total_Trans_Amt", "Total_Relationship_Count",
]
available_display = [c for c in display_cols if c in df_feat.columns]
final_risk_table = df_feat[available_display].merge(risk_scores, on="CLIENTNUM", how="left")
final_risk_table["Churn_Actual"] = df_feat["Churn"].values

pred_path = OUTPUTS_DIR / "predictions" / "customer_risk_scores.csv"
final_risk_table.to_csv(pred_path, index=False)
print(f"[OK] Saved customer_risk_scores.csv ({len(final_risk_table):,} customers)")

print("\n[DONE] Training complete. All artifacts saved.")
print(f"   High Risk customers: {(final_risk_table['Risk_Category']=='High Risk').sum():,}")
print(f"   Medium Risk:         {(final_risk_table['Risk_Category']=='Medium Risk').sum():,}")
print(f"   Low Risk:            {(final_risk_table['Risk_Category']=='Low Risk').sum():,}")
