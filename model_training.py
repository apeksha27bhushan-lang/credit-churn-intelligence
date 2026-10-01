"""
model_training.py
-----------------
Builds preprocessing pipeline, trains models, performs hyperparameter
tuning via cross-validation, selects the best threshold, and saves
artifacts for the Streamlit app.
"""

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier

RANDOM_SEED = 42
TEST_SIZE = 0.20
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_preprocessor(categorical_features: list, numerical_features: list):
    """
    Build a ColumnTransformer preprocessing pipeline.
    - StandardScaler for numerical features.
    - OneHotEncoder for categorical features.
    Fitted ONLY on training data to prevent leakage.
    """
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(df: pd.DataFrame, feature_cols: list, target_col: str = "Churn"):
    """
    Stratified train/test split with fixed seed.
    Returns X_train, X_test, y_train, y_test.
    """
    X = df[feature_cols]
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )
    return X_train, X_test, y_train, y_test


def build_model_pipelines(preprocessor):
    """
    Return a dict of sklearn pipelines for each model.
    class_weight='balanced' handles class imbalance without modifying
    the test distribution.
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_SEED,
                C=0.1,
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                max_depth=10,
                min_samples_leaf=5,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=5,   # ~8500/1627 ≈ 5.2
                eval_metric="logloss",
                random_state=RANDOM_SEED,
                n_jobs=-1,
                verbosity=0,
            )),
        ]),
    }
    return pipelines


def evaluate_pipeline_cv(pipeline, X_train, y_train, cv=5):
    """
    Evaluate a pipeline using stratified cross-validation on training data.
    Returns ROC-AUC and PR-AUC from out-of-fold predictions.
    """
    cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_SEED)
    oof_probs = cross_val_predict(
        pipeline, X_train, y_train,
        cv=cv_splitter,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]
    roc = roc_auc_score(y_train, oof_probs)
    precision, recall, _ = precision_recall_curve(y_train, oof_probs)
    pr = auc(recall, precision)
    return roc, pr


def find_optimal_threshold(pipeline, X_val, y_val):
    """
    Find the threshold that maximises F1 on the validation set.
    Uses only X_val / y_val — never the test set.
    """
    probs = pipeline.predict_proba(X_val)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_val, probs)
    # F1 = 2 * P * R / (P + R); avoid divide-by-zero
    f1_scores = np.where(
        (precision + recall) > 0,
        2 * precision * recall / (precision + recall),
        0.0,
    )
    # thresholds has one fewer element than precision/recall
    best_idx = np.argmax(f1_scores[:-1])
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])
    return best_threshold, best_f1


def save_artifacts(
    pipeline,
    threshold: float,
    categorical_features: list,
    numerical_features: list,
    model_name: str,
    metrics: dict,
):
    """Save model pipeline, threshold, and metadata to disk."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(pipeline, MODELS_DIR / "final_model.joblib")

    metadata = {
        "model_name": model_name,
        "threshold": threshold,
        "categorical_features": categorical_features,
        "numerical_features": numerical_features,
        "all_features": numerical_features + categorical_features,
        "target_col": "Churn",
        "target_mapping": {"Existing Customer": 0, "Attrited Customer": 1},
        "random_seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "metrics": metrics,
    }
    with open(MODELS_DIR / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("[OK] Saved final_model.joblib")
    print("[OK] Saved model_metadata.json")
    return MODELS_DIR / "final_model.joblib", MODELS_DIR / "model_metadata.json"


def load_artifacts():
    """Load saved model pipeline and metadata."""
    model_path = MODELS_DIR / "final_model.joblib"
    meta_path = MODELS_DIR / "model_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata not found: {meta_path}")

    pipeline = joblib.load(model_path)
    with open(meta_path) as f:
        metadata = json.load(f)

    return pipeline, metadata
