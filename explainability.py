"""
explainability.py
-----------------
SHAP-based global and local explainability for the final model.
Works with tree models (RandomForest, XGBoost) via TreeExplainer
and with linear models via LinearExplainer.
"""

import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline


def get_explainer(pipeline: Pipeline, X_train_raw: pd.DataFrame):
    """
    Build a SHAP explainer appropriate for the final estimator.
    X_train_raw is the raw (pre-preprocessing) training DataFrame.
    Returns (explainer, X_train_transformed).
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_train_transformed = preprocessor.transform(X_train_raw)
    feature_names = _get_feature_names(preprocessor)

    clf_class = type(classifier).__name__
    if clf_class in ("RandomForestClassifier", "XGBClassifier",
                     "GradientBoostingClassifier", "ExtraTreesClassifier"):
        explainer = shap.TreeExplainer(classifier)
    else:
        # Fallback: use KernelExplainer on a background summary
        background = shap.sample(X_train_transformed, 100, random_state=42)
        explainer = shap.KernelExplainer(classifier.predict_proba, background)

    return explainer, X_train_transformed, feature_names


def get_shap_values(explainer, X_transformed: np.ndarray):
    """
    Compute SHAP values for the positive class (churn=1).
    Returns a 2-D array of shape (n_samples, n_features).
    """
    sv = explainer.shap_values(X_transformed)
    # Tree models return list [class0_shap, class1_shap]
    if isinstance(sv, list):
        return sv[1]
    return sv


def global_feature_importance(shap_values: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature across all samples.
    Returns a DataFrame sorted by importance descending.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame(
        {"Feature": feature_names, "Mean_SHAP": mean_abs}
    ).sort_values("Mean_SHAP", ascending=False).reset_index(drop=True)
    return df


def local_explanation(
    explainer,
    pipeline: Pipeline,
    customer_df: pd.DataFrame,
    feature_names: list,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Compute SHAP values for a single customer.
    Returns a DataFrame of top contributing features (positive = increases risk).
    customer_df should contain exactly one row with the raw feature values.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(customer_df)
    sv = explainer.shap_values(X_transformed)

    if isinstance(sv, list):
        shap_vals = sv[1][0]
    else:
        shap_vals = sv[0]

    df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": shap_vals,
        "Feature_Value": X_transformed[0],
    })
    df["Abs_SHAP"] = df["SHAP_Value"].abs()
    df = df.sort_values("Abs_SHAP", ascending=False).head(top_n).reset_index(drop=True)
    df["Direction"] = df["SHAP_Value"].apply(
        lambda v: "↑ Increases Risk" if v > 0 else "↓ Decreases Risk"
    )
    return df[["Feature", "SHAP_Value", "Direction"]]


def _get_feature_names(preprocessor) -> list:
    """Extract feature names after ColumnTransformer."""
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names = transformer.get_feature_names_out(cols)
        else:
            names = cols
        feature_names.extend(names)
    return list(feature_names)
