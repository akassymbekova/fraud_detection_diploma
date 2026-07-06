import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def build_meta_features(predictions_df: pd.DataFrame, target_col: str = "y_true"):
    """
    Build meta-features from base model prediction probabilities.
    """
    X_meta = predictions_df.drop(columns=[target_col])
    y_meta = predictions_df[target_col].astype(int)
    return X_meta, y_meta

def train_logistic_meta_model(X_meta, y_meta):
    """
    Train interpretable Logistic Regression meta-model.
    """
    meta_model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    )
    meta_model.fit(X_meta, y_meta)
    return meta_model

def predict_meta_model(meta_model, predictions_df: pd.DataFrame, target_col: str = "y_true"):
    """
    Predict final fraud probability using meta-model.
    """
    X_meta = predictions_df.drop(columns=[target_col], errors="ignore")
    return meta_model.predict_proba(X_meta)[:, 1]

def get_meta_model_coefficients(meta_model, feature_names):
    """
    Extract coefficients from Logistic Regression inside pipeline.
    """
    logreg = meta_model.named_steps["logisticregression"]
    coef_df = pd.DataFrame({
        "base_model": feature_names,
        "coefficient": logreg.coef_[0],
    }).sort_values("coefficient", ascending=False)
    return coef_df