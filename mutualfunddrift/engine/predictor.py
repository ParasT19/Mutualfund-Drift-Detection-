"""
MutualFundDrift — XGBoost predictor module.
Trains the drift prediction classifier, loads saved models, generates predictions,
and explains predictions with SHAP values.
"""

import json
import logging
import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from backend.config import settings
from engine.feature_engineer import FEATURE_NAMES

logger = logging.getLogger(__name__)

MODEL_VERSION = "1.0.0"


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
) -> Tuple[XGBClassifier, float]:
    """
    Train an XGBoost classifier using time-series cross-validation.

    Uses TimeSeriesSplit to avoid data leakage, then retrains on the full dataset
    and saves the model artifact to the configured path.

    Args:
        X: Feature DataFrame with columns matching FEATURE_NAMES.
        y: Binary label Series (1 = will drift, 0 = won't drift).

    Returns:
        Tuple of (fitted_model, mean_auc_across_folds).
    """
    if X.empty or y.empty:
        raise ValueError("Cannot train on empty dataset.")

    # Handle class imbalance
    n_neg = int((y == 0).sum())
    n_pos = int((y == 1).sum())
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    model = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
        use_label_encoder=False,
        verbosity=0,
    )

    # Time-series cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    auc_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

        if len(y_val.unique()) > 1:
            proba = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, proba)
            auc_scores.append(auc)
            logger.info("Fold %d AUC: %.4f", fold + 1, auc)
        else:
            logger.warning("Fold %d has only one class; skipping AUC.", fold + 1)

    mean_auc = float(np.mean(auc_scores)) if auc_scores else 0.0
    logger.info("Mean AUC across %d folds: %.4f", len(auc_scores), mean_auc)

    # Final training on full dataset
    model.fit(X, y, verbose=False)

    # Save model
    os.makedirs(os.path.dirname(settings.model_save_path), exist_ok=True)
    joblib.dump(model, settings.model_save_path)
    logger.info("Model saved to %s (version %s)", settings.model_save_path, MODEL_VERSION)

    return model, mean_auc


def load_model() -> XGBClassifier:
    """
    Load the saved XGBoost model from the configured file path.

    Returns:
        The loaded XGBClassifier model object.

    Raises:
        FileNotFoundError: If the model file does not exist, with a clear message
                           instructing the user to train the model first.
    """
    if not os.path.exists(settings.model_save_path):
        raise FileNotFoundError(
            f"Model not found at '{settings.model_save_path}'. "
            "Run POST /api/drift/train to train the model first."
        )
    model = joblib.load(settings.model_save_path)
    logger.info("Model loaded from %s", settings.model_save_path)
    return model


def predict_drift(feature_vector: pd.DataFrame) -> dict:
    """
    Run the XGBoost model on a feature vector and return prediction with SHAP explanation.

    Loads the saved model, computes drift probability and binary prediction,
    then uses SHAP TreeExplainer to identify the top 5 most influential features.

    Args:
        feature_vector: Single-row DataFrame with 19 feature columns (from build_feature_vector).

    Returns:
        Dict with keys:
          - drift_probability: float (probability of significant drift)
          - will_drift: bool (True if probability > 0.5)
          - top_shap_features: dict of {feature_name: shap_value} (top 5 by |shap|)
    """
    model = load_model()

    # Ensure correct column order
    fv = feature_vector[FEATURE_NAMES]

    drift_probability = float(model.predict_proba(fv)[0][1])
    will_drift = drift_probability > 0.5

    # SHAP explanation
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(fv)

        # shap_values shape: (n_samples, n_features) or list for multi-class
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1 SHAP values
        else:
            sv = shap_values[0]

        shap_dict = {FEATURE_NAMES[i]: float(sv[i]) for i in range(len(FEATURE_NAMES))}
        top_shap = dict(
            sorted(shap_dict.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
        )
    except Exception as exc:
        logger.warning("SHAP computation failed: %s. Returning empty SHAP dict.", exc)
        top_shap = {}

    return {
        "drift_probability": drift_probability,
        "will_drift": will_drift,
        "top_shap_features": top_shap,
    }


def evaluate_model(X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the saved model on held-out test data and return key performance metrics.

    Args:
        X_test: Test feature DataFrame.
        y_test: Ground truth binary labels.

    Returns:
        Dict with keys: auc, precision, recall, f1, confusion_matrix (as list).
    """
    model = load_model()

    fv = X_test[FEATURE_NAMES]
    proba = model.predict_proba(fv)[:, 1]
    preds = (proba > 0.5).astype(int)

    auc = roc_auc_score(y_test, proba) if len(y_test.unique()) > 1 else 0.0
    precision = precision_score(y_test, preds, zero_division=0)
    recall = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    cm = confusion_matrix(y_test, preds).tolist()

    return {
        "auc": float(auc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }
