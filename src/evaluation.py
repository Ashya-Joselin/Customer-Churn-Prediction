import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)

def evaluate_model(
    model, X_test: pd.DataFrame, y_test: pd.Series, threshold: float = 0.30
) -> dict:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            f"Model of type {type(model).__name__} does not support predict_proba. "
            "A probability-capable model is required for threshold tuning."
        )

    y_pred_default = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred_custom = (y_prob >= threshold).astype(int)

    metrics = {
        "default": {
            "accuracy": accuracy_score(y_test, y_pred_default),
            "confusion_matrix": confusion_matrix(y_test, y_pred_default),
            "classification_report": classification_report(y_test, y_pred_default),
        },
        "custom": {
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, y_pred_custom),
            "confusion_matrix": confusion_matrix(y_test, y_pred_custom),
            "classification_report": classification_report(y_test, y_pred_custom),
            "churn_recall": recall_score(y_test, y_pred_custom),
            "roc_auc": roc_auc_score(y_test, y_prob),
        },
        "y_prob": y_prob,
    }

    logger.info("=== Default Threshold Results ===")
    logger.info("Accuracy: %.4f", metrics["default"]["accuracy"])
    logger.info("Confusion Matrix:\n%s", metrics["default"]["confusion_matrix"])
    logger.info("Classification Report:\n%s", metrics["default"]["classification_report"])

    logger.info("=== Custom Threshold Results (Threshold = %.2f) ===", threshold)
    logger.info("Accuracy: %.4f", metrics["custom"]["accuracy"])
    logger.info("Confusion Matrix:\n%s", metrics["custom"]["confusion_matrix"])
    logger.info("Classification Report:\n%s", metrics["custom"]["classification_report"])
    logger.info("Churn Recall: %.4f", metrics["custom"]["churn_recall"])
    logger.info("ROC-AUC: %.4f", metrics["custom"]["roc_auc"])

    return metrics