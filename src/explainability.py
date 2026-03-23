import logging
import shap
import matplotlib.pyplot as plt
import matplotlib.figure
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

def feature_importance(
    model: RandomForestClassifier, X: pd.DataFrame, top_n: int = 10
) -> tuple[pd.Series, matplotlib.figure.Figure]:
    if not hasattr(model, "feature_importances_"):
        raise TypeError(
            f"Model of type {type(model).__name__} does not expose feature_importances_."
        )
    if hasattr(model, "feature_names_in_") and list(X.columns) != list(model.feature_names_in_):
        raise ValueError(
            "X columns do not match the features the model was trained on."
        )

    importances = pd.Series(
        model.feature_importances_, index=X.columns
    ).sort_values(ascending=False)

    logger.info("Top %d important features:\n%s", top_n, importances.head(top_n))

    fig, ax = plt.subplots(figsize=(8, 5))
    importances.head(top_n).plot(kind="bar", ax=ax)
    ax.set_title(f"Top {top_n} Feature Importances")
    fig.tight_layout()

    return importances, fig


def shap_explain(
    model: RandomForestClassifier, X_test: pd.DataFrame
) -> tuple[np.ndarray, matplotlib.figure.Figure]:
    explainer = shap.TreeExplainer(model)
    explanation = explainer(X_test)

    shap_vals = (
        explanation.values[:, :, 1]
        if explanation.values.ndim == 3
        else explanation.values
    )

    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals, X_test, show=False)
    ax = plt.gca()
    fig = ax.get_figure()

    logger.info("SHAP summary plot generated.")

    return shap_vals, fig