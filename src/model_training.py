import logging
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)

def train_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[RandomForestClassifier, dict]:
    if len(X_train) != len(y_train):
        raise ValueError(
            f"X_train and y_train must have the same number of rows, "
            f"got {len(X_train)} and {len(y_train)}."
        )

    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid = GridSearchCV(
        estimator=RandomForestClassifier(class_weight="balanced", random_state=42),
        param_grid=param_grid,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)

    logger.info("Best parameters: %s", grid.best_params_)
    logger.info("Best CV ROC-AUC: %.4f", grid.best_score_)

    return grid.best_estimator_, grid.best_params_