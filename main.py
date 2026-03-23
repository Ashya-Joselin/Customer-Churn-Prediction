import logging
import joblib
import os
from src.data_preprocessing import load_and_clean_data, encode_features
from src.model_training import train_model
from src.evaluation import evaluate_model
from src.explainability import feature_importance, shap_explain
from src.utils import split_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

try:
    df = load_and_clean_data("data/Telco-Customer-Churn.csv")
except Exception as e:
    logger.error("Failed to load data: %s", e)
    raise

X, y = encode_features(df)
X_train, X_test, y_train, y_test = split_data(X, y)

model, best_params = train_model(X_train, y_train)
logger.info("Best parameters: %s", best_params)

metrics = evaluate_model(model, X_test, y_test, threshold=0.30)
logger.info("ROC-AUC: %.4f", metrics["custom"]["roc_auc"])
logger.info("Churn Recall: %.4f", metrics["custom"]["churn_recall"])

importances, importance_fig = feature_importance(model, X)
importance_fig.savefig("models/feature_importance.png", bbox_inches="tight")

shap_vals, shap_fig = shap_explain(model, X_test)
shap_fig.savefig("models/shap_summary.png", bbox_inches="tight")

os.makedirs("models", exist_ok=True)

try:
    joblib.dump(model, "models/churn_model.pkl")
    logger.info("Model saved to models/churn_model.pkl")
except Exception as e:
    logger.error("Failed to save model: %s", e)
    raise

logger.info("Pipeline executed successfully.")