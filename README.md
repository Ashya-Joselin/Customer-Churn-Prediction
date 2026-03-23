Customer Churn Prediction
Telecom companies lose a lot of money to churn — customers quietly cancelling and moving to a competitor. By the time you notice, it's often too late. This project tries to catch those customers before they leave.
It's an end-to-end ML pipeline built on the Telco Customer Churn dataset: data cleaning, model training, threshold-tuned evaluation, SHAP explainability, and a Streamlit dashboard for real-time predictions.

What's inside
```
├── data/
│   └── Telco-Customer-Churn.csv
├── models/
│   ├── churn_model.pkl
│   ├── feature_importance.png
│   └── shap_summary.png
├── src/
│   ├── data_preprocessing.py   # cleaning + encoding
│   ├── model_training.py       # GridSearchCV + Random Forest
│   ├── evaluation.py           # metrics + threshold tuning
│   ├── explainability.py       # feature importance + SHAP
│   └── utils.py                # train/test split
├── app.py                      # Streamlit dashboard
├── main.py                     # pipeline entry point
└── requirements.txt
```

Getting started
You'll need Python 3.10+.
```bash
git clone https://github.com/your-username/churn-prediction.git
cd churn-prediction

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Running it
Train the model
```bash
python main.py
```
This runs the full pipeline — loads and cleans the data, trains a Random Forest with cross-validated hyperparameter tuning, evaluates at two thresholds, generates explainability plots, and saves everything to `models/`.
Launch the dashboard
```bash
streamlit run app.py
```
Enter customer details and get a churn probability back instantly. Useful for demoing the model without touching any code.

How the model works
A Random Forest classifier with `class_weight="balanced"` to handle the fact that churners are a minority in the data. Hyperparameters are tuned with 5-fold GridSearchCV scored on ROC-AUC.
On the prediction threshold: instead of the default 0.5, predictions are made at 0.30. A missed churner is more costly than a false alarm, so lowering the threshold trades some precision for meaningfully higher churn recall (~48% → ~70–88%). The pipeline reports metrics at both thresholds so you can see the tradeoff directly.

Explainability
Two plots are saved to `models/` after training.
Feature importance shows which features the trees leaned on most. Fast and easy to explain, but biased toward high-cardinality features.
SHAP values show each feature's contribution to individual predictions. More reliable than impurity-based importance and better for explaining specific predictions to stakeholders.

Dataset
Telco Customer Churn — IBM Sample Dataset via Kaggle
~7,000 customers, 21 features, ~26% churn rate.

Limitations
No sklearn Pipeline. Preprocessing runs before the train/test split. Fine here since no encoding steps are stateful, but adding scaling or target encoding would introduce leakage without a proper Pipeline.
Hardcoded paths. File paths are baked into the code. A config file or `argparse` would make this easier across environments.
No tests. Nothing is tested beyond running the pipeline end to end.

What's next
Wrap preprocessing and model into a `sklearn.Pipeline`
Add a FastAPI backend for production inference
Deploy to Streamlit Cloud or Render
Add automated testing and model monitoring

Requirements
```
pandas
scikit-learn
shap
matplotlib
streamlit
joblib
```
