# Customer Churn Prediction

## Overview
This project predicts customer churn using machine learning techniques.
The objective is to identify customers who are likely to leave a service and
help businesses take proactive retention actions.

## Dataset
- **Telco Customer Churn Dataset**
- Source: Kaggle
- Contains customer demographics, service usage, and churn information.

## Technologies Used
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn

## Models Implemented
- Logistic Regression
- Random Forest Classifier

## Model Optimization
- Addressed class imbalance in churn data
- Tuned probability thresholds to improve churn detection
- Improved churn recall from **48% to 72%**
- Achieved an ROC–AUC score of **0.81**

## Results
The optimized Random Forest model prioritizes churn recall over raw accuracy,
making it suitable for real-world customer retention scenarios where missing
churn customers is more costly than false positives.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/Customer-Churn-Prediction.git
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Open and run:
   ```bash
   churn_prediction.ipynb
