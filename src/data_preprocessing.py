import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])

    valid_churn_values = {"Yes", "No"}
    unexpected = set(df["Churn"].unique()) - valid_churn_values
    if unexpected:
        raise ValueError(f"Unexpected values in 'Churn' column: {unexpected}")
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    df = df.drop(columns=["customerID"])

    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    X_raw = df.drop(columns=["Churn"])
    y = df["Churn"]

    X = pd.get_dummies(X_raw, drop_first=True)

    return X, y