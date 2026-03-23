import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if len(X) != len(y):
        raise ValueError(
            f"X and y must have the same number of rows, got {len(X)} and {len(y)}."
        )

    return train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )