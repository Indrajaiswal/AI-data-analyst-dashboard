import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------- CLEAN FUNCTION -------------------
def clean_numeric_data(df):
    df = df.copy()

    df = df.select_dtypes(include=np.number)

    df = df.replace([np.inf, -np.inf], np.nan)

    df = df.fillna(df.median())

    return df


# ------------------- Linear Regression -------------------
def linear_regression(df, target_col):
    clean_df = clean_numeric_data(df)

    if target_col not in clean_df.columns:
        raise ValueError("Target must be numeric")

    X = clean_df.drop(columns=[target_col])
    y = clean_df[target_col]

    if X.shape[1] == 0:
        raise ValueError("No valid features")

    if len(X) < 5:
        raise ValueError("Dataset too small")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "R2 Score": float(r2_score(y_test, y_pred))
    }

    return y_test, y_pred, metrics