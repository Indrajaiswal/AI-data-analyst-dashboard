import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# ------------------- Data Cleaning -------------------
def clean_data(df):
    df = df.copy()

    # Replace infinite values with NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill missing values with mean
    df = df.fillna(df.mean())

    return df


# ------------------- Linear Regression -------------------
def linear_regression(df, target_col):
    numeric_df = df.select_dtypes(include=np.number)

    if target_col not in numeric_df.columns:
        raise ValueError("Target column must be numeric")

    # Clean data
    numeric_df = clean_data(numeric_df)

    X = numeric_df.drop(columns=[target_col])
    y = numeric_df[target_col]

    # Safety checks
    if X.shape[1] == 0:
        raise ValueError("No features available for training")

    if len(X) < 5:
        raise ValueError("Not enough data for regression")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prediction
    y_pred = model.predict(X_test)

    # Metrics
    metrics = {
        "MSE": float(mean_squared_error(y_test, y_pred)),
        "R2 Score": float(r2_score(y_test, y_pred))
    }

    return y_test, y_pred, metrics