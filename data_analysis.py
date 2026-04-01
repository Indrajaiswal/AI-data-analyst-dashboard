import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


# ------------------ LOAD DATA ------------------
def load_data(file):
    """Load CSV or Excel file safely"""
    try:
        if file.name.endswith(".csv"):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)
        return df
    except Exception as e:
        raise ValueError(f"Error loading file: {e}")


# ------------------ CLEAN DATA ------------------
def clean_data(df):
    """Clean dataset: remove duplicates, fix types, handle missing values"""

    df = df.copy()

    # Clean column names
    df.columns = df.columns.str.strip()

    # Remove duplicates
    df = df.drop_duplicates()

    for col in df.columns:

        # ✅ SAFE numeric conversion (fix for pandas error)
        try:
            df[col] = pd.to_numeric(df[col])
        except:
            pass

        # Handle missing values
        if df[col].dtype == 'object':
            # Categorical/Text
            if df[col].isnull().sum() > 0:
                if not df[col].mode().empty:
                    df[col] = df[col].fillna(df[col].mode()[0])
                else:
                    df[col] = df[col].fillna("Unknown")
        else:
            # Numeric
            if df[col].isnull().sum() > 0:
                if df[col].notnull().sum() > 0:
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(0)

    return df


# ------------------ SCALE NUMERIC ------------------
def scale_numeric(df):
    """Scale numeric columns using StandardScaler"""

    df = df.copy()

    numeric_cols = df.select_dtypes(include=np.number).columns

    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    return df