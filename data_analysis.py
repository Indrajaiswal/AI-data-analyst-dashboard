import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file):
    """Load CSV or Excel file."""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def clean_data(df):
    """Remove duplicates and fill missing values intelligently."""
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].dtype == "string" or df[col].dtype == object:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            if df[col].isna().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
    return df

def scale_numeric(df):
    """Scale numeric columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df

def get_column_types(df):
    """Detect numeric, categorical, date, and text columns."""
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Use string for pandas 3.x compatibility
    categorical_cols = df.select_dtypes(include="string").columns.tolist()

    date_cols, text_cols = [], []
    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce", format="%Y-%m-%d")
            if parsed.notna().sum() > 0:
                date_cols.append(col)
            elif df[col].dtype == "string":
                # Consider high-cardinality strings as text
                if df[col].nunique() / len(df) > 0.5:
                    text_cols.append(col)
        except Exception:
            continue
    return numeric_cols, categorical_cols, date_cols, text_cols