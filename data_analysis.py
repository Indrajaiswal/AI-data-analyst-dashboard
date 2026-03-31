import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def load_data(file):
    """Load CSV or Excel"""
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return df

def clean_data(df):
    """Remove duplicates and fill missing values intelligently"""
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    return df

def scale_numeric(df):
    """Scale numeric columns"""
    numeric_cols = df.select_dtypes(include=np.number).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df