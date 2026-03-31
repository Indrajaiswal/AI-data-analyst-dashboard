import numpy as np

def generate_insights(df):
    numeric_cols = df.select_dtypes(include=np.number).columns
    insights = f"Dataset has {df.shape[0]} rows and {df.shape[1]} columns.\n"
    missing = df.isnull().sum().sum()
    insights += f"Total missing values: {missing}\n"
    
    if numeric_cols.any():
        for col in numeric_cols:
            insights += f"- Column '{col}': min={df[col].min():.2f}, max={df[col].max():.2f}, mean={df[col].mean():.2f}\n"
    return insights