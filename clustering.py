import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

# ------------------- CLEAN FUNCTION -------------------
def clean_numeric_data(df):
    df = df.copy()

    # Keep only numeric
    df = df.select_dtypes(include=np.number)

    # Replace inf → NaN
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with median
    df = df.fillna(df.median())

    return df


# ------------------- KMeans Clustering -------------------
def kmeans_clustering(df, n_clusters=3):
    clean_df = clean_numeric_data(df)

    if clean_df.shape[1] < 2:
        raise ValueError("Need at least 2 numeric columns")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    clusters = kmeans.fit_predict(clean_df)

    df = df.copy()
    df['Cluster'] = clusters

    return df, kmeans


# ------------------- Elbow Method -------------------
def calculate_elbow(X, max_k=10):
    X = clean_numeric_data(X)

    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(range(1, max_k + 1)),
        y=inertias,
        mode='lines+markers'
    ))

    fig.update_layout(
        title="Elbow Method",
        xaxis_title="Clusters",
        yaxis_title="Inertia"
    )

    return fig


# ------------------- Silhouette Score -------------------
def silhouette_score_kmeans(X, labels):
    X = clean_numeric_data(X)
    return silhouette_score(X, labels)