import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.graph_objects as go

# ------------------- Data Cleaning -------------------
def clean_data(X):
    X = X.copy()

    # Replace inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    # Fill missing values with column mean
    X = X.fillna(X.mean())

    return X


# ------------------- KMeans Clustering -------------------
def kmeans_clustering(df, n_clusters=3):
    numeric_cols = df.select_dtypes(include='number').columns

    if len(numeric_cols) < 2:
        raise ValueError("Need at least 2 numeric columns for clustering")

    X = df[numeric_cols]
    X = clean_data(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X)

    df = df.copy()
    df['Cluster'] = clusters

    return df, kmeans


# ------------------- Elbow Method -------------------
def calculate_elbow(X, max_k=10):
    X = clean_data(X)

    inertias = []

    for k in range(1, max_k + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X)
            inertias.append(kmeans.inertia_)
        except:
            inertias.append(None)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(1, max_k + 1)),
            y=inertias,
            mode='lines+markers'
        )
    )

    fig.update_layout(
        title="Elbow Method: Inertia vs Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia"
    )

    return fig


# ------------------- Silhouette Score -------------------
def silhouette_score_kmeans(X, labels):
    X = clean_data(X)

    try:
        score = silhouette_score(X, labels)
        return score
    except:
        return "Not enough data for silhouette score"