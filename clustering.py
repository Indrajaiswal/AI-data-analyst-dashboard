import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go

# ------------------- KMeans Clustering -------------------
def kmeans_clustering(df, n_clusters=3):
    numeric_cols = df.select_dtypes(include='number').columns
    X = df[numeric_cols]
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X)
    df['Cluster'] = clusters
    return df, kmeans

# ------------------- Elbow Method -------------------
def calculate_elbow(X, max_k=10):
    inertias = []
    for k in range(1, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(1, max_k+1)), y=inertias, mode='lines+markers'))
    fig.update_layout(
        title="Elbow Method: Inertia vs Number of Clusters",
        xaxis_title="Number of Clusters (k)",
        yaxis_title="Inertia"
    )
    return fig

# ------------------- Silhouette Score -------------------
def silhouette_score_kmeans(X, labels):
    score = silhouette_score(X, labels)
    return score