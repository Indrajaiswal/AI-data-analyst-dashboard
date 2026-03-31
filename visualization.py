import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd

# ---------- Basic Plots ----------

def plot_histogram(df, column):
    """Plot histogram of a numeric column"""
    fig = px.histogram(df, x=column, nbins=20, title=f"Histogram of {column}")
    return fig

def plot_correlation(df):
    """Plot correlation heatmap for numeric columns"""
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        return None
    fig = px.imshow(numeric_df.corr(), text_auto=True, color_continuous_scale='RdBu_r', title="Correlation Heatmap")
    return fig

def plot_scatter(df, x_col, y_col, color_col=None):
    """Scatter plot between two columns with optional color"""
    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"{x_col} vs {y_col}")
    return fig

# ---------- Professional Dashboard Plots ----------

def plot_top_sales(df, target_col='yhat', top_n=10):
    """
    Bar chart of top N values from a numeric column.
    Works for any dataset, not just sales forecasts.
    """
    if target_col not in df.columns:
        return None
    top_df = df.nlargest(top_n, target_col)
    fig = px.bar(top_df, x=top_df.index, y=target_col,
                 text=target_col, labels={'x':'Index','y':target_col},
                 title=f"Top {top_n} {target_col} Values")
    return fig

def plot_cluster_distribution(df, target_col='yhat'):
    """Box plot showing distribution of numeric column per cluster"""
    if 'Cluster' not in df.columns or target_col not in df.columns:
        return None
    fig = px.box(df, x='Cluster', y=target_col, color='Cluster',
                 title=f"{target_col} Distribution per Cluster")
    return fig

def plot_actual_vs_predicted(y_true, y_pred, top_n=None, clusters=None):
    """
    Scatter plot for regression datasets:
    - y_true vs y_pred
    - Optionally highlight top N predicted values
    - Optionally color by cluster
    """
    df = pd.DataFrame({'Actual': y_true, 'Predicted': y_pred})
    df['Index'] = df.index

    # Highlight top N predicted values
    if top_n:
        df['Top'] = False
        top_indices = df.nlargest(top_n, 'Predicted').index
        df.loc[top_indices, 'Top'] = True
    else:
        df['Top'] = False

    # Color by cluster if provided
    if clusters is not None and len(clusters) == len(df):
        df['Cluster'] = clusters
        color_col = 'Cluster'
    else:
        color_col = 'Top'

    fig = px.scatter(df, x='Actual', y='Predicted', color=color_col,
                     hover_data=['Index'],
                     title="Actual vs Predicted Values Scatter Plot")
    return fig