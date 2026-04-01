import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
from data_analysis import load_data, clean_data, scale_numeric, get_column_types
from visualization import (
    plot_histogram, plot_correlation, plot_scatter,
    plot_top_sales, plot_actual_vs_predicted
)
from clustering import kmeans_clustering, calculate_elbow, silhouette_score_kmeans
from ml_models import linear_regression
from classification import classify_churn
from insight import generate_insights
import numpy as np

# Optional libraries
try:
    from wordcloud import WordCloud
    wordcloud_available = True
except ModuleNotFoundError:
    wordcloud_available = False

try:
    from textblob import TextBlob
    textblob_available = True
except ModuleNotFoundError:
    textblob_available = False

try:
    from time_series import forecast_time_series
    prophet_available = True
except ModuleNotFoundError:
    prophet_available = False

# ------------------- Page Config -------------------
st.set_page_config(page_title="Enterprise AI Dashboard", page_icon="📊", layout="wide")

# ------------------- CSS -------------------
st.markdown("""
<style>
/* App background */
[data-testid="stAppViewContainer"] {
background: linear-gradient(#a29bfe, #6c5ce7)
            }
/* Sidebar */
[data-testid="stSidebar"] { background: #5D3FD3; }
[data-testid="stSidebar"] * { color: white !important; }
/* KPI Cards */
[data-testid="metric-container"] { background: white; border-radius: 12px; padding: 15px;
box-shadow: 0px 4px 15px rgba(0,0,0,0.1); border-left: 5px solid #3b82f6; color: black !important;}
/* Buttons */
.stButton>button { background: linear-gradient(90deg, #3b82f6, #6366f1); color: white; border-radius: 8px; border: none; font-weight: 600; }
.stButton>button:hover { background: linear-gradient(90deg, #2563eb, #4f46e5); }
.stDownloadButton>button { background: linear-gradient(90deg, #10b981, #059669); color: white; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ------------------- Cached Data Loading -------------------
@st.cache_data
def load_clean_scale(file):
    df = load_data(file)
    df = clean_data(df)
    df = scale_numeric(df)
    return df

# ------------------- File Upload -------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel dataset", type=["csv","xlsx"])
df = None
numeric_cols, categorical_cols, date_cols, text_cols = [], [], [], []

if uploaded_file:
    df = load_clean_scale(uploaded_file)
    numeric_cols, categorical_cols, date_cols, text_cols = get_column_types(df)

    # Limit dataset for speed on Streamlit Cloud
    if len(df) > 5000:
        df = df.sample(5000, random_state=42)

# ------------------- Dynamic Pages -------------------
pages = ["Dataset Overview"]
if numeric_cols:
    pages += ["Visualizations", "Clustering", "Regression Predictions"]
if categorical_cols and len(text_cols) < len(categorical_cols):
    pages += ["Churn / Classification"]
if prophet_available and date_cols and numeric_cols:
    pages += ["Time-Series Forecast"]
if wordcloud_available and text_cols:
    pages += ["Text / NLP Analysis"]

selected_page = st.sidebar.radio("Go to", pages)

# ------------------- Filters -------------------
if df is not None and categorical_cols:
    st.sidebar.subheader("Filter Dataset")
    for col in categorical_cols:
        options = df[col].unique().tolist()
        selected = st.sidebar.multiselect(f"{col} filter", options, default=options)
        df = df[df[col].isin(selected)]

# ------------------- Helper for Download -------------------
def download_plotly_chart(fig, filename="chart.png"):
    try:
        import kaleido
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        st.download_button(label="📥 Download Chart", data=buf.getvalue(), file_name=filename, mime="image/png")
    except Exception:
        st.warning("Install `kaleido` to enable chart downloads.")

# ------------------- Pages -------------------

# Dataset Overview
if selected_page == "Dataset Overview":
    st.header("📄 Dataset Overview")
    if df is not None:
        st.dataframe(df.head())
        st.subheader("🤖 AI Insights")
        st.text(generate_insights(df))
    else:
        st.info("👈 Upload a dataset to see overview and insights.")

# KPI Cards
if df is not None and numeric_cols:
    st.subheader("📊 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Records", df.shape[0])
    col2.metric("Numeric Columns", len(numeric_cols))
    col3.metric("Categorical Columns", len(categorical_cols))
    col4.metric("Date Columns", len(date_cols))

# ------------------- Visualizations -------------------
if selected_page == "Visualizations" and numeric_cols:
    st.header("📈 Visualizations")
    plot_df = df.sample(min(len(df), 2000), random_state=42)
    if len(numeric_cols) >= 2:
        st.subheader("Correlation Heatmap")
        fig_corr = plot_correlation(plot_df)
        st.plotly_chart(fig_corr)
        download_plotly_chart(fig_corr, "correlation_heatmap.png")

    st.subheader("Histograms")
    for col in numeric_cols:
        fig_hist = plot_histogram(plot_df, col)
        st.plotly_chart(fig_hist)

    if len(numeric_cols) >= 2:
        st.subheader("Scatter Plot")
        x_col = st.selectbox("X-axis", numeric_cols)
        y_col = st.selectbox("Y-axis", numeric_cols)
        color_col = None
        if categorical_cols:
            color_col = st.selectbox("Color by (optional)", [None] + categorical_cols)
        fig_scatter = plot_scatter(plot_df, x_col, y_col, color_col)
        st.plotly_chart(fig_scatter)
        download_plotly_chart(fig_scatter, "scatter_plot.png")

# ------------------- Clustering -------------------
if selected_page == "Clustering" and len(numeric_cols) >= 2:
    st.header("📌 KMeans Clustering")
    n_clusters = st.slider("Number of clusters", 2, 10, 3)
    st.subheader("Elbow Method")
    st.plotly_chart(calculate_elbow(df[numeric_cols], max_k=10))

    clustered_df, kmeans = kmeans_clustering(df, n_clusters)
    st.dataframe(clustered_df.head())
    score = silhouette_score_kmeans(clustered_df[numeric_cols], clustered_df['Cluster'])
    st.info(f"Silhouette Score: {score:.4f}")
    fig_clusters = plot_scatter(clustered_df, numeric_cols[0], numeric_cols[1], color_col='Cluster')
    st.plotly_chart(fig_clusters)

# ------------------- Regression -------------------
if selected_page == "Regression Predictions" and numeric_cols and len(numeric_cols) >= 2:
    st.header("🤖 Regression Predictions")
    target = st.selectbox("Target Column", numeric_cols)
    y_test, y_pred, metrics = linear_regression(df, target)
    st.write("Metrics:", metrics)
    pred_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    st.dataframe(pred_df.head())
    fig_pred = plot_actual_vs_predicted(y_test, y_pred)
    st.plotly_chart(fig_pred)
    download_plotly_chart(fig_pred, "regression_predictions.png")

# ------------------- Classification -------------------
if selected_page == "Churn / Classification" and categorical_cols:
    st.header("🤖 Churn / Classification")
    target_col = st.selectbox("Target Column", categorical_cols)
    y_test, y_pred, metrics, feat_imp = classify_churn(df, target_col)
    st.write("Metrics:", metrics)
    st.bar_chart(feat_imp)
    st.subheader("Predictions Preview")
    st.dataframe(pd.DataFrame({"Actual": y_test, "Predicted": y_pred}).head())

# ------------------- Time-Series Forecast -------------------
if selected_page == "Time-Series Forecast" and prophet_available:
    st.header("📈 Time-Series Forecast")
    if not date_cols:
        st.info("No date column detected for forecast.")
    else:
        date_col = st.selectbox("Date Column", date_cols)
        target_col = st.selectbox("Numeric Column to Forecast", numeric_cols)
        try:
            forecast_df, fig_forecast = forecast_time_series(df, date_col, target_col)
            st.plotly_chart(fig_forecast)
            st.subheader("Forecast Preview")
            st.dataframe(forecast_df.head())
            download_plotly_chart(fig_forecast, "time_series_forecast.png")
        except ValueError as e:
            st.error(str(e))

# ------------------- Text / NLP Analysis -------------------
if selected_page == "Text / NLP Analysis" and wordcloud_available:
    st.header("📰 Text / NLP Analysis")
    text_col = st.selectbox("Text Column", text_cols)
    st.subheader("Text Sample")
    st.dataframe(df[text_col].head())

    # WordCloud
    text_data = " ".join(df[text_col].astype(str))
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

    # Sentiment Analysis
    if textblob_available:
        st.subheader("Sentiment Analysis")
        df['Sentiment'] = df[text_col].astype(str).apply(lambda x: TextBlob(x).sentiment.polarity)
        df['Sentiment_Label'] = df['Sentiment'].apply(lambda p: "Positive" if p>0.1 else ("Negative" if p<-0.1 else "Neutral"))
        st.bar_chart(df['Sentiment_Label'].value_counts())