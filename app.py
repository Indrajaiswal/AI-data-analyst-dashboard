import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import numpy as np

# ------------------- Optional Libraries -------------------
try:
    from wordcloud import WordCloud
    wordcloud_available = True
except:
    wordcloud_available = False

try:
    from textblob import TextBlob
    textblob_available = True
except:
    textblob_available = False

try:
    from time_series import forecast_time_series
    prophet_available = True
except:
    prophet_available = False

# ------------------- Custom Modules -------------------
from data_analysis import load_data, clean_data
from visualization import (
    plot_histogram, plot_correlation, plot_scatter,
    plot_top_sales, plot_actual_vs_predicted
)
from clustering import kmeans_clustering, calculate_elbow, silhouette_score_kmeans
from ml_models import linear_regression
from classification import classify_churn
from insight import generate_insights

# ------------------- Page Config -------------------
st.set_page_config(page_title="Enterprise AI Dashboard", page_icon="📊", layout="wide")

# ------------------- YOUR CSS (UNCHANGED) -------------------
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
background: linear-gradient(#a29bfe, #6c5ce7)
}
[data-testid="stSidebar"] { background: #5D3FD3; }
[data-testid="stSidebar"] * { color: white !important; }
h1, h2, h3, h4, h5, h6 {
color: #1e3a8a !important;
font-weight: 700;
}
[data-testid="metric-container"] {
background: white;
border-radius: 12px;
padding: 15px;
box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
border-left: 5px solid #3b82f6;
color: black !important;
}
.stText, .stMarkdown { color: black !important; }
.stButton>button {
background: linear-gradient(90deg, #3b82f6, #6366f1);
color: white;
border-radius: 8px;
border: none;
font-weight: 600;
}
.stButton>button:hover {
background: linear-gradient(90deg, #2563eb, #4f46e5);
}
.stDownloadButton>button {
background: linear-gradient(90deg, #10b981, #059669);
color: white;
border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# ------------------- Banner -------------------
st.markdown("""
<div style="background: linear-gradient(90deg,#3b82f6,#6366f1);
padding:15px;border-radius:10px;color:white;
font-size:20px;font-weight:600">
🚀 AI Data Analyst Dashboard | Auto Data Analysis + ML Insights
</div>
""", unsafe_allow_html=True)

st.title("📊 Smart Data Dashboard")

# ------------------- Cached Loader -------------------
@st.cache_data
def load_dataset(file):
    df = load_data(file)
    df = clean_data(df)
    return df

# ------------------- Upload -------------------
uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel dataset", type=["csv","xlsx"])

df = None
numeric_cols, categorical_cols, date_cols, text_cols = [], [], [], []

if uploaded_file:
    try:
        df = load_dataset(uploaded_file)

        # 🔥 PERFORMANCE FIX
        if len(df) > 3000:
            df = df.sample(3000, random_state=42)

        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        categorical_cols = df.select_dtypes(include='object').columns.tolist()

        # Detect date/text
        for col in categorical_cols:
            try:
                pd.to_datetime(df[col])
                date_cols.append(col)
            except:
                if df[col].nunique() / len(df) > 0.5:
                    text_cols.append(col)

    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# ------------------- Pages -------------------
pages = ["Dataset Overview"]

if numeric_cols:
    pages += ["Visualizations", "Clustering", "Regression Predictions"]

if categorical_cols:
    pages += ["Churn / Classification"]

if prophet_available and date_cols:
    pages += ["Time-Series Forecast"]

if wordcloud_available and text_cols:
    pages += ["Text / NLP Analysis"]

selected_page = st.sidebar.radio("Go to", pages)

# ------------------- Filters -------------------
if df is not None and categorical_cols:
    st.sidebar.subheader("Filter Dataset")
    for col in categorical_cols:
        options = df[col].dropna().unique().tolist()
        selected = st.sidebar.multiselect(col, options, default=options)
        df = df[df[col].isin(selected)]

# ------------------- Download -------------------
def download_plotly_chart(fig, filename="chart.png"):
    try:
        import kaleido
        buf = io.BytesIO()
        fig.write_image(buf, format="png")
        st.download_button("📥 Download Chart", buf.getvalue(), filename)
    except:
        st.warning("Install kaleido to enable downloads")

# ------------------- Overview -------------------
# Dataset Overview
if selected_page == "Dataset Overview":
    st.header("📄 Dataset Overview")
    if df is not None:
        st.dataframe(df.head())
        st.subheader("🤖 AI Insights")
        st.text(generate_insights(df))
    else:
        st.markdown("""
            <div style="
                background-color: #f0f0f0;  /* light gray background */
                color: black;               /* text color black */
                padding: 10px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 16px;
            ">
            👈 Upload a dataset to see overview and insights.
            </div>
            """, unsafe_allow_html=True)

# ------------------- KPI -------------------
if df is not None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Numeric", len(numeric_cols))
    c3.metric("Categorical", len(categorical_cols))
    c4.metric("Date", len(date_cols))

# ------------------- Visualizations -------------------
if selected_page == "Visualizations" and numeric_cols:
    st.header("📈 Visualizations")

    try:
        plot_df = df.sample(min(len(df), 1500))

        # Correlation
        st.subheader("Correlation Heatmap")
        fig_corr = plot_correlation(plot_df)
        st.plotly_chart(fig_corr)
        download_plotly_chart(fig_corr, "correlation_heatmap.png")

        # Top N
        st.subheader("Top N Values")
        target_col = st.selectbox("Top N Column", numeric_cols)
        top_n = st.slider("Top N", 5, 50, 10)

        fig_top = plot_top_sales(plot_df, target_col, top_n)
        st.plotly_chart(fig_top)
        download_plotly_chart(fig_top, "top_values.png")

        # Histograms
        st.subheader("Histograms")
        for col in numeric_cols[:5]:
            fig_hist = plot_histogram(plot_df, col)
            st.plotly_chart(fig_hist)
            download_plotly_chart(fig_hist, f"{col}_hist.png")

        # Scatter
        if len(numeric_cols) >= 2:
            st.subheader("Scatter Plot")

            x_col = st.selectbox("X-axis", numeric_cols)
            y_col = st.selectbox("Y-axis", numeric_cols)

            color_col = None
            if categorical_cols:
                color_col = st.selectbox("Color by", [None] + categorical_cols)

            fig_scatter = plot_scatter(plot_df, x_col, y_col, color_col)
            st.plotly_chart(fig_scatter)
            download_plotly_chart(fig_scatter, "scatter_plot.png")

    except Exception as e:
        st.error(f"Visualization Error: {e}")

# ------------------- Clustering -------------------
if selected_page == "Clustering" and len(numeric_cols) >= 2:
    st.header("📌 Clustering")
    try:
        n_clusters = st.slider("Clusters", 2, 10, 3)

        st.plotly_chart(calculate_elbow(df[numeric_cols]))

        clustered_df, _ = kmeans_clustering(df, n_clusters)

        score = silhouette_score_kmeans(
            clustered_df[numeric_cols],
            clustered_df['Cluster']
        )

        st.success(f"Silhouette Score: {score:.3f}")

        st.plotly_chart(
            plot_scatter(clustered_df, numeric_cols[0], numeric_cols[1], 'Cluster')
        )

    except Exception as e:
        st.error(f"Clustering Error: {e}")

# ------------------- Regression -------------------
if selected_page == "Regression Predictions" and len(numeric_cols) >= 2:
    st.header("🤖 Regression")
    try:
        target = st.selectbox("Target", numeric_cols)

        y_test, y_pred, metrics = linear_regression(df, target)

        st.write(metrics)
        st.plotly_chart(plot_actual_vs_predicted(y_test, y_pred))

    except Exception as e:
        st.error(f"Regression Error: {e}")

# ------------------- Classification -------------------
if selected_page == "Churn / Classification":
    st.header("🤖 Classification")
    try:
        target = st.selectbox("Target", categorical_cols)

        y_test, y_pred, metrics, feat_imp = classify_churn(df, target)

        st.write(metrics)
        st.bar_chart(feat_imp)

    except Exception as e:
        st.error(f"Classification Error: {e}")

# ------------------- Forecast -------------------
if selected_page == "Time-Series Forecast" and prophet_available:
    st.header("📈 Forecast")
    try:
        date_col = st.selectbox("Date", date_cols)
        target = st.selectbox("Target", numeric_cols)

        forecast_df, fig = forecast_time_series(df, date_col, target)

        st.plotly_chart(fig)
        st.dataframe(forecast_df.head())

    except Exception as e:
        st.error(f"Forecast Error: {e}")

# ------------------- NLP -------------------
if selected_page == "Text / NLP Analysis" and wordcloud_available:
    st.header("🧠 NLP Analysis")

    text_col = st.selectbox("Text Column", text_cols)

    text_data = " ".join(df[text_col].astype(str))

    wc = WordCloud(width=800, height=400, background_color='white').generate(text_data)

    fig, ax = plt.subplots()
    ax.imshow(wc)
    ax.axis("off")
    st.pyplot(fig)

    if textblob_available:
        df['Sentiment'] = df[text_col].astype(str).apply(
            lambda x: TextBlob(x).sentiment.polarity
        )
        st.bar_chart(df['Sentiment'])