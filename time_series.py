import pandas as pd
from prophet import Prophet
import plotly.express as px

def forecast_time_series(df, date_col, target_col):
    """Forecast numeric column using Prophet"""
    df_ts = df[[date_col, target_col]].rename(columns={date_col:'ds', target_col:'y'})
    df_ts['ds'] = pd.to_datetime(df_ts['ds'], errors='coerce')
    df_ts = df_ts.dropna()
    
    model = Prophet()
    model.fit(df_ts)
    future = model.make_future_dataframe(periods=30)  # forecast next 30 periods
    forecast = model.predict(future)
    
    # Plot interactive forecast
    fig = px.line(forecast, x='ds', y='yhat', title=f"Forecast of {target_col}")
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Upper')
    fig.add_scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Lower')
    
    return forecast, fig