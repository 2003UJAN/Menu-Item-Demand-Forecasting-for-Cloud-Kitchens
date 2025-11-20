import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

st.set_page_config(page_title="Demand Forecasting App", layout="wide")

st.title("📈 Demand Forecasting Dashboard")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv("../data/synthetic_orders.csv", parse_dates=["order_date"])
    df["revenue"] = df["units_sold"] * df["price_per_unit"]
    return df

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# Aggregate daily demand
daily = df.groupby("order_date")["units_sold"].sum().reset_index()
daily["day_index"] = np.arange(len(daily))

# Train Model
model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(daily[["day_index"]], daily["units_sold"])

# Forecast next N days
st.subheader("Forecast Settings")
future_days = st.slider("Days to Forecast", 7, 90, 30)

last_index = daily["day_index"].iloc[-1]
future_index = np.arange(last_index+1, last_index+future_days+1)

forecast = model.predict(future_index.reshape(-1,1))

forecast_df = pd.DataFrame({
    "date": pd.date_range(daily["order_date"].iloc[-1] + pd.Timedelta(days=1),
                          periods=future_days),
    "forecast_units": forecast
})

st.subheader("Forecasted Demand")
st.line_chart(forecast_df.set_index("date"))

st.write(forecast_df)
