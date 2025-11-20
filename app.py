import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Demand Forecasting ML App", layout="wide")

st.title("📈 ML-Based Demand Forecasting")
st.write("Upload your CSV or manually enter data to train a forecasting model.")

# =============================
# OPTION 1 — FILE UPLOAD
# =============================
st.header("Upload Dataset")
uploaded = st.file_uploader("Upload CSV with columns: order_date, units_sold", type=["csv"])

# =============================
# OPTION 2 — MANUAL ENTRY
# =============================
st.header("Manual Data Entry")
manual_data = st.checkbox("Enter data manually instead of uploading file")

if manual_data:
    st.write("Enter your sales records:")
    records = st.number_input("How many rows do you want to enter?", 1, 50, 5)

    rows = []
    for i in range(records):
        st.write(f"Row {i+1}")
        date = st.date_input(f"Date {i+1}")
        units = st.number_input(f"Units Sold {i+1}", 0, 100000)
        rows.append([date, units])

    df = pd.DataFrame(rows, columns=["order_date", "units_sold"])

elif uploaded:
    df = pd.read_csv(uploaded)
    df["order_date"] = pd.to_datetime(df["order_date"])

else:
    st.stop()

# =============================
# DATA PREPROCESSING
# =============================
df = df.sort_values("order_date")
df["day_index"] = np.arange(len(df))

st.subheader("Preview Data")
st.dataframe(df.head())

# =============================
# MODEL TRAINING
# =============================
X = df[["day_index"]]
y = df["units_sold"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

st.subheader("Model Performance")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# =============================
# FORECAST FUTURE DEMAND
# =============================
st.header("Forecast Future Demand")
future_days = st.slider("How many days to forecast?", 7, 120, 30)

last_index = df["day_index"].iloc[-1]
future_index = np.arange(last_index + 1, last_index + future_days + 1)
forecast = model.predict(future_index.reshape(-1, 1))

forecast_df = pd.DataFrame({
    "date": pd.date_range(df["order_date"].iloc[-1], periods=future_days+1, closed="right"),
    "forecast_units": forecast
})

st.subheader("Forecast Results")
st.line_chart(forecast_df.set_index("date"))
st.dataframe(forecast_df)
