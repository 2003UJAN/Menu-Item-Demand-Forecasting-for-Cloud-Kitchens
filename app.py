import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cloud Kitchen Demand Forecasting", layout="wide")

st.title("🍽️ Cloud Kitchen Demand Forecasting (ML App)")
st.write("Upload your dataset and forecast future demand based on city, kitchen, and item.")

# -----------------------------------------------------------
# 1. FILE UPLOAD & VALIDATION
# -----------------------------------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload a dataset to continue.")
    st.stop()

required_cols = [
    "date","city","kitchen_id","item_name","category",
    "price","orders","weekday","promo_flag","temperature"
]

missing_cols = [c for c in required_cols if c not in df.columns]
if missing_cols:
    st.error(f"❌ Missing required columns: {missing_cols}")
    st.stop()

df["date"] = pd.to_datetime(df["date"])

# -----------------------------------------------------------
# 2. FILTERS (CITY → KITCHEN → ITEM)
# -----------------------------------------------------------
st.subheader("🔎 Select Filters")

city = st.selectbox("Select City", sorted(df["city"].unique()))
kitchen = st.selectbox("Select Kitchen", sorted(df["kitchen_id"].unique()))
item = st.selectbox("Select Menu Item", sorted(df["item_name"].unique()))

filtered = df[
    (df["city"] == city) &
    (df["kitchen_id"] == kitchen) &
    (df["item_name"] == item)
].copy()

if filtered.empty:
    st.error("No matching records found. Try different filters.")
    st.stop()

st.write("### 📌 Filtered Dataset Preview")
st.dataframe(filtered.head())

# -----------------------------------------------------------
# 3. FEATURE ENGINEERING
# -----------------------------------------------------------
filtered = filtered.sort_values("date")
filtered["day_index"] = np.arange(len(filtered))

# Features used for ML model
feature_cols = ["price", "promo_flag", "temperature", "day_index"]

X = filtered[feature_cols]
y = filtered["orders"]

# -----------------------------------------------------------
# 4. TRAIN ML MODEL
# -----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

model = RandomForestRegressor(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

st.subheader("📊 Model Performance")
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# -----------------------------------------------------------
# 5. FORECAST FUTURE DEMAND
# -----------------------------------------------------------
st.header("📈 Forecast Future Demand")

future_days = st.slider("Days to forecast:", 7, 120, 30)

last_index = filtered["day_index"].iloc[-1]
future_index = np.arange(last_index + 1, last_index + future_days + 1)

# Use last known values (stable prediction)
last_price = filtered["price"].iloc[-1]
last_promo = filtered["promo_flag"].iloc[-1]
last_temp = filtered["temperature"].iloc[-1]

# ----- FIXED VERSION -----
future_data = pd.DataFrame({
    "price": last_price,
    "promo_flag": last_promo,
    "temperature": last_temp,
    "day_index": future_index,
})

# Ensure correct column order
future_data = future_data[feature_cols]

forecast = model.predict(future_data)

forecast_df = pd.DataFrame({
    "date": pd.date_range(filtered["date"].iloc[-1], periods=future_days+1, closed="right"),
    "forecast_orders": forecast
})

st.subheader("📅 Forecasted Orders")
st.line_chart(forecast_df.set_index("date"))
st.dataframe(forecast_df)
