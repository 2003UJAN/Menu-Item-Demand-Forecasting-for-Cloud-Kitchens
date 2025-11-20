import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cloud Kitchen Demand Forecasting", layout="wide")

st.title("🍽️ Cloud Kitchen Demand Forecasting (ML App)")
st.write("Upload your dataset and forecast demand for any menu item across cities/kitchens.")

# ======================================
# 1. FILE UPLOAD
# ======================================
uploaded_file = st.file_uploader("Upload your CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload your dataset to continue.")
    st.stop()

# ======================================
# 2. VALIDATE DATASET STRUCTURE
# ======================================
required_cols = [
    "date","city","kitchen_id","item_name","category",
    "price","orders","weekday","promo_flag","temperature"
]

missing = [c for c in required_cols if c not in df.columns]

if missing:
    st.error(f"❌ Missing columns: {missing}")
    st.stop()

# Convert date column
df["date"] = pd.to_datetime(df["date"])

# ======================================
# 3. FILTERS
# ======================================
st.subheader("🔎 Apply Filters")

city = st.selectbox("Select City", df["city"].unique())
kitchen = st.selectbox("Select Kitchen", df["kitchen_id"].unique())
item = st.selectbox("Select Menu Item", df["item_name"].unique())

filtered_df = df[
    (df["city"] == city) &
    (df["kitchen_id"] == kitchen) &
    (df["item_name"] == item)
].copy()

if filtered_df.empty:
    st.error("No records found for this combination. Try different filters.")
    st.stop()

st.write("### 📌 Filtered Dataset Preview")
st.dataframe(filtered_df.head())

# ======================================
# 4. FEATURE ENGINEERING
# ======================================
filtered_df["day_index"] = np.arange(len(filtered_df))

feature_cols = ["price", "promo_flag", "temperature", "day_index"]

X = filtered_df[feature_cols]
y = filtered_df["orders"]

# ======================================
# 5. TRAIN ML MODEL
# ======================================
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

# ======================================
# 6. FORECAST FUTURE DEMAND
# ======================================
st.header("📈 Forecast Future Demand")

future_days = st.slider("How many days to forecast?", 7, 120, 30)

last_index = filtered_df["day_index"].iloc[-1]
future_index = np.arange(last_index + 1, last_index + future_days + 1)

# Use last known values
last_price = filtered_df["price"].iloc[-1]
last_promo = filtered_df["promo_flag"].iloc[-1]
last_temp = filtered_df["temperature"].iloc[-1]

future_data = pd.DataFrame({
    "day_index": future_index,
    "price": last_price,
    "promo_flag": last_promo,
    "temperature": last_temp,
})

forecast = model.predict(future_data)

forecast_df = pd.DataFrame({
    "date": pd.date_range(filtered_df["date"].iloc[-1], periods=future_days+1, closed="right"),
    "forecast_orders": forecast
})

st.subheader("📅 Forecasted Orders")
st.line_chart(forecast_df.set_index("date"))
st.dataframe(forecast_df)
