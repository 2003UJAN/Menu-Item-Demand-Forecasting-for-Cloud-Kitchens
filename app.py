import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Load Model
# -----------------------------
model = pickle.load(open("models/demand_forecast_model.pkl", "rb"))
required_features = list(model.feature_names_in_)

st.title("🍽️ Menu Item Demand Forecasting – Cloud Kitchen")

# ============================================================
#                OPTION 1 → UPLOAD CSV FILE
# ============================================================
st.header("Upload Sales File")
uploaded = st.file_uploader("Upload your CSV file", type=["csv"])

# ============================================================
#                OPTION 2 → MANUAL INPUT
# ============================================================
st.header("Or Enter Parameters Manually")

manual_date = st.date_input("Start Date for Forecast")
future_days = st.number_input("Days to Forecast", min_value=1, max_value=60, value=7)

menu_item = st.text_input("Menu Item Name (optional)", value="Item_1")

# ----------------------------
# When users click RUN
# ----------------------------
if st.button("Generate Forecast"):

    # ======================================================
    #         CASE 1: FILE UPLOADED
    # ======================================================
    if uploaded:
        df = pd.read_csv(uploaded)

        # Expecting columns: date, item, quantity
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        filtered = df[df["item"] == menu_item]

        if filtered.empty:
            st.error("No records found for this menu item!")
            st.stop()

        # create new future dates (FIXED)
        future_dates = pd.date_range(
            start=filtered["date"].iloc[-1] + pd.Timedelta(days=1),
            periods=future_days
        )

    # ======================================================
    #         CASE 2: MANUAL INPUT ONLY
    # ======================================================
    else:
        future_dates = pd.date_range(
            start=pd.to_datetime(manual_date),
            periods=future_days
        )

    # ======================================================
    #          CREATE FUTURE FEATURE DATAFRAME
    # ======================================================
    future_data = pd.DataFrame({
        "date": future_dates,
        "day_of_week": future_dates.dayofweek,
        "is_weekend": future_dates.dayofweek.isin([5, 6]).astype(int)
    })

    # ------------------------------------------------------
    # Ensure the feature order EXACTLY matches model
    # ------------------------------------------------------
    missing = [f for f in required_features if f not in future_data.columns]

    # If any missing (should not happen)
    for m in missing:
        future_data[m] = 0

    # Reorder columns → IMPORTANT FIX
    future_data = future_data[required_features]

    # ------------------------------------------------------
    # Predict
    # ------------------------------------------------------
    forecast = model.predict(future_data)

    # ------------------------------------------------------
    # Display results
    # ------------------------------------------------------
    result_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Demand": forecast.astype(int)
    })

    st.subheader("📈 Forecast Results")
    st.dataframe(result_df)

    st.line_chart(result_df.set_index("Date"))

