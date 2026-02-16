import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Sales Forecast", layout="centered")

st.title("📊 Sales & Demand Forecasting")
st.write(
    "Forecast future sales using a pre-trained SARIMA model to support "
    "inventory, staffing, and cash-flow planning."
)

st.divider()

# -----------------------------
# Forecast Settings
# -----------------------------
st.subheader("🔮 Forecast Settings")

forecast_months = st.slider(
    "Select number of months to forecast",
    min_value=3,
    max_value=12,
    value=6
)

generate = st.button("Generate Forecast")

# -----------------------------
# Load Model & Forecast
# -----------------------------
if generate:
    # Load trained model and historical data
    sarima_model = joblib.load("sarima_sales_model.pkl")
    monthly_sales = joblib.load("monthly_sales.pkl")

    # Generate forecast
    forecast = sarima_model.forecast(steps=forecast_months)

    # Create forecast dates
    forecast_dates = pd.date_range(
        start=monthly_sales['Order Date'].iloc[-1] + pd.offsets.MonthEnd(1),
        periods=forecast_months,
        freq="ME"
    )

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Forecasted Sales": forecast
    })

    st.divider()
    st.subheader("📈 Sales Forecast")

    # -----------------------------
    # Visualization (Improved)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))

    # Historical sales
    ax.plot(
        monthly_sales['Order Date'],
        monthly_sales['Total_Sales'],
        label="Historical Sales",
        linewidth=2
    )

    # Connect last historical point to forecast
    last_date = monthly_sales['Order Date'].iloc[-1]
    last_value = monthly_sales['Total_Sales'].iloc[-1]

    ax.plot(
        [last_date] + list(forecast_df['Date']),
        [last_value] + list(forecast_df['Forecasted Sales']),
        linestyle="--",
        marker="o",
        label="Forecasted Sales",
        linewidth=2
    )

    # Shade forecast region
    ax.axvspan(
        forecast_df['Date'].min(),
        forecast_df['Date'].max(),
        alpha=0.15
    )

    # Axis formatting
    ax.set_title("Monthly Sales Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.grid(True, alpha=0.3)
    ax.legend()

    st.pyplot(fig)

    # -----------------------------
    # Forecast Table
    # -----------------------------
    st.subheader("📋 Forecast Values")
    st.dataframe(forecast_df)

    # -----------------------------
    # Business Insight
    # -----------------------------
    st.subheader("💼 Business Insight")
    st.write(
        "The dashed line and shaded area represent forecasted sales. "
        "This projection helps businesses plan inventory levels, staffing, "
        "and cash flow based on expected demand."
    )

    st.success("Forecast generated successfully!")
