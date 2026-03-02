import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import joblib
import numpy as np

# -------------------------------------------------------
# Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="Sales & Demand Forecasting",
    page_icon="📊",
    layout="wide"
)

# -------------------------------------------------------
# Custom CSS Styling
# -------------------------------------------------------
st.markdown("""
    <style>
        .main { background-color: #0f1117; }
        .stApp { background-color: #0f1117; color: #e0e0e0; }
        h1 { color: #00d4ff; font-size: 2.4rem !important; font-weight: 700; }
        h2, h3 { color: #00d4ff; }
        .stButton>button {
            background: linear-gradient(135deg, #00d4ff, #0066cc);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 30px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background: linear-gradient(135deg, #00bcd4, #0052a3);
            transform: scale(1.03);
        }
        .metric-card {
            background: linear-gradient(135deg, #1c1f2e, #252840);
            border: 1px solid #2e3250;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 15px rgba(0, 212, 255, 0.1);
        }
        .metric-label {
            font-size: 0.85rem;
            color: #8899bb;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #00d4ff;
        }
        .insight-box {
            background: linear-gradient(135deg, #1a2230, #1e2840);
            border-left: 4px solid #00d4ff;
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 10px;
            color: #c0d0e0;
            font-size: 0.95rem;
            line-height: 1.6;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------------------------------
# Header
# -------------------------------------------------------
st.markdown("<h1>📊 Sales & Demand Forecasting</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='color:#8899bb; font-size:1.05rem;'>Powered by SARIMA — predict future sales to support "
    "inventory planning, staffing, and cash-flow decisions.</p>",
    unsafe_allow_html=True
)
st.divider()

# -------------------------------------------------------
# Sidebar Controls
# -------------------------------------------------------
with st.sidebar:
    st.markdown("<h2 style='color:#00d4ff;'>⚙️ Forecast Settings</h2>", unsafe_allow_html=True)
    forecast_months = st.slider(
        "Months to Forecast",
        min_value=3,
        max_value=12,
        value=6,
        help="Select how many months into the future you want to forecast."
    )
    st.markdown("---")
    st.markdown(
        "<small style='color:#556677;'>Built with Streamlit · SARIMA Model · Superstore Dataset</small>",
        unsafe_allow_html=True
    )
    generate = st.button("🔮 Generate Forecast", use_container_width=True)

# -------------------------------------------------------
# Load model & generate forecast on button click
# -------------------------------------------------------
if generate:
    with st.spinner("Loading model and generating forecast..."):
        # Load trained model and historical data
        sarima_model = joblib.load("sarima_sales_model.pkl")
        monthly_sales = joblib.load("monthly_sales.pkl")

        # --- Generate Point Forecast ---
        forecast_values = sarima_model.forecast(steps=forecast_months)

        # --- Build Forecast Dates ---
        last_date = monthly_sales["Order Date"].iloc[-1]
        forecast_dates = pd.date_range(
            start=last_date + pd.offsets.MonthEnd(1),
            periods=forecast_months,
            freq="ME"
        )

        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Forecasted Sales ($)": forecast_values.values
        })
        forecast_df["Month"] = forecast_df["Date"].dt.strftime("%b %Y")
        forecast_df["Forecasted Sales ($)"] = forecast_df["Forecasted Sales ($)"].round(2)

    st.divider()

    # -------------------------------------------------------
    # KPI Summary Cards
    # -------------------------------------------------------
    st.markdown("<h2>📈 Forecast Summary</h2>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)

    hist_avg  = monthly_sales["Total_Sales"].mean()
    fore_avg  = forecast_df["Forecasted Sales ($)"].mean()
    fore_max  = forecast_df["Forecasted Sales ($)"].max()
    fore_min  = forecast_df["Forecasted Sales ($)"].min()
    pct_change = ((fore_avg - hist_avg) / hist_avg) * 100

    with col1:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Avg Forecast / Month</div>
                <div class='metric-value'>${fore_avg:,.0f}</div>
            </div>""", unsafe_allow_html=True)
    with col2:
        direction = "📈" if pct_change >= 0 else "📉"
        color = "#00e676" if pct_change >= 0 else "#ff5252"
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>vs Historical Avg</div>
                <div class='metric-value' style='color:{color};'>{direction} {pct_change:+.1f}%</div>
            </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Peak Month Forecast</div>
                <div class='metric-value'>${fore_max:,.0f}</div>
            </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>Lowest Month Forecast</div>
                <div class='metric-value'>${fore_min:,.0f}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------
    # Main Chart
    # -------------------------------------------------------
    st.markdown("<h2>📉 Historical vs Forecasted Sales</h2>", unsafe_allow_html=True)

    hist_dates  = pd.to_datetime(monthly_sales["Order Date"])
    hist_values = monthly_sales["Total_Sales"]

    # Build connection bridge (last hist point → first forecast point)
    bridge_dates  = [hist_dates.iloc[-1]] + list(forecast_dates)
    bridge_values = [hist_values.iloc[-1]] + list(forecast_df["Forecasted Sales ($)"])

    fig, ax = plt.subplots(figsize=(14, 5.5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#141824")

    # Historical line
    ax.plot(
        hist_dates, hist_values,
        color="#00d4ff", linewidth=2.2,
        label="Historical Sales", zorder=3
    )
    ax.fill_between(hist_dates, hist_values, alpha=0.08, color="#00d4ff")

    # Forecast line (connected from last historical point)
    ax.plot(
        bridge_dates, bridge_values,
        color="#ff9a3c", linewidth=2.2,
        linestyle="--", marker="o", markersize=5,
        label="Forecasted Sales", zorder=4
    )

    # Shaded forecast region
    ax.axvspan(
        forecast_dates[0], forecast_dates[-1],
        alpha=0.08, color="#ff9a3c", label="_nolegend_"
    )

    # Vertical divider line between historical and forecast
    ax.axvline(
        x=hist_dates.iloc[-1],
        color="#556677", linewidth=1.2,
        linestyle=":", alpha=0.8
    )

    # Annotation for divider
    ax.text(
        hist_dates.iloc[-1], ax.get_ylim()[1] * 0.95,
        "  Forecast →",
        color="#8899bb", fontsize=9
    )

    # Axis formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=40, ha="right", color="#8899bb", fontsize=9)
    plt.yticks(color="#8899bb", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    ax.set_title("Monthly Sales: Historical & Forecast", color="#e0e0e0", fontsize=14, pad=14)
    ax.set_xlabel("Date", color="#8899bb", fontsize=10)
    ax.set_ylabel("Sales ($)", color="#8899bb", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#2e3250")
    ax.spines["bottom"].set_color("#2e3250")
    ax.tick_params(colors="#8899bb")

    ax.grid(True, linestyle="--", alpha=0.15, color="#8899bb")

    legend = ax.legend(
        facecolor="#1c1f2e",
        edgecolor="#2e3250",
        labelcolor="#e0e0e0",
        fontsize=10
    )

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    # -------------------------------------------------------
    # Forecast Table
    # -------------------------------------------------------
    st.markdown("<h2>📋 Detailed Forecast Values</h2>", unsafe_allow_html=True)

    display_df = forecast_df[["Month", "Forecasted Sales ($)"]].copy()
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = "#"

    st.dataframe(
        display_df.style.format({"Forecasted Sales ($)": "${:,.2f}"}),
        use_container_width=True
    )

    # -------------------------------------------------------
    # Business Insight
    # -------------------------------------------------------
    st.markdown("<h2>💼 Business Insight</h2>", unsafe_allow_html=True)

    trend_text = (
        "an upward trend" if pct_change > 5
        else "a downward trend" if pct_change < -5
        else "a relatively stable trend"
    )
    st.markdown(f"""
        <div class='insight-box'>
            The SARIMA model predicts <strong>{trend_text}</strong> over the next
            <strong>{forecast_months} months</strong>, with an average monthly forecast of
            <strong>${fore_avg:,.0f}</strong> (compared to a historical average of
            <strong>${hist_avg:,.0f}</strong>).<br><br>
            📦 <b>Inventory:</b> Align stock levels with the projected peak of ${fore_max:,.0f}.<br>
            👷 <b>Staffing:</b> Prepare for demand fluctuations — especially around
            {forecast_df.loc[forecast_df['Forecasted Sales ($)'] == fore_max, 'Month'].values[0]}.<br>
            💰 <b>Cash Flow:</b> Plan budgets considering a potential low of ${fore_min:,.0f} in
            {forecast_df.loc[forecast_df['Forecasted Sales ($)'] == fore_min, 'Month'].values[0]}.
        </div>
    """, unsafe_allow_html=True)

    st.success("✅ Forecast generated successfully!")

else:
    # Landing state
    st.markdown("""
        <div style='text-align:center; padding: 60px 0; color:#556677;'>
            <div style='font-size:4rem;'>📊</div>
            <h3 style='color:#8899bb;'>Select the forecast horizon and click <em>Generate Forecast</em></h3>
            <p>Use the controls in the sidebar on the left to get started.</p>
        </div>
    """, unsafe_allow_html=True)
