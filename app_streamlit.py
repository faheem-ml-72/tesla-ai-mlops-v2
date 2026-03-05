# ===== imports =====
import streamlit as st
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import time
import requests

API_URL = "http://127.0.0.1:8000/predict"

# ===== page config =====
st.set_page_config(
    page_title="Tesla AI Forecast App",
    page_icon="🚀",
    layout="wide"
)

# ===== UI STYLE =====
def set_ultra_premium_style():
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #020617 100%);
    }
    .glass-card {
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(14px);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.08);
    }
    </style>
    """, unsafe_allow_html=True)

set_ultra_premium_style()

# ===== HEADER =====
st.markdown("""
<h1 style='font-size:48px;'>🚀 Tesla AI Forecast Dashboard</h1>
<p style='color:#94a3b8;'>Advanced LSTM • Ensemble Intelligence • Real-Time Data</p>
""", unsafe_allow_html=True)

# ===== SIDEBAR =====
st.sidebar.header("⚙️ Controls")

days = st.sidebar.slider(
    "History Window",
    min_value=60,
    max_value=365,
    value=120
)

auto_refresh = st.sidebar.toggle("🔄 Live Mode", value=False)

if auto_refresh:
    time.sleep(10)
    st.rerun()

# ===== FETCH DATA =====
@st.cache_data
def load_tesla(days):
    return yf.download("TSLA", period=f"{days}d")

df = load_tesla(days)

# ===== LIVE PRICE =====
latest_price = float(df["Close"].iloc[-1])

st.markdown(f"""
<div class="glass-card">
    <div style="font-size:14px;">Live TSLA Price</div>
    <div style="font-size:42px;font-weight:700;">${latest_price:.2f}</div>
</div>
""", unsafe_allow_html=True)

# ===== SPARKLINE =====
spark = go.Figure()

spark.add_trace(go.Scatter(
    x=df.index[-60:],
    y=df["Close"].iloc[-60:],
    mode="lines"
))

spark.update_layout(
    template="plotly_dark",
    height=120,
    margin=dict(l=10, r=10, t=10, b=10),
    xaxis=dict(visible=False),
    yaxis=dict(visible=False)
)

st.plotly_chart(spark, use_container_width=True)

# ===== PRICE HISTORY =====
st.subheader("📈 Tesla Price History")

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df.index,
    y=df["Close"],
    mode="lines"
))

fig.update_layout(
    template="plotly_dark",
    height=380,
    xaxis_title="Date",
    yaxis_title="Price"
)

st.plotly_chart(fig, use_container_width=True)

# ===== PREDICTION =====
st.subheader("🤖 AI Prediction")

if len(df) < 30:
    st.error("Not enough data for prediction")
    st.stop()

if st.button("Predict Tesla Price"):

    with st.spinner("🔮 Generating AI forecast..."):

        # prepare last 30 days
        recent_close = df["Close"].values[-30:]

        # convert to 30x5 format expected by API
        data = np.tile(recent_close.reshape(-1,1),(1,5)).tolist()

        try:
            response = requests.post(
                API_URL,
                json={"data": data}
            )

            if response.status_code == 200:

                result = response.json()

                forecast = result["forecast_price"]
                lower = result["confidence_interval"][0]
                upper = result["confidence_interval"][1]
                risk = result["risk_level"]
                uncertainty = result["uncertainty"]

                st.success("Prediction generated")

                col1,col2,col3,col4 = st.columns(4)

                col1.metric("Forecast Price", f"${forecast:.2f}")
                col2.metric("Lower Bound", f"${lower:.2f}")
                col3.metric("Upper Bound", f"${upper:.2f}")
                col4.metric("Uncertainty", f"{uncertainty:.6f}")

                if risk == "Low":
                    st.success(f"Risk Level: {risk}")
                elif risk == "Moderate":
                    st.warning(f"Risk Level: {risk}")
                else:
                    st.error(f"Risk Level: {risk}")

            else:
                st.error("API error")

        except Exception as e:
            st.error(f"Connection error: {e}")