import streamlit as st
import pandas as pd
import plotly.express as px
import sqlalchemy
import os
from datetime import datetime
import requests

st.write("DEBUG: Secrets loaded:", list(st.secrets.keys()))


# --- Password Protection ---
if "auth_ok" not in st.session_state:
    st.session_state["auth_ok"] = False

if not st.session_state["auth_ok"]:
    password = st.text_input("Enter password:", type="password")
    if password == st.secrets["auth"]["password"]:
        st.session_state["auth_ok"] = True
        st.rerun()
    elif password:
        st.error("Incorrect password")
    st.stop()





# --- Page Config ---
st.set_page_config(page_title="SP500 Prediction Dashboard", layout="wide")

# --- Database Connection ---
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
               f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT', 5432)}/{os.getenv('DB_NAME')}?sslmode=prefer"

engine = sqlalchemy.create_engine(DATABASE_URL)

# --- App Header ---
st.title("📈 S&P 500 Option-Based Prediction Dashboard")
st.markdown("Visualize model signals, correlations, and live updates.")

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_data():
    query = "SELECT * FROM spy_option_backtest ORDER BY timestamp DESC LIMIT 300;"
    return pd.read_sql(query, engine)

try:
    df = load_data()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# --- Feature Selection ---
st.sidebar.header("🔍 Controls")
feature = st.sidebar.selectbox("Select feature to view correlation:", df.columns)
view_mode = st.sidebar.radio("View mode:", ["Model Predictions", "Feature Trend"])

# --- Display Feature Trend ---
if view_mode == "Feature Trend":
    fig = px.line(df, x="timestamp", y=feature, title=f"{feature} over time")
    st.plotly_chart(fig, use_container_width=True)

# --- Display Model Predictions ---
elif view_mode == "Model Predictions":
    if "prediction" not in df.columns:
        st.warning("⚠️ No model predictions available yet.")
    else:
        fig = px.scatter(
            df, x="timestamp", y="returns", color="prediction",
            title="Model Buy/Sell/Neutral Signals",
            color_discrete_map={"Buy": "green", "Sell": "red", "Neutral": "gray"}
        )
        st.plotly_chart(fig, use_container_width=True)

# --- Refresh Data ---
if st.button("🔄 Refresh Data"):
    try:
        response = requests.get("https://api.polygon.io/v2/aggs/ticker/SPY/prev?apiKey=" + os.getenv("POLYGON_API_KEY"))
        if response.status_code == 200:
            st.success("✅ Data refreshed successfully.")
            st.cache_data.clear()
        else:
            st.warning("⚠️ Could not fetch new data.")
    except Exception as e:
        st.error(f"Error refreshing: {e}")
