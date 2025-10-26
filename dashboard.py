import os
import pandas as pd
import plotly.express as px
import requests
import sqlalchemy as sa
import streamlit as st

# --- Page Config ---
st.set_page_config(page_title="SP500 Prediction Dashboard", layout="wide")

# --- Password Protection ---
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

if not st.session_state["authenticated"]:
    password = st.text_input("Enter password:", type="password")
    # Expecting [auth] section in Streamlit secrets
    if "auth" in st.secrets and password == st.secrets["auth"].get("password"):
        st.session_state["authenticated"] = True
        st.rerun()
    elif password:
        st.error("Incorrect password.")
    st.stop()

# --- Database Configuration (from Streamlit secrets) ---
DB_USER = st.secrets["database"]["DB_USER"]
DB_PASSWORD = st.secrets["database"]["DB_PASSWORD"]
DB_HOST = st.secrets["database"]["DB_HOST"]
DB_PORT = st.secrets["database"]["DB_PORT"]
DB_NAME = st.secrets["database"]["DB_NAME"]

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = sa.create_engine(DATABASE_URL)

# --- Diagnostics (expand to see) ---
with st.expander("🩺 Diagnostics"):
    has_auth = "auth" in st.secrets and "password" in st.secrets["auth"]
    has_db = "database" in st.secrets
    st.write("Has [auth] secret:", has_auth)
    st.write("Has [database] secrets:", has_db)
    if has_db:
        redacted_url = f"postgresql+psycopg2://{st.secrets['database'].get('DB_USER','?')}:***@" \
                       f"{st.secrets['database'].get('DB_HOST','?')}:" \
                       f"{st.secrets['database'].get('DB_PORT','?')}/" \
                       f"{st.secrets['database'].get('DB_NAME','?')}"
        st.write("DB URL (redacted):", redacted_url)
    try:
        with engine.connect() as conn:
            one = conn.execute(sa.text("SELECT 1")).scalar()
        st.success(f"DB ping OK (SELECT 1 → {one})")
    except Exception as e:
        st.error(f"DB ping failed: {e}")
    try:
        with engine.connect() as conn:
            tables = pd.read_sql(
                sa.text("SELECT table_schema, table_name FROM information_schema.tables "
                        "WHERE table_schema='public' ORDER BY 1,2"),
                conn,
            )
        st.write("Public tables:", tables)
    except Exception as e:
        st.error(f"Failed to list tables: {e}")

# --- App Header ---
st.title("📈 S&P 500 Option-Based Prediction Dashboard")
st.markdown("Visualize model signals, correlations, and live updates.")

# --- Load Data ---
@st.cache_data(ttl=3600)
def load_data(_engine):
    # IMPORTANT: exact, case-sensitive table name with quotes
    q = sa.text('SELECT * FROM "SPY_option_backtest" ORDER BY "timestamp" DESC LIMIT 300;')
    return pd.read_sql(q, _engine)

try:
    df = load_data(engine)
    # normalize timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    else:
        st.warning("No 'timestamp' column found.")
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

# --- Sidebar Controls ---
st.sidebar.header("🔍 Controls")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
default_feature = numeric_cols[0] if numeric_cols else None
feature = st.sidebar.selectbox("Select feature to view trend:", numeric_cols, index=0 if default_feature else None)
view_mode = st.sidebar.radio("View mode:", ["Model Predictions", "Feature Trend"], index=0)

# --- Feature Trend ---
if view_mode == "Feature Trend":
    if feature is None:
        st.info("No numeric features available to plot.")
    else:
        fig = px.line(df.sort_values("timestamp"), x="timestamp", y=feature, title=f"{feature} over time")
        st.plotly_chart(fig, use_container_width=True)

# --- Model Predictions view ---
else:
    if "prediction" not in df.columns:
        st.warning("⚠️ No model predictions available yet.")
    else:
        # Try to map numeric predictions to labels if needed
        pred_series = df["prediction"]
        if pd.api.types.is_numeric_dtype(pred_series):
            mapping = {1: "Buy", -1: "Sell", 0: "Neutral"}
            df["_pred_label"] = pred_series.map(mapping).fillna(pred_series.astype(str))
            color_col = "_pred_label"
        else:
            color_col = "prediction"

        if "returns" not in df.columns:
            st.warning("No 'returns' column; showing prediction markers over time.")
            fig = px.scatter(
                df.sort_values("timestamp"), x="timestamp", y=None, color=color_col,
                title="Model Signals over time"
            )
        else:
            fig = px.scatter(
                df.sort_values("timestamp"),
                x="timestamp",
                y="returns",
                color=color_col,
                title="Model Buy/Sell/Neutral Signals",
                color_discrete_map={"Buy": "green", "Sell": "red", "Neutral": "gray"},
            )
        st.plotly_chart(fig, use_container_width=True)

# --- Refresh Data Button ---
if st.button("🔄 Refresh Data"):
    try:
        # This just clears cache; replace with your own updater if desired
        st.cache_data.clear()
        st.success("✅ Cache cleared. Data will reload on next run.")
        st.rerun()
    except Exception as e:
        st.error(f"Error refreshing: {e}")
