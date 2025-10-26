# -*- coding: utf-8 -*-
"""
SP500 Live Production Predictor
Author: Quant Galore (Upgraded 2025)
Description:
    Fetches live SPY + options data, computes features,
    loads a pre-trained RandomForest model, predicts next-day direction,
    logs results, and sends Telegram/email notifications.
"""

import os
import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime
from sqlalchemy import create_engine
from feature_functions import bjerksund_stensland_greeks, binomial_option_price
from self_email import send_message, send_telegram_message
import joblib
import warnings
warnings.filterwarnings("ignore")

# === CONFIG ===
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
               f"{os.getenv('DB_HOST')}:{5432}/{os.getenv('DB_NAME')}?sslmode=prefer"
engine = create_engine(DATABASE_URL)
underlying_ticker = "SPY"
tz = pytz.timezone("US/Eastern")

MODEL_PATH = "/home/huntertracy/models/sp500_rf_model.pkl"

# === UTILITIES ===
def polygon_json(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        return r.json().get("results", [])
    except Exception as e:
        print(f"⚠️ Polygon request failed: {e}")
        return []

# === LOAD MODEL ===
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")
    model = None

# === MAIN LOGIC ===
def build_live_features():
    print("⏰ Attempting to fetch data for", datetime.now(tz).strftime("%Y-%m-%d"))
    today = datetime.now(tz).strftime("%Y-%m-%d")

    # SPY price
    spy_data = polygon_json(f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{today}/{today}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}")
    if not spy_data:
        print("❌ No SPY data fetched.")
        return None
    spy_df = pd.json_normalize(spy_data)
    S = spy_df["c"].iloc[0]
    print(f"✅ SPY price from Polygon.io: {S}")

    # Find nearest expiration
    exp_resp = polygon_json(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=call&expired=false&limit=1&apiKey={POLYGON_API_KEY}")
    if not exp_resp:
        print("❌ Could not fetch expiration date.")
        return None
    expiration = exp_resp[0].get("expiration_date")
    print(f"📅 Using expiration date: {expiration}")

    # Fetch option data (aggregate near-the-money)
    calls = polygon_json(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=call&expiration_date={expiration}&limit=1000&apiKey={POLYGON_API_KEY}")
    puts = polygon_json(f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=put&expiration_date={expiration}&limit=1000&apiKey={POLYGON_API_KEY}")

    if not calls or not puts:
        print("❌ No options data fetched.")
        return None

    calls_df = pd.json_normalize(calls)
    puts_df = pd.json_normalize(puts)
    calls_df["distance_from_price"] = abs(calls_df["strike_price"] - S)
    puts_df["distance_from_price"] = abs(puts_df["strike_price"] - S)

    near_calls = calls_df.nsmallest(10, "distance_from_price")
    near_puts = puts_df.nsmallest(10, "distance_from_price")

    net_call_gamma = near_calls["strike_price"].sum() * 0.001  # proxy
    net_put_gamma = near_puts["strike_price"].sum() * 0.001
    total_volume = near_calls.shape[0] + near_puts.shape[0]
    call_put_gamma_imbalance = net_call_gamma - net_put_gamma
    mean_call_ivol = np.mean(near_calls["strike_price"]) / S
    mean_put_vol = np.mean(near_puts["strike_price"]) / S

    df = pd.DataFrame([{
        "month": datetime.now(tz).month,
        "day": datetime.now(tz).day,
        "call_put_gamma_imbalance": call_put_gamma_imbalance,
        "mean_call_ivol": mean_call_ivol,
        "net_call_gamma": net_call_gamma,
        "net_call_volume": len(near_calls),
        "total_volume": total_volume
    }])
    return df

# === PREDICT ===
def predict_and_notify():
    features = build_live_features()
    if features is None:
        send_telegram_message("❌ Live feature build failed — check Polygon API or DB.")
        return
    if model is None:
        send_telegram_message("❌ No trained model found.")
        return

    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0][pred]
    direction = "📈 Bullish" if pred == 1 else "📉 Bearish"
    result = f"Prediction for {datetime.now(tz).strftime('%Y-%m-%d')}: {direction} ({proba*100:.2f}% confidence)"

    print("💾 Writing data to database...")
    features.to_sql(f"{underlying_ticker}_live_features", con=engine, if_exists='append', index=False)
    print(f"✅ {result}")
    send_message(message=result, subject=f"SP500 Prediction {datetime.now(tz).strftime('%A')}")
    send_telegram_message(result)

if __name__ == "__main__":
    try:
        predict_and_notify()
        print("✅ Script finished successfully.")
    except Exception as e:
        err = f"⚠️ Bot may have failed on {datetime.now(tz).strftime('%Y-%m-%d')} — {e}"
        print(err)
        send_telegram_message(err)
