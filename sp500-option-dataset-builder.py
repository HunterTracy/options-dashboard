# -*- coding: utf-8 -*-
"""
SP500 Option Dataset Builder
Created in 2023
Author: Quant Galore (updated for production reliability)
"""
# === Imports ===
import os
import pandas as pd
import numpy as np
import scipy.optimize as optimize
import requests
import yfinance as yf
import pytz
from datetime import timedelta, datetime
from pandas_market_calendars import get_calendar
from sqlalchemy import create_engine, text
from feature_functions import binomial_option_price, bjerksund_stensland_greeks, Binarizer, return_proba
from self_email import send_message, send_telegram_message

# === Config ===
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
DATABASE_URL = (
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@"
    f"{os.getenv('DB_HOST')}:{5432}/{os.getenv('DB_NAME')}?sslmode=prefer"
)
calendar = get_calendar("NYSE")
underlying_ticker = "SPY"
today = datetime.now(tz=pytz.timezone("US/Eastern")).date()
initial_time = datetime.now()
start_date = today - timedelta(days=365)

# === Helper: Safe Polygon request ===
def polygon_json(url, timeout=15):
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        # polygon v2/v3 endpoints commonly return {"results": [...]} OR sometimes other shapes
        if isinstance(data, dict):
            return data.get("results", [])
        # if already a list, return as-is
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        print(f"⚠️ Polygon request failed for {url}: {e}")
        return []

# small helper to sanitize values that may be numpy scalars/arrays
def clean_scalar(v):
    try:
        # numpy scalar
        if isinstance(v, np.generic):
            return v.item()
        # numpy array with single element
        if isinstance(v, np.ndarray):
            if v.size == 1:
                return v.item()
            else:
                # more-than-one arrays -> turn into list
                return v.tolist()
        # pandas/numpy nan -> None for SQL
        if pd.isna(v):
            return None
        return v
    except Exception:
        return v

# === Build trade date range ===
trade_dates = pd.DataFrame({
    "trade_dates": calendar.schedule(start_date=start_date, end_date=today)
    .index.strftime("%Y-%m-%d").values[-11:]  # for quick tests; change to [-101:] for full history
})

feature_price_data_list = []
option_chain_list = []
times = []

# === Main Loop ===
for date in trade_dates["trade_dates"]:
    start_time = datetime.now()
    # Determine expiration and previous/next trading days
    if date == trade_dates["trade_dates"].iloc[-1]:
        # last row special handling: use previous trading date in polygon query
        next_trading_date = calendar.schedule(
            start_date=date, end_date=pd.to_datetime(date) + timedelta(days=5)
        ).index.strftime("%Y-%m-%d").values[1]
        idx = trade_dates[trade_dates["trade_dates"] == date].index[0]
        previous_trading_date = trade_dates.iloc[idx - 1, 0]
        expiration_date = next_trading_date

        Underlying_Today = pd.json_normalize(
            polygon_json(
                f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{previous_trading_date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
            )
        ).set_index("t") if polygon_json(
            f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{previous_trading_date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
        ) else pd.DataFrame()

        if Underlying_Today.empty:
            # fallback — create a placeholder row so DB build doesn't break
            today_feature_price_dataframe = pd.DataFrame([{
                "timestamp": pd.to_datetime(date),
                "open": None,
                "close": None,
                "net_call_volume": np.nan,
                "net_put_volume": np.nan,
                "net_call_gamma": np.nan,
                "net_put_gamma": np.nan,
                "mean_call_ivol": np.nan,
                "mean_put_vol": np.nan,
                "total_gamma": np.nan,
                "total_volume": np.nan,
                "mean_ivol": np.nan,
                "call_put_gamma_imbalance": np.nan,
                "gamma_times_volume": np.nan,
                "ivol_adjusted_net_gamma": np.nan
            }])
            feature_price_data_list.append(today_feature_price_dataframe)
            break

        Underlying_Today["returns"] = ((Underlying_Today["o"] - Underlying_Today["c"].shift(1)) / Underlying_Today["c"].shift(1)).fillna(0)
        today_overnight_return = Underlying_Today["returns"].iloc[-1]

        today_feature_price_dataframe = pd.DataFrame([{
            "timestamp": pd.to_datetime(date),
            "open": Underlying_Today["o"].iloc[-1],
            "close": Underlying_Today["c"].iloc[0],
            "net_call_volume": np.nan,
            "net_put_volume": np.nan,
            "net_call_gamma": np.nan,
            "net_put_gamma": np.nan,
            "mean_call_ivol": np.nan,
            "mean_put_vol": np.nan,
            "total_gamma": np.nan,
            "total_volume": np.nan,
            "mean_ivol": np.nan,
            "call_put_gamma_imbalance": np.nan,
            "gamma_times_volume": np.nan,
            "ivol_adjusted_net_gamma": np.nan
        }])
        feature_price_data_list.append(today_feature_price_dataframe)
        break
    else:
        idx = trade_dates[trade_dates["trade_dates"] == date].index[0]
        expiration_date = trade_dates.iloc[idx + 1, 0]

    # === Underlying price ===
    underlying_results = polygon_json(
        f"https://api.polygon.io/v2/aggs/ticker/{underlying_ticker}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=50000&apiKey={POLYGON_API_KEY}"
    )
    Underlying = pd.json_normalize(underlying_results).set_index("t") if underlying_results else pd.DataFrame()

    if Underlying.empty:
        print(f"⚠️ No underlying data for {date}, skipping...")
        continue

    Underlying.index = pd.to_datetime(Underlying.index, unit="ms", utc=True).tz_convert("America/New_York")
    Underlying_Price = Underlying["c"].iloc[0]
    Underlying_Open = Underlying["o"].iloc[0]
    S = Underlying_Price

    # === Option Contracts ===
    call_contracts_results = polygon_json(
        f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=call&expiration_date={expiration_date}&as_of={expiration_date}&expired=false&limit=1000&apiKey={POLYGON_API_KEY}"
    )
    Call_Contracts = pd.json_normalize(call_contracts_results) if call_contracts_results else pd.DataFrame()
    if Call_Contracts.empty:
        print(f"⚠️ No call contracts on {date}")
        continue
    Call_Contracts["distance_from_price"] = abs(Call_Contracts["strike_price"] - S)
    Call_Symbols = Call_Contracts.nsmallest(n=10, columns="distance_from_price")["ticker"].values

    put_contracts_results = polygon_json(
        f"https://api.polygon.io/v3/reference/options/contracts?underlying_ticker={underlying_ticker}&contract_type=put&expiration_date={expiration_date}&as_of={expiration_date}&expired=false&limit=1000&apiKey={POLYGON_API_KEY}"
    )
    Put_Contracts = pd.json_normalize(put_contracts_results) if put_contracts_results else pd.DataFrame()
    if Put_Contracts.empty:
        print(f"⚠️ No put contracts on {date}")
        continue
    Put_Contracts["distance_from_price"] = abs(Put_Contracts["strike_price"] - S)
    Put_Symbols = Put_Contracts.nsmallest(n=10, columns="distance_from_price")["ticker"].values

    call_data_list = []
    put_data_list = []

    # === Collect call data ===
    for call in Call_Symbols:
        call_contract_info = Call_Contracts[Call_Contracts["ticker"] == call]
        call_ohlcv_results = polygon_json(
            f"https://api.polygon.io/v2/aggs/ticker/{call}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
        )
        if not call_ohlcv_results:
            continue
        call_ohlcv = pd.json_normalize(call_ohlcv_results).set_index("t")
        call_ohlcv.index = pd.to_datetime(call_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")
        K = call_contract_info["strike_price"].iloc[0]
        T = 1 / 252
        r = 0.05
        n = 500

        def f_call(volatility):
            return binomial_option_price(S, K, T, r, volatility, n, option_type="call") - call_ohlcv["c"].iloc[0]

        try:
            call_implied_vol = optimize.newton(f_call, x0=0.15, tol=0.05, maxiter=50)
        except Exception:
            continue

        call_delta, call_gamma, call_theta, call_vega = bjerksund_stensland_greeks(
            S, K, T, r, sigma=call_implied_vol, option_type="call"
        )

        # sanitize greeks (numpy scalars) into python scalars before creating DataFrame
        call_delta = clean_scalar(call_delta)
        call_gamma = clean_scalar(call_gamma)
        call_theta = clean_scalar(call_theta)
        call_vega = clean_scalar(call_vega)
        call_implied_vol = clean_scalar(call_implied_vol)

        call_dataframe = pd.DataFrame([{
            "timestamp": date,
            "strike_price": K,
            "call_symbol": call,
            "call_delta": call_delta,
            "call_gamma": call_gamma,
            "call_theta": call_theta,
            "call_vega": call_vega,
            "call_implied_vol": call_implied_vol,
            "call_open": clean_scalar(call_ohlcv["o"].iloc[0]),
            "call_high": clean_scalar(call_ohlcv["h"].iloc[0]),
            "call_low": clean_scalar(call_ohlcv["l"].iloc[0]),
            "call_close": clean_scalar(call_ohlcv["c"].iloc[0]),
            "call_vw": clean_scalar(call_ohlcv["vw"].iloc[0]),
            "call_volume": clean_scalar(call_ohlcv["v"].iloc[0])
        }])
        call_data_list.append(call_dataframe)

    # === Collect put data ===
    for put in Put_Symbols:
        put_contract_info = Put_Contracts[Put_Contracts["ticker"] == put]
        put_ohlcv_results = polygon_json(
            f"https://api.polygon.io/v2/aggs/ticker/{put}/range/1/day/{date}/{date}?adjusted=true&sort=asc&limit=5000&apiKey={POLYGON_API_KEY}"
        )
        if not put_ohlcv_results:
            continue
        put_ohlcv = pd.json_normalize(put_ohlcv_results).set_index("t")
        put_ohlcv.index = pd.to_datetime(put_ohlcv.index, unit="ms", utc=True).tz_convert("America/New_York")
        K = put_contract_info["strike_price"].iloc[0]
        T = 1 / 252
        r = 0.05
        n = 500

        def f_put(volatility):
            return binomial_option_price(S, K, T, r, volatility, n, option_type="put") - put_ohlcv["c"].iloc[0]

        try:
            put_implied_vol = optimize.newton(f_put, x0=0.15, tol=0.05, maxiter=50)
        except Exception:
            continue

        if put_implied_vol < 0:
            last_vol = put_data_list[-1]["put_implied_vol"].iloc[0] if put_data_list else 0.15
            put_implied_vol = last_vol

        put_delta, put_gamma, put_theta, put_vega = bjerksund_stensland_greeks(
            S, K, T, r, sigma=put_implied_vol, option_type="put"
        )

        put_delta = clean_scalar(put_delta)
        put_gamma = clean_scalar(put_gamma)
        put_theta = clean_scalar(put_theta)
        put_vega = clean_scalar(put_vega)
        put_implied_vol = clean_scalar(put_implied_vol)

        put_dataframe = pd.DataFrame([{
            "timestamp": date,
            "strike_price": K,
            "put_symbol": put,
            "put_delta": put_delta,
            "put_gamma": put_gamma,
            "put_theta": put_theta,
            "put_vega": put_vega,
            "put_implied_vol": put_implied_vol,
            "put_open": clean_scalar(put_ohlcv["o"].iloc[0]),
            "put_high": clean_scalar(put_ohlcv["h"].iloc[0]),
            "put_low": clean_scalar(put_ohlcv["l"].iloc[0]),
            "put_close": clean_scalar(put_ohlcv["c"].iloc[0]),
            "put_vw": clean_scalar(put_ohlcv["vw"].iloc[0]),
            "put_volume": clean_scalar(put_ohlcv["v"].iloc[0])
        }])
        put_data_list.append(put_dataframe)

    # === Merge call/put data ===
    if len(put_data_list) == len(call_data_list) and len(put_data_list) > 0:
        Call_Data = pd.concat(call_data_list).set_index("timestamp")
        Put_Data = pd.concat(put_data_list)
        Option_Chain = pd.merge(Put_Data, Call_Data, on="strike_price").set_index("timestamp")
        # ensure all values inside Option_Chain are scalars (no numpy arrays)
        for col in Option_Chain.columns:
            Option_Chain[col] = Option_Chain[col].apply(clean_scalar)

        Option_Chain["underlying_price"] = S
        Option_Chain["distance_from_price"] = abs(Option_Chain["strike_price"] - S)

        net_call_gamma = Option_Chain["call_gamma"].sum()
        net_call_volume = Option_Chain["call_volume"].sum()
        mean_call_vol = Option_Chain["call_implied_vol"].mean()
        net_put_gamma = Option_Chain["put_gamma"].sum()
        net_put_volume = Option_Chain["put_volume"].sum()
        mean_put_vol = Option_Chain["put_implied_vol"].mean()
        total_gamma = net_call_gamma + net_put_gamma
        total_volume = net_call_volume + net_put_volume
        total_mean_vol = (mean_call_vol + mean_put_vol) / 2
        call_put_gamma_imbalance = net_call_gamma - net_put_gamma
        gamma_times_volume = total_gamma * total_volume
        ivol_adjusted_gamma = total_mean_vol * total_gamma

        feature_price_dataframe = pd.DataFrame([{
            "timestamp": pd.to_datetime(date),
            "open": Underlying_Open,
            "close": Underlying_Price,
            "net_call_volume": net_call_volume,
            "net_put_volume": net_put_volume,
            "net_call_gamma": net_call_gamma,
            "net_put_gamma": net_put_gamma,
            "mean_call_ivol": mean_call_vol,
            "mean_put_vol": mean_put_vol,
            "total_gamma": total_gamma,
            "total_volume": total_volume,
            "mean_ivol": total_mean_vol,
            "call_put_gamma_imbalance": call_put_gamma_imbalance,
            "gamma_times_volume": gamma_times_volume,
            "ivol_adjusted_net_gamma": ivol_adjusted_gamma
        }])
        feature_price_data_list.append(feature_price_dataframe)
        option_chain_list.append(Option_Chain)
    else:
        print(f"⚠️ call/put list mismatch for {date}: calls={len(call_data_list)}, puts={len(put_data_list)} — skipping this date")
        continue

    end_time = datetime.now()
    seconds_to_complete = (end_time - start_time).total_seconds()
    times.append(seconds_to_complete)
    iteration = round((np.where(trade_dates["trade_dates"] == date)[0][0] / len(trade_dates)) * 100, 2)
    iterations_remaining = len(trade_dates) - np.where(trade_dates["trade_dates"] == date)[0][0]
    average_time_to_complete = np.mean(times)
    estimated_completion_time = datetime.now() + timedelta(seconds=int(average_time_to_complete * iterations_remaining))
    time_remaining = estimated_completion_time - datetime.now()
    print(f"{iteration}% complete, {time_remaining} left, ETA: {estimated_completion_time}")

# === Final writes ===
full_feature_price_data = pd.concat(feature_price_data_list).set_index("timestamp")
Pre_Training_Dataset = full_feature_price_data.copy()
Pre_Training_Dataset["returns"] = ((Pre_Training_Dataset["open"] - Pre_Training_Dataset["close"].shift(1)) / Pre_Training_Dataset["close"].shift(1)).fillna(0).shift(-1)
Pre_Training_Dataset["month"] = Pre_Training_Dataset.index.month
Pre_Training_Dataset["day"] = Pre_Training_Dataset.index.day

engine = create_engine(DATABASE_URL)
from sqlalchemy import text

# --- Drop and rebuild tables cleanly ---
with engine.begin() as conn:
    conn.execute(text(f"DROP TABLE IF EXISTS {underlying_ticker}_option_backtest"))
    conn.execute(text(f"DROP TABLE IF EXISTS {underlying_ticker}_option_chain"))

# --- Recreate tables ---
Pre_Training_Dataset.to_sql(f"{underlying_ticker}_option_backtest", con=engine, if_exists='replace', index=True)
Option_Chain_DataFrame = pd.concat(option_chain_list).reset_index()

# Convert all numpy arrays to scalars for safe SQL write
for col in Option_Chain_DataFrame.columns:
    Option_Chain_DataFrame[col] = Option_Chain_DataFrame[col].apply(
        lambda x: float(x) if isinstance(x, np.ndarray) and x.size == 1 else (x.tolist() if isinstance(x, np.ndarray) else x)
    )

Option_Chain_DataFrame.to_sql(f"{underlying_ticker}_option_chain", con=engine, if_exists='replace', index=True)

# --- Completion tracking + notifications ---
final_time = datetime.now(tz=pytz.timezone("US/Eastern"))
elapsed = final_time - pytz.timezone("US/Eastern").localize(initial_time)


database_string = (
    f"✅ Database rebuilt on {today} in {elapsed}, "
    f"last data point: {Pre_Training_Dataset.index[-1].strftime('%Y-%m-%d')}"
)
print(database_string)

# Email & Telegram notifications
send_message(message=database_string, subject=f"Database Operation on {today.strftime('%A')}, {today}")
send_telegram_message(message=database_string, subject="📊 SP500 Dataset Builder Update")
print("✅ Dataset build completed and notifications sent.")
import subprocess

print("🧠 Starting model retraining...")
try:
    subprocess.run(["python3.10", "/home/huntertracy/train_model.py"], check=True)
    print("✅ Model retrained successfully.")
    send_telegram_message("✅ Model retrained successfully after dataset update.")
except subprocess.CalledProcessError as e:
    print(f"⚠️ Model retraining failed: {e}")
    send_telegram_message(f"⚠️ Model retraining failed: {e}")




