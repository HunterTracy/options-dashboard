# train_model.py — trains and saves the SP500 model
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sqlalchemy import create_engine, text
import joblib

DATABASE_URL = f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@" \
               f"{os.getenv('DB_HOST')}:{5432}/{os.getenv('DB_NAME')}?sslmode=prefer"

MODEL_PATH = "/home/huntertracy/models/sp500_rf_model.pkl"
engine = create_engine(DATABASE_URL)
underlying_ticker = "SPY"

print("🔍 Fetching historical data from database...")

# --- FIX: use a connection context ---
with engine.connect() as conn:
	df = pd.read_sql(text(f'SELECT * FROM "{underlying_ticker}_option_backtest"'), con=conn)

# Basic cleaning
df = df.dropna().reset_index(drop=True)

# Define target and features
df["target"] = (df["returns"] > 0).astype(int)
features = [
    "net_call_volume", "net_put_volume",
    "net_call_gamma", "net_put_gamma",
    "mean_call_ivol", "mean_put_vol",
    "total_gamma", "total_volume",
    "mean_ivol", "call_put_gamma_imbalance",
    "gamma_times_volume", "ivol_adjusted_net_gamma",
    "month", "day"
]

X = df[features]
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train Random Forest
print("🌲 Training RandomForest model...")
model = RandomForestClassifier(n_estimators=500, max_depth=8, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Model trained. Test accuracy: {acc:.4f}")

# Save model
joblib.dump(model, MODEL_PATH)
print(f"💾 Model saved to {MODEL_PATH}")
