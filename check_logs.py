import os, datetime as dt, requests

log_path = os.path.expanduser("~/sp500_logs/production.log")
msg = ""

if not os.path.exists(log_path):
    msg = f"⚠️ No production log found for {dt.date.today()}."
else:
    with open(log_path) as f:
        log = f.read()
    if "✅ Script completed successfully" in log or "⚠️ No SPY option data" in log:
        msg = f"✅ Bot ran successfully on {dt.date.today()}."
    else:
        msg = f"⚠️ Bot may have failed on {dt.date.today()} — check logs."

requests.post(
    f"https://api.telegram.org/bot{os.getenv('TG_BOT_TOKEN')}/sendMessage",
    data={"chat_id": os.getenv("TG_CHAT_ID"), "text": msg}
)
