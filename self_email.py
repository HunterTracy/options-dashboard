# -*- coding: utf-8 -*-
"""
Created in 2023

@author: Quant Galore
"""
import os
import smtplib
from email.mime.text import MIMEText

EMAIL_ADDRESS = os.getenv("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

def send_message(message, subject):
    msg = MIMEText(message)
    msg["Subject"] = subject
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = EMAIL_ADDRESS
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, EMAIL_ADDRESS, msg.as_string())

if __name__ == "__main__":
    send_message("Test Email", "This is a test email")

import os, requests

def send_telegram_message(message, subject="📊 SP500 Update"):
    """
    Sends a Telegram message using bot token and chat ID from environment variables.
    """
    token = os.getenv("TG_BOT_TOKEN")
    chat_id = os.getenv("TG_CHAT_ID")
    if not token or not chat_id:
        print("⚠️ Telegram credentials not set in environment.")
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={"chat_id": chat_id, "text": f"{subject}\n{message}"}
        )
        print("✅ Telegram message sent.")
    except Exception as e:
        print(f"⚠️ Telegram send failed: {e}")
