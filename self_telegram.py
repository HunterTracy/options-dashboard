# -*- coding: utf-8 -*-
"""
Telegram message sender for automated trading notifications
"""

import os
import requests

BOT_TOKEN = os.getenv("TG_BOT_TOKEN")
CHAT_ID = os.getenv("TG_CHAT_ID")

def send_telegram_message(message, subject=None):
    """Send a formatted message to your Telegram chat."""
    if not BOT_TOKEN or not CHAT_ID:
        raise EnvironmentError("Missing TG_BOT_TOKEN or TG_CHAT_ID environment variables.")
    
    header = f"📈 *{subject or 'Trade Notification'}*\n\n"
    text = header + message
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": text, "parse_mode": "Markdown"}
    try:
        response = requests.post(url, json=payload, timeout=15)
        response.raise_for_status()
        print("✅ Telegram message sent successfully.")
    except Exception as e:
        print(f"❌ Failed to send Telegram message: {e}")
