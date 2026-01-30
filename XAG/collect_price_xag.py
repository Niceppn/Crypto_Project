#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import websocket
import json
import csv
import os
import time
from datetime import datetime

# =========================
# Configuration
# =========================
SYMBOL = "btcusdc"
SOCKET = f"wss://demo-fstream.binance.com/ws/{SYMBOL}@aggTrade"
FILENAME = "btcusdc_training_data.csv"
RECONNECT_DELAY = 5  # seconds

# =========================
# CSV Helpers
# =========================
def init_csv():
    """Create CSV file with header if not exists"""
    if not os.path.exists(FILENAME):
        with open(FILENAME, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp_ms",
                "readable_time",
                "price",
                "quantity",
                "side",
                "is_maker"
            ])
        print(f"CSV file created: {FILENAME}", flush=True)

def save_to_csv(row):
    """Append one row to CSV"""
    with open(FILENAME, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)

# =========================
# WebSocket Callbacks
# =========================
def on_message(ws, message):
    try:
        data = json.loads(message)

        timestamp = data["T"]                 # trade time (ms)
        price = float(data["p"])              # price
        qty = float(data["q"])                # quantity
        is_maker = data["m"]                  # maker flag

        side = "SELL" if is_maker else "BUY"
        readable_time = datetime.fromtimestamp(
            timestamp / 1000
        ).strftime("%Y-%m-%d %H:%M:%S.%f")

        print(f"[{side}] Price={price} Qty={qty}", flush=True)

        row = [
            timestamp,
            readable_time,
            price,
            qty,
            side,
            is_maker
        ]
        save_to_csv(row)

    except Exception as e:
        print(f"Message error: {e}", flush=True)

def on_error(ws, error):
    print(f"WebSocket error: {error}", flush=True)

def on_close(ws, close_status_code, close_msg):
    print("WebSocket closed. Reconnecting...", flush=True)

def on_open(ws):
    print("Connected to Binance. Collecting data...", flush=True)

# =========================
# Main Loop (Auto Reconnect)
# =========================
if __name__ == "__main__":
    init_csv()

    while True:
        try:
            ws = websocket.WebSocketApp(
                SOCKET,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            ws.run_forever(ping_interval=20, ping_timeout=10)

        except Exception as e:
            print(f"Fatal error: {e}", flush=True)

        print(f"Reconnect in {RECONNECT_DELAY}s...", flush=True)
        time.sleep(RECONNECT_DELAY)