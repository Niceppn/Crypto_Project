#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BTCUSDC Data Collector - aggTrade + depth + markPrice
"""

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

# 3 Streams
STREAMS = [
    f"{SYMBOL}@aggTrade",
    f"{SYMBOL}@depth@500ms",
    f"{SYMBOL}@markPrice",
]

SOCKET = f"wss://demo-fstream.binance.com/stream?streams={'/'.join(STREAMS)}"

FILENAME = "btcusdc_full_data.csv"
row_count = 0

# =========================
# Global State
# =========================
# Order Book
order_book = {
    'bids': {},
    'asks': {},
    'best_bid': 0,
    'best_ask': 0,
    'bid_volume': 0,
    'ask_volume': 0,
    'spread': 0,
    'book_imbalance': 0
}

# Mark Price & Funding
mark_data = {
    'mark_price': 0,
    'index_price': 0,
    'funding_rate': 0,
    'next_funding_time': 0
}

# Second aggregation
second_data = {
    'price': 0,
    'buy_volume': 0,
    'sell_volume': 0,
    'buy_count': 0,
    'sell_count': 0,
    'high': 0,
    'low': float('inf'),
    'vwap_sum': 0,
    'total_qty': 0,
    'last_ts': None
}

# =========================
# CSV
# =========================
def init_csv():
    global row_count
    
    if not os.path.exists(FILENAME):
        with open(FILENAME, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                # Time
                "timestamp",
                "readable_time",
                
                # Price
                "price",
                "high",
                "low",
                
                # Volume & Flow
                "buy_volume",
                "sell_volume",
                "net_flow",
                "volume_imbalance",
                
                # Trade Count
                "buy_count",
                "sell_count",
                "total_trades",
                
                # VWAP
                "vwap",
                
                # Order Book
                "best_bid",
                "best_ask",
                "spread",
                "spread_pct",
                "bid_volume",
                "ask_volume",
                "book_imbalance",
                
                # Mark Price & Funding
                "mark_price",
                "index_price",
                "funding_rate",
            ])
        print(f"✅ CSV created: {FILENAME}", flush=True)
    else:
        with open(FILENAME, 'r') as f:
            row_count = sum(1 for _ in f) - 1
        print(f"✅ CSV exists: {FILENAME} ({row_count} rows)", flush=True)

def save_second():
    global row_count, second_data, order_book, mark_data
    
    if second_data['last_ts'] is None or second_data['price'] == 0:
        return
    
    ts = second_data['last_ts']
    price = second_data['price']
    
    # Volume
    buy_vol = second_data['buy_volume']
    sell_vol = second_data['sell_volume']
    total_vol = buy_vol + sell_vol
    net_flow = buy_vol - sell_vol
    vol_imbalance = net_flow / total_vol if total_vol > 0 else 0
    
    # High/Low
    high = second_data['high'] if second_data['high'] > 0 else price
    low = second_data['low'] if second_data['low'] != float('inf') else price
    
    # VWAP
    vwap = second_data['vwap_sum'] / second_data['total_qty'] if second_data['total_qty'] > 0 else price
    
    # Spread
    spread_pct = order_book['spread'] / order_book['best_bid'] * 100 if order_book['best_bid'] > 0 else 0
    
    readable_time = datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")
    
    row = [
        # Time
        ts,
        readable_time,
        
        # Price
        round(price, 2),
        round(high, 2),
        round(low, 2),
        
        # Volume & Flow
        round(buy_vol, 6),
        round(sell_vol, 6),
        round(net_flow, 6),
        round(vol_imbalance, 4),
        
        # Trade Count
        second_data['buy_count'],
        second_data['sell_count'],
        second_data['buy_count'] + second_data['sell_count'],
        
        # VWAP
        round(vwap, 2),
        
        # Order Book
        round(order_book['best_bid'], 2),
        round(order_book['best_ask'], 2),
        round(order_book['spread'], 2),
        round(spread_pct, 4),
        round(order_book['bid_volume'], 4),
        round(order_book['ask_volume'], 4),
        round(order_book['book_imbalance'], 4),
        
        # Mark Price & Funding
        round(mark_data['mark_price'], 2),
        round(mark_data['index_price'], 2),
        mark_data['funding_rate'],
    ]
    
    with open(FILENAME, mode="a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)
    
    row_count += 1
    
    # Show funding rate nicely
    funding_pct = mark_data['funding_rate'] * 100
    funding_str = f"{funding_pct:+.4f}%" if mark_data['funding_rate'] != 0 else "N/A"
    
    print(f"\n💾 #{row_count} | {readable_time} | P=${price:.2f} | "
          f"Net={net_flow:+.4f} | Spread={spread_pct:.4f}% | "
          f"BookImb={order_book['book_imbalance']:+.2f} | "
          f"Fund={funding_str}", flush=True)

def reset_second():
    global second_data
    second_data = {
        'price': 0,
        'buy_volume': 0,
        'sell_volume': 0,
        'buy_count': 0,
        'sell_count': 0,
        'high': 0,
        'low': float('inf'),
        'vwap_sum': 0,
        'total_qty': 0,
        'last_ts': None
    }

# =========================
# Process Functions
# =========================
def process_agg_trade(data):
    """aggTrade: p, q, m, T"""
    global second_data
    
    ts = int(data['T'] / 1000)
    price = float(data['p'])
    qty = float(data['q'])
    is_maker = data['m']
    side = "SELL" if is_maker else "BUY"
    
    # New second → Save previous
    if second_data['last_ts'] is not None and ts > second_data['last_ts']:
        save_second()
        reset_second()
    
    second_data['last_ts'] = ts
    second_data['price'] = price
    second_data['vwap_sum'] += price * qty
    second_data['total_qty'] += qty
    
    # High/Low
    if second_data['high'] == 0 or price > second_data['high']:
        second_data['high'] = price
    if price < second_data['low']:
        second_data['low'] = price
    
    if is_maker:  # SELL
        second_data['sell_volume'] += qty
        second_data['sell_count'] += 1
    else:  # BUY
        second_data['buy_volume'] += qty
        second_data['buy_count'] += 1
    
    print(f"\r📥 Trade: ${price:.2f} x {qty:.4f} {side} | Rows: {row_count}", end="", flush=True)

def process_depth(data):
    """depth: b, a"""
    global order_book
    
    bids = data.get('b', [])
    asks = data.get('a', [])
    
    # Update bids
    for p, q in bids:
        price, qty = float(p), float(q)
        if qty == 0:
            order_book['bids'].pop(price, None)
        else:
            order_book['bids'][price] = qty
    
    # Update asks
    for p, q in asks:
        price, qty = float(p), float(q)
        if qty == 0:
            order_book['asks'].pop(price, None)
        else:
            order_book['asks'][price] = qty
    
    # Calculate metrics
    if order_book['bids']:
        order_book['best_bid'] = max(order_book['bids'].keys())
        order_book['bid_volume'] = sum(order_book['bids'].values())
    else:
        order_book['best_bid'] = 0
        order_book['bid_volume'] = 0
    
    if order_book['asks']:
        order_book['best_ask'] = min(order_book['asks'].keys())
        order_book['ask_volume'] = sum(order_book['asks'].values())
    else:
        order_book['best_ask'] = 0
        order_book['ask_volume'] = 0
    
    # Spread
    if order_book['best_bid'] > 0 and order_book['best_ask'] > 0:
        order_book['spread'] = order_book['best_ask'] - order_book['best_bid']
    else:
        order_book['spread'] = 0
    
    # Book imbalance (-1 to +1)
    total = order_book['bid_volume'] + order_book['ask_volume']
    if total > 0:
        order_book['book_imbalance'] = (order_book['bid_volume'] - order_book['ask_volume']) / total
    else:
        order_book['book_imbalance'] = 0

def process_mark_price(data):
    """markPrice: p, i, r, T"""
    global mark_data
    
    mark_data['mark_price'] = float(data.get('p', 0))
    mark_data['index_price'] = float(data.get('i', 0))
    mark_data['funding_rate'] = float(data.get('r', 0))
    mark_data['next_funding_time'] = int(data.get('T', 0))

# =========================
# WebSocket
# =========================
def on_message(ws, message):
    try:
        msg = json.loads(message)
        stream = msg.get('stream', '')
        data = msg.get('data', {})
        
        if '@aggTrade' in stream:
            process_agg_trade(data)
        elif '@depth' in stream:
            process_depth(data)
        elif '@markPrice' in stream:
            process_mark_price(data)
    
    except Exception as e:
        print(f"\n❌ Error: {e}", flush=True)

def on_open(ws):
    print(f"\n✅ Connected!", flush=True)
    print(f"📊 Streams: {STREAMS}", flush=True)
    print(f"💾 Saving to: {FILENAME}", flush=True)
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", flush=True)

def on_error(ws, error):
    print(f"\n❌ Error: {error}", flush=True)

def on_close(ws, code, msg):
    print(f"\n⚠️ Closed: {code} - {msg}", flush=True)

# =========================
# Main
# =========================
if __name__ == "__main__":
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", flush=True)
    print(f"🚀 BTCUSDC Data Collector", flush=True)
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", flush=True)
    print(f"📊 Symbol: {SYMBOL}", flush=True)
    print(f"📊 Streams:", flush=True)
    print(f"   • aggTrade (Trades)", flush=True)
    print(f"   • depth@500ms (Order Book)", flush=True)
    print(f"   • markPrice (Funding Rate)", flush=True)
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", flush=True)
    
    init_csv()
    
    # Auto reconnect loop
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
            print(f"\n❌ Fatal: {e}", flush=True)
        
        print(f"\n🔄 Reconnecting in 5s...", flush=True)
        time.sleep(5)