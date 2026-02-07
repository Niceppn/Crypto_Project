#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Crypto Price Collector V2 - Multi-Stream Edition
Collects: aggTrade + Order Book (depth) + Funding Rate (markPrice)
Uses REST API snapshot for Order Book initialization

Usage:
    python collect_price_v2.py --bot-id 1 --symbol btcusdc --socket-type demo
"""

import json
import csv
import os
import sys
import time
import argparse
import ssl
import requests
import threading
from datetime import datetime
from collections import defaultdict

try:
    from websocket import WebSocketApp
except ImportError:
    import websocket
    WebSocketApp = websocket.WebSocketApp

# =========================
# Configuration
# =========================
CSV_DIR = os.path.dirname(os.path.abspath(__file__))
RECONNECT_DELAY = 5

# CSV Headers
CSV_HEADERS = [
    'timestamp', 'readable_time', 'symbol',
    'open', 'high', 'low', 'close',
    'buy_volume', 'sell_volume', 'total_volume', 'net_flow',
    'buy_count', 'sell_count', 'trade_count',
    'best_bid', 'best_ask', 'bid_qty', 'ask_qty',
    'spread', 'book_imbalance', 'funding_rate'
]

# WebSocket endpoints (combined streams)
SOCKET_TYPES = {
    "spot": {
        "ws": "wss://stream.binance.com:9443/stream?streams={symbol}@aggTrade/{symbol}@depth@500ms/{symbol}@markPrice",
        "rest": "https://api.binance.com/api/v3/depth?symbol={symbol_upper}&limit=20"
    },
    "future": {
        "ws": "wss://fstream.binance.com/stream?streams={symbol}@aggTrade/{symbol}@depth@500ms/{symbol}@markPrice",
        "rest": "https://fapi.binance.com/fapi/v1/depth?symbol={symbol_upper}&limit=20"
    },
    "demo": {
        "ws": "wss://demo-fstream.binance.com/stream?streams={symbol}@aggTrade/{symbol}@depth@500ms/{symbol}@markPrice",
        "rest": "https://demo-fapi.binance.com/fapi/v1/depth?symbol={symbol_upper}&limit=20"
    }
}

# =========================
# Logging
# =========================
def log(level, message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] [{level}] {message}", flush=True)

# =========================
# Multi-Stream Collector
# =========================
class MultiStreamCollector:
    def __init__(self, bot_id, symbol, socket_type, batch_size=50):
        self.bot_id = bot_id
        self.symbol = symbol.lower()
        self.symbol_upper = symbol.upper()
        self.socket_type = socket_type
        
        # URLs
        config = SOCKET_TYPES[socket_type]
        self.ws_url = config["ws"].format(symbol=self.symbol, symbol_upper=self.symbol_upper)
        self.rest_url = config["rest"].format(symbol=self.symbol, symbol_upper=self.symbol_upper)
        
        self.ws = None
        self.trade_count = 0
        self.batch_size = batch_size
        self.trade_buffer = []
        self.last_flush_time = time.time()
        
        # CSV file setup
        today = datetime.now().strftime('%Y%m%d')
        self.csv_filename = os.path.join(CSV_DIR, f"{self.symbol_upper}_{socket_type}_{today}.csv")
        self.csv_file = None
        self.csv_writer = None
        
        # Order Book (will be initialized from REST API)
        self.order_book = {
            'bids': {},  # price -> quantity
            'asks': {}   # price -> quantity
        }
        self.order_book_initialized = False
        self.depth_last_update_id = 0
        
        # Funding Rate
        self.funding_rate = 0.0
        
        # Aggregation per second
        self.current_second = None
        self.second_data = self._empty_second_data()
        
    def _empty_second_data(self):
        """Empty template for per-second aggregation"""
        return {
            'trades': [],
            'buy_volume': 0.0,
            'sell_volume': 0.0,
            'buy_count': 0,
            'sell_count': 0,
            'open_price': None,
            'high_price': None,
            'low_price': None,
            'close_price': None,
        }
    
    def initialize_order_book(self):
        """Fetch Order Book snapshot from REST API"""
        log("INFO", f"Fetching Order Book snapshot from REST API...")
        log("INFO", f"URL: {self.rest_url}")
        
        try:
            response = requests.get(self.rest_url, timeout=10)
            data = response.json()
            
            if 'bids' not in data or 'asks' not in data:
                log("ERROR", f"Invalid depth response: {data}")
                return False
            
            # Clear and initialize order book
            self.order_book['bids'].clear()
            self.order_book['asks'].clear()
            
            for bid in data['bids']:
                price = float(bid[0])
                qty = float(bid[1])
                if qty > 0:
                    self.order_book['bids'][price] = qty
                    
            for ask in data['asks']:
                price = float(ask[0])
                qty = float(ask[1])
                if qty > 0:
                    self.order_book['asks'][price] = qty
            
            # Save lastUpdateId for sync
            self.depth_last_update_id = data.get('lastUpdateId', 0)
            self.order_book_initialized = True
            
            # Log snapshot info
            best_bid = max(self.order_book['bids'].keys()) if self.order_book['bids'] else 0
            best_ask = min(self.order_book['asks'].keys()) if self.order_book['asks'] else 0
            
            log("INFO", f"Order Book initialized:")
            log("INFO", f"  - Bids: {len(self.order_book['bids'])} levels, Best Bid: {best_bid}")
            log("INFO", f"  - Asks: {len(self.order_book['asks'])} levels, Best Ask: {best_ask}")
            log("INFO", f"  - Spread: ${best_ask - best_bid:.2f}")
            log("INFO", f"  - lastUpdateId: {self.depth_last_update_id}")
            
            return True
            
        except Exception as e:
            log("ERROR", f"Failed to get Order Book snapshot: {e}")
            return False
    
    def get_best_bid_ask(self):
        """Get current best bid/ask from order book"""
        best_bid = max(self.order_book['bids'].keys()) if self.order_book['bids'] else 0
        best_ask = min(self.order_book['asks'].keys()) if self.order_book['asks'] else 0
        
        bid_qty = self.order_book['bids'].get(best_bid, 0) if best_bid else 0
        ask_qty = self.order_book['asks'].get(best_ask, 0) if best_ask else 0
        
        return best_bid, bid_qty, best_ask, ask_qty
    
    def calculate_book_imbalance(self, levels=5):
        """Calculate order book imbalance"""
        if not self.order_book['bids'] or not self.order_book['asks']:
            return 0.0
        
        # Get top N levels
        sorted_bids = sorted(self.order_book['bids'].keys(), reverse=True)[:levels]
        sorted_asks = sorted(self.order_book['asks'].keys())[:levels]
        
        bid_volume = sum(self.order_book['bids'][p] for p in sorted_bids)
        ask_volume = sum(self.order_book['asks'][p] for p in sorted_asks)
        
        total = bid_volume + ask_volume
        if total == 0:
            return 0.0
        
        # Imbalance: +1 = all bids, -1 = all asks
        return (bid_volume - ask_volume) / total
    
    def process_depth_update(self, data):
        """Process incremental depth update from WebSocket"""
        if not self.order_book_initialized:
            return
        
        # Update bids
        for bid in data.get('b', []):
            price = float(bid[0])
            qty = float(bid[1])
            if qty == 0:
                self.order_book['bids'].pop(price, None)  # Remove
            else:
                self.order_book['bids'][price] = qty  # Add/Update
        
        # Update asks
        for ask in data.get('a', []):
            price = float(ask[0])
            qty = float(ask[1])
            if qty == 0:
                self.order_book['asks'].pop(price, None)  # Remove
            else:
                self.order_book['asks'][price] = qty  # Add/Update
    
    def process_trade(self, data):
        """Process aggTrade data"""
        price = float(data['p'])
        quantity = float(data['q'])
        is_sell = data['m']  # True = seller is maker = SELL
        timestamp_ms = data['T']
        
        # Aggregate by second
        trade_second = timestamp_ms // 1000
        
        if self.current_second is None:
            self.current_second = trade_second
            
        # New second? Save previous and reset
        if trade_second != self.current_second:
            self.save_second_data()
            self.current_second = trade_second
            self.second_data = self._empty_second_data()
        
        # Accumulate trade data
        self.second_data['trades'].append({
            'price': price,
            'qty': quantity,
            'is_sell': is_sell,
            'timestamp_ms': timestamp_ms
        })
        
        if is_sell:
            self.second_data['sell_volume'] += quantity
            self.second_data['sell_count'] += 1
        else:
            self.second_data['buy_volume'] += quantity
            self.second_data['buy_count'] += 1
        
        # OHLC
        if self.second_data['open_price'] is None:
            self.second_data['open_price'] = price
        if self.second_data['high_price'] is None or price > self.second_data['high_price']:
            self.second_data['high_price'] = price
        if self.second_data['low_price'] is None or price < self.second_data['low_price']:
            self.second_data['low_price'] = price
        self.second_data['close_price'] = price
    
    def process_mark_price(self, data):
        """Process markPrice data for funding rate"""
        self.funding_rate = float(data.get('r', 0))
    
    def init_csv_file(self):
        """Initialize CSV file with headers if needed"""
        file_exists = os.path.exists(self.csv_filename)
        self.csv_file = open(self.csv_filename, 'a', newline='', buffering=1)
        self.csv_writer = csv.writer(self.csv_file)
        
        if not file_exists:
            self.csv_writer.writerow(CSV_HEADERS)
            log("INFO", f"Created new CSV file: {self.csv_filename}")
        else:
            log("INFO", f"Appending to existing CSV: {self.csv_filename}")
    
    def save_second_data(self):
        """Save aggregated second data to CSV"""
        if not self.second_data['trades']:
            return
        
        try:
            # Calculate features
            best_bid, bid_qty, best_ask, ask_qty = self.get_best_bid_ask()
            spread = best_ask - best_bid if best_bid and best_ask else 0
            book_imbalance = self.calculate_book_imbalance()
            
            total_volume = self.second_data['buy_volume'] + self.second_data['sell_volume']
            net_flow = self.second_data['buy_volume'] - self.second_data['sell_volume']
            trade_count = self.second_data['buy_count'] + self.second_data['sell_count']
            
            # Prepare CSV row
            readable_time = datetime.fromtimestamp(self.current_second).strftime("%Y-%m-%d %H:%M:%S")
            
            row = [
                self.current_second * 1000,  # timestamp_ms
                readable_time,
                self.symbol_upper,
                self.second_data['open_price'],
                self.second_data['high_price'],
                self.second_data['low_price'],
                self.second_data['close_price'],
                round(self.second_data['buy_volume'], 6),
                round(self.second_data['sell_volume'], 6),
                round(total_volume, 6),
                round(net_flow, 6),
                self.second_data['buy_count'],
                self.second_data['sell_count'],
                trade_count,
                best_bid,
                best_ask,
                round(bid_qty, 4),
                round(ask_qty, 4),
                round(spread, 2),
                round(book_imbalance, 4),
                self.funding_rate
            ]
            
            self.trade_buffer.append(row)
            self.trade_count += 1
            
            # Log periodically
            if self.trade_count % 10 == 0:
                log("INFO", f"[{readable_time}] Price={self.second_data['close_price']:.1f} | "
                    f"Vol={total_volume:.4f} | NetFlow={net_flow:+.4f} | "
                    f"Bid={best_bid:.1f} Ask={best_ask:.1f} | "
                    f"Imbalance={book_imbalance:+.3f} | "
                    f"Records={self.trade_count}")
            
            # Flush to CSV
            if len(self.trade_buffer) >= self.batch_size or time.time() - self.last_flush_time >= 5:
                self.flush_trades()
                
        except Exception as e:
            log("ERROR", f"Error saving second data: {e}")
    
    def flush_trades(self):
        """Flush trades to CSV file"""
        if not self.trade_buffer:
            return 0
        
        try:
            if not self.csv_writer:
                self.init_csv_file()
            
            for row in self.trade_buffer:
                self.csv_writer.writerow(row)
            
            self.csv_file.flush()
            batch_count = len(self.trade_buffer)
            self.trade_buffer.clear()
            self.last_flush_time = time.time()
            
            return batch_count
            
        except Exception as e:
            log("ERROR", f"CSV write error: {e}")
            self.trade_buffer.clear()
            return 0
    
    def on_message(self, ws, message):
        """Handle incoming WebSocket messages"""
        try:
            msg = json.loads(message)
            stream = msg.get('stream', '')
            data = msg.get('data', {})
            
            if '@aggTrade' in stream:
                self.process_trade(data)
            elif '@depth' in stream:
                self.process_depth_update(data)
            elif '@markPrice' in stream:
                self.process_mark_price(data)
                
        except Exception as e:
            log("ERROR", f"Message processing error: {e}")
    
    def on_error(self, ws, error):
        log("ERROR", f"WebSocket error: {error}")
    
    def on_close(self, ws, close_status_code, close_msg):
        # Save remaining data
        if self.second_data['trades']:
            self.save_second_data()
        if self.trade_buffer:
            batch_count = self.flush_trades()
            log("INFO", f"Flushed {batch_count} remaining records")
        
        if self.csv_file:
            try:
                self.csv_file.close()
            except:
                pass
            self.csv_file = None
            self.csv_writer = None
        
        log("WARNING", "WebSocket closed. Will reconnect...")
    
    def on_open(self, ws):
        log("INFO", f"Connected to {self.socket_type.upper()} WebSocket")
        log("INFO", f"Streams: aggTrade + depth@500ms + markPrice")
        log("INFO", f"Collecting data (Bot ID: {self.bot_id})...")
    
    def start(self):
        """Start collecting data"""
        log("INFO", "=" * 60)
        log("INFO", "Crypto Price Collector V2 - Multi-Stream Edition")
        log("INFO", "=" * 60)
        log("INFO", f"Symbol: {self.symbol_upper}")
        log("INFO", f"Socket Type: {self.socket_type.upper()}")
        log("INFO", f"Bot ID: {self.bot_id}")
        log("INFO", f"WebSocket URL: {self.ws_url}")
        log("INFO", "=" * 60)
        
        sslopt = {"cert_reqs": ssl.CERT_NONE}
        
        while True:
            try:
                # Step 1: Initialize Order Book from REST API
                if not self.initialize_order_book():
                    log("WARNING", f"Order Book init failed. Retrying in {RECONNECT_DELAY}s...")
                    time.sleep(RECONNECT_DELAY)
                    continue
                
                # Step 2: Connect to WebSocket
                self.ws = WebSocketApp(
                    self.ws_url,
                    on_open=self.on_open,
                    on_message=self.on_message,
                    on_error=self.on_error,
                    on_close=self.on_close,
                )
                self.ws.run_forever(
                    ping_interval=20,
                    ping_timeout=10,
                    sslopt=sslopt
                )
                
            except KeyboardInterrupt:
                log("INFO", "Bot stopped by user")
                if self.second_data['trades']:
                    self.save_second_data()
                if self.trade_buffer:
                    batch_count = self.flush_trades()
                    log("INFO", f"Flushed {batch_count} remaining records")
                if self.csv_file:
                    self.csv_file.close()
                sys.exit(0)
                
            except Exception as e:
                log("ERROR", f"Fatal error: {e}")
            
            # Reset order book for reconnect
            self.order_book_initialized = False
            log("INFO", f"Reconnecting in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)

# =========================
# Main
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Binance Multi-Stream Price Collector V2')
    parser.add_argument('--bot-id', type=int, default=1, help='Bot ID (default: 1)')
    parser.add_argument('--symbol', type=str, default='btcusdc', help='Trading symbol (default: btcusdc)')
    parser.add_argument('--socket-type', type=str, choices=['spot', 'future', 'demo'],
                        default='demo', help='Socket type (default: demo)')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for CSV writes')
    
    args = parser.parse_args()
    
    if args.socket_type not in SOCKET_TYPES:
        log("ERROR", f"Invalid socket type: {args.socket_type}")
        sys.exit(1)
    
    collector = MultiStreamCollector(args.bot_id, args.symbol, args.socket_type, args.batch_size)
    collector.start()
