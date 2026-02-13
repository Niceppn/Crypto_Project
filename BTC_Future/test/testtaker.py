#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot V3 FUTURES - Taker Entry (Long + Short)
Uses V3 AI Model (33 features, no funding)
3 WebSocket streams: aggTrade + depth@500ms + markPrice@1s

Taker Strategy (zero fee on FDUSD):
  - Long Entry:  Market BUY  when prob >= LONG_THRESHOLD
  - Short Entry: Market SELL when prob <= SHORT_THRESHOLD
  - Exit: Limit order at TP / Market order at SL or Time Exit

Backtest confirmed: Taker > Maker when fee = 0 (FDUSD pair)
"""

import websocket, json, datetime, sys, requests, threading, time, os
import pandas as pd
import numpy as np
import lightgbm as lgb
from collections import deque
from binance.client import Client
from binance.enums import *

# ==========================================
# 1. CONFIGURATION
# ==========================================

SYMBOL_WS = "btcfdusd"
SYMBOL_TRADE = "BTCFDUSD"

# --- Futures Settings ---
LEVERAGE = 10
MARGIN_TYPE = "ISOLATED"

# --- Model ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_FILE = os.path.join(MODEL_DIR, "v3_model_20260211_205205.txt")
META_FILE  = os.path.join(MODEL_DIR, "v3_model_20260211_205205_meta.json")

# --- TELEGRAM ---
TG_TOKEN = "8515779063:AAGMORwNT0FdPuYlQRxUx4sGh-yhYok5Wcc"
TG_CHAT_ID = "8440162744"

# --- API KEYS ---
API_KEY = "eJVrU1CjLCKTOuS3QtQXfAVlxYBFWV1JHctSEEYDo3WL5uHBb2mbks6OUNytJmT2"
SECRET_KEY = "b7EX7kRfTxGmyVi7JePsvWnt1AFWlgXGy9mhedJhtVptfquIzHqrZADSzauWKqOM"

# --- Strategy ---
LONG_THRESHOLD = 0.033          # prob >= this → LONG
SHORT_THRESHOLD = 0.008         # prob <= this → SHORT
CAPITAL_PER_TRADE = 15          # USDT per trade (before leverage)
HOLDING_TIME = 1800             # 30 min
PROFIT_TARGET_PCT = 0.0015      # 0.15% TP
STOP_LOSS_PCT = 0.01            # 1.00% SL
STATUS_REPORT_INTERVAL = 1800

# --- Taker Settings ---
MAKER_ORDER_TIMEOUT = 60        # timeout for TP limit order check

# --- Concurrent Positions (shared pool for Long + Short) ---
MAX_POSITIONS = 4
COOLDOWN_SECONDS = 180
SLOT2_COOLDOWN_SECONDS = 120
SLOT3_COOLDOWN_SECONDS = 120
SLOT4_COOLDOWN_SECONDS = 180

# ==========================================
# 2. CONNECT TO BINANCE FUTURES
# ==========================================
try:
    client = Client(API_KEY, SECRET_KEY, testnet=True)
    client.FUTURES_URL = 'https://demo-fapi.binance.com'

    # Set leverage
    try:
        client.futures_change_leverage(symbol=SYMBOL_TRADE, leverage=LEVERAGE)
        print(f"  Leverage: {LEVERAGE}x")
    except Exception as e:
        print(f"  Leverage (already set or error): {e}")

    # Set margin type
    try:
        client.futures_change_margin_type(symbol=SYMBOL_TRADE, marginType=MARGIN_TYPE)
        print(f"  Margin: {MARGIN_TYPE}")
    except Exception as e:
        if "No need to change" in str(e):
            print(f"  Margin: {MARGIN_TYPE} (already set)")
        else:
            print(f"  Margin error: {e}")

    # Check balance
    balance = client.futures_account_balance()
    fdusd_bal = next((item for item in balance if item["asset"] == "FDUSD"), None)
    usdt_bal = next((item for item in balance if item["asset"] == "USDT"), None)
    bal_asset = fdusd_bal or usdt_bal
    bal_name = "FDUSD" if fdusd_bal else "USDT"
    bal_val = float(bal_asset['balance']) if bal_asset else 0

    # Get precision from exchange info
    exchange_info = client.futures_exchange_info()
    QTY_STEP = 0.001
    PRICE_TICK = 0.01
    QTY_PRECISION = 3
    PRICE_PRECISION = 2
    for s in exchange_info['symbols']:
        if s['symbol'] == SYMBOL_TRADE:
            for f in s['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    QTY_STEP = float(f['stepSize'])
                    QTY_PRECISION = max(0, len(f['stepSize'].rstrip('0').split('.')[-1]))
                elif f['filterType'] == 'PRICE_FILTER':
                    PRICE_TICK = float(f['tickSize'])
                    PRICE_PRECISION = max(0, len(f['tickSize'].rstrip('0').split('.')[-1]))
            break

    print(f"\n  Connected to Binance Futures Demo")
    print(f"  {bal_name}: {bal_val:.2f}")
    print(f"  {SYMBOL_TRADE}: qty_step={QTY_STEP} ({QTY_PRECISION} dp) | price_tick={PRICE_TICK} ({PRICE_PRECISION} dp)")

except Exception as e:
    print(f"  Connect failed: {e}")
    sys.exit()

# ==========================================
# GLOBAL VARS
# ==========================================
IS_RUNNING = True
HAS_EXISTING_POSITION = False
stats = {'win': 0, 'loss': 0, 'breakeven': 0}
total_pnl_cash = 0.0
active_orders = []      # each has 'direction': 'LONG' or 'SHORT'
pending_orders = []     # not used for taker (fills instantly), kept for compatibility
timeout_history = []
loss_history = []
last_trade_time_per_slot = [0] * MAX_POSITIONS
last_status_report_time = time.time()
last_update_id = 0
mark_price = 0.0

# --- V3 Multi-Stream Data ---
BUFFER_SIZE = 60
buffer = deque(maxlen=BUFFER_SIZE)

current_sec = {
    'ts': None,
    'open': 0.0, 'high': 0.0, 'low': 999999.0, 'close': 0.0,
    'buy_volume': 0.0, 'sell_volume': 0.0, 'total_volume': 0.0,
    'net_flow': 0.0,
    'buy_count': 0, 'sell_count': 0, 'trade_count': 0,
}

order_book = {
    'best_bid': 0.0, 'best_ask': 0.0,
    'bid_qty': 0.0, 'ask_qty': 0.0,
    'spread': 0.0, 'book_imbalance': 0.0,
}

# 33 features matching V3 model (no funding)
FEATURE_COLS = [
    'candle_body', 'candle_range', 'upper_shadow', 'lower_shadow',
    'price_change_1', 'price_change_5',
    'dist_ma15', 'dist_ma30',
    'std_5', 'std_15',
    'rsi_14',
    'momentum_5', 'momentum_15',
    'trade_count',
    'volume_ratio', 'net_flow_ma5', 'net_flow_ma15', 'net_flow_diff',
    'volume_ma5', 'volume_spike', 'cum_flow_10',
    'total_volume',
    'spread', 'spread_ma5', 'spread_change',
    'book_imbalance', 'imbalance_ma5', 'imbalance_ma15', 'imbalance_change',
    'bid_ask_ratio',
    'imbalance_x_flow', 'spread_x_volume', 'price_vol_corr',
]

# ==========================================
# 3. LOAD V3 MODEL
# ==========================================
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    with open(META_FILE, 'r') as f:
        model_meta = json.load(f)

    meta_features = model_meta.get('features', [])
    if meta_features != FEATURE_COLS:
        print(f"  Feature mismatch! Meta: {len(meta_features)}, Code: {len(FEATURE_COLS)}")

    print(f"  Loaded V3 Model: {os.path.basename(MODEL_FILE)}")
    print(f"   Features: {model_meta.get('n_features', '?')}")
    print(f"   Precision: {model_meta.get('precision', 0)*100:.1f}%")
except Exception as e:
    print(f"  Model Load Failed: {e}")
    sys.exit()

# ==========================================
# 4. TELEGRAM FUNCTIONS
# ==========================================
def send_tg_msg(msg):
    try:
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage",
            data={'chat_id': TG_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=5
        )
    except: pass

def send_status_report():
    global last_status_report_time
    current_time = time.time()
    if current_time - last_status_report_time >= STATUS_REPORT_INTERVAL:
        total_trades = stats['win'] + stats['loss'] + stats['breakeven']
        win_rate = (stats['win'] / total_trades * 100) if total_trades > 0 else 0
        longs = sum(1 for o in active_orders if o['direction'] == 'LONG')
        shorts = sum(1 for o in active_orders if o['direction'] == 'SHORT')
        send_tg_msg(
            f"<b>AUTO REPORT (V3 FUTURES)</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"{datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
            f"Win: {stats['win']} | Loss: {stats['loss']}\n"
            f"Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"Active: {len(active_orders)} (L:{longs} S:{shorts})\n"
            f"Buffer: {len(buffer)}/{BUFFER_SIZE}"
        )
        last_status_report_time = current_time

def get_telegram_updates():
    global last_update_id
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
            params={'offset': last_update_id + 1, 'timeout': 5},
            timeout=10
        )
        data = response.json()
        if data.get('ok') and data.get('result'):
            return data['result']
    except: pass
    return []

def handle_telegram_commands():
    global last_update_id, IS_RUNNING, active_orders, pending_orders, stats, total_pnl_cash
    global HOLDING_TIME, STOP_LOSS_PCT, PROFIT_TARGET_PCT, LONG_THRESHOLD, SHORT_THRESHOLD
    global MAKER_ORDER_TIMEOUT, HAS_EXISTING_POSITION, CAPITAL_PER_TRADE, LEVERAGE

    updates = get_telegram_updates()
    for update in updates:
        if 'update_id' not in update: continue
        last_update_id = update['update_id']
        if 'message' not in update or not update['message']: continue
        if 'text' not in update['message'] or not update['message']['text']: continue

        message = update['message']['text'].strip()

        if message == '/status':
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            win_rate = (stats['win'] / total_trades * 100) if total_trades > 0 else 0
            try:
                bal = client.futures_account_balance()
                fdusd_b = next((item for item in bal if item["asset"] == "FDUSD"), None)
                usdt_b = next((item for item in bal if item["asset"] == "USDT"), None)
                b = fdusd_b or usdt_b
                b_name = "FDUSD" if fdusd_b else "USDT"
                balance_text = f"Balance: ${float(b['balance']):.2f} {b_name}" if b else "Balance: N/A"
            except:
                balance_text = "Balance: N/A"

            longs = sum(1 for o in active_orders if o['direction'] == 'LONG')
            shorts = sum(1 for o in active_orders if o['direction'] == 'SHORT')

            send_tg_msg(
                f"<b>BOT V3 FUTURES STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"Status: {'RUNNING' if IS_RUNNING else 'STOPPED'}\n"
                f"Market: <b>FUTURES {LEVERAGE}x {MARGIN_TYPE}</b>\n"
                f"Model: V3 (33 features)\n"
                f"Mark: ${mark_price:.2f}\n"
                f"{balance_text}\n"
                f"Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"Win: {stats['win']} | Loss: {stats['loss']} | BE: {stats['breakeven']}\n"
                f"Win Rate: <b>{win_rate:.1f}%</b>\n"
                f"Active: {len(active_orders)} (L:{longs} S:{shorts})\n"
                f"Buffer: {len(buffer)}/{BUFFER_SIZE}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<b>SETTINGS:</b>\n"
                f"Long: >={LONG_THRESHOLD*100:.1f}% | Short: <={SHORT_THRESHOLD*100:.1f}%\n"
                f"TP: {PROFIT_TARGET_PCT*100:.3f}% | SL: {STOP_LOSS_PCT*100:.2f}%\n"
                f"Capital: ${CAPITAL_PER_TRADE} x {LEVERAGE}x"
            )

        elif message == '/position':
            try:
                positions = client.futures_position_information(symbol=SYMBOL_TRADE)
                pos_lines = []
                for pos in positions:
                    amt = float(pos['positionAmt'])
                    if amt != 0:
                        entry = float(pos['entryPrice'])
                        pnl = float(pos['unRealizedProfit'])
                        side = "LONG" if amt > 0 else "SHORT"
                        pos_lines.append(
                            f"  {side}: {abs(amt)} BTC @ ${entry:.2f}\n"
                            f"  PNL: ${pnl:.4f}"
                        )
                if not pos_lines:
                    pos_lines.append("  No open positions")

                open_orders = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
                buy_orders = [o for o in open_orders if o['side'] == 'BUY']
                sell_orders = [o for o in open_orders if o['side'] == 'SELL']

                longs = sum(1 for o in active_orders if o['direction'] == 'LONG')
                shorts = sum(1 for o in active_orders if o['direction'] == 'SHORT')

                send_tg_msg(
                    f"<b>FUTURES POSITION</b>\n━━━━━━━━━━━━━━━━\n"
                    f"{''.join(pos_lines)}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"Bot Active: {len(active_orders)} (L:{longs} S:{shorts})\n"
                    f"Exchange Orders: BUY={len(buy_orders)} SELL={len(sell_orders)}"
                )
            except Exception as e:
                send_tg_msg(f"Error: {e}")

        elif message.startswith('/set_long'):
            try:
                val = float(message.split()[1])
                LONG_THRESHOLD = val / 100 if val > 1 else val
                send_tg_msg(f"Long threshold: <b>{LONG_THRESHOLD*100:.2f}%</b>")
            except: send_tg_msg("Usage: /set_long 3.3")

        elif message.startswith('/set_short'):
            try:
                val = float(message.split()[1])
                SHORT_THRESHOLD = val / 100 if val > 1 else val
                send_tg_msg(f"Short threshold: <b>{SHORT_THRESHOLD*100:.2f}%</b>")
            except: send_tg_msg("Usage: /set_short 0.8")

        elif message.startswith('/set_leverage'):
            try:
                new_lev = int(message.split()[1])
                client.futures_change_leverage(symbol=SYMBOL_TRADE, leverage=new_lev)
                LEVERAGE = new_lev
                send_tg_msg(f"Leverage: <b>{LEVERAGE}x</b>")
            except Exception as e: send_tg_msg(f"Error: {e}")

        elif message.startswith('/set_conf'):
            try:
                val = float(message.split()[1])
                LONG_THRESHOLD = val / 100 if val > 1 else val
                send_tg_msg(f"Long threshold: <b>{LONG_THRESHOLD*100:.2f}%</b>")
            except: send_tg_msg("Usage: /set_conf 3.3")

        elif message.startswith('/set_cap'):
            try:
                CAPITAL_PER_TRADE = float(message.split()[1])
                send_tg_msg(f"Capital/Trade: <b>${CAPITAL_PER_TRADE:.0f}</b>")
            except: send_tg_msg("Usage: /set_cap 100")

        elif message.startswith('/set_sl'):
            try:
                STOP_LOSS_PCT = float(message.split()[1]) / 100
                send_tg_msg(f"SL: <b>{STOP_LOSS_PCT*100:.2f}%</b>")
            except: send_tg_msg("Usage: /set_sl 0.7")

        elif message.startswith('/set_tp'):
            try:
                PROFIT_TARGET_PCT = float(message.split()[1]) / 100
                send_tg_msg(f"TP: <b>{PROFIT_TARGET_PCT*100:.3f}%</b>")
            except: send_tg_msg("Usage: /set_tp 0.15")

        elif message == '/stop':
            IS_RUNNING = False
            send_tg_msg("<b>BOT STOPPED</b>")

        elif message == '/start':
            IS_RUNNING = True
            send_tg_msg("<b>BOT STARTED</b>")

        elif message == '/reset':
            HAS_EXISTING_POSITION = False
            active_orders.clear(); pending_orders.clear()
            send_tg_msg("<b>BOT RESET</b>\nCleared internal state")

        elif message == '/closeall':
            try:
                # Cancel all open orders
                open_ords = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
                for o in open_ords:
                    cancel_order_futures(SYMBOL_TRADE, o['orderId'])

                # Close all positions (Long + Short)
                positions = client.futures_position_information(symbol=SYMBOL_TRADE)
                closed = 0
                for pos in positions:
                    amt = float(pos['positionAmt'])
                    if amt > 0:  # Long → SELL to close
                        client.futures_create_order(
                            symbol=SYMBOL_TRADE, side='SELL',
                            type='MARKET', quantity=abs(amt)
                        )
                        closed += 1
                    elif amt < 0:  # Short → BUY to close
                        client.futures_create_order(
                            symbol=SYMBOL_TRADE, side='BUY',
                            type='MARKET', quantity=abs(amt)
                        )
                        closed += 1

                active_orders.clear(); pending_orders.clear()
                HAS_EXISTING_POSITION = False
                send_tg_msg(
                    f"<b>CLOSE ALL</b>\n"
                    f"Cancelled {len(open_ords)} orders\n"
                    f"Closed {closed} positions"
                )
            except Exception as e:
                send_tg_msg(f"Error: {e}")

        elif message == '/help':
            send_tg_msg(
                f"<b>BOT V3 FUTURES COMMANDS</b>\n━━━━━━━━━━━━━━━━\n"
                f"/status - Bot status\n/position - Positions\n"
                f"/stop /start - Stop/Start\n/closeall - Close all\n/reset - Reset\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"/set_long [%] - Long threshold\n"
                f"/set_short [%] - Short threshold\n"
                f"/set_leverage [n] - Leverage\n"
                f"/set_cap [$] | /set_sl [%] | /set_tp [%]"
            )

def telegram_command_loop():
    while True:
        try: handle_telegram_commands()
        except Exception as e: print(f"  TG Error: {e}")
        time.sleep(5)

# ==========================================
# 5. SYNC EXISTING POSITIONS
# ==========================================
def sync_existing_positions():
    """Check for existing futures positions"""
    global HAS_EXISTING_POSITION
    try:
        positions = client.futures_position_information(symbol=SYMBOL_TRADE)
        has_pos = False
        for pos in positions:
            amt = float(pos['positionAmt'])
            if amt != 0:
                entry = float(pos['entryPrice'])
                side = "LONG" if amt > 0 else "SHORT"
                print(f"\n  SYNC: {side} {abs(amt)} BTC @ ${entry:.2f}")
                has_pos = True

        open_orders = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
        print(f"  SYNC: Open Orders={len(open_orders)}")

        if has_pos:
            HAS_EXISTING_POSITION = True
            send_tg_msg(
                f"<b>EXISTING POSITION</b>\n━━━━━━━━━━━━━━━━\n"
                f"Will not open new positions\nUse /closeall then /reset"
            )
            return True
        else:
            HAS_EXISTING_POSITION = False
            if len(open_orders) > 0:
                for o in open_orders:
                    cancel_order_futures(SYMBOL_TRADE, o['orderId'])
                send_tg_msg(f"Cancelled {len(open_orders)} stale orders")
            print(f"  No existing position - ready")
            return False
    except Exception as e:
        print(f"  Sync Error: {e}"); return False

def check_position_before_trade():
    """Check net position size doesn't exceed limit"""
    try:
        positions = client.futures_position_information(symbol=SYMBOL_TRADE)
        total_abs = 0
        for pos in positions:
            total_abs += abs(float(pos['positionAmt']))
        current_price = buffer[-1]['close'] if len(buffer) > 0 else 100000
        max_allowed = MAX_POSITIONS * (CAPITAL_PER_TRADE * LEVERAGE / current_price) * 1.5
        if total_abs > max_allowed:
            print(f"\n  Position limit: {total_abs:.6f} > {max_allowed:.6f}")
            return False
        return True
    except: return True

# ==========================================
# 6. TRADING FUNCTIONS (FUTURES - TAKER ENTRY)
# ==========================================
def place_market_order(symbol, side, quantity):
    """Market order (taker) for entry"""
    try:
        return client.futures_create_order(
            symbol=symbol, side=side, type='MARKET',
            quantity=round(quantity, QTY_PRECISION)
        )
    except Exception as e:
        print(f"  Market {side} Error: {e}"); return None

def place_limit_order(symbol, side, quantity, price):
    """Limit order for TP exit"""
    try:
        return client.futures_create_order(
            symbol=symbol, side=side, type='LIMIT',
            quantity=round(quantity, QTY_PRECISION),
            price=str(round(price, PRICE_PRECISION)),
            timeInForce='GTC'
        )
    except Exception as e:
        print(f"  Limit {side} Error: {e}"); return None

def cancel_order_futures(symbol, order_id):
    """Cancel a futures order"""
    try:
        client.futures_cancel_order(symbol=symbol, orderId=order_id); return True
    except Exception as e:
        if "Unknown order" in str(e): return True
        print(f"  Cancel Error: {e}"); return False

def close_position_futures(symbol, quantity, direction, reason):
    """Market close: SELL for long, BUY for short"""
    try:
        close_side = 'SELL' if direction == 'LONG' else 'BUY'
        # Verify position still exists
        positions = client.futures_position_information(symbol=symbol)
        for pos in positions:
            amt = float(pos['positionAmt'])
            if direction == 'LONG' and amt > 0:
                close_qty = min(quantity, amt)
                client.futures_create_order(
                    symbol=symbol, side=close_side,
                    type='MARKET', quantity=round(close_qty, QTY_PRECISION)
                )
                return True
            elif direction == 'SHORT' and amt < 0:
                close_qty = min(quantity, abs(amt))
                client.futures_create_order(
                    symbol=symbol, side=close_side,
                    type='MARKET', quantity=round(close_qty, QTY_PRECISION)
                )
                return True
        return True  # position already closed
    except Exception as e:
        print(f"  Close Error: {e}"); return False

last_check_active_ts = 0

def check_active_orders(current_price, current_ts):
    """Check TP/SL/Time exit for active orders (Long + Short)"""
    global stats, total_pnl_cash, loss_history, HAS_EXISTING_POSITION, last_check_active_ts

    if not active_orders: return
    # Throttle API checks to every 3 seconds
    if current_ts - last_check_active_ts < 3: return
    last_check_active_ts = current_ts

    for order in active_orders[:]:
        direction = order['direction']
        is_exit = False; reason = ""

        if direction == 'LONG':
            if current_price >= order['take_profit']:
                is_exit = True; reason = "TP WIN"
            elif current_price <= order['stop_loss']:
                is_exit = True; reason = "STOP LOSS"
            elif current_ts >= order['exit_ts']:
                is_exit = True; reason = "TIME EXIT"
        else:  # SHORT
            if current_price <= order['take_profit']:
                is_exit = True; reason = "TP WIN"
            elif current_price >= order['stop_loss']:
                is_exit = True; reason = "STOP LOSS"
            elif current_ts >= order['exit_ts']:
                is_exit = True; reason = "TIME EXIT"

        if not is_exit:
            continue

        # Handle exit
        if "TP" in reason and order.get('tp_order_id'):
            # Check if TP limit order actually filled
            try:
                tp_status = client.futures_get_order(
                    symbol=SYMBOL_TRADE, orderId=order['tp_order_id']
                )
                status = tp_status.get('status', '')
                if status == 'FILLED':
                    pass  # good, already closed
                elif status in ('NEW', 'PARTIALLY_FILLED'):
                    continue  # wait for fill
                else:
                    # Cancelled/expired → market close
                    close_position_futures(SYMBOL_TRADE, order['quantity'], direction, reason)
            except:
                continue
        else:
            # SL/TIME EXIT: cancel TP order, market close
            if order.get('tp_order_id'):
                cancel_order_futures(SYMBOL_TRADE, order['tp_order_id'])
            close_position_futures(SYMBOL_TRADE, order['quantity'], direction, reason)

        # Calculate PNL
        if direction == 'LONG':
            profit = (current_price - order['entry']) * order['quantity']
        else:  # SHORT
            profit = (order['entry'] - current_price) * order['quantity']

        total_pnl_cash += profit
        if profit > 0:
            stats['win'] += 1
        elif profit < 0:
            stats['loss'] += 1
            loss_history.append({
                'time': datetime.datetime.now().strftime('%H:%M:%S'),
                'entry': order['entry'], 'exit': current_price,
                'pnl': profit, 'reason': reason, 'direction': direction
            })
        else:
            stats['breakeven'] += 1

        longs_after = sum(1 for o in active_orders if o != order and o['direction'] == 'LONG')
        shorts_after = sum(1 for o in active_orders if o != order and o['direction'] == 'SHORT')

        send_tg_msg(
            f"{'WIN' if profit >= 0 else 'LOSS'} <b>CLOSED {direction}</b>\n━━━━━━━━━━━━━━━━\n"
            f"Entry: ${order['entry']:.2f}\nExit: ${current_price:.2f}\n"
            f"PNL: <b>${profit:.4f}</b>\n{reason}\n"
            f"Active: {len(active_orders)-1} (L:{longs_after} S:{shorts_after})"
        )
        active_orders.remove(order)
        if len(active_orders) == 0 and len(pending_orders) == 0:
            HAS_EXISTING_POSITION = False

# ==========================================
# 7. V3 FEATURE ENGINEERING (33 features)
# ==========================================
def compute_features(data_list):
    df = pd.DataFrame(data_list)
    if len(df) < 30:
        return None

    i = len(df) - 1
    close = df['close']; high = df['high']; low = df['low']; opn = df['open']

    candle_body = close.iloc[i] - opn.iloc[i]
    candle_range = high.iloc[i] - low.iloc[i]
    upper_shadow = high.iloc[i] - max(opn.iloc[i], close.iloc[i])
    lower_shadow = min(opn.iloc[i], close.iloc[i]) - low.iloc[i]

    price_change_1 = (close.iloc[i] / close.iloc[i-1] - 1) * 100 if close.iloc[i-1] != 0 else 0
    price_change_5 = (close.iloc[i] / close.iloc[i-5] - 1) * 100 if i >= 5 and close.iloc[i-5] != 0 else 0

    ma15 = close.rolling(15).mean().iloc[i]
    ma30 = close.rolling(30).mean().iloc[i]
    dist_ma15 = close.iloc[i] - ma15
    dist_ma30 = close.iloc[i] - ma30

    std_5 = close.rolling(5).std().iloc[i]
    std_15 = close.rolling(15).std().iloc[i]

    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[i]
    loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[i]
    rs = gain / (loss_val + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))

    momentum_5 = close.iloc[i] - close.iloc[i-5] if i >= 5 else 0
    momentum_15 = close.iloc[i] - close.iloc[i-15] if i >= 15 else 0
    trade_count_val = df['trade_count'].iloc[i]

    buy_vol = df['buy_volume']; sell_vol = df['sell_volume']
    total_vol = df['total_volume']; net_f = df['net_flow']

    volume_ratio = buy_vol.iloc[i] / (sell_vol.iloc[i] + 1e-10)
    net_flow_ma5 = net_f.rolling(5).mean().iloc[i]
    net_flow_ma15 = net_f.rolling(15).mean().iloc[i]
    net_flow_diff = net_f.diff().iloc[i]
    volume_ma5 = total_vol.rolling(5).mean().iloc[i]
    volume_ma15 = total_vol.rolling(15).mean().iloc[i]
    volume_spike = total_vol.iloc[i] / (volume_ma15 + 1e-10)
    cum_flow_10 = net_f.rolling(10).sum().iloc[i]
    total_volume_val = total_vol.iloc[i]

    spread_series = df['spread']; imb_series = df['book_imbalance']
    spread_val = spread_series.iloc[i]
    spread_ma5 = spread_series.rolling(5).mean().iloc[i]
    spread_change = spread_series.diff().iloc[i]
    book_imbalance_val = imb_series.iloc[i]
    imbalance_ma5 = imb_series.rolling(5).mean().iloc[i]
    imbalance_ma15 = imb_series.rolling(15).mean().iloc[i]
    imbalance_change = imb_series.diff().iloc[i]
    bid_ask_ratio = df['bid_qty'].iloc[i] / (df['ask_qty'].iloc[i] + 1e-10)

    imbalance_x_flow = book_imbalance_val * net_f.iloc[i]
    spread_x_volume = spread_val * total_vol.iloc[i]
    price_vol_corr = close.rolling(10).corr(total_vol).iloc[i]

    feat = {
        'candle_body': candle_body, 'candle_range': candle_range,
        'upper_shadow': upper_shadow, 'lower_shadow': lower_shadow,
        'price_change_1': price_change_1, 'price_change_5': price_change_5,
        'dist_ma15': dist_ma15, 'dist_ma30': dist_ma30,
        'std_5': std_5, 'std_15': std_15, 'rsi_14': rsi_14,
        'momentum_5': momentum_5, 'momentum_15': momentum_15,
        'trade_count': trade_count_val,
        'volume_ratio': volume_ratio, 'net_flow_ma5': net_flow_ma5,
        'net_flow_ma15': net_flow_ma15, 'net_flow_diff': net_flow_diff,
        'volume_ma5': volume_ma5, 'volume_spike': volume_spike,
        'cum_flow_10': cum_flow_10, 'total_volume': total_volume_val,
        'spread': spread_val, 'spread_ma5': spread_ma5,
        'spread_change': spread_change,
        'book_imbalance': book_imbalance_val, 'imbalance_ma5': imbalance_ma5,
        'imbalance_ma15': imbalance_ma15, 'imbalance_change': imbalance_change,
        'bid_ask_ratio': bid_ask_ratio,
        'imbalance_x_flow': imbalance_x_flow, 'spread_x_volume': spread_x_volume,
        'price_vol_corr': price_vol_corr,
    }

    for k, v in feat.items():
        if pd.isna(v): feat[k] = 0.0

    return feat

# ==========================================
# 8. PREDICTION & SLOT MANAGEMENT
# ==========================================
def get_available_slot(current_ts):
    total_open = len(active_orders) + len(pending_orders)
    if total_open >= MAX_POSITIONS: return None
    if total_open == 0: return 0
    if total_open == 1 and len(active_orders) == 1:
        entry_time = active_orders[0].get('entry_ts', current_ts)
        if (current_ts - entry_time) >= SLOT2_COOLDOWN_SECONDS: return 1
    if total_open == 2 and len(active_orders) == 2:
        slot2 = next((o for o in active_orders if o['slot'] == 1), None)
        if slot2 and (current_ts - slot2.get('entry_ts', current_ts)) >= SLOT3_COOLDOWN_SECONDS: return 2
    if total_open == 3 and len(active_orders) == 3:
        slot3 = next((o for o in active_orders if o['slot'] == 2), None)
        if slot3 and (current_ts - slot3.get('entry_ts', current_ts)) >= SLOT4_COOLDOWN_SECONDS: return 3
    return None

def predict_and_trade(current_ts):
    """V3 prediction — Futures Taker (Long + Short)"""
    global last_trade_time_per_slot, active_orders, HAS_EXISTING_POSITION

    if not IS_RUNNING: return
    if HAS_EXISTING_POSITION and len(active_orders) == 0 and len(pending_orders) == 0: return
    if current_ts % 30 == 0:
        if not check_position_before_trade(): return

    available_slot = get_available_slot(current_ts)
    if available_slot is None: return

    if len(buffer) < 30: return

    feat = compute_features(list(buffer))
    if feat is None: return

    feat_df = pd.DataFrame([feat])[FEATURE_COLS]
    prob = model.predict(feat_df)[0]

    last_price = buffer[-1]['close']
    longs = sum(1 for o in active_orders if o['direction'] == 'LONG')
    shorts = sum(1 for o in active_orders if o['direction'] == 'SHORT')

    print(f"\rFUT V3 | ${last_price:.1f} | AI: {prob*100:.2f}% | "
          f"OB: bid={order_book['best_bid']:.1f} ask={order_book['best_ask']:.1f} "
          f"imb={order_book['book_imbalance']:.3f} | "
          f"L:{longs} S:{shorts} Buf:{len(buffer)}", end="")

    direction = None
    if prob >= LONG_THRESHOLD:
        direction = 'LONG'
    elif prob <= SHORT_THRESHOLD:
        direction = 'SHORT'

    if direction is None:
        return

    try:
        qty_notional = CAPITAL_PER_TRADE * LEVERAGE
        raw_qty = qty_notional / last_price
        qty = float(int(raw_qty / QTY_STEP) * QTY_STEP)
        qty = round(qty, QTY_PRECISION)

        if direction == 'LONG':
            entry_side = 'BUY'
            tp_price = round(last_price * (1 + PROFIT_TARGET_PCT), PRICE_PRECISION)
            sl_price = round(last_price * (1 - STOP_LOSS_PCT), PRICE_PRECISION)
            tp_side = 'SELL'
        else:  # SHORT
            entry_side = 'SELL'
            tp_price = round(last_price * (1 - PROFIT_TARGET_PCT), PRICE_PRECISION)
            sl_price = round(last_price * (1 + STOP_LOSS_PCT), PRICE_PRECISION)
            tp_side = 'BUY'

        print(f"\n  [{direction} TAKER] Slot {available_slot} @ ~${last_price:.2f} | AI: {prob*100:.2f}%")

        # Market entry (taker)
        order_response = place_market_order(SYMBOL_TRADE, entry_side, qty)

        if order_response:
            # Get actual fill price from order response
            fill_price = last_price
            if order_response.get('avgPrice'):
                fill_price = float(order_response['avgPrice'])
            elif order_response.get('price') and float(order_response['price']) > 0:
                fill_price = float(order_response['price'])

            # Recalculate TP/SL from actual fill
            if direction == 'LONG':
                tp_price = round(fill_price * (1 + PROFIT_TARGET_PCT), PRICE_PRECISION)
                sl_price = round(fill_price * (1 - STOP_LOSS_PCT), PRICE_PRECISION)
            else:
                tp_price = round(fill_price * (1 - PROFIT_TARGET_PCT), PRICE_PRECISION)
                sl_price = round(fill_price * (1 + STOP_LOSS_PCT), PRICE_PRECISION)

            # Place TP limit order
            tp_order = place_limit_order(SYMBOL_TRADE, tp_side, qty, tp_price)
            tp_order_id = tp_order.get('orderId') if tp_order else None

            active_orders.append({
                'entry': fill_price, 'quantity': qty,
                'take_profit': tp_price, 'stop_loss': sl_price,
                'exit_ts': current_ts + HOLDING_TIME, 'entry_ts': current_ts,
                'tp_order_id': tp_order_id,
                'confidence': prob, 'slot': available_slot,
                'direction': direction,
            })
            last_trade_time_per_slot[available_slot] = current_ts

            longs_now = sum(1 for o in active_orders if o['direction'] == 'LONG')
            shorts_now = sum(1 for o in active_orders if o['direction'] == 'SHORT')

            send_tg_msg(
                f"<b>{direction} ENTRY (TAKER)</b>\n━━━━━━━━━━━━━━━━\n"
                f"Entry: ${fill_price:.2f}\n"
                f"TP: ${tp_price:.2f} ({'+' if direction=='LONG' else '-'}{PROFIT_TARGET_PCT*100:.3f}%)\n"
                f"SL: ${sl_price:.2f}\n"
                f"AI: {prob*100:.2f}%\n"
                f"Qty: {qty} ({LEVERAGE}x)\n"
                f"Slot: {available_slot + 1}/{MAX_POSITIONS}\n"
                f"Active: L:{longs_now} S:{shorts_now}"
            )
    except Exception as e:
        print(f"\n  Trade Error: {e}")

# ==========================================
# 9. MULTI-STREAM WEBSOCKET (3 streams)
# ==========================================
def flush_current_sec():
    global current_sec
    if current_sec['ts'] is None: return

    row = {
        'open': current_sec['open'], 'high': current_sec['high'],
        'low': current_sec['low'], 'close': current_sec['close'],
        'buy_volume': current_sec['buy_volume'],
        'sell_volume': current_sec['sell_volume'],
        'total_volume': current_sec['total_volume'],
        'net_flow': current_sec['net_flow'],
        'buy_count': current_sec['buy_count'],
        'sell_count': current_sec['sell_count'],
        'trade_count': current_sec['trade_count'],
        'best_bid': order_book['best_bid'],
        'best_ask': order_book['best_ask'],
        'bid_qty': order_book['bid_qty'],
        'ask_qty': order_book['ask_qty'],
        'spread': order_book['spread'],
        'book_imbalance': order_book['book_imbalance'],
    }
    buffer.append(row)

def reset_current_sec(ts, price):
    global current_sec
    current_sec = {
        'ts': ts,
        'open': price, 'high': price, 'low': price, 'close': price,
        'buy_volume': 0.0, 'sell_volume': 0.0, 'total_volume': 0.0,
        'net_flow': 0.0,
        'buy_count': 0, 'sell_count': 0, 'trade_count': 0,
    }

def on_message(ws, msg):
    """Handle 3 streams: aggTrade + depth + markPrice"""
    global current_sec, order_book, mark_price

    data = json.loads(msg)
    stream = data.get('stream', '')
    d = data.get('data', data)

    # --- Stream 1: aggTrade ---
    if 'aggTrade' in stream or d.get('e') == 'aggTrade':
        price = float(d['p'])
        qty = float(d['q'])
        is_seller = d['m']
        ts = int(d['T'] / 1000)

        check_active_orders(price, ts)
        send_status_report()

        if current_sec['ts'] is None:
            reset_current_sec(ts, price)
        elif ts > current_sec['ts']:
            flush_current_sec()
            predict_and_trade(ts)
            reset_current_sec(ts, price)

        if is_seller:
            current_sec['sell_volume'] += qty
            current_sec['sell_count'] += 1
            current_sec['net_flow'] -= qty
        else:
            current_sec['buy_volume'] += qty
            current_sec['buy_count'] += 1
            current_sec['net_flow'] += qty

        current_sec['total_volume'] += qty
        current_sec['trade_count'] += 1
        current_sec['close'] = price
        current_sec['high'] = max(current_sec['high'], price)
        current_sec['low'] = min(current_sec['low'], price)

    # --- Stream 2: depth (order book) ---
    elif 'depth' in stream:
        bids = d.get('b', d.get('bids', []))
        asks = d.get('a', d.get('asks', []))

        if bids and asks:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            bid_qty = float(bids[0][1])
            ask_qty = float(asks[0][1])

            order_book['best_bid'] = best_bid
            order_book['best_ask'] = best_ask
            order_book['bid_qty'] = bid_qty
            order_book['ask_qty'] = ask_qty
            order_book['spread'] = best_ask - best_bid
            total_qty = bid_qty + ask_qty
            order_book['book_imbalance'] = (bid_qty - ask_qty) / total_qty if total_qty > 0 else 0

    # --- Stream 3: markPrice ---
    elif 'markPrice' in stream:
        mark_price = float(d.get('p', 0))

def on_error(ws, error):
    print(f"\n  WebSocket Error: {error}")

def on_close(ws, close_status, close_msg):
    print(f"\n  WebSocket Closed: {close_status} - {close_msg}")

def on_open(ws):
    print("  WebSocket Connected (aggTrade + depth + markPrice)")

# ==========================================
# 10. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  TRADING BOT V3 FUTURES - TAKER (Long + Short)")
    print(f"{'='*55}")
    print(f"  Model: V3 ({len(FEATURE_COLS)} features, no funding)")
    print(f"  Market: FUTURES DEMO ({LEVERAGE}x {MARGIN_TYPE})")
    print(f"  Entry: TAKER (Market Order)")
    print(f"  Long: >= {LONG_THRESHOLD*100:.2f}%")
    print(f"  Short: <= {SHORT_THRESHOLD*100:.2f}%")
    print(f"  TP: {PROFIT_TARGET_PCT*100:.3f}% | SL: {STOP_LOSS_PCT*100:.2f}%")
    print(f"  Max Positions: {MAX_POSITIONS}")
    print(f"{'='*55}")

    sync_existing_positions()

    longs = sum(1 for o in active_orders if o['direction'] == 'LONG')
    shorts = sum(1 for o in active_orders if o['direction'] == 'SHORT')

    send_tg_msg(
        f"<b>BOT V3 FUTURES STARTED</b>\n━━━━━━━━━━━━━━━━\n"
        f"Model: V3 (33 features)\n"
        f"Market: FUTURES DEMO ({LEVERAGE}x {MARGIN_TYPE})\n"
        f"Entry: TAKER (Market Order)\n"
        f"Long: >={LONG_THRESHOLD*100:.2f}% | Short: <={SHORT_THRESHOLD*100:.2f}%\n"
        f"TP: {PROFIT_TARGET_PCT*100:.3f}% | SL: {STOP_LOSS_PCT*100:.2f}%\n"
        f"Max: {MAX_POSITIONS} positions\n"
        f"Sync: {'Has Position' if HAS_EXISTING_POSITION else 'Clean'}\n"
        f"━━━━━━━━━━━━━━━━\n/help for commands"
    )

    tg_thread = threading.Thread(target=telegram_command_loop, daemon=True)
    tg_thread.start()
    print("  Telegram Handler Started")

    # 3 streams: aggTrade + depth@500ms + markPrice@1s
    ws_url = (
        f"wss://demo-fstream.binance.com/stream?streams="
        f"{SYMBOL_WS}@aggTrade/"
        f"{SYMBOL_WS}@depth@500ms/"
        f"{SYMBOL_WS}@markPrice@1s"
    )

    print(f"  Connecting to 3 streams...")

    while True:
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open
            )
            ws.run_forever(ping_interval=30, ping_timeout=10)
        except Exception as e:
            print(f"\n  Fatal Error: {e}")

        print("\n  Reconnecting in 5 seconds...")
        time.sleep(5)
