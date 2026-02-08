#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trading Bot V3 - Maker Only (Multi-Stream)
Uses V3 AI Model (33 features: Price + Volume + Order Book + Cross)
3 WebSocket streams: aggTrade + depth@500ms + markPrice@1s

Maker Strategy:
  - Entry: Limit BUY at (market_price - offset)
  - Exit: Limit SELL at TP / Market SELL at SL or Time Exit
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
SYMBOL_WS = "btcusdc"
SYMBOL_TRADE = "BTCUSDC"

# --- Model ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODEL_FILE = os.path.join(MODEL_DIR, "v3_model_20260208_001401.txt")
META_FILE  = os.path.join(MODEL_DIR, "v3_model_20260208_001401_meta.json")

# --- TELEGRAM ---
TG_TOKEN = "8552406124:AAGhfHsvF0B65FeefrvEPHxzlW3pwZcmMkY"
TG_CHAT_ID = "8440162744"

# --- API KEYS ---
API_KEY = "eJVrU1CjLCKTOuS3QtQXfAVlxYBFWV1JHctSEEYDo3WL5uHBb2mbks6OUNytJmT2"
SECRET_KEY = "b7EX7kRfTxGmyVi7JePsvWnt1AFWlgXGy9mhedJhtVptfquIzHqrZADSzauWKqOM"

# --- Strategy (match training settings) ---
CONFIDENCE_THRESHOLD = 0.60
CAPITAL_PER_TRADE = 200
HOLDING_TIME = 1500
PROFIT_TARGET_PCT = 0.0005   # 0.05% (match training)
STOP_LOSS_PCT = 0.02
STATUS_REPORT_INTERVAL = 1800

# --- Maker Buy Settings ---
MAKER_BUY_OFFSET_PCT = 0.0003
MAKER_ORDER_TIMEOUT = 60

# --- Concurrent Positions ---
MAX_POSITIONS = 4
COOLDOWN_SECONDS = 180
SLOT2_COOLDOWN_SECONDS = 120
SLOT3_COOLDOWN_SECONDS = 180
SLOT4_COOLDOWN_SECONDS = 180

# ==========================================
# 2. CONNECT TO BINANCE
# ==========================================
try:
    client = Client(API_KEY, SECRET_KEY, testnet=True)
    client.FUTURES_URL = 'https://demo-fapi.binance.com'
    
    balance = client.futures_account_balance()
    usdt = next((item for item in balance if item["asset"] == "USDT"), None)
    print(f"\n✅ เชื่อมต่อ Binance Testnet สำเร็จ!")
    print(f"💰 เงินในพอร์ต: {usdt['balance']} USDT")

except Exception as e:
    print(f"❌ เชื่อมต่อไม่ได้: {e}")
    sys.exit()

# ==========================================
# GLOBAL VARS
# ==========================================
IS_RUNNING = True
HAS_EXISTING_POSITION = False
stats = {'win': 0, 'loss': 0, 'breakeven': 0, 'unfilled': 0}
total_pnl_cash = 0.0
active_orders = []
pending_orders = []
timeout_history = []
loss_history = []
last_trade_time_per_slot = [0] * MAX_POSITIONS
last_status_report_time = time.time()
last_update_id = 0

# --- V3 Multi-Stream Data ---
BUFFER_SIZE = 60
buffer = deque(maxlen=BUFFER_SIZE)

# Per-second aggregation (same structure as V2 collector)
current_sec = {
    'ts': None,
    'open': 0.0, 'high': 0.0, 'low': 999999.0, 'close': 0.0,
    'buy_volume': 0.0, 'sell_volume': 0.0, 'total_volume': 0.0,
    'net_flow': 0.0,
    'buy_count': 0, 'sell_count': 0, 'trade_count': 0,
}

# Order book (updated from depth stream)
order_book = {
    'best_bid': 0.0, 'best_ask': 0.0,
    'bid_qty': 0.0, 'ask_qty': 0.0,
    'spread': 0.0, 'book_imbalance': 0.0,
}

# Funding rate (updated from markPrice stream)
funding_rate = 0.0

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
    
    # Load metadata
    with open(META_FILE, 'r') as f:
        model_meta = json.load(f)
    
    # Verify features match
    meta_features = model_meta.get('features', [])
    if meta_features != FEATURE_COLS:
        print(f"⚠️  Feature mismatch! Meta: {len(meta_features)}, Code: {len(FEATURE_COLS)}")
        diff = set(meta_features) ^ set(FEATURE_COLS)
        if diff:
            print(f"   Diff: {diff}")
    
    print(f"✅ Loaded V3 Model: {os.path.basename(MODEL_FILE)}")
    print(f"   Features: {model_meta.get('n_features', '?')}")
    print(f"   Precision: {model_meta.get('precision', 0)*100:.1f}%")
    print(f"   Threshold: {model_meta.get('best_threshold', '?')}")
except Exception as e:
    print(f"❌ Model Load Failed: {e}")
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
        send_tg_msg(
            f"📊 <b>AUTO REPORT (V3)</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"💰 Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
            f"✅ Win: {stats['win']} | ❌ Loss: {stats['loss']}\n"
            f"📈 Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"📋 Active: {len(active_orders)} | Pending: {len(pending_orders)}\n"
            f"📊 Buffer: {len(buffer)}/{BUFFER_SIZE}"
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
    except:
        pass
    return []

def handle_telegram_commands():
    global last_update_id, IS_RUNNING, active_orders, pending_orders, stats, total_pnl_cash
    global HOLDING_TIME, STOP_LOSS_PCT, PROFIT_TARGET_PCT, CONFIDENCE_THRESHOLD, CAPITAL_PER_TRADE
    global MAKER_BUY_OFFSET_PCT, MAKER_ORDER_TIMEOUT, HAS_EXISTING_POSITION
    
    updates = get_telegram_updates()
    for update in updates:
        if 'update_id' not in update:
            continue
        last_update_id = update['update_id']
        if 'message' not in update or not update['message']:
            continue
        if 'text' not in update['message'] or not update['message']['text']:
            continue
        
        message = update['message']['text'].strip()
        
        if message == '/status':
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            win_rate = (stats['win'] / total_trades * 100) if total_trades > 0 else 0
            try:
                bal = client.futures_account_balance()
                usdt_bal = next((item for item in bal if item["asset"] == "USDT"), None)
                balance_text = f"💰 Balance: ${float(usdt_bal['balance']):.2f}"
                positions = client.futures_position_information(symbol=SYMBOL_TRADE)
                pos_amt = 0; pos_entry = 0; pos_pnl = 0
                for pos in positions:
                    amt = float(pos['positionAmt'])
                    if amt > 0:
                        pos_amt = amt; pos_entry = float(pos['entryPrice']); pos_pnl = float(pos['unRealizedProfit']); break
                pos_text = f"📊 Pos: {pos_amt} BTC @ ${pos_entry:.2f}\n💹 Unrealized: ${pos_pnl:.2f}" if pos_amt > 0 else "📊 Position: None"
            except:
                balance_text = "💰 Balance: N/A"; pos_text = "📊 Position: N/A"
            
            send_tg_msg(
                f"📊 <b>BOT V3 STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🤖 Status: {'🟢 RUNNING' if IS_RUNNING else '🔴 STOPPED'}\n"
                f"🧠 Model: V3 (33 features)\n"
                f"🔒 Existing Pos: {'⚠️ YES' if HAS_EXISTING_POSITION else '✅ NO'}\n"
                f"{balance_text}\n{pos_text}\n"
                f"💵 Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"✅ Win: {stats['win']} | ❌ Loss: {stats['loss']}\n"
                f"😐 BE: {stats['breakeven']} | ⏳ Unfilled: {stats['unfilled']}\n"
                f"📈 Win Rate: <b>{win_rate:.1f}%</b>\n"
                f"📋 Active: {len(active_orders)} | Pending: {len(pending_orders)}\n"
                f"📊 Buffer: {len(buffer)}/{BUFFER_SIZE}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⚙️ <b>SETTINGS:</b>\n"
                f"📉 Offset: -{MAKER_BUY_OFFSET_PCT*100:.2f}%\n"
                f"🎯 TP: {PROFIT_TARGET_PCT*100:.3f}% | 🛑 SL: {STOP_LOSS_PCT*100:.2f}%\n"
                f"🤖 Conf: {CONFIDENCE_THRESHOLD*100:.0f}%"
            )
        
        elif message == '/position':
            try:
                positions = client.futures_position_information(symbol=SYMBOL_TRADE)
                open_orders = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
                pos_amt = 0; pos_entry = 0; pos_pnl = 0
                for pos in positions:
                    amt = float(pos['positionAmt'])
                    if amt > 0:
                        pos_amt = amt; pos_entry = float(pos['entryPrice']); pos_pnl = float(pos['unRealizedProfit']); break
                buy_orders = [o for o in open_orders if o['side'] == 'BUY']
                sell_orders = [o for o in open_orders if o['side'] == 'SELL']
                send_tg_msg(
                    f"📊 <b>POSITION DETAILS</b>\n━━━━━━━━━━━━━━━━\n"
                    f"📈 Size: <b>{pos_amt} BTC</b>\n💰 Entry: ${pos_entry:.2f}\n💹 PNL: ${pos_pnl:.2f}\n"
                    f"━━━━━━━━━━━━━━━━\n📋 Buy Orders: {len(buy_orders)}\n📋 Sell Orders: {len(sell_orders)}"
                )
            except Exception as e:
                send_tg_msg(f"❌ Error: {e}")
        
        elif message.startswith('/set_offset'):
            try:
                val = float(message.split()[1]); MAKER_BUY_OFFSET_PCT = val / 100
                send_tg_msg(f"✅ Offset: <b>-{val}%</b>")
            except: send_tg_msg("❌ ใช้: /set_offset 0.03")
        
        elif message.startswith('/set_timeout'):
            try:
                MAKER_ORDER_TIMEOUT = int(message.split()[1])
                send_tg_msg(f"✅ Timeout: <b>{MAKER_ORDER_TIMEOUT}s</b>")
            except: send_tg_msg("❌ ใช้: /set_timeout 60")
        
        elif message.startswith('/set_conf'):
            try:
                CONFIDENCE_THRESHOLD = float(message.split()[1]) / 100
                send_tg_msg(f"✅ Confidence: <b>{CONFIDENCE_THRESHOLD*100:.0f}%</b>")
            except: send_tg_msg("❌ ใช้: /set_conf 60")
        
        elif message.startswith('/set_sl'):
            try:
                STOP_LOSS_PCT = float(message.split()[1]) / 100
                send_tg_msg(f"✅ SL: <b>{STOP_LOSS_PCT*100:.2f}%</b>")
            except: send_tg_msg("❌ ใช้: /set_sl 0.7")
        
        elif message.startswith('/set_tp'):
            try:
                PROFIT_TARGET_PCT = float(message.split()[1]) / 100
                send_tg_msg(f"✅ TP: <b>{PROFIT_TARGET_PCT*100:.3f}%</b>")
            except: send_tg_msg("❌ ใช้: /set_tp 0.05")
        
        elif message == '/stop':
            IS_RUNNING = False
            send_tg_msg("🔴 <b>BOT STOPPED</b>")
        
        elif message == '/start':
            IS_RUNNING = True
            send_tg_msg("🟢 <b>BOT STARTED</b>")
        
        elif message == '/reset':
            HAS_EXISTING_POSITION = False
            active_orders.clear(); pending_orders.clear()
            send_tg_msg(f"🔄 <b>BOT RESET</b>\n✅ Cleared internal state\nใช้ /position เพื่อตรวจสอบ")
        
        elif message == '/closeall':
            try:
                open_ords = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
                for o in open_ords: cancel_order(SYMBOL_TRADE, o['orderId'])
                positions = client.futures_position_information(symbol=SYMBOL_TRADE)
                for pos in positions:
                    amt = float(pos['positionAmt'])
                    if amt > 0:
                        client.futures_create_order(symbol=SYMBOL_TRADE, side='SELL', type='MARKET', quantity=amt)
                active_orders.clear(); pending_orders.clear(); HAS_EXISTING_POSITION = False
                send_tg_msg(f"✅ <b>CLOSE ALL COMPLETE</b>\nCancelled {len(open_ords)} orders + closed positions")
            except Exception as e:
                send_tg_msg(f"❌ Error: {e}")
        
        elif message == '/help':
            send_tg_msg(
                f"📚 <b>BOT V3 COMMANDS</b>\n━━━━━━━━━━━━━━━━\n"
                f"/status - สถานะ\n/position - ดู Position\n"
                f"/stop /start - หยุด/เริ่ม\n/closeall - ปิดทั้งหมด\n/reset - Reset\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"/set_offset [%] | /set_timeout [s]\n"
                f"/set_conf [%] | /set_sl [%] | /set_tp [%]"
            )

def telegram_command_loop():
    while True:
        try: handle_telegram_commands()
        except Exception as e: print(f"❌ TG Error: {e}")
        time.sleep(5)

# ==========================================
# 5. SYNC EXISTING POSITIONS
# ==========================================
def sync_existing_positions():
    global active_orders, HAS_EXISTING_POSITION
    try:
        positions = client.futures_position_information(symbol=SYMBOL_TRADE)
        current_position = 0; entry_price = 0
        for pos in positions:
            amt = float(pos['positionAmt'])
            if amt > 0: current_position = amt; entry_price = float(pos['entryPrice']); break
        
        open_orders = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
        print(f"\n📊 SYNC: Position={current_position} BTC @ ${entry_price:.2f} | Open Orders={len(open_orders)}")
        
        if current_position > 0:
            HAS_EXISTING_POSITION = True
            send_tg_msg(
                f"⚠️ <b>EXISTING POSITION</b>\n━━━━━━━━━━━━━━━━\n"
                f"📊 {current_position} BTC @ ${entry_price:.2f}\n"
                f"⛔ ไม่เปิด Position ใหม่\nใช้ /closeall → /reset"
            )
            return True
        else:
            HAS_EXISTING_POSITION = False
            if len(open_orders) > 0:
                for o in open_orders: cancel_order(SYMBOL_TRADE, o['orderId'])
                send_tg_msg(f"🧹 ยกเลิก {len(open_orders)} Orders ค้าง")
            print(f"✅ ไม่มี Position เดิม - พร้อมเริ่ม")
            return False
    except Exception as e:
        print(f"❌ Sync Error: {e}"); return False

def check_position_before_trade():
    global HAS_EXISTING_POSITION
    try:
        positions = client.futures_position_information(symbol=SYMBOL_TRADE)
        current_position = 0
        for pos in positions:
            amt = float(pos['positionAmt'])
            if amt > 0: current_position = amt; break
        max_allowed = MAX_POSITIONS * (CAPITAL_PER_TRADE / 70000) * 1.5
        if current_position > max_allowed:
            if not HAS_EXISTING_POSITION:
                HAS_EXISTING_POSITION = True
                send_tg_msg(f"⚠️ <b>POSITION LIMIT</b>\nCurrent: {current_position} BTC > Max: {max_allowed:.4f}")
            return False
        return True
    except: return False

# ==========================================
# 6. TRADING FUNCTIONS (MAKER ONLY)
# ==========================================
def place_limit_buy(symbol, quantity, limit_price):
    try:
        return client.futures_create_order(
            symbol=symbol, side='BUY', type='LIMIT',
            quantity=quantity, price=str(round(limit_price, 1)), timeInForce='GTC'
        )
    except Exception as e: print(f"❌ Limit Buy Error: {e}"); return None

def place_limit_sell(symbol, quantity, limit_price):
    try:
        return client.futures_create_order(
            symbol=symbol, side='SELL', type='LIMIT',
            quantity=quantity, price=str(round(limit_price, 1)), timeInForce='GTC'
        )
    except Exception as e: print(f"❌ Limit Sell Error: {e}"); return None

def cancel_order(symbol, order_id):
    try: client.futures_cancel_order(symbol=symbol, orderId=order_id); return True
    except Exception as e:
        if "Unknown order" in str(e): return True
        print(f"❌ Cancel Error: {e}"); return False

def close_position(symbol, quantity, reason):
    try:
        positions = client.futures_position_information(symbol=symbol)
        current_position = 0
        for pos in positions:
            if pos['positionSide'] in ('BOTH', 'LONG'):
                current_position = float(pos['positionAmt']); break
        if current_position <= 0 or current_position < quantity * 0.9: return True
        close_qty = min(quantity, current_position)
        client.futures_create_order(symbol=symbol, side='SELL', type='MARKET', quantity=close_qty)
        return True
    except Exception as e: print(f"❌ Close Error: {e}"); return False

def check_pending_orders(current_price, current_ts):
    global pending_orders, active_orders, stats, timeout_history
    for order in pending_orders[:]:
        if current_price <= order['limit_price']:
            sell_order = place_limit_sell(SYMBOL_TRADE, order['quantity'], order['take_profit'])
            sell_order_id = sell_order.get('orderId') if sell_order else None
            active_orders.append({
                'entry': order['limit_price'], 'quantity': order['quantity'],
                'take_profit': order['take_profit'], 'stop_loss': order['stop_loss'],
                'exit_ts': current_ts + HOLDING_TIME, 'entry_ts': current_ts,
                'sell_order_id': sell_order_id,
                'confidence': order.get('confidence', 0), 'slot': order.get('slot', 0)
            })
            pending_orders.remove(order)
            send_tg_msg(
                f"🟢 <b>FILLED (MAKER)</b>\n━━━━━━━━━━━━━━━━\n"
                f"📥 Entry: ${order['limit_price']:.2f}\n🎯 TP: ${order['take_profit']:.2f}\n"
                f"🛑 SL: ${order['stop_loss']:.2f}\n🤖 AI: {order['confidence']*100:.1f}%\n"
                f"📊 Active: {len(active_orders)}/{MAX_POSITIONS}"
            )
            continue
        if current_ts >= order['timeout_ts']:
            stats['unfilled'] += 1
            if order.get('order_id'): cancel_order(SYMBOL_TRADE, order['order_id'])
            timeout_history.append({
                'time': datetime.datetime.now().strftime('%H:%M:%S'),
                'limit_price': order['limit_price'], 'confidence': order.get('confidence', 0)
            })
            pending_orders.remove(order)

def check_active_orders(current_price, current_ts):
    global stats, total_pnl_cash, loss_history, HAS_EXISTING_POSITION
    for order in active_orders[:]:
        is_exit = False; reason = ""
        if current_price >= order['take_profit']:
            is_exit = True; reason = "TP WIN 🎯"
        elif current_price <= order['stop_loss']:
            is_exit = True; reason = "STOP LOSS 🛑"
        elif current_ts >= order['exit_ts']:
            is_exit = True; reason = "TIME EXIT ⏳"
        
        if is_exit:
            if order.get('sell_order_id'): cancel_order(SYMBOL_TRADE, order['sell_order_id'])
            if "TP" not in reason: close_position(SYMBOL_TRADE, order['quantity'], reason)
            profit = (current_price - order['entry']) * order['quantity']
            total_pnl_cash += profit
            if profit > 0: stats['win'] += 1
            elif profit < 0:
                stats['loss'] += 1
                loss_history.append({
                    'time': datetime.datetime.now().strftime('%H:%M:%S'),
                    'entry': order['entry'], 'exit': current_price, 'pnl': profit, 'reason': reason
                })
            else: stats['breakeven'] += 1
            send_tg_msg(
                f"{'✅' if profit >= 0 else '❌'} <b>CLOSED</b>\n━━━━━━━━━━━━━━━━\n"
                f"📥 Entry: ${order['entry']:.2f}\n📤 Exit: ${current_price:.2f}\n"
                f"💰 PNL: <b>${profit:.4f}</b>\n📝 {reason}\n"
                f"📊 Active: {len(active_orders)-1}/{MAX_POSITIONS}"
            )
            active_orders.remove(order)
            if len(active_orders) == 0 and len(pending_orders) == 0:
                HAS_EXISTING_POSITION = False

# ==========================================
# 7. V3 FEATURE ENGINEERING (33 features)
# ==========================================
def compute_features(data_list):
    """
    Compute 33 features from buffer (list of per-second dicts)
    Must match EXACTLY what train_model_v3.py creates
    """
    df = pd.DataFrame(data_list)
    if len(df) < 30:
        return None
    
    i = len(df) - 1  # latest row index
    
    close = df['close']
    high = df['high']
    low = df['low']
    opn = df['open']
    
    # --- Group 1: Price/OHLC ---
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
    
    # RSI 14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean().iloc[i]
    loss_val = (-delta.where(delta < 0, 0)).rolling(14).mean().iloc[i]
    rs = gain / (loss_val + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    
    momentum_5 = close.iloc[i] - close.iloc[i-5] if i >= 5 else 0
    momentum_15 = close.iloc[i] - close.iloc[i-15] if i >= 15 else 0
    
    trade_count_val = df['trade_count'].iloc[i]
    
    # --- Group 2: Volume Flow ---
    buy_vol = df['buy_volume']
    sell_vol = df['sell_volume']
    total_vol = df['total_volume']
    net_f = df['net_flow']
    
    volume_ratio = buy_vol.iloc[i] / (sell_vol.iloc[i] + 1e-10)
    net_flow_ma5 = net_f.rolling(5).mean().iloc[i]
    net_flow_ma15 = net_f.rolling(15).mean().iloc[i]
    net_flow_diff = net_f.diff().iloc[i]
    volume_ma5 = total_vol.rolling(5).mean().iloc[i]
    volume_ma15 = total_vol.rolling(15).mean().iloc[i]
    volume_spike = total_vol.iloc[i] / (volume_ma15 + 1e-10)
    cum_flow_10 = net_f.rolling(10).sum().iloc[i]
    total_volume_val = total_vol.iloc[i]
    
    # --- Group 3: Order Book ---
    spread_series = df['spread']
    imb_series = df['book_imbalance']
    
    spread_val = spread_series.iloc[i]
    spread_ma5 = spread_series.rolling(5).mean().iloc[i]
    spread_change = spread_series.diff().iloc[i]
    
    book_imbalance_val = imb_series.iloc[i]
    imbalance_ma5 = imb_series.rolling(5).mean().iloc[i]
    imbalance_ma15 = imb_series.rolling(15).mean().iloc[i]
    imbalance_change = imb_series.diff().iloc[i]
    
    bid_ask_ratio = df['bid_qty'].iloc[i] / (df['ask_qty'].iloc[i] + 1e-10)
    
    # --- Group 5: Cross Features ---
    imbalance_x_flow = book_imbalance_val * net_f.iloc[i]
    spread_x_volume = spread_val * total_vol.iloc[i]
    price_vol_corr = close.rolling(10).corr(total_vol).iloc[i]
    
    # Build feature dict (order must match FEATURE_COLS exactly)
    feat = {
        'candle_body': candle_body,
        'candle_range': candle_range,
        'upper_shadow': upper_shadow,
        'lower_shadow': lower_shadow,
        'price_change_1': price_change_1,
        'price_change_5': price_change_5,
        'dist_ma15': dist_ma15,
        'dist_ma30': dist_ma30,
        'std_5': std_5,
        'std_15': std_15,
        'rsi_14': rsi_14,
        'momentum_5': momentum_5,
        'momentum_15': momentum_15,
        'trade_count': trade_count_val,
        'volume_ratio': volume_ratio,
        'net_flow_ma5': net_flow_ma5,
        'net_flow_ma15': net_flow_ma15,
        'net_flow_diff': net_flow_diff,
        'volume_ma5': volume_ma5,
        'volume_spike': volume_spike,
        'cum_flow_10': cum_flow_10,
        'total_volume': total_volume_val,
        'spread': spread_val,
        'spread_ma5': spread_ma5,
        'spread_change': spread_change,
        'book_imbalance': book_imbalance_val,
        'imbalance_ma5': imbalance_ma5,
        'imbalance_ma15': imbalance_ma15,
        'imbalance_change': imbalance_change,
        'bid_ask_ratio': bid_ask_ratio,
        'imbalance_x_flow': imbalance_x_flow,
        'spread_x_volume': spread_x_volume,
        'price_vol_corr': price_vol_corr,
    }
    
    # Replace NaN with 0
    for k, v in feat.items():
        if pd.isna(v):
            feat[k] = 0.0
    
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
    """V3 prediction with 33 features from multi-stream data"""
    global last_trade_time_per_slot, pending_orders, HAS_EXISTING_POSITION
    
    if not IS_RUNNING: return
    if HAS_EXISTING_POSITION: return
    if current_ts % 30 == 0:
        if not check_position_before_trade(): return
    
    available_slot = get_available_slot(current_ts)
    if available_slot is None: return
    
    if len(buffer) < 30: return
    
    # Compute 33 features
    feat = compute_features(list(buffer))
    if feat is None: return
    
    # Predict
    feat_df = pd.DataFrame([feat])[FEATURE_COLS]
    prob = model.predict(feat_df)[0]
    
    last_price = buffer[-1]['close']
    
    print(f"\rV3 | ${last_price:.1f} | AI: {prob*100:.1f}% | "
          f"OB: bid={order_book['best_bid']:.1f} ask={order_book['best_ask']:.1f} "
          f"imb={order_book['book_imbalance']:.3f} | "
          f"Active: {len(active_orders)} Pending: {len(pending_orders)} "
          f"Buf: {len(buffer)}", end="")
    
    if prob >= CONFIDENCE_THRESHOLD:
        try:
            limit_buy_price = last_price * (1 - MAKER_BUY_OFFSET_PCT)
            qty = round(CAPITAL_PER_TRADE / last_price, 3)
            tp = limit_buy_price * (1 + PROFIT_TARGET_PCT)
            sl = limit_buy_price * (1 - STOP_LOSS_PCT)
            
            print(f"\n⚡ [V3 MAKER BUY] Slot {available_slot} @ ${limit_buy_price:.2f} | AI: {prob*100:.1f}%")
            
            order_response = place_limit_buy(SYMBOL_TRADE, qty, limit_buy_price)
            
            if order_response:
                pending_orders.append({
                    'limit_price': limit_buy_price, 'quantity': qty,
                    'take_profit': tp, 'stop_loss': sl,
                    'timeout_ts': current_ts + MAKER_ORDER_TIMEOUT,
                    'order_id': order_response.get('orderId'),
                    'confidence': prob, 'slot': available_slot
                })
                last_trade_time_per_slot[available_slot] = current_ts
                
                send_tg_msg(
                    f"⚡ <b>V3 MAKER BUY</b>\n━━━━━━━━━━━━━━━━\n"
                    f"📉 Limit: ${limit_buy_price:.2f}\n📊 Market: ${last_price:.2f}\n"
                    f"🎯 TP: ${tp:.2f} (+{PROFIT_TARGET_PCT*100:.3f}%)\n🛑 SL: ${sl:.2f}\n"
                    f"🤖 AI: {prob*100:.1f}%\n⏱️ Timeout: {MAKER_ORDER_TIMEOUT}s\n"
                    f"📋 Slot: {available_slot + 1}/{MAX_POSITIONS}"
                )
        except Exception as e:
            print(f"\n❌ Trade Error: {e}")

# ==========================================
# 9. MULTI-STREAM WEBSOCKET
# ==========================================
def flush_current_sec():
    """Flush current second data to buffer (called when new second starts)"""
    global current_sec
    
    if current_sec['ts'] is None:
        return
    
    # Merge with latest order book snapshot
    row = {
        'open': current_sec['open'],
        'high': current_sec['high'],
        'low': current_sec['low'],
        'close': current_sec['close'],
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
        'funding_rate': funding_rate,
    }
    
    buffer.append(row)

def reset_current_sec(ts, price):
    """Reset current_sec for new second"""
    global current_sec
    current_sec = {
        'ts': ts,
        'open': price, 'high': price, 'low': price, 'close': price,
        'buy_volume': 0.0, 'sell_volume': 0.0, 'total_volume': 0.0,
        'net_flow': 0.0,
        'buy_count': 0, 'sell_count': 0, 'trade_count': 0,
    }

def on_message(ws, msg):
    """Handle all 3 streams: aggTrade, depth, markPrice"""
    global current_sec, order_book, funding_rate
    
    data = json.loads(msg)
    stream = data.get('stream', '')
    d = data.get('data', data)
    
    # --- Stream 1: aggTrade ---
    if 'aggTrade' in stream or 'e' in d and d.get('e') == 'aggTrade':
        price = float(d['p'])
        qty = float(d['q'])
        is_seller = d['m']  # True = seller is maker (taker bought)
        ts = int(d['T'] / 1000)
        
        # Check pending/active orders with every trade
        check_pending_orders(price, ts)
        check_active_orders(price, ts)
        send_status_report()
        
        # New second → flush old, start new
        if current_sec['ts'] is None:
            reset_current_sec(ts, price)
        elif ts > current_sec['ts']:
            flush_current_sec()
            predict_and_trade(ts)
            reset_current_sec(ts, price)
        
        # Aggregate trade into current second
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
        bids = d.get('b', [])
        asks = d.get('a', [])
        
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
    
    # --- Stream 3: markPrice (funding rate) ---
    elif 'markPrice' in stream:
        funding_rate = float(d.get('r', 0))

def on_error(ws, error):
    print(f"\n❌ WebSocket Error: {error}")

def on_close(ws, close_status, close_msg):
    print(f"\n⚠️ WebSocket Closed: {close_status} - {close_msg}")

def on_open(ws):
    print("✅ Multi-Stream WebSocket Connected (aggTrade + depth + markPrice)")

# ==========================================
# 10. MAIN
# ==========================================
if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"🚀 TRADING BOT V3 - MAKER ONLY (Multi-Stream)")
    print(f"{'='*55}")
    print(f"🧠 Model: V3 ({len(FEATURE_COLS)} features, no funding)")
    print(f"📉 Buy Offset: -{MAKER_BUY_OFFSET_PCT*100:.2f}%")
    print(f"🎯 TP: {PROFIT_TARGET_PCT*100:.3f}% | 🛑 SL: {STOP_LOSS_PCT*100:.2f}%")
    print(f"🤖 Confidence: {CONFIDENCE_THRESHOLD*100:.0f}%")
    print(f"📊 Max Positions: {MAX_POSITIONS}")
    print(f"{'='*55}")
    
    # Sync positions
    sync_existing_positions()
    
    send_tg_msg(
        f"🚀 <b>BOT V3 STARTED</b>\n━━━━━━━━━━━━━━━━\n"
        f"🧠 Model: V3 (33 features)\n"
        f"📉 Offset: -{MAKER_BUY_OFFSET_PCT*100:.2f}%\n"
        f"🎯 TP: {PROFIT_TARGET_PCT*100:.3f}% | 🛑 SL: {STOP_LOSS_PCT*100:.2f}%\n"
        f"🤖 Conf: {CONFIDENCE_THRESHOLD*100:.0f}%\n"
        f"📊 Max: {MAX_POSITIONS} positions\n"
        f"🔄 Sync: {'⚠️ Has Position' if HAS_EXISTING_POSITION else '✅ Clean'}\n"
        f"━━━━━━━━━━━━━━━━\n/help เพื่อดูคำสั่ง"
    )
    
    # Start Telegram handler
    tg_thread = threading.Thread(target=telegram_command_loop, daemon=True)
    tg_thread.start()
    print("✅ Telegram Handler Started")
    
    # Multi-stream WebSocket URL
    # 3 streams: aggTrade + depth@500ms + markPrice@1s
    ws_url = (
        f"wss://demo-fstream.binance.com/stream?streams="
        f"{SYMBOL_WS}@aggTrade/"
        f"{SYMBOL_WS}@depth@500ms/"
        f"{SYMBOL_WS}@markPrice@1s"
    )
    
    print(f"🌐 Connecting to 3 streams...")
    
    # Start WebSocket with reconnect
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
            print(f"\n❌ Fatal Error: {e}")
        
        print("\n🔄 Reconnecting in 5 seconds...")
        time.sleep(5)
