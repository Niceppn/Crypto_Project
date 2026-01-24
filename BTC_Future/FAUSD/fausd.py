import websocket, json, datetime, sys, requests, threading
import pandas as pd
import lightgbm as lgb
from collections import deque

# ==========================================
# CONFIGURATION & TELEGRAM
# ==========================================
SYMBOL = "btcfdusd"
MODEL_FILE = "BTCFDUSD.txt"

TG_TOKEN = "8555159238:AAFQPvIFMqqvi7PxhBvXv1zfurF7XaF_kWY"
TG_CHAT_ID = "8440162744" 

# --- Dynamic Strategy Parameters ---
CONFIDENCE_THRESHOLD = 0.40  
CAPITAL_PER_TRADE = 5.1      
HOLDING_TIME = 90            
PROFIT_TARGET_PCT = 0.0001   
COOLDOWN_SECONDS = 30        

# --- Stats & State ---
IS_RUNNING = True            
stats = {'win': 0, 'loss': 0, 'breakeven': 0}
total_pnl_cash = 0.0         

active_orders = []
last_trade_time = 0
last_report_time = datetime.datetime.now()

# Terminal Colors
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_RESET = "\033[0m"

# ==========================================
# TELEGRAM CONTROL CENTER
# ==========================================
def send_tg_msg(msg):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try: requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': msg}, timeout=5)
    except: pass

def telegram_worker():
    global IS_RUNNING, stats, total_pnl_cash, active_orders, last_report_time
    global CONFIDENCE_THRESHOLD, CAPITAL_PER_TRADE, current_sec
    last_update_id = 0
    while True:
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=10"
            res = requests.get(url, timeout=15).json()
            if res.get("result"):
                for update in res["result"]:
                    last_update_id = update["update_id"]
                    if "message" in update and "text" in update["message"]:
                        args = update["message"]["text"].strip().split()
                        cmd = args[0].lower()
                        
                        # --- COMMANDS ---
                        if cmd == "/start_bot":
                            IS_RUNNING = True
                            send_tg_msg("✅ BOT STARTED (Trading ON)")
                        
                        elif cmd == "/stop_bot":
                            IS_RUNNING = False
                            send_tg_msg("🛑 BOT STOPPED (Trading OFF)")
                        
                        elif cmd == "/status":
                            run_stat = "RUNNING" if IS_RUNNING else "STOPPED"
                            cur_price = current_sec['close'] # ดึงราคา Realtime ล่าสุด
                            
                            # --- MODIFIED PART: Show Price if Holding + Realtime Price ---
                            if active_orders:
                                current_order = active_orders[0]
                                # แสดง Buy / Sell / Current(Realtime)
                                holding_msg = (f"🟢 Buy: {current_order['entry']:.2f}\n"
                                               f"🎯 Sell: {current_order['take_profit']:.2f}\n"
                                               f"⚡ Current: {cur_price:.2f}")
                            else:
                                holding_msg = f"📦 Holding: NO | ⚡ Price: {cur_price:.2f}"
                            # --------------------------------------------

                            msg = (f"📊 **STATUS REPORT** 📊\n"
                                   f"State: {run_stat}\n"
                                   f"💰 Net PNL: {total_pnl_cash:.4f} FDUSD\n"
                                   f"🏆 Win: {stats['win']} | ❌ Loss: {stats['loss']}\n"
                                   f"⚙️ Conf: {CONFIDENCE_THRESHOLD} | Cap: {CAPITAL_PER_TRADE}\n"
                                   f"--------------------\n"
                                   f"{holding_msg}")
                            send_tg_msg(msg)
                        
                        elif cmd == "/reset":
                            stats = {'win': 0, 'loss': 0, 'breakeven': 0}
                            total_pnl_cash = 0.0
                            active_orders = []
                            send_tg_msg("♻️ Statistics & PNL Reset Done.")
                        
                        elif cmd == "/set_conf" and len(args) > 1:
                            try:
                                val = float(args[1])
                                CONFIDENCE_THRESHOLD = val
                                send_tg_msg(f"⚙️ Confidence Threshold set to: {val}")
                            except: send_tg_msg("Error: Invalid number")

                        elif cmd == "/set_cap" and len(args) > 1:
                            try:
                                val = float(args[1])
                                CAPITAL_PER_TRADE = val
                                send_tg_msg(f"💰 Capital per trade set to: {val} FDUSD")
                            except: send_tg_msg("Error: Invalid number")

        except: pass

threading.Thread(target=telegram_worker, daemon=True).start()

# ==========================================
# LOAD MODEL
# ==========================================
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    print(f"{C_CYAN}Loaded Model: {SYMBOL.upper()}{C_RESET}")
except:
    print(f"Model File Not Found"); sys.exit()

buffer = deque(maxlen=60)
current_sec = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'close': 0.0, 'low': 999999.0, 'ts': None}

# ==========================================
# TRADING LOGIC
# ==========================================

def check_orders(current_price, current_ts):
    global stats, active_orders, total_pnl_cash
    for order in active_orders[:]:
        is_exit = False
        reason = ""
        
        # 1. TP Exit (Maker Logic)
        if current_price >= order['take_profit']:
            is_exit = True
            reason = "TP WIN (MAKER)"
        # 2. Time Exit
        elif current_ts >= order['exit_ts']:
            is_exit = True
            reason = "TIME EXIT"

        if is_exit:
            profit = (current_price - order['entry']) * order['quantity']
            total_pnl_cash += profit
            
            if profit > 0:
                stats['win'] += 1
                color = C_GREEN
            elif profit < 0:
                stats['loss'] += 1
                color = C_RED
            else:
                stats['breakeven'] += 1
                color = C_YELLOW
            
            # Print to Terminal ONLY (No Telegram Spam)
            print(f"\n{color}[{reason}] Sold: {current_price:.2f} | Net: {profit:.4f} FDUSD{C_RESET}")
            active_orders.remove(order)

def predict(data_list, last_price, current_ts):
    global active_orders, last_trade_time, total_pnl_cash, last_report_time
    
    # --- 30-Minute Report Routine ---
    now = datetime.datetime.now()
    if (now - last_report_time).total_seconds() >= 1800:
        msg = (f"🕒 **30-Min Report**\n"
               f"PNL: {total_pnl_cash:.4f} FDUSD\n"
               f"W: {stats['win']} | L: {stats['loss']}")
        send_tg_msg(msg)
        last_report_time = now

    if not IS_RUNNING: return
    
    # Check if we can trade (Must be empty hand & Cooldown finished)
    if len(active_orders) > 0 or (current_ts - last_trade_time < COOLDOWN_SECONDS):
        return

    df = pd.DataFrame(data_list)
    if len(df) < 15: return
    
    # Feature Engineering
    feat = {
        'total_volume': df['total_volume'].iloc[-1], 'net_flow': df['net_flow'].iloc[-1],
        'trade_count': df['trade_count'].iloc[-1],
        'net_flow_ma5': df['net_flow'].rolling(5).mean().iloc[-1],
        'net_flow_ma15': df['net_flow'].rolling(15).mean().iloc[-1],
        'volume_ma5': df['total_volume'].rolling(5).mean().iloc[-1],
        'net_flow_diff': df['net_flow'].diff().iloc[-1],
        'price_change': df['close'].pct_change().iloc[-1] * 100,
        'std_5': df['close'].rolling(5).std().iloc[-1],
        'dist_ma15': df['close'].iloc[-1] - df['close'].rolling(15).mean().iloc[-1]
    }
    delta = df['close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    feat['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10)))).iloc[-1]

    X = pd.DataFrame([feat])
    prob = model.predict(X)[0]
    
    pnl_color = C_GREEN if total_pnl_cash >= 0 else C_RED
    print(f"\rPrice: {last_price:.2f} | Prob: {prob*100:.2f}% | Net PNL: {pnl_color}{total_pnl_cash:.4f}{C_RESET}", end="")

    # Signal BUY
    if prob >= CONFIDENCE_THRESHOLD:
        target_sell = last_price * (1 + PROFIT_TARGET_PCT)
        quantity = CAPITAL_PER_TRADE / last_price
        
        # Print to Terminal ONLY (No Telegram Spam)
        print(f"\n{C_CYAN}[BUY] Entry: {last_price:.2f} | Set Maker Sell: {target_sell:.2f}{C_RESET}")
        
        active_orders.append({
            'entry': last_price, 'quantity': quantity,
            'take_profit': target_sell, 'exit_ts': current_ts + HOLDING_TIME
        })
        last_trade_time = current_ts

def on_message(ws, msg):
    global current_sec
    d = json.loads(msg)
    p, q, m, t = float(d['p']), float(d['q']), d['m'], int(d['T']/1000)
    if current_sec['ts'] is None: current_sec['ts'] = t
    
    check_orders(p, t)
    
    if t > current_sec['ts']:
        buffer.append(current_sec.copy())
        predict(list(buffer), p, t)
        current_sec = {'net_flow':0.0, 'total_volume':0.0, 'trade_count':0, 'close':p, 'low':p, 'ts':t}
    
    current_sec['net_flow'] += -q if m else q
    current_sec['total_volume'] += q
    current_sec['trade_count'] += 1
    current_sec['close'] = p
    if p < current_sec['low']: current_sec['low'] = min(current_sec['low'], p)

# ==========================================
# START
# ==========================================
print(f"{C_CYAN}--- Bot Started: Silent Mode (30m Report) ---{C_RESET}")
send_tg_msg("🚀 BOT ONLINE (Silent Mode)\nReports every 30 mins.\nUse /status to check manually.")

ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade", on_message=on_message)
ws.run_forever()