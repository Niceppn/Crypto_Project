import websocket, json, datetime, sys, requests, threading
import pandas as pd
import lightgbm as lgb
from collections import deque

# ==========================================
# CONFIGURATION
# ==========================================
SYMBOL = "btcfdusd"
MODEL1_FILE = "BTCFDUSD.txt"
MODEL2_FILE = "BTCFDUSD2.txt"

# --- Telegram Config ---
TG_TOKEN = "8555159238:AAFQPvIFMqqvi7PxhBvXv1zfurF7XaF_kWY"
TG_CHAT_ID = "8440162744" 

# --- Strategy Parameters ---
CONF1 = 0.40                 # Confidence for Model 1
CONF2 = 0.40                 # Confidence for Model 2
CAPITAL_PER_TRADE = 5.1      # FDUSD per trade
HOLDING_TIME = 90            
PROFIT_TARGET_PCT = 0.0001   
COOLDOWN_SECONDS = 30        

# --- Stats & State (Separated for A/B Testing) ---
IS_RUNNING = True            

# Model 1 Stats
stats1 = {'win': 0, 'loss': 0, 'be': 0}
pnl1 = 0.0
orders1 = []
last_trade1 = 0

# Model 2 Stats
stats2 = {'win': 0, 'loss': 0, 'be': 0}
pnl2 = 0.0
orders2 = []
last_trade2 = 0

last_report_time = datetime.datetime.now()

# Terminal Colors
C_GREEN = "\033[92m"
C_RED = "\033[91m"
C_YELLOW = "\033[93m"
C_RESET = "\033[0m"

# ==========================================
# TELEGRAM CONTROL CENTER
# ==========================================
def send_tg_msg(msg):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try: requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': msg}, timeout=5)
    except: pass

def telegram_worker():
    global IS_RUNNING, stats1, stats2, pnl1, pnl2, orders1, orders2
    global CONF1, CONF2, CAPITAL_PER_TRADE, last_report_time
    last_update_id = 0
    while True:
        try:
            url = f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=10"
            res = requests.get(url, timeout=15).json()
            if res.get("result"):
                for update in res["result"]:
                    last_update_id = update["update_id"]
                    if "message" in update and "text" in update["message"]:
                        full_text = update["message"]["text"].strip()
                        args = full_text.split()
                        cmd = args[0].lower()
                        
                        if cmd == "/start_bot":
                            IS_RUNNING = True
                            send_tg_msg("BOT STARTED (Dual Models)")
                            
                        elif cmd == "/stop_bot":
                            IS_RUNNING = False
                            send_tg_msg("BOT STOPPED")
                            
                        elif cmd == "/status":
                            run_stat = "RUNNING" if IS_RUNNING else "STOPPED"
                            msg = (f"STATUS: {run_stat} | CAP: {CAPITAL_PER_TRADE}\n"
                                   f"------------------------\n"
                                   f"MODEL 1 (Conf {CONF1:.2f})\n"
                                   f"PNL: {pnl1:.4f} | W:{stats1['win']} L:{stats1['loss']}\n"
                                   f"Active: {len(orders1)}\n"
                                   f"------------------------\n"
                                   f"MODEL 2 (Conf {CONF2:.2f})\n"
                                   f"PNL: {pnl2:.4f} | W:{stats2['win']} L:{stats2['loss']}\n"
                                   f"Active: {len(orders2)}")
                            send_tg_msg(msg)
                            
                        elif cmd == "/set_conf1" and len(args) > 1:
                            try: CONF1 = float(args[1]); send_tg_msg(f"M1 Conf set to {CONF1}")
                            except: pass
                        elif cmd == "/set_conf2" and len(args) > 1:
                            try: CONF2 = float(args[1]); send_tg_msg(f"M2 Conf set to {CONF2}")
                            except: pass
                        elif cmd == "/set_cap" and len(args) > 1:
                            try: CAPITAL_PER_TRADE = float(args[1]); send_tg_msg(f"Capital set to {CAPITAL_PER_TRADE}")
                            except: pass
                            
                        elif cmd == "/reset":
                            stats1 = {'win': 0, 'loss': 0, 'be': 0}
                            stats2 = {'win': 0, 'loss': 0, 'be': 0}
                            pnl1 = 0.0; pnl2 = 0.0
                            orders1 = []; orders2 = []
                            last_report_time = datetime.datetime.now()
                            send_tg_msg("ALL STATS RESET")

        except: pass

threading.Thread(target=telegram_worker, daemon=True).start()

# ==========================================
# LOAD MODELS
# ==========================================
try:
    model1 = lgb.Booster(model_file=MODEL1_FILE)
    print(f"Loaded Model 1: {MODEL1_FILE}")
    
    model2 = lgb.Booster(model_file=MODEL2_FILE)
    print(f"Loaded Model 2: {MODEL2_FILE}")
except Exception as e:
    print(f"Model Load Error: {e}"); sys.exit()

buffer = deque(maxlen=60)
current_sec = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'close': 0.0, 'low': 999999.0, 'ts': None}

# ==========================================
# TRADING LOGIC
# ==========================================

def check_orders(current_price, current_ts):
    global stats1, stats2, orders1, orders2, pnl1, pnl2
    
    # --- CHECK MODEL 1 ORDERS ---
    for order in orders1[:]:
        if current_price >= order['tp']:
            stats1['win'] += 1
            profit = (current_price - order['entry']) * order['qty']
            pnl1 += profit
            print(f"\n{C_GREEN}[M1 TP WIN] +{profit:.4f} FDUSD{C_RESET}")
            orders1.remove(order)
        elif current_ts >= order['exit']:
            diff = current_price - order['entry']
            amt = diff * order['qty']
            pnl1 += amt
            if diff > 0: stats1['win'] += 1
            elif diff < 0: stats1['loss'] += 1
            else: stats1['be'] += 1
            print(f"\n{C_YELLOW}[M1 TIME] PNL: {amt:.4f} FDUSD{C_RESET}")
            orders1.remove(order)

    # --- CHECK MODEL 2 ORDERS ---
    for order in orders2[:]:
        if current_price >= order['tp']:
            stats2['win'] += 1
            profit = (current_price - order['entry']) * order['qty']
            pnl2 += profit
            print(f"\n{C_GREEN}[M2 TP WIN] +{profit:.4f} FDUSD{C_RESET}")
            orders2.remove(order)
        elif current_ts >= order['exit']:
            diff = current_price - order['entry']
            amt = diff * order['qty']
            pnl2 += amt
            if diff > 0: stats2['win'] += 1
            elif diff < 0: stats2['loss'] += 1
            else: stats2['be'] += 1
            print(f"\n{C_YELLOW}[M2 TIME] PNL: {amt:.4f} FDUSD{C_RESET}")
            orders2.remove(order)

def predict(data_list, last_price, current_ts):
    global orders1, orders2, last_trade1, last_trade2, pnl1, pnl2, last_report_time
    
    # --- Telegram Report every 30 mins ---
    now = datetime.datetime.now()
    if (now - last_report_time).total_seconds() >= 1800:
        msg = (f"[30-Min Report]\nM1 PNL: {pnl1:.4f}\nM2 PNL: {pnl2:.4f}")
        send_tg_msg(msg)
        last_report_time = now

    if not IS_RUNNING: return

    df = pd.DataFrame(data_list)
    if len(df) < 15: 
        print(f"\rGathering Data... {len(df)}/15", end="")
        return
    
    # --- Feature Engineering (Shared) ---
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
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    feat['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10)))).iloc[-1]

    X = pd.DataFrame([feat])

    # --- MODEL 1 PREDICTION ---
    if current_ts - last_trade1 >= COOLDOWN_SECONDS:
        prob1 = model1.predict(X)[0]
        if prob1 >= CONF1:
            qty = CAPITAL_PER_TRADE / last_price
            orders1.append({
                'entry': last_price, 'qty': qty,
                'tp': last_price * (1 + PROFIT_TARGET_PCT),
                'exit': current_ts + HOLDING_TIME
            })
            last_trade1 = current_ts
            print(f"\n{C_GREEN}[M1 BUY] Conf:{prob1:.2%}{C_RESET}")

    # --- MODEL 2 PREDICTION ---
    if current_ts - last_trade2 >= COOLDOWN_SECONDS:
        prob2 = model2.predict(X)[0]
        if prob2 >= CONF2:
            qty = CAPITAL_PER_TRADE / last_price
            orders2.append({
                'entry': last_price, 'qty': qty,
                'tp': last_price * (1 + PROFIT_TARGET_PCT),
                'exit': current_ts + HOLDING_TIME
            })
            last_trade2 = current_ts
            print(f"\n{C_YELLOW}[M2 BUY] Conf:{prob2:.2%}{C_RESET}")

    # Status Line
    print(f"\rPrice: {last_price:.2f} | M1 PNL: {pnl1:.4f} | M2 PNL: {pnl2:.4f}", end="")

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
    if p < current_sec['low']: current_sec['low'] = p

# ==========================================
# START
# ==========================================
print(f"--- Dual Bot Started (Model 1 & Model 2) ---")
send_tg_msg(f"BOT ONLINE (Dual Models)\nConf1: {CONF1} | Conf2: {CONF2}")

ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade", on_message=on_message)
ws.run_forever()