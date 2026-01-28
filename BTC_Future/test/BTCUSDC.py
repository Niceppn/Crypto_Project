import websocket, json, datetime, sys, requests, threading, time
import pandas as pd
import lightgbm as lgb
from collections import deque

# ==========================================
# CONFIGURATION & TELEGRAM
# ==========================================
SYMBOL = "btcusdc"
MODEL_FILE = "BTCUSDC.txt"

TG_TOKEN = "8552406124:AAGhfHsvF0B65FeefrvEPHxzlW3pwZcmMkY"
TG_CHAT_ID = "8440162744" 

# --- Dynamic Strategy Parameters ---
CONFIDENCE_THRESHOLD = 0.40  
CAPITAL_PER_TRADE = 5.1      
HOLDING_TIME = 180            
PROFIT_TARGET_PCT = 0.0001   
STOP_LOSS_PCT = 0.01       
COOLDOWN_SECONDS = 30        

# --- Stats & State ---
IS_RUNNING = True            
stats = {'win': 0, 'loss': 0, 'breakeven': 0}
total_pnl_cash = 0.0         
timeout_probs = deque(maxlen=100) 

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
# AUTO UPDATE MODEL (EVERY 20 MINS)
# ==========================================
def model_reload_worker():
    global model
    while True:
        time.sleep(6000) # แก้เป็น 1200 วินาที (20 นาที)
        try:
            new_model = lgb.Booster(model_file=MODEL_FILE)
            model = new_model
            now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n{C_CYAN}[{now_str}] 🔄 Auto-Update: Model reloaded from disk.{C_RESET}")
        except Exception as e:
            print(f"\n{C_RED}❌ Error updating model: {e}{C_RESET}")

threading.Thread(target=model_reload_worker, daemon=True).start()

# ==========================================
# TELEGRAM CONTROL CENTER
# ==========================================
def send_tg_msg(msg):
    url = f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage"
    try: requests.post(url, data={'chat_id': TG_CHAT_ID, 'text': msg}, timeout=5)
    except: pass

def telegram_worker():
    global IS_RUNNING, stats, total_pnl_cash, active_orders, last_report_time
    global CONFIDENCE_THRESHOLD, CAPITAL_PER_TRADE, current_sec, model, STOP_LOSS_PCT, timeout_probs
    global HOLDING_TIME
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
                        
                        if cmd == "/start_bot":
                            IS_RUNNING = True
                            send_tg_msg("✅ BOT STARTED (Trading ON)")
                        
                        elif cmd == "/stop_bot":
                            IS_RUNNING = False
                            send_tg_msg("🛑 BOT STOPPED (Trading OFF)")

                        elif cmd == "/reload":
                            try:
                                model = lgb.Booster(model_file=MODEL_FILE)
                                send_tg_msg("🔄 Manual Reload: Model updated.")
                            except: send_tg_msg("❌ Reload Failed.")
                        
                        elif cmd == "/status":
                            run_stat = "RUNNING" if IS_RUNNING else "STOPPED"
                            cur_price = current_sec['close']
                            if active_orders:
                                current_order = active_orders[0]
                                holding_msg = (f"🟢 Buy: {current_order['entry']:.2f}\n"
                                               f"🎯 TP: {current_order['take_profit']:.2f} | SL: {current_order['stop_loss']:.2f}\n"
                                               f"⚡ Current: {cur_price:.2f}")
                            else:
                                holding_msg = f"📦 Holding: NO | ⚡ Price: {cur_price:.2f}"

                            msg = (f"📊 **STATUS REPORT** 📊\n"
                                   f"State: {run_stat}\n"
                                   f"💰 Net PNL: {total_pnl_cash:.4f} FDUSD\n"
                                   f"🏆 Win: {stats['win']} | ❌ Loss: {stats['loss']}\n"
                                   f"⚙️ Conf: {CONFIDENCE_THRESHOLD} | Hold: {HOLDING_TIME}s\n"
                                   f"⚠️ SL: {STOP_LOSS_PCT*100}%\n"
                                   f"--------------------\n"
                                   f"{holding_msg}")
                            send_tg_msg(msg)

                        elif cmd == "/set_hold" or cmd == "/holding" and len(args) > 1:
                            try:
                                val = int(args[1])
                                HOLDING_TIME = val
                                send_tg_msg(f"⏳ Holding Time set to: {val} seconds")
                            except: send_tg_msg("Error: Invalid number")

                        elif cmd == "/set_conf" and len(args) > 1:
                            try:
                                val = float(args[1])
                                CONFIDENCE_THRESHOLD = val
                                send_tg_msg(f"⚙️ Confidence Threshold set to: {val}")
                            except: send_tg_msg("Error: Invalid number")

                        # [คำสั่งอื่นๆ ยังคงเดิมเพื่อความสมบูรณ์]
                        elif cmd == "/reset":
                            stats = {'win': 0, 'loss': 0, 'breakeven': 0}; total_pnl_cash = 0.0; active_orders = []; timeout_probs.clear()
                            send_tg_msg("♻️ Statistics & PNL Reset Done.")
        except: pass

threading.Thread(target=telegram_worker, daemon=True).start()

# ==========================================
# LOAD MODEL (FIRST TIME)
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
    global stats, active_orders, total_pnl_cash, timeout_probs
    for order in active_orders[:]:
        is_exit = False
        reason = ""
        now_str = datetime.datetime.now().strftime("%H:%M:%S")
        
        if current_price >= order['take_profit']:
            is_exit, reason = True, "TP WIN (MAKER)"
        elif current_price <= order['stop_loss']:
            is_exit, reason = True, "STOP LOSS (SELL)"
        elif current_ts >= order['exit_ts']:
            is_exit, reason = True, "TIME EXIT"
            if 'prob' in order: timeout_probs.append(order['prob'])

        if is_exit:
            profit = (current_price - order['entry']) * order['quantity']
            total_pnl_cash += profit
            if profit > 0: stats['win'] += 1; color = C_GREEN
            elif profit < 0: stats['loss'] += 1; color = C_RED
            else: stats['breakeven'] += 1; color = C_YELLOW
            
            print(f"\n{color}[{now_str}][{reason}] Sold: {current_price:.2f} | Net: {profit:.4f} FDUSD{C_RESET}")
            active_orders.remove(order)

def predict(data_list, last_price, current_ts):
    global active_orders, last_trade_time, total_pnl_cash, last_report_time, model, STOP_LOSS_PCT, HOLDING_TIME
    
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S") # เก็บวันที่และเวลา

    if (now - last_report_time).total_seconds() >= 1800:
        msg = (f"🕒 **30-Min Report**\nPNL: {total_pnl_cash:.4f} FDUSD\nW: {stats['win']} | L: {stats['loss']}")
        send_tg_msg(msg)
        last_report_time = now

    if not IS_RUNNING: return
    if len(active_orders) > 0 or (current_ts - last_trade_time < COOLDOWN_SECONDS):
        return

    df = pd.DataFrame(data_list)
    if len(df) < 15: return
    
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
    # แสดงเวลาใน Log Prediction
    print(f"\r[{now_str}] Price: {last_price:.2f} | Prob: {prob*100:.2f}% | Net PNL: {pnl_color}{total_pnl_cash:.4f}{C_RESET}", end="")

    if prob >= CONFIDENCE_THRESHOLD:
        target_sell = last_price * (1 + PROFIT_TARGET_PCT)
        stop_loss_price = last_price * (1 - STOP_LOSS_PCT)
        quantity = CAPITAL_PER_TRADE / last_price
        
        # แสดงเวลาตอนมีสัญญาณ BUY
        print(f"\n{C_CYAN}[{now_str}][BUY SIGNAL] Entry: {last_price:.2f} | TP: {target_sell:.2f} | Hold: {HOLDING_TIME}s{C_RESET}")
        
        active_orders.append({
            'entry': last_price, 'quantity': quantity,
            'take_profit': target_sell, 
            'stop_loss': stop_loss_price,
            'exit_ts': current_ts + HOLDING_TIME,
            'prob': prob
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
print(f"{C_CYAN}--- Bot Started: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---{C_RESET}")
send_tg_msg("🚀 BOT ONLINE\nReports every 30 mins.\nCommands: /status, /holding [sec], /set_conf [val]")

ws = websocket.WebSocketApp(f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade", on_message=on_message)
ws.run_forever()