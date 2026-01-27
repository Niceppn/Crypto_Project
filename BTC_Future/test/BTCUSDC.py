import websocket, json, datetime, sys, requests, threading, time
import pandas as pd
import lightgbm as lgb
from collections import deque

# ==========================================
# CONFIGURATION & TELEGRAM
# ==========================================
SYMBOL = "btcusdc"
MODEL_FILE = "/Users/Macbook/Collect_Crypto/BTC_Future/test/btcusdc_training_data.txt"

TG_TOKEN = "8552406124:AAGhfHsvF0B65FeefrvEPHxzlW3pwZcmMkY"
TG_CHAT_ID = "8440162744" 

# --- Dynamic Strategy Parameters ---
CONFIDENCE_THRESHOLD = 0.40  
CAPITAL_PER_TRADE = 5.1      
HOLDING_TIME = 150            
PROFIT_TARGET_PCT = 0.0001   
STOP_LOSS_PCT = 0.01       # [เพิ่ม] เปอร์เซ็นต์ตัดขาดทุน (0.01 = 1%)
COOLDOWN_SECONDS = 30        

# --- Stats & State ---
IS_RUNNING = True            
stats = {'win': 0, 'loss': 0, 'breakeven': 0}
total_pnl_cash = 0.0         
timeout_probs = deque(maxlen=50) # เก็บค่า Prob ของไม้ที่ Time Exit

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
        time.sleep(1200) # แก้กลับเป็น 1200 วินาที (20 นาที) ตามคอมเมนต์
        try:
            new_model = lgb.Booster(model_file=MODEL_FILE)
            model = new_model
            now_str = datetime.datetime.now().strftime("%H:%M:%S")
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
                                   f"⚙️ Conf: {CONFIDENCE_THRESHOLD} | SL: {STOP_LOSS_PCT*100}%\n"
                                   f"--------------------\n"
                                   f"{holding_msg}")
                            send_tg_msg(msg)

                        elif cmd == "/timeout":
                            if not timeout_probs:
                                send_tg_msg("⏳ No timeout data yet.")
                            else:
                                avg_p = sum(timeout_probs)/len(timeout_probs)
                                # แก้ไขจากแสดง 5 ค่า เป็นแสดงทั้งหมดที่มีใน deque
                                all_probs_str = ", ".join([f"{p*100:.1f}%" for p in timeout_probs])
                                msg = (f"⏳ **TIMEOUT ANALYSIS (ALL)** ⏳\n"
                                       f"Count: {len(timeout_probs)} trades\n"
                                       f"Avg Prob: {avg_p*100:.2f}%\n"
                                       f"--------------------\n"
                                       f"History: {all_probs_str}")
                                send_tg_msg(msg)
                        
                        elif cmd == "/reset":
                            stats = {'win': 0, 'loss': 0, 'breakeven': 0}
                            total_pnl_cash = 0.0
                            active_orders = []
                            timeout_probs.clear()
                            send_tg_msg("♻️ Statistics & PNL Reset Done.")
                        
                        elif cmd == "/set_conf" and len(args) > 1:
                            try:
                                val = float(args[1])
                                CONFIDENCE_THRESHOLD = val
                                send_tg_msg(f"⚙️ Confidence Threshold set to: {val}")
                            except: send_tg_msg("Error: Invalid number")

                        elif cmd == "/set_sl" and len(args) > 1:
                            try:
                                val = float(args[1])
                                STOP_LOSS_PCT = val
                                send_tg_msg(f"⚠️ Stop Loss set to: {val} (e.g. 0.001 = 0.1%)")
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
        
        # 1. เช็คเป้ากำไร
        if current_price >= order['take_profit']:
            is_exit = True
            reason = "TP WIN (MAKER)"
        
        # 2. เช็คจุดตัดขาดทุน
        elif current_price <= order['stop_loss']:
            is_exit = True
            reason = "STOP LOSS (SELL)"

        # 3. เช็คหมดเวลา
        elif current_ts >= order['exit_ts']:
            is_exit = True
            reason = "TIME EXIT"
            if 'prob' in order:
                timeout_probs.append(order['prob'])

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
            
            print(f"\n{color}[{reason}] Sold: {current_price:.2f} | Net: {profit:.4f} FDUSD{C_RESET}")
            active_orders.remove(order)

def predict(data_list, last_price, current_ts):
    global active_orders, last_trade_time, total_pnl_cash, last_report_time, model, STOP_LOSS_PCT
    
    now = datetime.datetime.now()
    if (now - last_report_time).total_seconds() >= 1800:
        msg = (f"🕒 **30-Min Report**\n"
               f"PNL: {total_pnl_cash:.4f} FDUSD\n"
               f"W: {stats['win']} | L: {stats['loss']}")
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
    print(f"\rPrice: {last_price:.2f} | Prob: {prob*100:.2f}% | Net PNL: {pnl_color}{total_pnl_cash:.4f}{C_RESET}", end="")

    if prob >= CONFIDENCE_THRESHOLD:
        target_sell = last_price * (1 + PROFIT_TARGET_PCT)
        stop_loss_price = last_price * (1 - STOP_LOSS_PCT)
        quantity = CAPITAL_PER_TRADE / last_price
        
        print(f"\n{C_CYAN}[BUY] Entry: {last_price:.2f} | TP: {target_sell:.2f} | SL: {stop_loss_price:.2f}{C_RESET}")
        
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
print(f"{C_CYAN}--- Bot Started: Silent Mode (30m Report / 20m Auto-Reload) ---{C_RESET}")
send_tg_msg("🚀 BOT ONLINE (Silent Mode)\nReports every 30 mins.\nAuto-Reload Model every 20 mins.\nStop Loss & Timeout Stats Active.")

ws = websocket.WebSocketApp(f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade", on_message=on_message)
ws.run_forever()