import websocket
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import datetime
import os
import sys
import time
from collections import deque

# ==========================================
# ⚙️ ตั้งค่า (CONFIG)
# ==========================================
MODEL_PATH = '/Users/Macbook/Collect_Crypto/BTC_USDT/btc_model_70acc.keras'
SCALER_PATH = '/Users/Macbook/Collect_Crypto/BTC_USDT/scaler_btc.pkl'
SYMBOL = 'btcusdt'
CONFIDENCE_THRESHOLD = 0.60
HOLDING_TIME = 5
SEQUENCE_LENGTH = 30

# --- 🛡️ ตั้งค่าความปลอดภัย (Safety) ---
MAX_OPEN_ORDERS = 1       # ถือได้สูงสุดทีละ 1 ไม้
COOLDOWN_SECONDS = 10     # พัก 10 วิ หลังจบไม้
# ------------------------------------

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'

# ==========================================
# 🚀 โหลดทรัพยากร
# ==========================================
print(f"{Colors.CYAN}--- เริ่มระบบ BTC AI Scalping (With PnL Log) ---{Colors.RESET}")

try:
    if not os.path.exists(MODEL_PATH): raise FileNotFoundError(f"ไม่เจอไฟล์ {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ โหลดโมเดลสำเร็จ!")
except Exception as e:
    print(f"{Colors.RED}❌ Error โมเดล: {e}{Colors.RESET}")
    sys.exit()

try:
    if not os.path.exists(SCALER_PATH): raise FileNotFoundError(f"ไม่เจอไฟล์ {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ โหลด Scaler สำเร็จ!")
except Exception as e:
    print(f"{Colors.RED}❌ Error Scaler: {e}{Colors.RESET}")
    sys.exit()

# ==========================================
# 📊 ตัวแปร Global
# ==========================================
history_buffer = deque(maxlen=100)
current_candle = {
    'quantity': 0.0, 'signed_qty': 0.0, 'trade_count': 0,
    'close_price': 0.0, 'timestamp_sec': None
}

active_orders = []
stats = {'win': 0, 'loss': 0, 'draw': 0}
total_pnl = 0.0    # <--- 💰 ตัวแปรเก็บกำไร/ขาดทุนรวม (หน่วยเป็นจุด USDT)
last_trade_time = 0

# ==========================================
# 🧮 Calculate Features
# ==========================================
def calculate_features(buffer_list):
    df = pd.DataFrame(buffer_list)
    df['total_volume'] = df['quantity']
    df['net_flow'] = df['signed_qty']
    df['net_flow_ma5'] = df['net_flow'].rolling(window=5).mean()
    df['net_flow_ma15'] = df['net_flow'].rolling(window=15).mean()
    df['volume_ma5'] = df['total_volume'].rolling(window=5).mean()
    df['price_change'] = df['close_price'].pct_change().fillna(0) * 100
    
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    df['rsi_14'] = df['rsi_14'].fillna(50)
    
    df['middle_band'] = df['close_price'].rolling(window=20).mean()
    df['std_dev'] = df['close_price'].rolling(window=20).std()
    df['upper_band'] = df['middle_band'] + (df['std_dev'] * 2)
    df['lower_band'] = df['middle_band'] - (df['std_dev'] * 2)
    df['bb_width'] = np.where(df['middle_band'] != 0, (df['upper_band'] - df['lower_band']) / df['middle_band'], 0)
    
    df = df.fillna(0)
    feature_cols = [
        'total_volume', 'net_flow', 'trade_count', 'net_flow_ma5', 'net_flow_ma15', 
        'volume_ma5', 'price_change', 'rsi_14', 'bb_width',
        'upper_band', 'lower_band', 'close_price'
    ]
    return df[feature_cols].tail(SEQUENCE_LENGTH)

# ==========================================
# 🔮 Predict Signal
# ==========================================
def predict_signal(price, timestamp):
    global last_trade_time
    
    if len(active_orders) >= MAX_OPEN_ORDERS: return
    if (timestamp - last_trade_time) < COOLDOWN_SECONDS: return

    if len(history_buffer) < 50: return
    features_df = calculate_features(list(history_buffer))
    if len(features_df) < SEQUENCE_LENGTH: return

    try:
        X_input = scaler.transform(features_df.values)
        X_input = np.array([X_input])
    except: return

    prob = model.predict(X_input, verbose=0)[0][0]
    now_str = datetime.datetime.now().strftime('%H:%M:%S')

    if prob >= CONFIDENCE_THRESHOLD:
        print(f"{Colors.GREEN}🚀 {now_str} | SIGNAL BUY! ({prob*100:.1f}%) | ราคา {price:.2f}{Colors.RESET}")
        active_orders.append({
            'entry_price': price,
            'check_time': timestamp + HOLDING_TIME,
        })
    elif prob > 0.4:
        print(f"👀 {now_str} | ... ({prob*100:.1f}%)")

# ==========================================
# 🏁 Check Orders & Log PnL
# ==========================================
def check_active_orders(current_price, current_ts):
    global active_orders, stats, last_trade_time, total_pnl
    
    for order in active_orders[:]:
        if current_ts >= order['check_time']:
            # 1. คำนวณกำไร/ขาดทุน ของไม้นี้
            diff = current_price - order['entry_price']
            
            # 2. อัปเดต PnL รวม
            total_pnl += diff
            
            # 3. ตัดสินผลแพ้ชนะ
            if diff > 0: 
                stats['win'] += 1
                res = f"{Colors.GREEN}✅ WIN{Colors.RESET}"
            elif diff < 0:
                stats['loss'] += 1
                res = f"{Colors.RED}❌ LOSS{Colors.RESET}"
            else:
                stats['draw'] += 1
                res = f"{Colors.YELLOW}➖ DRAW{Colors.RESET}"
                
            # 4. เลือกสีสำหรับ PnL รวม (เขียวถ้าบวก, แดงถ้าลบ)
            pnl_color = Colors.GREEN if total_pnl >= 0 else Colors.RED
            
            print("-" * 50)
            print(f"🏁 ปิดออเดอร์: {order['entry_price']:.1f} -> {current_price:.1f} ({diff:+.1f}) | {res}")
            print(f"📊 Score: {stats['win']}W - {stats['loss']}L - {stats['draw']}D")
            print(f"💰 {Colors.BOLD}Total PnL: {pnl_color}{total_pnl:+.2f} points{Colors.RESET} (ประมาณ ${total_pnl:.2f} ถ้า 1 BTC)")
            print("-" * 50)
            
            last_trade_time = current_ts
            print(f"{Colors.MAGENTA}⏳ Cooldown {COOLDOWN_SECONDS}s...{Colors.RESET}")
            
            active_orders.remove(order)

# ==========================================
# 📡 WebSocket Logic
# ==========================================
def on_message(ws, message):
    global current_candle, history_buffer
    try:
        data = json.loads(message)
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m']
        ts = int(data['T'] / 1000)
        
        check_active_orders(price, ts)
        
        if current_candle['timestamp_sec'] is None:
            current_candle['timestamp_sec'] = ts
            
        if ts > current_candle['timestamp_sec']:
            history_buffer.append({
                'quantity': current_candle['quantity'],
                'signed_qty': current_candle['signed_qty'],
                'trade_count': current_candle['trade_count'],
                'close_price': current_candle['close_price']
            })
            predict_signal(price, ts)
            current_candle = {
                'quantity': 0.0, 'signed_qty': 0.0, 'trade_count': 0,
                'close_price': price, 'timestamp_sec': ts
            }
            
        flow = qty if not is_maker else -qty
        current_candle['quantity'] += qty
        current_candle['signed_qty'] += flow
        current_candle['trade_count'] += 1
        current_candle['close_price'] = price
        
    except Exception as e:
        print(f"Error: {e}")

def on_error(ws, error): print(f"{Colors.RED}WebSocket Error: {error}{Colors.RESET}")
def on_close(ws, close_status_code, close_msg): print(f"{Colors.YELLOW}### Disconnected ###{Colors.RESET}")
def on_open(ws): 
    print(f"{Colors.GREEN}✅ เชื่อมต่อ Binance ({SYMBOL.upper()}) สำเร็จ!{Colors.RESET}")
    print(f"⏳ รอสะสมข้อมูล 50 วินาที...")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade",
        on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()