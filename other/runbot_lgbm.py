import json
import time
import websocket
import lightgbm as lgb
import numpy as np
from collections import deque

# ==========================================
# ⚙️ ตั้งค่า (Configuration)
# ==========================================
MODEL_FILE = 'btc_scalping_model_lgbm.txt'
CONFIDENCE_THRESHOLD = 0.75
HOLDING_TIME = 5  # 5 วินาที (ตรงกับตอนเทรน)
SYMBOL = "btcusdt"
SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"

# โหลดโมเดล LightGBM
print(f"🧠 กำลังโหลด LightGBM จาก {MODEL_FILE}...")
model = lgb.Booster(model_file=MODEL_FILE)
print("✅ พร้อมทำงาน!")

# ==========================================
# 📊 เก็บข้อมูลสำหรับคำนวณ Features
# ==========================================
WINDOW_SIZE = 35  # ต้องเก็บ 30+ วินาทีสำหรับ rolling

# เก็บข้อมูลรายวินาที
history = {
    'net_flow': deque(maxlen=WINDOW_SIZE),
    'total_volume': deque(maxlen=WINDOW_SIZE),
    'trade_count': deque(maxlen=WINDOW_SIZE),
    'close_price': deque(maxlen=WINDOW_SIZE),
}

# ข้อมูลวินาทีปัจจุบัน
current_second = {
    'net_flow': 0.0,
    'total_volume': 0.0,
    'trade_count': 0,
    'close_price': 0.0,
    'timestamp_sec': None
}

# Active orders & stats
active_orders = []
stats = {'win': 0, 'loss': 0}

# ==========================================
# 🔧 ฟังก์ชันคำนวณ Features
# ==========================================
def calculate_features():
    """คำนวณ Features ทั้ง 14 ตัว"""
    if len(history['net_flow']) < 30:
        return None
    
    nf = list(history['net_flow'])
    vol = list(history['total_volume'])
    tc = list(history['trade_count'])
    price = list(history['close_price'])
    
    # Basic features
    total_volume = vol[-1]
    net_flow = nf[-1]
    trade_count = tc[-1]
    
    # Moving Averages
    net_flow_ma5 = np.mean(nf[-5:])
    net_flow_ma15 = np.mean(nf[-15:])
    net_flow_ma30 = np.mean(nf[-30:])
    volume_ma5 = np.mean(vol[-5:])
    volume_ma15 = np.mean(vol[-15:])
    
    # Momentum
    net_flow_diff = nf[-1] - nf[-2] if len(nf) >= 2 else 0
    net_flow_diff5 = nf[-1] - nf[-5] if len(nf) >= 5 else 0
    
    # Price Change
    price_change = ((price[-1] - price[-2]) / price[-2] * 100) if len(price) >= 2 and price[-2] != 0 else 0
    price_changes = []
    for i in range(-5, 0):
        if len(price) >= abs(i) + 1 and price[i-1] != 0:
            price_changes.append((price[i] - price[i-1]) / price[i-1] * 100)
    price_change_ma5 = np.mean(price_changes) if price_changes else 0
    
    # Ratio & Cumulative
    net_flow_ratio = net_flow / (total_volume + 1e-10)
    cumulative_net_flow_30 = sum(nf[-30:])
    
    # Return features ตาม feature_cols ที่เทรน
    features = [
        total_volume, net_flow, trade_count,
        net_flow_ma5, net_flow_ma15, net_flow_ma30,
        volume_ma5, volume_ma15,
        net_flow_diff, net_flow_diff5,
        price_change, price_change_ma5,
        net_flow_ratio, cumulative_net_flow_30
    ]
    
    return np.array(features).reshape(1, -1)

# ==========================================
# 🎯 ฟังก์ชันตรวจ Orders
# ==========================================
def check_active_orders(current_price, current_ts):
    global active_orders, stats
    
    completed = []
    for order in active_orders:
        if current_ts >= order['exit_time']:
            entry_price = order['entry_price']
            if current_price > entry_price:
                stats['win'] += 1
                result = "✅ WIN"
            else:
                stats['loss'] += 1
                result = "❌ LOSS"
            
            pnl = (current_price - entry_price) / entry_price * 100
            total = stats['win'] + stats['loss']
            winrate = (stats['win'] / total * 100) if total > 0 else 0
            
            print(f"  {result} | Entry: {entry_price:.2f} → Exit: {current_price:.2f} | PnL: {pnl:+.3f}% | WinRate: {winrate:.1f}% ({stats['win']}/{total})")
            completed.append(order)
    
    for order in completed:
        active_orders.remove(order)

# ==========================================
# 📡 WebSocket Handlers
# ==========================================
def on_message(ws, message):
    global current_second
    
    data = json.loads(message)
    price = float(data['p'])
    qty = float(data['q'])
    is_buyer_maker = data['m']
    timestamp_ms = data['T']
    timestamp_sec = timestamp_ms // 1000
    
    current_ts = time.time()
    
    # ตรวจ active orders
    check_active_orders(price, current_ts)
    
    # เริ่มวินาทีใหม่
    if current_second['timestamp_sec'] is None:
        current_second['timestamp_sec'] = timestamp_sec
    
    if timestamp_sec != current_second['timestamp_sec']:
        # บันทึกวินาทีที่ผ่านมา
        history['net_flow'].append(current_second['net_flow'])
        history['total_volume'].append(current_second['total_volume'])
        history['trade_count'].append(current_second['trade_count'])
        history['close_price'].append(current_second['close_price'] if current_second['close_price'] > 0 else price)
        
        # คำนวณ Features & Predict
        features = calculate_features()
        if features is not None and len(active_orders) == 0:
            prob = model.predict(features)[0]
            
            if prob >= CONFIDENCE_THRESHOLD:
                print(f"\n🚀 BUY SIGNAL! | Price: {price:.2f} | Confidence: {prob*100:.1f}%")
                active_orders.append({
                    'entry_price': price,
                    'entry_time': current_ts,
                    'exit_time': current_ts + HOLDING_TIME
                })
        
        # Reset สำหรับวินาทีใหม่
        current_second['net_flow'] = 0.0
        current_second['total_volume'] = 0.0
        current_second['trade_count'] = 0
        current_second['close_price'] = 0.0
        current_second['timestamp_sec'] = timestamp_sec
    
    # สะสมข้อมูล
    signed_qty = qty if not is_buyer_maker else -qty
    current_second['net_flow'] += signed_qty
    current_second['total_volume'] += qty
    current_second['trade_count'] += 1
    current_second['close_price'] = price

def on_error(ws, error):
    print(f"❌ Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("🔴 Connection closed")

def on_open(ws):
    print("🟢 Connected to Binance!")
    print(f"⏱️ Holding Time: {HOLDING_TIME} วินาที")
    print(f"🎯 Threshold: {CONFIDENCE_THRESHOLD*100}%")
    print("=" * 50)
    print("📊 กำลังเก็บข้อมูล 30 วินาทีแรก...")

# ==========================================
# 🚀 Main
# ==========================================
if __name__ == "__main__":
    ws = websocket.WebSocketApp(
        SOCKET,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
        on_open=on_open
    )
    ws.run_forever()