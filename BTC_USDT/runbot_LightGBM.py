import websocket
import json
import pandas as pd
import datetime
import lightgbm as lgb
from collections import deque

# ==========================================
# ⚙️ ส่วนตั้งค่า (Configuration) - ปรับจูนใหม่ 🔧
# ==========================================
MODEL_FILE = '/Users/Macbook/Collect_Crypto/BTC_USDT/btc_scalping_model_lgbm.txt'

# 1. ปรับเกณฑ์ความมั่นใจสูงขึ้น (กรองไม้ให้แม่นขึ้น)
CONFIDENCE_THRESHOLD = 0.5  # จากเดิม 0.55 -> 0.65

# 2. ตั้งค่าการเงินและ Money Management
INVESTMENT_USDT = 21.85     # ทุน 1,000 บาท
LEVERAGE = 20               # Leverage 20x
COMMISSION_RATE = 0.00045   # ค่าคอม 0.045% (VIP0 + BNB Discount)

# 3. ตั้งเป้าหมายกำไร/ขาดทุน (Strategy)
TARGET_TP_PERCENT = 0.0010  # เป้ากำไร 0.15% (เพื่อให้คุ้มค่าคอม 0.09%)
STOP_LOSS_PERCENT = 0.0010  # ยอมแพ้ที่ 0.10%
MAX_HOLD_SEC = 300          # ถือนานสุด 5 นาที (300 วิ) ถ้าไม่ไปไหนให้ปิด

MIN_FLOW = 0.01
SYMBOL = "btcusdt"
SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"

# ==========================================
# 🚀 เริ่มต้นระบบ
# ==========================================
print(f"🧠 กำลังโหลดสมอง AI...")
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    print("✅ โมเดลพร้อมทำงาน!")
except Exception as e:
    print(f"❌ โหลดโมเดลไม่ได้: {e}")
    exit()

print("="*50)
print(f"💰 Config: ทุน {INVESTMENT_USDT} USDT | Lev {LEVERAGE}x")
print(f"🎯 Strategy: TP {TARGET_TP_PERCENT*100}% | SL {STOP_LOSS_PERCENT*100}%")
print(f"🛡️ Confidence: > {CONFIDENCE_THRESHOLD*100}%")
print("="*50)

# ==========================================
# 📊 ตัวแปรเก็บข้อมูล
# ==========================================
history_buffer = deque(maxlen=30)
current_second_data = {
    'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 
    'close_price': 0.0, 'timestamp_sec': None
}

active_orders = []
stats = {
    'win': 0, 'loss': 0, 'breakeven': 0,
    'total_net_pnl': 0.0
}

# ==========================================
# 🧮 ฟังก์ชันคำนวณ Features
# ==========================================
def calculate_features(current_data, history):
    all_data = list(history) + [current_data]
    df = pd.DataFrame(all_data)
    if len(df) < 2: return None
    
    features = {
        'total_volume': current_data['total_volume'],
        'net_flow': current_data['net_flow'],
        'trade_count': current_data['trade_count'],
        'net_flow_ratio': current_data['net_flow'] / (current_data['total_volume'] + 1e-10),
    }
    
    features['net_flow_ma5'] = df['net_flow'].tail(5).mean()
    features['net_flow_ma15'] = df['net_flow'].tail(15).mean()
    features['net_flow_ma30'] = df['net_flow'].tail(30).mean()
    features['volume_ma5'] = df['total_volume'].tail(5).mean()
    features['volume_ma15'] = df['total_volume'].tail(15).mean()
    features['cumulative_net_flow_30'] = df['net_flow'].tail(30).sum()
    
    if len(df) >= 2:
        features['net_flow_diff'] = current_data['net_flow'] - df['net_flow'].iloc[-2]
        features['price_change'] = ((current_data['close_price'] - df['close_price'].iloc[-2]) / df['close_price'].iloc[-2]) * 100
    else:
        features['net_flow_diff'] = 0.0
        features['price_change'] = 0.0
        
    if len(df) >= 6:
        features['net_flow_diff5'] = current_data['net_flow'] - df['net_flow'].iloc[-6]
        features['price_change_ma5'] = df['close_price'].pct_change().tail(5).mean() * 100
    else:
        features['net_flow_diff5'] = 0.0
        features['price_change_ma5'] = 0.0
        
    return features

# ==========================================
# 🏁 ตรวจผลลัพธ์ (Logic ใหม่: TP/SL)
# ==========================================
def check_active_orders(current_price, current_ts):
    global active_orders, stats
    
    for order in active_orders[:]:
        entry_price = order['entry_price']
        position_size_btc = order['size_btc']
        entry_ts = order['entry_ts']
        
        # 1. คำนวณ % การเปลี่ยนแปลงของราคา (Unrealized PnL %)
        price_change_pct = (current_price - entry_price) / entry_price
        
        # 2. ตรวจสอบเงื่อนไขการขาย (TP หรือ SL หรือ หมดเวลา)
        is_take_profit = price_change_pct >= TARGET_TP_PERCENT
        is_stop_loss = price_change_pct <= -STOP_LOSS_PERCENT
        is_timeout = (current_ts - entry_ts) >= MAX_HOLD_SEC
        
        if is_take_profit or is_stop_loss or is_timeout:
            
            # --- คำนวณกำไร/ขาดทุนจริง (Net PnL) ---
            gross_pnl = (current_price - entry_price) * position_size_btc
            
            # ค่าคอม (คิดจากมูลค่าพอร์ตทั้งขาเข้าและออก)
            entry_val = entry_price * position_size_btc
            exit_val = current_price * position_size_btc
            total_fee = (entry_val + exit_val) * COMMISSION_RATE
            
            net_pnl = gross_pnl - total_fee
            stats['total_net_pnl'] += net_pnl
            net_pnl_thb = net_pnl * 45.8
            
            # ระบุเหตุผลที่ปิด
            reason = "🎯 TP" if is_take_profit else ("🛑 SL" if is_stop_loss else "⏰ Timeout")
            
            if net_pnl > 0:
                stats['win'] += 1
                color = "\033[92m" # เขียว
            else:
                stats['loss'] += 1
                color = "\033[91m" # แดง

            print(f"{color}{reason} | เข้า {entry_price:.1f} ออก {current_price:.1f} ({price_change_pct*100:+.2f}%) | "
                  f"Net: {net_pnl:+.4f} USDT ({net_pnl_thb:+.1f} บาท)\033[0m")
            
            active_orders.remove(order)

# ==========================================
# 🔮 ทำนายและออกออเดอร์
# ==========================================
def predict_signal(current_data, current_price, current_ts):
    global active_orders, history_buffer
    
    features = calculate_features(current_data, history_buffer)
    if features is None: return
    
    feature_cols = [
        'total_volume', 'net_flow', 'trade_count',
        'net_flow_ma5', 'net_flow_ma15', 'net_flow_ma30',
        'volume_ma5', 'volume_ma15',
        'net_flow_diff', 'net_flow_diff5',
        'price_change', 'price_change_ma5',
        'net_flow_ratio', 'cumulative_net_flow_30'
    ]
    
    X = pd.DataFrame([features])[feature_cols]
    prob_buy = model.predict(X)[0]
    
    # ⚡️ เข้าเมื่อมั่นใจ > 65% เท่านั้น
    if prob_buy >= CONFIDENCE_THRESHOLD and current_data['net_flow'] >= MIN_FLOW:
        
        # ป้องกันการเปิดออเดอร์ซ้ำซ้อน (ถ้ามีของอยู่แล้ว ไม่เปิดเพิ่ม)
        if len(active_orders) > 0: 
            return

        position_value_usdt = INVESTMENT_USDT * LEVERAGE
        size_btc = position_value_usdt / current_price
        
        print(f"🚀 BUY SIGNAL! ({prob_buy*100:.1f}%) | Price: {current_price:.2f} | Target: {current_price * (1+TARGET_TP_PERCENT):.2f}")
        
        active_orders.append({
            'entry_price': current_price,
            'size_btc': size_btc,
            'entry_ts': current_ts  # เก็บเวลาเข้า เพื่อเช็ค Timeout
        })

# ==========================================
# 📡 WebSocket Handlers
# ==========================================
def on_message(ws, message):
    global current_second_data
    try:
        data = json.loads(message)
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m']
        ts = int(data['T'] / 1000)
        
        check_active_orders(price, ts)
        
        if current_second_data['timestamp_sec'] is None:
            current_second_data['timestamp_sec'] = ts
            
        if ts > current_second_data['timestamp_sec']:
            current_second_data['close_price'] = price
            history_buffer.append(current_second_data.copy())
            if len(history_buffer) >= 5:
                predict_signal(current_second_data, price, ts)
            
            current_second_data = {
                'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0,
                'close_price': price, 'timestamp_sec': ts
            }
            
        signed_vol = qty if not is_maker else -qty
        current_second_data['net_flow'] += signed_vol
        current_second_data['total_volume'] += qty
        current_second_data['trade_count'] += 1
        current_second_data['close_price'] = price
        
    except Exception as e:
        print(f"Error: {e}")

def on_close(ws, close_status_code, close_msg):
    print(f"\n🛑 จบการทำงาน | Net PnL: {stats['total_net_pnl']:.4f} USDT")
    print(f"📊 Win: {stats['win']} | Loss: {stats['loss']}")

def on_open(ws):
    print(f"--- เชื่อมต่อ Binance ({SYMBOL.upper()}) สำเร็จ! ---")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()