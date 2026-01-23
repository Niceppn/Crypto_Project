import websocket
import json
import pandas as pd
import datetime
import lightgbm as lgb
from collections import deque

# ==========================================
# ⚙️ ส่วนตั้งค่า (Maker Configuration) 🔧
# ==========================================
MODEL_FILE = '/Users/Macbook/Collect_Crypto/BTC_USDT/btc_scalping_model_lgbm.txt'

# 1. ปรับเกณฑ์ความมั่นใจ
CONFIDENCE_THRESHOLD = 0.65 

# 2. ตั้งค่าการเงิน (Maker Fee ต่ำกว่า Taker มาก)
INVESTMENT_USDT = 21.85     
LEVERAGE = 20               
# ปกติ Maker Fee Binance Futures คือ 0.02% (0.0002) 
# ถ้าเป็น Spot BTC/FDUSD อาจจะเป็น 0.0
COMMISSION_RATE = 0.00000 

# 3. Strategy
TARGET_TP_PERCENT = 0.0001  # เป้า 0.15%
STOP_LOSS_PERCENT = 0.0010  # ยอมแพ้ 0.10%
MAX_HOLD_SEC = 100          

# --- 🆕 การตั้งค่า Maker ---
MAKER_TIMEOUT_SEC = 10      # ถ้าราคาไม่ลงมา Match ใน 10 วิ ให้ยกเลิก (ตกรถ)
MAKER_OFFSET = 0.0          # 0.0 = ตั้งราคาที่ราคาปัจจุบัน (Best Bid)

MIN_FLOW = 0.01
SYMBOL = "btcusdt"
SOCKET = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"

# ==========================================
# 🚀 เริ่มต้นระบบ
# ==========================================
print(f"🧠 กำลังโหลดสมอง AI...")
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    print("✅ โมเดลพร้อมทำงาน! (โหมด Maker - ประหยัดค่าคอม)")
except Exception as e:
    print(f"❌ โหลดโมเดลไม่ได้: {e}")
    exit()

# ==========================================
# 📊 ตัวแปรเก็บข้อมูล
# ==========================================
history_buffer = deque(maxlen=30)
current_second_data = {
    'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 
    'close_price': 0.0, 'timestamp_sec': None
}

active_orders = []  # ออเดอร์ที่ Match แล้ว (ถือของอยู่)
pending_orders = [] # 🆕 ออเดอร์ที่ตั้งรอ (Limit Order)

stats = {
    'win': 0, 'loss': 0, 'missed': 0, # เพิ่มสถิติ "ตกรถ"
    'total_net_pnl': 0.0
}

# ==========================================
# 🧮 ฟังก์ชันคำนวณ Features (เหมือนเดิม)
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
# 🆕 จัดการ Pending Orders (Maker Logic)
# ==========================================
def check_pending_orders(current_price, current_ts):
    global pending_orders, active_orders, stats

    for order in pending_orders[:]:
        limit_price = order['limit_price']
        order_ts = order['order_ts']
        
        # 1. เช็คว่า Match หรือยัง? (Maker Buy จะ Match เมื่อราคาตลาด <= ราคาที่เราตั้ง)
        if current_price <= limit_price:
            print(f"✅ Filled! ได้ของที่ {limit_price:.2f} (Market dip to {current_price:.2f})")
            
            # ย้ายจาก Pending -> Active
            active_orders.append({
                'entry_price': limit_price, # ได้ราคาที่ตั้งไว้
                'size_btc': order['size_btc'],
                'entry_ts': current_ts
            })
            pending_orders.remove(order)
            
        # 2. เช็คว่ารอนานเกินไปไหม? (ตกรถ)
        elif (current_ts - order_ts) >= MAKER_TIMEOUT_SEC:
            print(f"💨 Cancel Order @ {limit_price:.2f} (Price ran away to {current_price:.2f}) - Missed Train")
            stats['missed'] += 1
            pending_orders.remove(order)

# ==========================================
# 🏁 ตรวจผลลัพธ์ Active Orders (TP/SL)
# ==========================================
def check_active_orders(current_price, current_ts):
    global active_orders, stats
    
    for order in active_orders[:]:
        entry_price = order['entry_price']
        position_size_btc = order['size_btc']
        entry_ts = order['entry_ts']
        
        price_change_pct = (current_price - entry_price) / entry_price
        
        is_take_profit = price_change_pct >= TARGET_TP_PERCENT
        is_stop_loss = price_change_pct <= -STOP_LOSS_PERCENT
        is_timeout = (current_ts - entry_ts) >= MAX_HOLD_SEC
        
        if is_take_profit or is_stop_loss or is_timeout:
            
            # คำนวณกำไร (Maker Fee ขาเข้า, Taker Fee ขาออก หรือ Limit ขาออกก็ได้ถ้าเก่ง)
            # ในที่นี้สมมติขาออกเป็น Taker (เพื่อให้ปิดของได้ชัวร์ๆ) หรือ Maker ถ้าตั้ง TP ล่วงหน้า
            # *เพื่อความปลอดภัย บอทส่วนใหญ่มักใช้ Taker ตอนออกของเพื่อหนีตาย* # แต่ถ้าอยากประหยัดสุดๆ ต้องตั้ง Limit ขายรอ
            
            gross_pnl = (current_price - entry_price) * position_size_btc
            
            # ค่าคอม: เข้า Maker (ถูก) + ออก Taker (แพงกว่า) หรือ Maker (ถูก)
            # สมมติเฉลี่ยๆ หรือใช้ rate ที่ตั้งไว้
            entry_val = entry_price * position_size_btc
            exit_val = current_price * position_size_btc
            total_fee = (entry_val + exit_val) * COMMISSION_RATE 
            
            net_pnl = gross_pnl - total_fee
            stats['total_net_pnl'] += net_pnl
            net_pnl_thb = net_pnl * 34.0 # เรทบาทสมมติ
            
            reason = "🎯 TP" if is_take_profit else ("🛑 SL" if is_stop_loss else "⏰ Timeout")
            color = "\033[92m" if net_pnl > 0 else "\033[91m"

            print(f"{color}{reason} | Entry {entry_price:.1f} Exit {current_price:.1f} | "
                  f"Net: {net_pnl:+.4f} USDT\033[0m")
            
            active_orders.remove(order)

# ==========================================
# 🔮 ทำนายและวางบิล (Place Limit Order)
# ==========================================
def predict_signal(current_data, current_price, current_ts):
    global pending_orders, active_orders, history_buffer
    
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
    
    # เงื่อนไขการเข้า
    if prob_buy >= CONFIDENCE_THRESHOLD and current_data['net_flow'] >= MIN_FLOW:
        
        # ห้ามเปิดเพิ่มถ้ามี Pending หรือ Active อยู่
        if len(active_orders) > 0 or len(pending_orders) > 0: 
            return

        position_value_usdt = INVESTMENT_USDT * LEVERAGE
        
        # 🆕 MAKER STRATEGY: ตั้งราคาที่ราคาปัจจุบัน (หรือต่ำกว่านิดหน่อย)
        limit_price = current_price - MAKER_OFFSET 
        size_btc = position_value_usdt / limit_price
        
        print(f"⏳ PLACING MAKER ORDER ({prob_buy*100:.1f}%) | Bid @ {limit_price:.2f} | Waiting for fill...")
        
        pending_orders.append({
            'limit_price': limit_price,
            'size_btc': size_btc,
            'order_ts': current_ts 
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
        
        # 1. เช็คว่าออเดอร์ที่รอ (Pending) ได้ของหรือยัง?
        check_pending_orders(price, ts)
        
        # 2. เช็คออเดอร์ที่มีของ (Active) ว่าต้องขายไหม?
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
    print(f"\n🛑 จบการทำงาน")
    print(f"📊 Win: {stats['win']} | Loss: {stats['loss']} | Missed (ตกรถ): {stats['missed']}")
    print(f"💰 Total Net PnL: {stats['total_net_pnl']:.4f} USDT")

def on_open(ws):
    print(f"--- Maker Bot Started ({SYMBOL.upper()}) ---")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()