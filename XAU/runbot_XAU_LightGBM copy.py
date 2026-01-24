import websocket
import json
import pandas as pd
import datetime
import lightgbm as lgb
from collections import deque

# ==========================================
# ⚙️ ตั้งค่า (Configuration)
# ==========================================
MODEL_FILE = '/Users/Macbook/Collect_Crypto/XAU/xauusdt_scalping_model_lgbm.txt'
CONFIDENCE_THRESHOLD = 0.55
HOLDING_TIME = 5
MIN_PROFIT = 0.01
MIN_FLOW = 0.01
SYMBOL = "xauusdt"
SOCKET = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"# ชื่อไฟล์ที่จะบันทึก

# โหลดโมเดล LightGBM
print(f"🧠 กำลังโหลดสมอง AI จาก {MODEL_FILE}...")
model = lgb.Booster(model_file=MODEL_FILE)
print("✅ พร้อมทำงาน!")
print(f"⚙️ Threshold: {CONFIDENCE_THRESHOLD*100}% | Min Profit: ${MIN_PROFIT} | Min Flow: {MIN_FLOW}")

# ==========================================
# 📊 ตัวแปรเก็บข้อมูล (Data Storage)
# ==========================================
# เก็บข้อมูลย้อนหลัง 30 วินาที สำหรับคำนวณ Moving Averages
history_buffer = deque(maxlen=30)

current_second_data = {
    'net_flow': 0.0, 
    'total_volume': 0.0, 
    'trade_count': 0, 
    'close_price': 0.0, 
    'timestamp_sec': None
}

active_orders = []
stats = {'win': 0, 'loss': 0, 'breakeven': 0}
total_profit = 0.0  # กำไร/ขาดทุนรวม

# ==========================================
# 🧮 ฟังก์ชันคำนวณ Features
# ==========================================
def calculate_features(current_data, history):
    """คำนวณ features ทั้ง 14 ตัวจากข้อมูล real-time"""
    
    # สร้าง DataFrame จาก history + current
    all_data = list(history) + [current_data]
    df = pd.DataFrame(all_data)
    
    if len(df) < 2:
        return None  # ยังไม่มีข้อมูลพอ
    
    # Features พื้นฐาน
    features = {
        'total_volume': current_data['total_volume'],
        'net_flow': current_data['net_flow'],
        'trade_count': current_data['trade_count'],
    }
    
    # Moving Averages ของ Net Flow
    features['net_flow_ma5'] = df['net_flow'].tail(5).mean() if len(df) >= 5 else df['net_flow'].mean()
    features['net_flow_ma15'] = df['net_flow'].tail(15).mean() if len(df) >= 15 else df['net_flow'].mean()
    features['net_flow_ma30'] = df['net_flow'].tail(30).mean() if len(df) >= 30 else df['net_flow'].mean()
    
    # Volume Moving Averages
    features['volume_ma5'] = df['total_volume'].tail(5).mean() if len(df) >= 5 else df['total_volume'].mean()
    features['volume_ma15'] = df['total_volume'].tail(15).mean() if len(df) >= 15 else df['total_volume'].mean()
    
    # Net Flow Momentum
    if len(df) >= 2:
        features['net_flow_diff'] = current_data['net_flow'] - df['net_flow'].iloc[-2]
    else:
        features['net_flow_diff'] = 0.0
        
    if len(df) >= 6:
        features['net_flow_diff5'] = current_data['net_flow'] - df['net_flow'].iloc[-6]
    else:
        features['net_flow_diff5'] = 0.0
    
    # Price Change
    if len(df) >= 2 and df['close_price'].iloc[-2] > 0:
        features['price_change'] = ((current_data['close_price'] - df['close_price'].iloc[-2]) / df['close_price'].iloc[-2]) * 100
    else:
        features['price_change'] = 0.0
    
    # Price Change MA5
    if len(df) >= 6:
        price_changes = df['close_price'].pct_change().tail(5) * 100
        features['price_change_ma5'] = price_changes.mean() if not price_changes.isna().all() else 0.0
    else:
        features['price_change_ma5'] = 0.0
    
    # Net Flow Ratio
    features['net_flow_ratio'] = current_data['net_flow'] / (current_data['total_volume'] + 1e-10)
    
    # Cumulative Net Flow 30 วินาที
    features['cumulative_net_flow_30'] = df['net_flow'].tail(30).sum()
    
    return features

# ==========================================
# 🏁 ฟังก์ชันตรวจผล Orders
# ==========================================
def check_active_orders(current_price, current_ts):
    """ตรวจสอบผลลัพธ์ของ orders ที่เปิดไว้"""
    global active_orders, stats, total_profit
    
    for order in active_orders[:]:
        if current_ts >= order['check_time']:
            diff = current_price - order['entry_price']
            
            if diff >= MIN_PROFIT:
                stats['win'] += 1
                result_text = f"\033[92m✅ WIN \033[0m"
                profit_text = f"(+{diff:.2f})"
            elif diff > -MIN_PROFIT:
                stats['breakeven'] += 1
                result_text = f"\033[93m➖ BREAKEVEN\033[0m"
                profit_text = f"({diff:+.2f})"
            else:
                stats['loss'] += 1
                result_text = f"\033[91m❌ LOSS\033[0m"
                profit_text = f"({diff:.2f})"
            
            total_profit += diff  # สะสมกำไร/ขาดทุน
            
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            real_trades = stats['win'] + stats['loss']
            win_rate = (stats['win'] / real_trades * 100) if real_trades > 0 else 0
            
            print(f"🏁 ตรวจผล: เข้า {order['entry_price']:.2f} -> ออก {current_price:.2f} {profit_text} | ผล: {result_text} | Win Rate: {win_rate:.1f}% ({stats['win']}W/{stats['loss']}L/{stats['breakeven']}BE) | กำไร/ขาดทุนรวม: {total_profit:.2f}")
            
            active_orders.remove(order)

# ==========================================
# 🔮 ฟังก์ชันทำนาย
# ==========================================
def predict_signal(current_data, current_entry_price, current_real_time):
    """ทำนายสัญญาณซื้อจาก features"""
    global active_orders, history_buffer
    
    # คำนวณ features
    features = calculate_features(current_data, history_buffer)
    
    if features is None:
        return  # ยังไม่มีข้อมูลพอ
    
    # เรียงลำดับ features ให้ตรงกับตอน train
    feature_cols = [
        'total_volume', 'net_flow', 'trade_count',
        'net_flow_ma5', 'net_flow_ma15', 'net_flow_ma30',
        'volume_ma5', 'volume_ma15',
        'net_flow_diff', 'net_flow_diff5',
        'price_change', 'price_change_ma5',
        'net_flow_ratio', 'cumulative_net_flow_30'
    ]
    
    X = pd.DataFrame([features])[feature_cols]
    
    # LightGBM Booster predict() คืนค่า probability โดยตรง
    prob_buy = model.predict(X)[0]
    
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    
    # ตรวจสอบเงื่อนไขเข้าซื้อ
    if prob_buy >= CONFIDENCE_THRESHOLD :
        target_time = current_real_time + HOLDING_TIME
        
        print(f"🚀 {timestamp} | สัญญาณซื้อ! (Flow: {current_data['net_flow']:.4f}) | มั่นใจ {prob_buy*100:.1f}% | ราคาเข้า: {current_entry_price:.2f}")
        
        active_orders.append({
            'entry_price': current_entry_price,
            'check_time': target_time
        })

# ==========================================
# 📡 WebSocket Handlers
# ==========================================
def on_message(ws, message):
    global current_second_data, history_buffer
    
    try:
        data = json.loads(message)
        
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m']
        ts = int(data['T'] / 1000)
        
        # ตรวจ orders ที่เปิดไว้
        check_active_orders(price, ts)
        
        # เริ่มต้นวินาทีแรก
        if current_second_data['timestamp_sec'] is None:
            current_second_data['timestamp_sec'] = ts
        
        # ถ้าเปลี่ยนวินาที → บันทึกและทำนาย
        if ts > current_second_data['timestamp_sec']:
            current_second_data['close_price'] = price
            
            # เก็บเข้า history buffer
            history_buffer.append(current_second_data.copy())
            
            # ทำนาย (ต้องมีอย่างน้อย 5 วินาที)
            if len(history_buffer) >= 5:
                predict_signal(current_second_data, price, ts)
            
            # รีเซ็ตข้อมูลวินาทีใหม่
            current_second_data = {
                'net_flow': 0.0,
                'total_volume': 0.0,
                'trade_count': 0,
                'close_price': price,
                'timestamp_sec': ts
            }
        
        # สะสมข้อมูลในวินาทีปัจจุบัน
        signed_vol = qty if not is_maker else -qty
        current_second_data['net_flow'] += signed_vol
        current_second_data['total_volume'] += qty
        current_second_data['trade_count'] += 1
        current_second_data['close_price'] = price
        
    except Exception as e:
        print(f"❌ Error: {e}")

def on_error(ws, error):
    print(f"❌ WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### Disconnected ###")
    print(f"📊 สรุปผล: {stats['win']}W / {stats['loss']}L / {stats['breakeven']}BE | กำไร/ขาดทุนรวม: {total_profit:.2f}")

def on_open(ws):
    print(f"--- เชื่อมต่อ Binance ({SYMBOL.upper()}) สำเร็จ! ---")
    print(f"⏳ รอสะสมข้อมูล 5 วินาทีก่อนเริ่มทำนาย...")

# ==========================================
# 🚀 Main
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("🤖 BTC Scalping Bot (LightGBM)")
    print("="*50)
    
    ws = websocket.WebSocketApp(
        SOCKET,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()