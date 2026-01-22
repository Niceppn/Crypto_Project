import websocket
import json
import pandas as pd
import datetime
import lightgbm as lgb
from collections import deque
import time

# ==========================================
# ⚙️ ตั้งค่า (Configuration) - Maker Version
# ==========================================
MODEL_FILE = 'BTC_USDT/btc_scalping_model_lgbm.txt'
CONFIDENCE_THRESHOLD = 0.65
HOLDING_TIME = 10  # เพิ่มจาก 5 → 10 วินาทีสำหรับ maker orders
MIN_PROFIT = 0.008  # เพิ่มจาก 0.01 → 0.008 คำนึงค่าธรรมเนียม 0.1%
MIN_FLOW = 0.01
SYMBOL = "btcusdt"
SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"

# Maker order settings
ORDER_TIMEOUT = 30  # วินาทีที่รอให้ order ถูก match
SPREAD_BUFFER = 0.0002  # 0.02% ห่างจาก mid price
MAX_ORDER_AGE = 5  # วินาทีที่สามารถแก้ไขราคา order

# โหลดโมเดล LightGBM
print(f"🧠 กำลังโหลดสมอง AI จาก {MODEL_FILE}...")
model = lgb.Booster(model_file=MODEL_FILE)
print("✅ พร้อมทำงาน!")
print(f"⚙️ Threshold: {CONFIDENCE_THRESHOLD*100}% | Min Profit: {MIN_PROFIT*100}% | Min Flow: {MIN_FLOW}")
print(f"📋 Maker Mode: Holding {HOLDING_TIME}s | Timeout {ORDER_TIMEOUT}s | Buffer {SPREAD_BUFFER*100}%")

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
    'bid_price': 0.0,
    'ask_price': 0.0,
    'timestamp_sec': None
}

# Maker orders tracking
maker_orders = []  # Orders ที่วางไว้รอ match
active_positions = []  # Positions ที่เปิดแล้ว
stats = {'win': 0, 'loss': 0, 'breakeven': 0, 'expired': 0}

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
# 📋 Maker Order Management
# ==========================================
def calculate_maker_price(current_price, is_buy):
    """คำนวณราคาสำหรับ maker order"""
    if is_buy:
        # Buy order: วางต่ำกว่า mid price
        return current_price * (1 - SPREAD_BUFFER)
    else:
        # Sell order: วางสูงกว่า mid price  
        return current_price * (1 + SPREAD_BUFFER)

def check_maker_orders(current_price, current_ts):
    """ตรวจสอบว่า maker orders ถูก match หรือหมดอายุ"""
    global maker_orders, stats
    
    for order in maker_orders[:]:
        # ตรวจสอบว่า order ถูก match หรือไม่
        if order['is_buy'] and current_price <= order['limit_price']:
            # Buy order ถูก match
            position = {
                'entry_price': order['limit_price'],
                'check_time': current_ts + HOLDING_TIME,
                'direction': 'buy',
                'confidence': order['confidence']
            }
            active_positions.append(position)
            
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"🟢 {timestamp} | Maker BUY ถูก match! ราคา: {order['limit_price']:.2f} | มั่นใจ: {order['confidence']*100:.1f}%")
            
            maker_orders.remove(order)
            
        elif not order['is_buy'] and current_price >= order['limit_price']:
            # Sell order ถูก match - ตรวจสอบผลลัพธ์
            entry_price = order['entry_price']
            exit_price = order['limit_price']
            diff = exit_price - entry_price
            
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
                result_text = f"\033[91m❌ LOSS \033[0m"
                profit_text = f"({diff:.2f})"
            
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            real_trades = stats['win'] + stats['loss']
            win_rate = (stats['win'] / real_trades * 100) if real_trades > 0 else 0
            
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"🔴 {timestamp} | Maker SELL ถูก match! เข้า {entry_price:.2f} -> ออก {exit_price:.2f} {profit_text} | ผล: {result_text} | Win Rate: {win_rate:.1f}%")
            
            maker_orders.remove(order)
            
        # ตรวจสอบ timeout
        elif current_ts >= order['timeout']:
            stats['expired'] += 1
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            order_type = "BUY" if order['is_buy'] else "SELL"
            print(f"⏰ {timestamp} | Maker {order_type} order หมดอายุ | ราคา: {order['limit_price']:.2f}")
            maker_orders.remove(order)

def update_maker_orders(current_price, current_ts):
    """อัพเดตราคา maker orders ถ้าจำเป็น"""
    for order in maker_orders:
        age = current_ts - order['created_time']
        if age >= MAX_ORDER_AGE:
            # ปรับราคาให้ใกล้ current price มากขึ้น
            new_price = calculate_maker_price(current_price, order['is_buy'])
            if abs(new_price - order['limit_price']) > current_price * 0.0001:
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"🔄 {timestamp} | อัพเดตราคา maker: {order['limit_price']:.2f} → {new_price:.2f}")
                order['limit_price'] = new_price
                order['created_time'] = current_ts

# ==========================================
# 🏁 ฟังก์ชันตรวจผล Positions
# ==========================================
def check_active_positions(current_price, current_ts):
    """ตรวจสอบผลลัพธ์ของ positions ที่เปิดไว้"""
    global active_positions, stats
    
    for position in active_positions[:]:
        if current_ts >= position['check_time']:
            # ถึงเวลาปิด position → วาง maker sell order
            place_maker_sell_order(position, current_ts)
            active_positions.remove(position)

# ==========================================
# 🔮 ฟังก์ชันทำนายและวาง Maker Order
# ==========================================
def predict_and_place_order(current_data, current_price, current_ts):
    """ทำนายสัญญาณและวาง maker order"""
    global maker_orders, history_buffer, active_positions
    
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
    
    # ตรวจสอบเงื่อนไขเข้าซื้อและวาง maker order
    if prob_buy >= CONFIDENCE_THRESHOLD and current_data['net_flow'] >= MIN_FLOW:
        # คำนวณราคา maker order
        maker_price = calculate_maker_price(current_price, is_buy=True)
        
        # สร้าง maker order
        maker_order = {
            'limit_price': maker_price,
            'is_buy': True,
            'created_time': current_ts,
            'timeout': current_ts + ORDER_TIMEOUT,
            'confidence': prob_buy
        }
        
        maker_orders.append(maker_order)
        
        print(f"📋 {timestamp} | วาง MAKER BUY! (Flow: {current_data['net_flow']:.4f}) | มั่นใจ {prob_buy*100:.1f}% | ราคา: {maker_price:.2f} | รอ match...")

def place_maker_sell_order(position, current_ts):
    """วาง maker sell order เพื่อปิด position"""
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    current_price = current_second_data['close_price']
    
    # คำนวณราคา sell order (สูงกว่า current price)
    sell_price = calculate_maker_price(current_price, is_buy=False)
    
    # สร้าง maker sell order
    sell_order = {
        'limit_price': sell_price,
        'is_buy': False,
        'created_time': current_ts,
        'timeout': current_ts + ORDER_TIMEOUT,
        'position_id': id(position),  # อ้างอิงถึง position นี้
        'entry_price': position['entry_price']
    }
    
    maker_orders.append(sell_order)
    
    print(f"📋 {timestamp} | วาง MAKER SELL! ปิด position @ {position['entry_price']:.2f} | ราคา: {sell_price:.2f} | รอ match...")

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
        
        # ตรวจสอบ maker orders และ positions
        check_maker_orders(price, ts)
        check_active_positions(price, ts)
        update_maker_orders(price, ts)
        
        # เริ่มต้นวินาทีแรก
        if current_second_data['timestamp_sec'] is None:
            current_second_data['timestamp_sec'] = ts
        
        # ถ้าเปลี่ยนวินาที → บันทึกและทำนาย
        if ts > current_second_data['timestamp_sec']:
            current_second_data['close_price'] = price
            
            # เก็บเข้า history buffer
            history_buffer.append(current_second_data.copy())
            
            # ทำนายและวาง maker order (ต้องมีอย่างน้อย 5 วินาที)
            if len(history_buffer) >= 5:
                predict_and_place_order(current_second_data, price, ts)
            
            # รีเซ็ตข้อมูลวินาทีใหม่
            current_second_data = {
                'net_flow': 0.0,
                'total_volume': 0.0,
                'trade_count': 0,
                'close_price': price,
                'bid_price': 0.0,
                'ask_price': 0.0,
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
    total_trades = stats['win'] + stats['loss'] + stats['breakeven']
    real_trades = stats['win'] + stats['loss']
    win_rate = (stats['win'] / real_trades * 100) if real_trades > 0 else 0
    print(f"📊 สรุปผล: {stats['win']}W / {stats['loss']}L / {stats['breakeven']}BE / {stats['expired']}EX")
    print(f"📈 Win Rate: {win_rate:.1f}% | ค่าธรรมเนียมประหยัด: ~{(0.5-0.1)*100}%")

def on_open(ws):
    print(f"--- เชื่อมต่อ Binance ({SYMBOL.upper()}) สำเร็จ! ---")
    print(f"📋 MAKER MODE: รอสะสมข้อมูล 5 วินาทีก่อนเริ่มวาง order...")
    print(f"💰 ค่าธรรมเนียม: 0.1% (Maker) vs 0.5% (Market)")

# ==========================================
# 🚀 Main
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("🤖 BTC Scalping Bot (LightGBM) - MAKER VERSION")
    print("="*50)
    
    ws = websocket.WebSocketApp(
        SOCKET,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()
