import websocket
import json
import pandas as pd
import datetime
import lightgbm as lgb
from collections import deque
import threading
import time

# ==========================================
# ⚙️ ตั้งค่า (Configuration) - Real-time Maker Version
# ==========================================
MODEL_FILE = 'BTC_USDT/btc_scalping_model_lgbm.txt'
CONFIDENCE_THRESHOLD = 0.65
HOLDING_TIME = 10  # เพิ่มจาก 5 → 10 วินาทีสำหรับ maker orders
MIN_PROFIT = 0.008  # เพิ่มจาก 0.01 → 0.008 คำนึงค่าธรรมเนียม 0.1%
MIN_FLOW = 0.01
SYMBOL = "btcusdt"

# WebSocket streams
TRADE_SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"
DEPTH_SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth5@100ms"

# Maker order settings
ORDER_TIMEOUT = 30  # วินาทีที่รอให้ order ถูก match
SPREAD_BUFFER = 0.00005  # 0.005% ห่างจาก bid/ask
MAX_ORDER_AGE = 3  # วินาทีที่สามารถแก้ไขราคา order

# โหลดโมเดล LightGBM
print(f"🧠 กำลังโหลดสมอง AI จาก {MODEL_FILE}...")
model = lgb.Booster(model_file=MODEL_FILE)
print("✅ พร้อมทำงาน!")
print(f"⚙️ Threshold: {CONFIDENCE_THRESHOLD*100}% | Min Profit: {MIN_PROFIT*100}% | Min Flow: {MIN_FLOW}")
print(f"📋 Real-time Maker Mode: Holding {HOLDING_TIME}s | Timeout {ORDER_TIMEOUT}s | Buffer {SPREAD_BUFFER*100}%")

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

# Real-time order book data
order_book = {
    'bids': [],  # [[price, qty], ...]
    'asks': [],  # [[price, qty], ...]
    'last_update': 0
}

# Maker orders tracking
maker_orders = []  # Orders ที่วางไว้รอ match
active_positions = []  # Positions ที่เปิดแล้ว
stats = {'win': 0, 'loss': 0, 'breakeven': 0, 'expired': 0}

# Thread synchronization
data_lock = threading.Lock()

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
# 📋 Real-time Maker Order Management
# ==========================================
def calculate_maker_price_from_orderbook(is_buy):
    """คำนวณราคาสำหรับ maker order จาก real-time order book"""
    with data_lock:
        if not order_book['bids'] or not order_book['asks']:
            return None
        
        best_bid = float(order_book['bids'][0][0])  # ราคาซื้อสูงสุด
        best_ask = float(order_book['asks'][0][0])  # ราคาขายต่ำสุด
        
        if is_buy:
            # Buy order: วางต่ำกว่า best_ask เล็กน้อย
            return best_ask * (1 - SPREAD_BUFFER)
        else:
            # Sell order: วางสูงกว่า best_bid เล็กน้อย  
            return best_bid * (1 + SPREAD_BUFFER)

def check_maker_orders(current_ts):
    """ตรวจสอบว่า maker orders ถูก match หรือหมดอายุ"""
    global maker_orders, stats
    
    with data_lock:
        if not order_book['bids'] or not order_book['asks']:
            return
        
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
    
    for order in maker_orders[:]:
        # ตรวจสอบว่า order ถูก match หรือไม่
        if order['is_buy'] and best_ask <= order['limit_price']:
            # Buy order ถูก match (ราคาขายต่ำกว่า/เท่ากับราคาซื้อของเรา)
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
            
        elif not order['is_buy'] and best_bid >= order['limit_price']:
            # Sell order ถูก match (ราคาซื้อสูงกว่า/เท่ากับราคาขายของเรา)
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"🔴 {timestamp} | Maker SELL ถูก match! ราคา: {order['limit_price']:.2f}")
            maker_orders.remove(order)
            
        # ตรวจสอบ timeout
        elif current_ts >= order['timeout']:
            stats['expired'] += 1
            timestamp = datetime.datetime.now().strftime('%H:%M:%S')
            print(f"⏰ {timestamp} | Maker order หมดอายุ | ราคา: {order['limit_price']:.2f}")
            maker_orders.remove(order)

def update_maker_orders(current_ts):
    """อัพเดตราคา maker orders ถ้าจำเป็น"""
    for order in maker_orders:
        age = current_ts - order['created_time']
        if age >= MAX_ORDER_AGE:
            # ปรับราคาจาก order book ปัจจุบัน
            new_price = calculate_maker_price_from_orderbook(order['is_buy'])
            if new_price and abs(new_price - order['limit_price']) > order['limit_price'] * 0.0001:
                timestamp = datetime.datetime.now().strftime('%H:%M:%S')
                print(f"🔄 {timestamp} | อัพเดตราคา maker: {order['limit_price']:.2f} → {new_price:.2f}")
                order['limit_price'] = new_price
                order['created_time'] = current_ts

# ==========================================
# 🏁 ฟังก์ชันตรวจผล Positions
# ==========================================
def check_active_positions(current_ts):
    """ตรวจสอบผลลัพธ์ของ positions ที่เปิดไว้"""
    global active_positions, stats
    
    with data_lock:
        if not order_book['bids'] or not order_book['asks']:
            return
        
        # ใช้ mid price จาก order book
        best_bid = float(order_book['bids'][0][0])
        best_ask = float(order_book['asks'][0][0])
        current_price = (best_bid + best_ask) / 2
    
    for position in active_positions[:]:
        if current_ts >= position['check_time']:
            diff = current_price - position['entry_price']
            
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
            
            print(f"🏁 ตรวจผล: เข้า {position['entry_price']:.2f} -> ออก {current_price:.2f} {profit_text} | ผล: {result_text} | Win Rate: {win_rate:.1f}% ({stats['win']}W/{stats['loss']}L/{stats['breakeven']}BE/{stats['expired']}EX)")
            
            active_positions.remove(position)

# ==========================================
# 🔮 ฟังก์ชันทำนายและวาง Maker Order
# ==========================================
def predict_and_place_order(current_data, current_ts):
    """ทำนายสัญญาณและวาง maker order"""
    global maker_orders, history_buffer
    
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
        # คำนวณราคา maker order จาก order book
        maker_price = calculate_maker_price_from_orderbook(is_buy=True)
        
        if maker_price:
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

# ==========================================
# 📡 Trade Stream Handler
# ==========================================
def on_trade_message(ws, message):
    global current_second_data, history_buffer
    
    try:
        data = json.loads(message)
        
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m']
        ts = int(data['T'] / 1000)
        
        # ตรวจสอบ maker orders และ positions
        check_maker_orders(ts)
        check_active_positions(ts)
        update_maker_orders(ts)
        
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
                predict_and_place_order(current_second_data, ts)
            
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
        print(f"❌ Trade Error: {e}")

# ==========================================
# 📊 Depth Stream Handler
# ==========================================
def on_depth_message(ws, message):
    global order_book
    
    try:
        data = json.loads(message)
        
        with data_lock:
            order_book['bids'] = data['bids']
            order_book['asks'] = data['asks']
            order_book['last_update'] = data.get('T', int(time.time() * 1000))
        
    except Exception as e:
        print(f"❌ Depth Error: {e}")

# ==========================================
# 📡 WebSocket Handlers
# ==========================================
def on_trade_error(ws, error):
    print(f"❌ Trade WebSocket Error: {error}")

def on_trade_close(ws, close_status_code, close_msg):
    print("### Trade Stream Disconnected ###")
    total_trades = stats['win'] + stats['loss'] + stats['breakeven']
    real_trades = stats['win'] + stats['loss']
    win_rate = (stats['win'] / real_trades * 100) if real_trades > 0 else 0
    print(f"📊 สรุปผล: {stats['win']}W / {stats['loss']}L / {stats['breakeven']}BE / {stats['expired']}EX")
    print(f"📈 Win Rate: {win_rate:.1f}% | ค่าธรรมเนียมประหยัด: ~{(0.5-0.1)*100}%")

def on_trade_open(ws):
    print(f"--- เชื่อมต่อ Trade Stream สำเร็จ! ---")
    print(f"📋 REAL-TIME MAKER MODE: รอสะสมข้อมูล 5 วินาทีก่อนเริ่มวาง order...")
    print(f"💰 ค่าธรรมเนียม: 0.1% (Maker) vs 0.5% (Market)")

def on_depth_error(ws, error):
    print(f"❌ Depth WebSocket Error: {error}")

def on_depth_close(ws, close_status_code, close_msg):
    print("### Depth Stream Disconnected ###")

def on_depth_open(ws):
    print(f"--- เชื่อมต่อ Depth Stream สำเร็จ! ---")

# ==========================================
# 🚀 Main
# ==========================================
if __name__ == "__main__":
    print("="*60)
    print("🤖 BTC Scalping Bot (LightGBM) - REAL-TIME MAKER VERSION")
    print("="*60)
    
    # Create WebSocket connections
    trade_ws = websocket.WebSocketApp(
        TRADE_SOCKET,
        on_open=on_trade_open,
        on_message=on_trade_message,
        on_error=on_trade_error,
        on_close=on_trade_close
    )
    
    depth_ws = websocket.WebSocketApp(
        DEPTH_SOCKET,
        on_open=on_depth_open,
        on_message=on_depth_message,
        on_error=on_depth_error,
        on_close=on_depth_close
    )
    
    # Run both WebSockets in separate threads
    trade_thread = threading.Thread(target=trade_ws.run_forever)
    depth_thread = threading.Thread(target=depth_ws.run_forever)
    
    print("🚀 Starting Real-time Trading Bot...")
    print("📡 Connecting to Trade & Depth Streams...")
    
    trade_thread.start()
    depth_thread.start()
    
    # Wait for threads to complete
    trade_thread.join()
    depth_thread.join()
