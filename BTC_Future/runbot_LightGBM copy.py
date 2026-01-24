import websocket
import json
import pandas as pd
import lightgbm as lgb
from collections import deque
import sys
import time

# ==========================================
# ⚙️ CONFIGURATION (ตั้งค่าระบบ) 🔧
# ==========================================
MODEL_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/btc_maker_model_lgbm_maker.txt' 

# 💰 Money Management
INVESTMENT_USDT = 21.85     
LEVERAGE = 20               
POSITION_SIZE_USDT = INVESTMENT_USDT * LEVERAGE 

# 🎯 Strategy Settings
CONFIDENCE_THRESHOLD = 0.15
TARGET_TP_PERCENT = 0.0001  
STOP_LOSS_PERCENT = 0.0010  
MAX_HOLD_SEC = 300          
MAKER_TIMEOUT_SEC = 10      

# 💸 Fee Settings
FEE_MAKER = 0.0000 
FEE_TAKER = 0.0000

# 📡 Network Settings
SYMBOL = "btcusdt"
SOCKET = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"

# ==========================================
# 🚀 INITIALIZATION
# ==========================================
print(f"🧠 Loading AI Model from: {MODEL_FILE}")
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    print("✅ Model Loaded Successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit()

# State Variables
history_buffer = deque(maxlen=30)
current_second_data = {
    'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 
    'close_price': 0.0, 'timestamp_sec': None
}

pending_orders = [] 
active_orders = []  
stats = {'win': 0, 'loss': 0, 'missed': 0, 'total_net_pnl': 0.0}

# ==========================================
# 🧮 FEATURE CALCULATION
# ==========================================
def calculate_features(current_data, history):
    all_data = list(history) + [current_data]
    df = pd.DataFrame(all_data)
    
    if len(df) < 15: return None
    
    features = {
        'total_volume': current_data['total_volume'],
        'net_flow': current_data['net_flow'],
        'trade_count': current_data['trade_count']
    }
    
    features['net_flow_ma5'] = df['net_flow'].tail(5).mean()
    features['net_flow_ma15'] = df['net_flow'].tail(15).mean()
    features['volume_ma5'] = df['total_volume'].tail(5).mean()
    
    prev_flow = df['net_flow'].iloc[-2] if len(df) >= 2 else 0
    features['net_flow_diff'] = current_data['net_flow'] - prev_flow
    
    prev_price = df['close_price'].iloc[-2]
    features['price_change'] = ((current_data['close_price'] - prev_price) / prev_price) * 100
    features['std_5'] = df['close_price'].tail(5).std()
    features['dist_ma15'] = current_data['close_price'] - df['close_price'].tail(15).mean()
    
    delta = df['close_price'].diff()
    gain = (delta.where(delta > 0, 0)).tail(14).mean()
    loss = (-delta.where(delta < 0, 0)).tail(14).mean()
    
    if loss == 0:
        features['rsi'] = 100
    else:
        rs = gain / loss
        features['rsi'] = 100 - (100 / (1 + rs))
        
    return features

# ==========================================
# 🔄 ORDER MANAGEMENT (แก้ไขส่วนนี้)
# ==========================================
def manage_orders(current_price, current_ts):
    global pending_orders, active_orders, stats
    
    # 1. เช็ค Pending Orders
    for order in pending_orders[:]:
        limit_price = order['limit_price']
        
        if current_price < limit_price:
            # --- คำนวณเป้าหมายราคา ---
            tp_price = limit_price * (1 + TARGET_TP_PERCENT)
            sl_price = limit_price * (1 - STOP_LOSS_PERCENT)
            
            # --- Print แจ้งเตือนพร้อม Target ---
            print(f"✅ FILLED! | Entry: {limit_price:.2f} | 🎯 TP: {tp_price:.2f} | 🛑 SL: {sl_price:.2f}")
            
            active_orders.append({
                'entry_price': limit_price,
                'tp_price': tp_price, # บันทึกไว้ใช้ตอนโชว์ status
                'size_btc': order['size_btc'],
                'entry_ts': current_ts
            })
            pending_orders.remove(order)
            
        elif (current_ts - order['order_ts']) >= MAKER_TIMEOUT_SEC:
            print(f"💨 Cancel Pending (Timeout) @ {limit_price:.2f}")
            stats['missed'] += 1
            pending_orders.remove(order)

    # 2. เช็ค Active Orders
    for order in active_orders[:]:
        entry_price = order['entry_price']
        size = order['size_btc']
        
        pct_change = (current_price - entry_price) / entry_price
        
        is_tp = pct_change >= TARGET_TP_PERCENT
        is_sl = pct_change <= -STOP_LOSS_PERCENT
        is_time_limit = (current_ts - order['entry_ts']) >= MAX_HOLD_SEC
        
        if is_tp or is_sl or is_time_limit:
            gross_pnl = (current_price - entry_price) * size
            comm_entry = 0.0 
            comm_exit = current_price * size * FEE_TAKER 
            net_pnl = gross_pnl - (comm_entry + comm_exit)
            stats['total_net_pnl'] += net_pnl
            
            if net_pnl > 0:
                stats['win'] += 1
                result_str = f"\033[92mWIN 💰\033[0m"
            else:
                stats['loss'] += 1
                result_str = f"\033[91mLOSS 💸\033[0m"
            
            reason = "TP" if is_tp else ("SL" if is_sl else "TIME")
            print(f"{result_str} ({reason}) | Exit: {current_price:.1f} | Net: {net_pnl:+.4f} USDT")
            active_orders.remove(order)

# ==========================================
# 🔮 PREDICTION LOGIC (เพิ่มการโชว์ราคาปัจจุบันเทียบเป้า)
# ==========================================
def run_prediction(current_data, current_price, current_ts):
    global pending_orders, active_orders, history_buffer
    
    # CASE 1: มีออเดอร์ค้างอยู่
    if len(pending_orders) > 0:
        print(f"⏳ Waiting for Match... (Pending: {len(pending_orders)})")
        return
        
    # --- เพิ่มส่วน Monitor สถานะตอนถือของ ---
    if len(active_orders) > 0:
        order = active_orders[0]
        entry = order['entry_price']
        tp = order.get('tp_price', entry * (1 + TARGET_TP_PERCENT))
        
        # คำนวณ PnL % (คูณ Leverage)
        pnl_pct = ((current_price - entry) / entry) * 100 * LEVERAGE
        
        # โชว์ว่าตอนนี้ราคาอยู่ตรงไหน เทียบกับ TP
        # print(f"🛡️ Holding... Price: {current_price:.2f} (TP: {tp:.2f}) | PnL: {pnl_pct:+.2f}%")
        return
    # -------------------------------------

    features = calculate_features(current_data, history_buffer)
    
    if features is None: 
        print(f"📥 Collecting Data: {len(history_buffer)}/15 frames...")
        return
    
    feature_cols = [
        'total_volume', 'net_flow', 'trade_count',
        'net_flow_ma5', 'net_flow_ma15', 'volume_ma5',
        'net_flow_diff', 'price_change', 
        'std_5', 'dist_ma15', 'rsi'
    ]
    
    try:
        X = pd.DataFrame([features])[feature_cols]
        prob = model.predict(X)[0]
    except Exception as e:
        print(f"Prediction Error: {e}")
        return
    
    # --- Decision Logic ---
    if prob >= CONFIDENCE_THRESHOLD:
        limit_price = current_price 
        size_btc = POSITION_SIZE_USDT / limit_price
        
        print(f"🚀 SIGNAL FOUND! Prob: {prob*100:.2f}% | Placing Bid @ {limit_price:.2f}")
        
        pending_orders.append({
            'limit_price': limit_price,
            'size_btc': size_btc,
            'order_ts': current_ts
        })
    else:
        print(f"👀 Watching... Prob: {prob*100:.2f}% (Need > {CONFIDENCE_THRESHOLD*100}%) | Price: {current_price:.2f}")

# ==========================================
# 📡 WEBSOCKET HANDLER
# ==========================================
def on_message(ws, message):
    global current_second_data
    try:
        data = json.loads(message)
        
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m']
        ts_ms = data['T']
        ts_sec = int(ts_ms / 1000)
        
        manage_orders(price, ts_sec)
        
        if current_second_data['timestamp_sec'] is None:
            current_second_data['timestamp_sec'] = ts_sec
            
        if ts_sec > current_second_data['timestamp_sec']:
            current_second_data['close_price'] = price
            history_buffer.append(current_second_data.copy())
            
            run_prediction(current_second_data, price, ts_sec)
            
            current_second_data = {
                'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0,
                'close_price': price, 'timestamp_sec': ts_sec
            }
        
        signed_vol = qty if not is_maker else -qty
        
        current_second_data['net_flow'] += signed_vol
        current_second_data['total_volume'] += qty
        current_second_data['trade_count'] += 1
        current_second_data['close_price'] = price
        
    except Exception as e:
        print(f"WS Error: {e}")

def on_close(ws, close_status_code, close_msg):
    print("\n" + "="*50)
    print(f"🛑 Bot Stopped Summary")
    print(f"💰 Net PnL: {stats['total_net_pnl']:.4f} USDT")
    print(f"📊 Win: {stats['win']} | Loss: {stats['loss']} | Missed: {stats['missed']}")
    print("="*50)

def on_open(ws):
    print(f"⚡ Connected to Binance: {SYMBOL.upper()}")
    print(f"🎯 Maker Strategy | Conf > {CONFIDENCE_THRESHOLD*100}%")
    print("-" * 50)

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message, on_close=on_close)
    try:
        ws.run_forever()
    except KeyboardInterrupt:
        print("User Interrupted")