import websocket
import json
import pandas as pd
import lightgbm as lgb
from collections import deque
import sys

# ==========================================
# ⚙️ CONFIGURATION (ปรับจูนตรงนี้) 🔧
# ==========================================
MODEL_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/btc_maker_model_lgbm_maker.txt' # 🔴 ต้องตรงกับไฟล์ที่เทรนมา

# Strategy Settings
CONFIDENCE_THRESHOLD = 0.65 # ความมั่นใจขั้นต่ำ 65%
INVESTMENT_USDT = 21.85     # ทุนต่อไม้
LEVERAGE = 20
TARGET_TP_PERCENT = 0.0001  # เป้ากำไร 0.15% (Net ประมาณ 10 ดอลลาร์ BTC)
STOP_LOSS_PERCENT = 0.0010  # ยอมขาดทุน 0.10%
MAX_HOLD_SEC = 300          # ถือสูงสุด 5 นาที
MAKER_TIMEOUT_SEC = 10      # รอออเดอร์ Match สูงสุด 10 วิ (ถ้านานกว่านี้คือตกรถ ยกเลิก)

# Fee Settings (คำนวณกำไรสุทธิ)
FEE_MAKER = 0.0000 # 0.02% (ขาเข้า)
FEE_TAKER = 0.0000 # 0.05% (ขาออก - เผื่อไว้แบบ Taker)

SYMBOL = "btcusdt"
SOCKET = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"

# ==========================================
# 🚀 INITIALIZATION
# ==========================================
print(f"🧠 Loading AI Model...")
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

pending_orders = [] # ออเดอร์ที่ตั้งรอ (Maker)
active_orders = []  # ออเดอร์ที่ได้ของแล้ว (Positions)

stats = {'win': 0, 'loss': 0, 'missed': 0, 'total_net_pnl': 0.0}

# ==========================================
# 🧮 FEATURE CALCULATION (ต้องเหมือนตอนเทรนเป๊ะๆ)
# ==========================================
def calculate_features(current_data, history):
    all_data = list(history) + [current_data]
    df = pd.DataFrame(all_data)
    
    # ต้องมีข้อมูลอย่างน้อย 15-20 แถวเพื่อคำนวณ Indicator
    if len(df) < 15: return None
    
    # คำนวณ Features (Copy logic มาจากไฟล์ Train)
    features = {
        'total_volume': current_data['total_volume'],
        'net_flow': current_data['net_flow'],
        'trade_count': current_data['trade_count']
    }
    
    # Rolling Features
    features['net_flow_ma5'] = df['net_flow'].tail(5).mean()
    features['net_flow_ma15'] = df['net_flow'].tail(15).mean()
    features['volume_ma5'] = df['total_volume'].tail(5).mean()
    features['net_flow_diff'] = current_data['net_flow'] - df['net_flow'].iloc[-2]
    
    # Price Features
    prev_price = df['close_price'].iloc[-2]
    features['price_change'] = ((current_data['close_price'] - prev_price) / prev_price) * 100
    features['std_5'] = df['close_price'].tail(5).std()
    features['dist_ma15'] = current_data['close_price'] - df['close_price'].tail(15).mean()
    
    # RSI Calculation
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
# 🔄 ORDER MANAGEMENT
# ==========================================
def manage_orders(current_price, current_ts):
    global pending_orders, active_orders, stats
    
    # 1. เช็ค Pending Orders (รอ Match)
    for order in pending_orders[:]:
        limit_price = order['limit_price']
        
        # เงื่อนไข Match: ราคาปัจจุบันย่อลงมาต่ำกว่าหรือเท่ากับราคาที่เราตั้ง
        if current_price <= limit_price:
            print(f"✅ Filled! Got entry @ {limit_price:.2f}")
            active_orders.append({
                'entry_price': limit_price,
                'size_btc': order['size_btc'],
                'entry_ts': current_ts
            })
            pending_orders.remove(order)
            
        # เงื่อนไข Timeout: รอนานเกินไป (ตกรถ)
        elif (current_ts - order['order_ts']) >= MAKER_TIMEOUT_SEC:
            print(f"💨 Cancel Pending (Missed) @ {limit_price:.2f}")
            stats['missed'] += 1
            pending_orders.remove(order)

    # 2. เช็ค Active Orders (ถือของอยู่ -> หาจังหวะขาย)
    for order in active_orders[:]:
        entry_price = order['entry_price']
        size = order['size_btc']
        
        # คำนวณ % PnL
        pct_change = (current_price - entry_price) / entry_price
        
        is_tp = pct_change >= TARGET_TP_PERCENT
        is_sl = pct_change <= -STOP_LOSS_PERCENT
        is_time = (current_ts - order['entry_ts']) >= MAX_HOLD_SEC
        
        if is_tp or is_sl or is_time:
            # คำนวณกำไร/ขาดทุน
            gross_pnl = (current_price - entry_price) * size
            
            # ค่าคอม: เข้า Maker + ออก Taker (เพื่อความชัวร์ในการปิด)
            fee = (entry_price * size * FEE_MAKER) + (current_price * size * FEE_TAKER)
            net_pnl = gross_pnl - fee
            
            stats['total_net_pnl'] += net_pnl
            
            if net_pnl > 0:
                stats['win'] += 1
                color = "\033[92m" # Green
            else:
                stats['loss'] += 1
                color = "\033[91m" # Red
                
            reason = "TP 🎯" if is_tp else ("SL 🛑" if is_sl else "Time ⏰")
            print(f"{color}{reason} | Entry {entry_price:.1f} -> Exit {current_price:.1f} | Net: {net_pnl:+.4f} USDT\033[0m")
            
            active_orders.remove(order)

# ==========================================
# 🔮 PREDICTION LOGIC
# ==========================================
def run_prediction(current_data, current_price, current_ts):
    global pending_orders, active_orders, history_buffer
    
    # ห้ามเปิดเพิ่มถ้ามีออเดอร์ค้างอยู่ (Focus ทีละไม้)
    if len(pending_orders) > 0 or len(active_orders) > 0:
        return

    features = calculate_features(current_data, history_buffer)
    if features is None: return
    
    # เรียงลำดับ Feature ให้ตรงกับตอนเทรน
    feature_cols = [
        'total_volume', 'net_flow', 'trade_count',
        'net_flow_ma5', 'net_flow_ma15', 'volume_ma5',
        'net_flow_diff', 'price_change', 
        'std_5', 'dist_ma15', 'rsi'
    ]
    
    # Predict
    X = pd.DataFrame([features])[feature_cols]
    prob = model.predict(X)[0]
    
    # Signal Logic
    if prob >= CONFIDENCE_THRESHOLD:
        position_value = INVESTMENT_USDT * LEVERAGE
        
        # Maker Entry: ตั้งราคาที่ Close Price ปัจจุบัน (หวังให้มันย่อมา match)
        limit_price = current_price 
        size_btc = position_value / limit_price
        
        print(f"📡 AI Signal ({prob*100:.1f}%) | Placing Bid @ {limit_price:.2f}...")
        
        pending_orders.append({
            'limit_price': limit_price,
            'size_btc': size_btc,
            'order_ts': current_ts
        })

# ==========================================
# 📡 WEBSOCKET LOGIC
# ==========================================
def on_message(ws, message):
    global current_second_data
    try:
        data = json.loads(message)
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m'] # True=คนขายเป็น Maker(เราซื้อ Taker), False=คนซื้อเป็น Maker
        ts = int(data['T'] / 1000)
        
        # อัพเดทสถานะออเดอร์ทุกครั้งที่มีราคาไหลเข้ามา
        manage_orders(price, ts)
        
        # เริ่มวินาทีใหม่
        if current_second_data['timestamp_sec'] is None:
            current_second_data['timestamp_sec'] = ts
            
        if ts > current_second_data['timestamp_sec']:
            # 1. จบวินาทีเก่า -> บันทึกเข้า History
            current_second_data['close_price'] = price # ใช้ราคาล่าสุดเป็น Close
            history_buffer.append(current_second_data.copy())
            
            # 2. ให้ AI ทำนายจากข้อมูลที่เพิ่งจบไป
            run_prediction(current_second_data, price, ts)
            
            # 3. รีเซ็ตตัวแปรสำหรับวินาทีใหม่
            current_second_data = {
                'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0,
                'close_price': price, 'timestamp_sec': ts
            }
            
        # สะสมข้อมูลในวินาทีปัจจุบัน
        # is_maker=True แปลว่าคนส่งคำสั่งขายตั้งรอไว้ (เราไปเคาะขวาซื้อ = buy flow)
        # แต่ใน AggTrade: m=True คือ "The buyer was the maker" -> แปลว่าเป็น Sell Order (เคาะซ้าย)
        # m=False คือ "The seller was the maker" -> แปลว่าเป็น Buy Order (เคาะขวา)
        
        signed_vol = qty if not is_maker else -qty
        
        current_second_data['net_flow'] += signed_vol
        current_second_data['total_volume'] += qty
        current_second_data['trade_count'] += 1
        current_second_data['close_price'] = price
        
    except Exception as e:
        print(f"Error: {e}")

def on_close(ws, close_status_code, close_msg):
    print("\n" + "="*50)
    print(f"🛑 Bot Stopped")
    print(f"💰 Total Net PnL: {stats['total_net_pnl']:.4f} USDT")
    print(f"📊 Win: {stats['win']} | Loss: {stats['loss']} | Missed: {stats['missed']}")
    print("="*50)

def on_open(ws):
    print(f"⚡ Connection Opened: {SYMBOL.upper()} (Strategy: Maker Entry + Taker Exit)")
    print(f"🎯 Target: {TARGET_TP_PERCENT*100}% | SL: {STOP_LOSS_PERCENT*100}%")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message, on_close=on_close)
    ws.run_forever()