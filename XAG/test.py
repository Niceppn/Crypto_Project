import websocket
import json
import pandas as pd
import datetime
import lightgbm as lgb
from collections import deque

# ==========================================
# ⚙️ ตั้งค่า (Configuration)
# ==========================================
MODEL_FILE = '/Users/Macbook/Collect_Crypto/XAG/xagusdt_scalping_model_lgbm.txt'
CONFIDENCE_THRESHOLD = 0.55
HOLDING_TIME = 5
MIN_PROFIT = 0.1

# ตั้งค่าจำลอง Spread ของ XM (Silver เฉลี่ย 0.04)
# หมายความว่า ราคาซื้อจะแพงกว่าราคาขายอยู่ 0.04
XM_SPREAD = 0.04 

SYMBOL = "xagusdt"
SOCKET = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"

# โหลดโมเดล
print(f"🧠 กำลังโหลดสมอง AI...")
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    print("✅ โมเดลพร้อมทำงานบน Mac!")
except:
    print(f"❌ ไม่เจอไฟล์โมเดลที่: {MODEL_FILE}")
    exit()

# ==========================================
# 📊 ตัวแปรเก็บข้อมูล
# ==========================================
history_buffer = deque(maxlen=30)
current_second_data = {
    'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 
    'close_price': 0.0, 'timestamp_sec': None
}
active_orders = []
stats = {'win': 0, 'loss': 0, 'breakeven': 0}
total_profit_binance = 0.0
total_profit_xm = 0.0  # กำไรจำลองของ XM

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
        'net_flow_ma5': df['net_flow'].tail(5).mean(),
        'net_flow_ma15': df['net_flow'].tail(15).mean(),
        'net_flow_ma30': df['net_flow'].tail(30).mean(),
        'volume_ma5': df['total_volume'].tail(5).mean(),
        'volume_ma15': df['total_volume'].tail(15).mean(),
        'net_flow_ratio': current_data['net_flow'] / (current_data['total_volume'] + 1e-10),
        'cumulative_net_flow_30': df['net_flow'].tail(30).sum()
    }
    features['net_flow_diff'] = current_data['net_flow'] - df['net_flow'].iloc[-2]
    features['net_flow_diff5'] = current_data['net_flow'] - df['net_flow'].iloc[-6] if len(df) >= 6 else 0
    features['price_change'] = ((current_data['close_price'] - df['close_price'].iloc[-2]) / df['close_price'].iloc[-2]) * 100
    features['price_change_ma5'] = df['close_price'].pct_change().tail(5).mean() * 100 if len(df) >= 6 else 0
    return features

# ==========================================
# 🏁 ตรวจผล Orders (จำลอง XM)
# ==========================================
def check_active_orders(bn_current_price, current_ts):
    global active_orders, stats, total_profit_binance, total_profit_xm
    
    # จำลองราคา Bid ของ XM (ราคาที่เราจะขายคืน)
    # สมมติราคา Binance คือราคากลาง -> Bid จะต่ำกว่ากลางครึ่ง Spread
    xm_bid_simulated = bn_current_price - (XM_SPREAD / 2)
    
    for order in active_orders[:]:
        if current_ts >= order['check_time']:
            # 1. ผล Binance
            diff_bn = bn_current_price - order['entry_bn']
            
            # 2. ผล XM (จำลอง)
            # กำไร = ราคาขาย(Bid) - ราคาซื้อ(Ask ตอนเข้า)
            diff_xm = xm_bid_simulated - order['entry_xm_ask']
            
            # เก็บสถิติ
            if diff_bn >= MIN_PROFIT:
                stats['win'] += 1
                res = "\033[92mWIN\033[0m"
            elif diff_bn > -MIN_PROFIT:
                stats['breakeven'] += 1
                res = "\033[93mBE\033[0m"
            else:
                stats['loss'] += 1
                res = "\033[91mLOSS\033[0m"
            
            total_profit_binance += diff_bn
            total_profit_xm += diff_xm
            
            print(f"🏁 ผลลัพธ์: {res}")
            print(f"   🔸 Binance: เข้า {order['entry_bn']:.3f} -> ออก {bn_current_price:.3f} | กำไร: {diff_bn:+.3f}")
            print(f"   🔹 XM (Sim): เข้า {order['entry_xm_ask']:.3f} -> ออก {xm_bid_simulated:.3f} | กำไร: {diff_xm:+.3f} (โดน Spread)")
            print(f"   💰 Total PN: BN={total_profit_binance:.2f} | XM={total_profit_xm:.2f}")
            print("-" * 50)
            
            active_orders.remove(order)

# ==========================================
# 🔮 ฟังก์ชันทำนาย
# ==========================================
def predict_signal(current_data, bn_price, current_ts):
    global active_orders, history_buffer
    
    features = calculate_features(current_data, history_buffer)
    if features is None: return
    
    feature_cols = [
        'total_volume', 'net_flow', 'trade_count', 'net_flow_ma5', 'net_flow_ma15', 'net_flow_ma30',
        'volume_ma5', 'volume_ma15', 'net_flow_diff', 'net_flow_diff5', 'price_change', 
        'price_change_ma5', 'net_flow_ratio', 'cumulative_net_flow_30'
    ]
    
    try:
        prob_buy = model.predict(pd.DataFrame([features])[feature_cols])[0]
    except: return

    if prob_buy >= CONFIDENCE_THRESHOLD:
        target_time = current_ts + HOLDING_TIME
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        # จำลองราคา Ask ของ XM (ราคาที่เราต้องซื้อ แพงกว่าราคากลาง)
        xm_ask_simulated = bn_price + (XM_SPREAD / 2)
        
        print(f"🚀 {timestamp} | BUY SIGNAL ({prob_buy*100:.1f}%)")
        print(f"   🔸 BN Entry: {bn_price:.3f}")
        print(f"   🔹 XM Entry: {xm_ask_simulated:.3f} (รวม Spread 0.04)")
        
        active_orders.append({
            'entry_bn': bn_price,
            'entry_xm_ask': xm_ask_simulated,
            'check_time': target_time
        })

# ==========================================
# 📡 Main WebSocket
# ==========================================
def on_message(ws, message):
    global current_second_data
    try:
        data = json.loads(message)
        price = float(data['p'])
        qty = float(data['q'])
        ts = int(data['T'] / 1000)
        
        check_active_orders(price, ts)
        
        if current_second_data['timestamp_sec'] is None:
            current_second_data['timestamp_sec'] = ts
            
        if ts > current_second_data['timestamp_sec']:
            current_second_data['close_price'] = price
            history_buffer.append(current_second_data.copy())
            if len(history_buffer) >= 5:
                predict_signal(current_second_data, price, ts)
            current_second_data = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'close_price': price, 'timestamp_sec': ts}
        
        signed_vol = qty if not data['m'] else -qty
        current_second_data['net_flow'] += signed_vol
        current_second_data['total_volume'] += qty
        current_second_data['trade_count'] += 1
        current_second_data['close_price'] = price
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print(f"🤖 Bot Started on Mac (XM Spread Simulation Mode: {XM_SPREAD})")
    ws = websocket.WebSocketApp(SOCKET, on_message=on_message, on_open=lambda ws: print("--- Connected ---"))
    ws.run_forever()