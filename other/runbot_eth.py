import websocket
import json
import pandas as pd
import datetime
from xgboost import XGBClassifier
from collections import deque

# ==========================================
# ⚙️ ตั้งค่า (Configuration)
# ==========================================
MODEL_FILE = 'eth_scalping_model.json'
CONFIDENCE_THRESHOLD = 0.65   # เพิ่มจาก 0.65 → 0.75
HOLDING_TIME = 5
MIN_PROFIT = 0.01              # 🆕 กำไรขั้นต่ำ (USD) ถึงจะนับว่า WIN
MIN_FLOW = 0.01                # 🆕 Net Flow ขั้นต่ำ (เพิ่มจาก 0.01)
SYMBOL = "ethusdt"
SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"


# โหลดโมเดล
print(f"🧠 กำลังโหลดสมอง AI จาก {MODEL_FILE}...")
model = XGBClassifier()
model.load_model(MODEL_FILE)
print("✅ พร้อมทำงาน!")
print(f"⚙️ Threshold: {CONFIDENCE_THRESHOLD*100}% | Min Profit: ${MIN_PROFIT} | Min Flow: {MIN_FLOW}")

# ตัวแปรเก็บสถานะ
current_second_data = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'close_price': 0.0, 'timestamp_sec': None}
active_orders = [] 
stats = {'win': 0, 'loss': 0, 'breakeven': 0}  # 🆕 เพิ่ม breakeven

def check_active_orders(current_price, current_ts):
    """ฟังก์ชันตรวจการบ้าน"""
    global active_orders, stats
    
    for order in active_orders[:]:
        if current_ts >= order['check_time']:
            
            diff = current_price - order['entry_price']
            
            # 🆕 แยก 3 กรณี: WIN / BREAKEVEN / LOSS
            if diff >= MIN_PROFIT:
                stats['win'] += 1
                result_text = f"\033[92m✅ WIN \033[0m"
                profit_text = f"(+{diff:.2f})"
            elif diff > -MIN_PROFIT:  # อยู่ระหว่าง -5 ถึง +5
                stats['breakeven'] += 1
                result_text = f"\033[93m➖ BREAKEVEN\033[0m"
                profit_text = f"({diff:+.2f})"
            else:
                stats['loss'] += 1
                result_text = f"\033[91m❌ LOSS\033[0m"
                profit_text = f"({diff:.2f})"
            
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            real_trades = stats['win'] + stats['loss']  # ไม่นับ breakeven
            win_rate = (stats['win'] / real_trades * 100) if real_trades > 0 else 0
            
            print(f"🏁 ตรวจผล: เข้า {order['entry_price']:.2f} -> ออก {current_price:.2f} {profit_text} | ผล: {result_text} | Win Rate: {win_rate:.1f}% ({stats['win']}W/{stats['loss']}L/{stats['breakeven']}BE)")
            
            active_orders.remove(order)

def predict_signal(data_row, current_entry_price, current_real_time):
    """ฟังก์ชันทำนาย"""
    global active_orders
    
    df = pd.DataFrame([data_row])
    feature_cols = ['total_volume', 'net_flow', 'trade_count']
    X = df[feature_cols]
    
    probs = model.predict_proba(X)[0]
    prob_buy = probs[1]
    
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    
    # 🆕 เงื่อนไขเข้มงวดขึ้น
    if prob_buy >= CONFIDENCE_THRESHOLD and data_row['net_flow'] >= MIN_FLOW:
        
        target_time = current_real_time + HOLDING_TIME
        
        print(f"🚀 {timestamp} | สัญญาณซื้อ! (Flow: {data_row['net_flow']:.2f}) | มั่นใจ {prob_buy*100:.0f}% | ราคาเข้า: {current_entry_price:.2f}")
        
        active_orders.append({
            'entry_price': current_entry_price,
            'check_time': target_time
        })

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
            predict_signal(current_second_data, price, ts)
            
            current_second_data = {
                'net_flow': 0.0, 
                'total_volume': 0.0, 
                'trade_count': 0, 
                'close_price': price, 
                'timestamp_sec': ts
            }
        
        signed_vol = qty if not is_maker else -qty
        current_second_data['net_flow'] += signed_vol
        current_second_data['total_volume'] += qty
        current_second_data['trade_count'] += 1
        
    except Exception as e:
        print(f"Error: {e}")

def on_error(ws, error): print(f"Error: {error}")
def on_close(ws, close_status_code, close_msg): print("### Disconnected ###")
def on_open(ws): print(f"--- เชื่อมต่อ Binance ({SYMBOL}) สำเร็จ! กำลังรอสัญญาณ... ---")

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()