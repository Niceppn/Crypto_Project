import websocket
import json
import pandas as pd
import threading
import time
from xgboost import XGBClassifier

# ==========================================
# ⚙️ Configuration
# ==========================================
MODEL_FILE = 'btc_scalping_model.json'
CONFIDENCE_THRESHOLD = 0.65
HOLDING_TIME = 5
MIN_PROFIT = 0.01

# URLs
BINANCE_SOCKET = "wss://stream.binance.com:9443/ws/btcusdt@aggTrade"
# หมายเหตุ: หาก Exness ยังติด 400 ให้ใช้ราคาจาก Binance เป็นตัวทดสอบก่อน
EXNESS_SOCKET = "wss://rtapi-sg.eccweb.mobi/rtapi/mt5/trial17/v2/ws/ticks/accounts/270824782"

# 🔑 Auth Headers (Update จาก Browser ของคุณ)
EXNESS_HEADERS = [
    "Authorization: Bearer eyJhbGciOiJSUzI1Ni...",
    "Cookie: __cf_bm=YZyuuDeZH92QVQHCTKtLpwT5rO..."
]

# Shared State
market_data = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'ts': 0}
exness_price = {'bid': 0.0, 'ts': 0}
active_orders = []

model = XGBClassifier()
model.load_model(MODEL_FILE)

def binance_worker():
    """ดึงข้อมูล Net Flow และ Volume จาก Binance (เสถียรและแม่นยำ)"""
    global market_data
    def on_message(ws, message):
        data = json.loads(message)
        qty = float(data['q'])
        is_maker = data['m']
        market_data['net_flow'] += -qty if is_maker else qty
        market_data['total_volume'] += qty
        market_data['trade_count'] += 1
        market_data['ts'] = int(data['T'] / 1000)

    ws = websocket.WebSocketApp(BINANCE_SOCKET, on_message=on_message)
    ws.run_forever()

def exness_worker():
    """ดึงราคาจาก Exness เพื่อใช้เข้าเทรดจริง"""
    global exness_price
    def on_message(ws, message):
        data = json.loads(message)
        if 'b' in data: # ราคา Bid จาก Exness
            exness_price['bid'] = float(data['b'])
            exness_price['ts'] = int(data['t'] / 1000)
            
    ws = websocket.WebSocketApp(EXNESS_SOCKET, header=EXNESS_HEADERS, on_message=on_message)
    ws.run_forever(origin="https://trading.exness.com")

def brain_worker():
    """Logic การตัดสินใจ (ใช้ข้อมูล Binance ทำนาย -> เทรดบนราคา Exness)"""
    global market_data, exness_price, active_orders
    last_processed_ts = 0
    
    while True:
        curr_ts = market_data['ts']
        if curr_ts > last_processed_ts:
            # 1. ตรวจผลออเดอร์เก่าบนราคา Exness
            for order in active_orders[:]:
                if curr_ts >= order['check_time']:
                    diff = exness_price['bid'] - order['entry_price']
                    print(f"🏁 Exness Result: {'WIN' if diff >= MIN_PROFIT else 'LOSS'} ({diff:.2f})")
                    active_orders.remove(order)

            # 2. ทำนายสัญญาณใหม่จากข้อมูล Binance
            X = pd.DataFrame([market_data])[['total_volume', 'net_flow', 'trade_count']]
            prob = model.predict_proba(X)[0][1]
            
            if prob >= CONFIDENCE_THRESHOLD and exness_price['bid'] > 0:
                print(f"🚀 Signal! Conf: {prob*100:.0f}% | Entry Exness: {exness_price['bid']}")
                active_orders.append({'entry_price': exness_price['bid'], 'check_time': curr_ts + HOLDING_TIME})

            # Reset ข้อมูลรายวินาที
            last_processed_ts = curr_ts
            market_data = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'ts': curr_ts}
        
        time.sleep(0.1)

if __name__ == "__main__":
    threading.Thread(target=binance_worker, daemon=True).start()
    threading.Thread(target=exness_worker, daemon=True).start()
    brain_worker()