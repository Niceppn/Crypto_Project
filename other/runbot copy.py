import websocket
import json
import pandas as pd
import datetime
from xgboost import XGBClassifier

# ==========================================
# ⚙️ ตั้งค่า
# ==========================================
MODEL_FILE = 'btc_scalping_model.json'
CONFIDENCE_THRESHOLD = 0.55
HOLDING_TIME = 5
MIN_PROFIT = 0.02
MIN_FLOW = 0.01
SYMBOL = "btcusdt"
SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"

# 🆕 Hybrid Mode Settings
TAKE_PROFIT = 0.01
MAX_SELL_WAIT_TIME = 15
STOP_LOSS = -10.0

# โหลดโมเดล
print(f"🧠 โหลด AI...")
model = XGBClassifier()
model.load_model(MODEL_FILE)
print(f"✅ พร้อม! | TP: +${TAKE_PROFIT} | SL: ${STOP_LOSS} | Wait: {MAX_SELL_WAIT_TIME}s")

# ตัวแปร
current_second_data = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'close_price': 0.0, 'timestamp_sec': None}
pending_sell_orders = []
active_position = None
stats = {'win': 0, 'loss': 0, 'breakeven': 0, 'sell_matched': 0, 'sell_timeout': 0, 'stop_loss': 0}
trade_count = 0  # นับเทรด

def check_pending_sell_orders(current_price, current_ts):
    global pending_sell_orders, active_position, stats
    
    for order in pending_sell_orders[:]:
        entry_price = order['entry_price']
        sell_price = order['sell_price']
        current_pnl = current_price - entry_price
        trade_id = order['trade_id']
        
        # ✅ Take Profit - ขายที่ตั้งไว้สำเร็จ!
        if current_price >= sell_price:
            profit = sell_price - entry_price
            stats['win'] += 1
            stats['sell_matched'] += 1
            
            real = stats['win'] + stats['loss']
            wr = (stats['win'] / real * 100) if real > 0 else 0
            
            print(f"   4️⃣ ขายที่ตั้ง: ✅ สำเร็จ @ ${sell_price:.2f}")
            print(f"   💰 ผล: +${profit:.2f} | WR: {wr:.0f}% ({stats['win']}W/{stats['loss']}L)")
            print(f"-" * 40)
            
            pending_sell_orders.remove(order)
            active_position = None
        
        # 🛑 Stop Loss
        elif current_pnl <= STOP_LOSS:
            stats['loss'] += 1
            stats['stop_loss'] += 1
            
            real = stats['win'] + stats['loss']
            wr = (stats['win'] / real * 100) if real > 0 else 0
            
            print(f"   4️⃣ ขายที่ตั้ง: 🛑 STOP LOSS @ ${current_price:.2f} (ตั้งไว้ ${sell_price:.2f})")
            print(f"   💰 ผล: ${current_pnl:.2f} | WR: {wr:.0f}% ({stats['win']}W/{stats['loss']}L)")
            print(f"-" * 40)
            
            pending_sell_orders.remove(order)
            active_position = None
            
        # ⏱️ Timeout
        elif current_ts >= order['expire_time']:
            profit = current_price - entry_price
            stats['sell_timeout'] += 1
            
            if profit > 0:
                stats['win'] += 1
                result = "✅ กำไร"
            elif profit > -1:
                stats['breakeven'] += 1
                result = "➖ เสมอตัว"
            else:
                stats['loss'] += 1
                result = "❌ ขาดทุน"
            
            real = stats['win'] + stats['loss']
            wr = (stats['win'] / real * 100) if real > 0 else 0
            
            print(f"   4️⃣ ขายที่ตั้ง: ⏱️ หมดเวลา - ขาย Market @ ${current_price:.2f} (ตั้งไว้ ${sell_price:.2f})")
            print(f"   💰 ผล: {profit:+.2f} {result} | WR: {wr:.0f}% ({stats['win']}W/{stats['loss']}L)")
            print(f"-" * 40)
            
            pending_sell_orders.remove(order)
            active_position = None

def predict_signal(data_row, current_price, current_ts):
    global pending_sell_orders, active_position, stats, trade_count
    
    if pending_sell_orders or active_position:
        return
    
    df = pd.DataFrame([data_row])
    X = df[['total_volume', 'net_flow', 'trade_count']]
    
    probs = model.predict_proba(X)[0]
    prob_buy = probs[1]
    
    if prob_buy >= CONFIDENCE_THRESHOLD and data_row['net_flow'] >= MIN_FLOW:
        trade_count += 1
        entry_price = current_price
        sell_price = entry_price + TAKE_PROFIT
        
        ts = datetime.datetime.now().strftime('%H:%M:%S')
        
        # 1️⃣ สัญญาณซื้อ
        print(f"\n📍 Trade #{trade_count} | {ts}")
        print(f"   1️⃣ สัญญาณซื้อ: ✅ มั่นใจ {prob_buy*100:.0f}% | Flow: {data_row['net_flow']:.4f}")
        
        # 2️⃣ เข้าซื้อ
        print(f"   2️⃣ เข้าซื้อ: ✅ สำเร็จ @ ${entry_price:.2f}")
        
        # 3️⃣ ตั้งขาย
        print(f"   3️⃣ ตั้งขาย: 📝 @ ${sell_price:.2f} (TP: +${TAKE_PROFIT}) | รอ {MAX_SELL_WAIT_TIME}s")
        
        active_position = {'entry_price': entry_price, 'entry_time': current_ts}
        
        pending_sell_orders.append({
            'trade_id': trade_count,
            'entry_price': entry_price,
            'sell_price': sell_price,
            'expire_time': current_ts + MAX_SELL_WAIT_TIME
        })

def on_message(ws, message):
    global current_second_data
    try:
        data = json.loads(message)
        
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m']
        ts = int(data['T'] / 1000)
        
        check_pending_sell_orders(price, ts)
        
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
        print(f"Err: {e}")

def on_error(ws, error): print(f"Err: {error}")

def on_close(ws, a, b): 
    real = stats['win'] + stats['loss']
    wr = (stats['win'] / real * 100) if real > 0 else 0
    print(f"\n{'='*40}")
    print(f"📊 สรุป: {stats['win']}W / {stats['loss']}L / {stats['breakeven']}BE | WR: {wr:.0f}%")
    print(f"   ขายที่ตั้ง: {stats['sell_matched']} | หมดเวลา: {stats['sell_timeout']} | SL: {stats['stop_loss']}")
    print(f"{'='*40}")

def on_open(ws): 
    print(f"🤖 เริ่มเทรด... (TP: +${TAKE_PROFIT} | SL: ${STOP_LOSS})")
    print(f"=" * 40)

if __name__ == "__main__":
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()