import websocket
import json
import pandas as pd
import datetime
import requests
import time
from xgboost import XGBClassifier

# ==========================================
# ⚙️ ตั้งค่า (Configuration)
# ==========================================
MODEL_FILE = 'btc_scalping_model.json'
CONFIDENCE_THRESHOLD = 0.65
HOLDING_TIME = 5
MIN_FLOW = 0.01
SYMBOL = "btcusdt"
SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"

# 🔑 TradingView Paper Trade Settings
TRADINGVIEW_ACCOUNT_ID = "26969300"
TRADINGVIEW_SESSION_ID = "3gz2chewk5w7tp9cuq6f3c7bwvxeu1yd"
TRADINGVIEW_SESSION_SIGN = "v3:NqI0XEOsTQzh0dflQV9cXtwbBa+cW4NVNCGoFNq03vg="
DEVICE_TOKEN = "cU1EcEFROjA.VNGLVg1eShYJogpQ_dyv8jsafgC1UzNvT212Kpi2jVw"

TRADE_QTY = 1

TV_BASE_URL = "https://papertrading.tradingview.com"

# โหลดโมเดล
print(f"🧠 กำลังโหลดสมอง AI จาก {MODEL_FILE}...")
model = XGBClassifier()
model.load_model(MODEL_FILE)
print("✅ พร้อมทำงาน!")

# ตัวแปรเก็บสถานะ
current_second_data = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'close_price': 0.0, 'timestamp_sec': None}
active_position = None
stats = {'win': 0, 'loss': 0, 'total_pnl': 0.0}

# สร้าง Session
session = requests.Session()

# ==========================================
# 🔧 TradingView API Functions
# ==========================================
def get_headers():
    """สร้าง Headers สำหรับ TradingView - ตรงตาม Browser!"""
    return {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9,th;q=0.8",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "origin": "https://th.tradingview.com",
        "referer": "https://th.tradingview.com/",
        "sec-ch-ua": '"Google Chrome";v="143", "Chromium";v="143", "Not A(Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"macOS"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36",
        "cookie": f"sessionid={TRADINGVIEW_SESSION_ID}; sessionid_sign={TRADINGVIEW_SESSION_SIGN}; device_t={DEVICE_TOKEN}"
    }

def place_order(side, qty=TRADE_QTY):
    """ส่งคำสั่ง Buy/Sell ไป TradingView - Format ตรงตาม Browser!"""
    url = f"{TV_BASE_URL}/trading/place/{TRADINGVIEW_ACCOUNT_ID}"
    
    # 🔧 Payload ตรงตามที่ Browser ส่ง!
    payload = {
        "symbol": "BINANCE:BTCUSDT",
        "type": "market",
        "qty": qty,
        "side": side,
        "outside_rth": False,
        "outside_rth_tp": False
    }
    
    # ส่งเป็น JSON string ใน body
    body = json.dumps(payload)
    
    try:
        response = session.post(
            url,
            headers=get_headers(),
            data=body,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            order_id = result.get('id', 'N/A')
            status = result.get('status', 'N/A')
            print(f"   ✅ TradingView: {side.upper()} {qty} BTC | Order: {order_id} | Status: {status}")
            return result
        else:
            print(f"   ❌ TradingView Error: {response.status_code}")
            print(f"   📄 Response: {response.text[:300]}")
            return None
            
    except requests.exceptions.Timeout:
        print(f"   ⏱️ Timeout...")
        return None
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return None

def open_long():
    """เปิด Long (ซื้อ)"""
    return place_order("buy", TRADE_QTY)

def close_long():
    """ปิด Long (ขาย)"""
    return place_order("sell", TRADE_QTY)

# ==========================================
# 🎯 Trading Logic
# ==========================================
def check_active_position(current_price, current_ts):
    """ตรวจสอบและปิด position"""
    global active_position, stats
    
    if active_position and current_ts >= active_position['check_time']:
        
        entry_price = active_position['entry_price']
        diff = current_price - entry_price
        pnl = diff * TRADE_QTY
        
        print(f"\n🏁 ปิด Position...")
        result = close_long()
        
        stats['total_pnl'] += pnl
        
        if pnl > 0:
            stats['win'] += 1
            result_text = "\033[92m✅ WIN\033[0m"
        else:
            stats['loss'] += 1
            result_text = "\033[91m❌ LOSS\033[0m"
        
        total = stats['win'] + stats['loss']
        win_rate = (stats['win'] / total * 100) if total > 0 else 0
        
        print(f"   📊 Entry: ${entry_price:.2f} → Exit: ${current_price:.2f}")
        print(f"   💰 PnL: ${pnl:+.2f} | Total: ${stats['total_pnl']:+.2f} | {result_text}")
        print(f"   🎯 Win Rate: {win_rate:.1f}% ({stats['win']}W/{stats['loss']}L)")
        print("-" * 50)
        
        active_position = None

def predict_and_trade(data_row, current_price, current_ts):
    """ทำนายและเปิด position"""
    global active_position
    
    if active_position:
        return
    
    df = pd.DataFrame([data_row])
    feature_cols = ['total_volume', 'net_flow', 'trade_count']
    X = df[feature_cols]
    
    probs = model.predict_proba(X)[0]
    prob_buy = probs[1]
    
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    
    if prob_buy >= CONFIDENCE_THRESHOLD and data_row['net_flow'] >= MIN_FLOW:
        
        print(f"\n🚀 {timestamp} | สัญญาณซื้อ! | มั่นใจ {prob_buy*100:.0f}% | Flow: {data_row['net_flow']:.2f}")
        
        result = open_long()
        
        if result:
            active_position = {
                'entry_price': current_price,
                'check_time': current_ts + HOLDING_TIME
            }

def on_message(ws, message):
    global current_second_data
    try:
        data = json.loads(message)
        
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m']
        ts = int(data['T'] / 1000)
        
        check_active_position(price, ts)
        
        if current_second_data['timestamp_sec'] is None:
            current_second_data['timestamp_sec'] = ts
            
        if ts > current_second_data['timestamp_sec']:
            current_second_data['close_price'] = price 
            predict_and_trade(current_second_data, price, ts)
            
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

def on_error(ws, error): 
    print(f"Error: {error}")

def on_close(ws, close_status_code, close_msg): 
    print("\n" + "=" * 50)
    print("📊 สรุปผล TradingView Paper Trade")
    print("=" * 50)
    print(f"💰 Total PnL: ${stats['total_pnl']:+.2f}")
    print(f"📊 Trades: {stats['win']}W / {stats['loss']}L")
    total = stats['win'] + stats['loss']
    if total > 0:
        print(f"🎯 Win Rate: {stats['win']/total*100:.1f}%")
    print("=" * 50)

def on_open(ws): 
    print("=" * 50)
    print("🤖 AUTO TRADE BOT → TradingView Paper Trade")
    print("=" * 50)
    print(f"📊 Account ID: {TRADINGVIEW_ACCOUNT_ID}")
    print(f"💹 Quantity: {TRADE_QTY} BTC")
    print(f"🎯 Threshold: {CONFIDENCE_THRESHOLD*100}%")
    print(f"⏱️ Holding Time: {HOLDING_TIME} วินาที")
    print("=" * 50)
    print("🟢 เริ่มเทรดอัตโนมัติ...\n")

if __name__ == "__main__":
    print("🔄 เริ่มต้น Bot...")
    print("✅ เชื่อมต่อ Binance WebSocket...")
    ws = websocket.WebSocketApp(SOCKET, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
    ws.run_forever()