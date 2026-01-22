import websocket
import json
import pandas as pd
import datetime
import requests
import time
import lightgbm as lgb
from collections import deque

# ==========================================
# Configuration
# ==========================================
MODEL_FILE = '/Users/Macbook/Collect_Crypto/XAU/xauusdt_scalping_model_lgbm.txt'
CONFIDENCE_THRESHOLD = 0.65
HOLDING_TIME = 5
MIN_FLOW = 0.01
SYMBOL = "xauusdt"
SOCKET = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"

# TradingView Paper Trade Settings
TRADINGVIEW_ACCOUNT_ID = "26969300"
TRADINGVIEW_SESSION_ID = "3gz2chewk5w7tp9cuq6f3c7bwvxeu1yd"
TRADINGVIEW_SESSION_SIGN = "v3:NqI0XEOsTQzh0dflQV9cXtwbBa+cW4NVNCGoFNq03vg="
DEVICE_TOKEN = "cU1EcEFROjA.VNGLVg1eShYJogpQ_dyv8jsafgC1UzNvT212Kpi2jVw"

# Updated cookies from browser (ล่าสุด)
COOKIE_STRING = "cookiePrivacyPreferenceBannerProduction=notApplicable; _ga=GA1.1.1729562086.1767587987; cookiesSettings={\"analytics\":true,\"advertising\":true}; device_t=cU1EcEFROjA.VNGLVg1eShYJogpQ_dyv8jsafgC1UzNvT212Kpi2jVw; sessionid=3gz2chewk5w7tp9cuq6f3c7bwvxeu1yd; sessionid_sign=v3:NqI0XEOsTQzh0dflQV9cXtwbBa+cW4NVNCGoFNq03vg=; etg=undefined; cachec=undefined; _sp_ses.cf1a=*; _sp_id.cf1a=59fe9406-0d6a-47df-998d-ba97f2735c2d.1767587987.17.1768994993.1768979119.20ccd534-80a0-4eda-a3dd-bb962285af57.bb689f90-3a6f-4b7b-a809-b271cbb1c524.f0aa93a7-c4d1-4b29-87b7-f6085dcf6574.1768994984747.2; _ga_YVVRYGL0E0=GS2.1.s1768991547$o39$g1$t1768994993$j43$l0$h0"

TRADE_QTY = 1  # 0.01 XAUUSD (ประมาณ $20-30 ต่อ trade)
TV_BASE_URL = "https://papertrading.tradingview.com"

# โหลดโมเดล LightGBM
print(f"🧠 กำลังโหลดสมอง AI จาก {MODEL_FILE}...")
model = lgb.Booster(model_file=MODEL_FILE)
print("✅ พร้อมทำงาน!")
print(f"⚙️ Threshold: {CONFIDENCE_THRESHOLD*100}% | Holding Time: {HOLDING_TIME}s | Min Flow: {MIN_FLOW}")

# State variables
history_buffer = deque(maxlen=30)

current_second_data = {
    'net_flow': 0.0,
    'total_volume': 0.0,
    'trade_count': 0,
    'close_price': 0.0,
    'timestamp_sec': None
}

active_position = None
stats = {'win': 0, 'loss': 0, 'breakeven': 0, 'total_pnl': 0.0}

# Create session
session = requests.Session()

# ==========================================
# TradingView API Functions
# ==========================================
def get_headers():
    """Create TradingView headers (browser identical)"""
    return {
        "accept": "*/*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-US,en;q=0.9,th;q=0.8",
        "content-type": "application/x-www-form-urlencoded; charset=UTF-8",
        "origin": "https://th.tradingview.com",
        "referer": "https://th.tradingview.com/chart/xDnVosHH/?symbol=BINANCE%3ABTCUSDT",
        "sec-ch-ua": '"Not(A:Brand";v="8", "Chromium";v="144", "Google Chrome";v="144"',
        "sec-ch-ua-mobile": "?1",
        "sec-ch-ua-platform": '"Android"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-site",
        "user-agent": "Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/144.0.0.0 Mobile Safari/537.36",
        "x-language": "th_TH",
        "x-requested-with": "XMLHttpRequest",
        "cookie": COOKIE_STRING
    }

def place_order(side, qty=TRADE_QTY):
    """Send Buy/Sell order to TradingView"""
    url = f"{TV_BASE_URL}/trading/place/{TRADINGVIEW_ACCOUNT_ID}"

    payload = {
        "symbol": "OANDA:XAUUSD",  # XAU/USD บน TradingView
        "type": "market",
        "qty": qty,
        "side": side,
        "outside_rth": False,
        "outside_rth_tp": False
    }

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
            print(f"📋 TradingView {side.upper()} {qty} XAU | Order {order_id} | Status {status}")
            return result
        else:
            print(f"❌ TradingView Error: {response.status_code}")
            print(f"Response: {response.text[:300]}")
            return None

    except requests.exceptions.Timeout:
        print("⏰ Request timeout")
        return None
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def open_long():
    """Open long position"""
    return place_order("buy", TRADE_QTY)

def close_long():
    """Close long position"""
    return place_order("sell", TRADE_QTY)

# ==========================================
# Feature Calculation (จากโค้ดเดิม)
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
# Trading Logic
# ==========================================
def check_active_position(current_price, current_ts):
    """Check and close active position"""
    global active_position, stats

    if active_position and current_ts >= active_position['check_time']:

        entry_price = active_position['entry_price']
        diff = current_price - entry_price
        pnl = diff * TRADE_QTY

        print("🔄 Closing position...")
        close_long()

        stats['total_pnl'] += pnl

        if diff > 0.01:  # ใช้ MIN_PROFIT = 0.01
            stats['win'] += 1
            result_text = "✅ WIN"
        elif diff > -0.01:
            stats['breakeven'] += 1
            result_text = "➖ BREAKEVEN"
        else:
            stats['loss'] += 1
            result_text = "❌ LOSS"

        total = stats['win'] + stats['loss'] + stats['breakeven']
        real_trades = stats['win'] + stats['loss']
        win_rate = (stats['win'] / real_trades * 100) if real_trades > 0 else 0

        print("-" * 50)
        print(f"🏁 ปิดออเดอร์: {entry_price:.2f} -> {current_price:.2f} ({diff:+.2f}) | {result_text}")
        print(f"📊 Score: {stats['win']}W - {stats['loss']}L - {stats['breakeven']}BE")
        print(f"💰 Total PnL: {stats['total_pnl']:+.2f} USD")
        print(f"📈 Win Rate: {win_rate:.1f}%")
        print("-" * 50)

        active_position = None

def predict_and_trade(data_row, current_price, current_ts):
    """Predict and open position"""
    global active_position, history_buffer

    if active_position:
        return

    # คำนวณ features
    features = calculate_features(data_row, history_buffer)
    
    if features is None:
        return

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

    if prob_buy >= CONFIDENCE_THRESHOLD :
        print(f"🚀 {timestamp} | XAU BUY signal! | Confidence {prob_buy*100:.1f}% | Flow {data_row['net_flow']:.2f} | Price {current_price:.2f}")

        result = open_long()

        if result:
            active_position = {
                'entry_price': current_price,
                'check_time': current_ts + HOLDING_TIME
            }

def on_message(ws, message):
    global current_second_data, history_buffer
    
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
            
            # เก็บเข้า history buffer
            history_buffer.append(current_second_data.copy())
            
            # ทำนาย (ต้องมีอย่างน้อย 5 วินาที)
            if len(history_buffer) >= 5:
                predict_and_trade(current_second_data, price, ts)

            # รีเซ็ตข้อมูลวินาทีใหม่
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
        current_second_data['close_price'] = price

    except Exception as e:
        print(f"❌ Error: {e}")

def on_error(ws, error):
    print(f"❌ WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### TradingView Paper Trade Summary ###")
    total = stats['win'] + stats['loss'] + stats['breakeven']
    real_trades = stats['win'] + stats['loss']
    win_rate = (stats['win'] / real_trades * 100) if real_trades > 0 else 0
    
    print("="*50)
    print("📊 สรุปผลการเทรด XAU")
    print("="*50)
    print(f"📈 จำนวน trades: {stats['win']}W / {stats['loss']}L / {stats['breakeven']}BE")
    print(f"🎯 Win Rate: {win_rate:.1f}%")
    print(f"💰 กำไร/ขาดทุนรวม: {stats['total_pnl']:+.2f} USD")
    
    if total > 0:
        avg_pnl = stats['total_pnl'] / total
        print(f"� กำไรเฉลี่ยต่อ trade: {avg_pnl:+.2f} USD")
        
        if stats['total_pnl'] > 0:
            print("🎉 ทำกำไรสุทธิ!")
        elif stats['total_pnl'] < 0:
            print("📉 ขาดทุนสุทธิ")
        else:
            print("➖ ทำกำไร/ขาดทุนตั้งศูนย์")
    
    print(f"⏰ เวลาทำงาน: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)

def on_open(ws):
    print("="*60)
    print("🤖 XAU Scalping Bot - TradingView Paper Trade Edition")
    print("="*60)
    print(f"📡 Account ID: {TRADINGVIEW_ACCOUNT_ID}")
    print(f"💰 Quantity: {TRADE_QTY} XAU per trade")
    print(f"⚙️ Threshold: {CONFIDENCE_THRESHOLD*100}%")
    print(f"⏰ Holding Time: {HOLDING_TIME} seconds")
    print(f"🎯 Symbol: OANDA:XAUUSD")
    print("🚀 Auto trading started...")
    print(f"⏳ รอสะสมข้อมูล 5 วินาทีก่อนเริ่มทำนาย...")
    
    # # ทดสอบส่ง buy order ทันที
    # print("\n" + "="*50)
    # print("🧪 TESTING BUY ORDER NOW...")
    # print("="*50)
    # test_result = place_order("buy", TRADE_QTY)
    # if test_result:
    #     print("✅ Test buy successful! ไปดูใน TradingView ได้เลย")
    # else:
    #     print("❌ Test buy failed! ตรวจสอบ session/headers")
    # print("="*50 + "\n")

if __name__ == "__main__":
    print("🔄 Starting XAU TradingView Bot...")
    print("📡 Connecting to Binance WebSocket for XAU data...")
    
    ws = websocket.WebSocketApp(
        SOCKET,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()
