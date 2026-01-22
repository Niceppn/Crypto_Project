import websocket
import json
import pandas as pd
import datetime
import lightgbm as lgb
from collections import deque

# ==========================================
# Configuration
# ==========================================
MODEL_FILE = '/Users/Macbook/Collect_Crypto/XAU/xauusdt_scalping_model_lgbm.txt'
CONFIDENCE_THRESHOLD = 0.55
HOLDING_TIME = 5
MIN_PROFIT = 0.01
MIN_FLOW = 0.01
SYMBOL = "xauusdt"
SOCKET = f"wss://fstream.binance.com/ws/{SYMBOL}@aggTrade"

# Load LightGBM model
print(f"Loading AI model from {MODEL_FILE}...")
model = lgb.Booster(model_file=MODEL_FILE)
print("Ready to work!")
print(f"Threshold: {CONFIDENCE_THRESHOLD*100}% | Min Profit: ${MIN_PROFIT} | Min Flow: {MIN_FLOW}")

# ==========================================
# Data Storage
# ==========================================
# Store last 30 seconds data for Moving Average calculations
history_buffer = deque(maxlen=30)

current_second_data = {
    'net_flow': 0.0, 
    'total_volume': 0.0, 
    'trade_count': 0, 
    'close_price': 0.0, 
    'timestamp_sec': None
}

active_orders = []
stats = {'win': 0, 'loss': 0, 'breakeven': 0}
total_profit = 0.0  # Total profit/loss

# ==========================================
# Feature Calculation Functions
# ==========================================
def calculate_features(current_data, history):
    """Calculate all 14 features from real-time data"""
    
    # Create DataFrame from history + current
    all_data = list(history) + [current_data]
    df = pd.DataFrame(all_data)
    
    if len(df) < 2:
        return None  # Not enough data yet
    
    # Basic features
    features = {
        'total_volume': current_data['total_volume'],
        'net_flow': current_data['net_flow'],
        'trade_count': current_data['trade_count'],
    }
    
    # Moving Averages of Net Flow
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
    
    # Cumulative Net Flow 30 seconds
    features['cumulative_net_flow_30'] = df['net_flow'].tail(30).sum()
    
    return features

# ==========================================
# Order Result Check Functions
# ==========================================
def check_active_orders(current_price, current_ts):
    """Check results of active orders"""
    global active_orders, stats, total_profit
    
    for order in active_orders[:]:
        if current_ts >= order['check_time']:
            diff = current_price - order['entry_price']
            
            if diff >= MIN_PROFIT:
                stats['win'] += 1
                result_text = "WIN"
                profit_text = f"(+{diff:.2f})"
            elif diff > -MIN_PROFIT:
                stats['breakeven'] += 1
                result_text = "BREAKEVEN"
                profit_text = f"({diff:+.2f})"
            else:
                stats['loss'] += 1
                result_text = "LOSS"
                profit_text = f"({diff:.2f})"
            
            total_profit += diff  # Accumulate profit/loss
            
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            real_trades = stats['win'] + stats['loss']
            win_rate = (stats['win'] / real_trades * 100) if real_trades > 0 else 0
            
            print(f"Check result: Entry {order['entry_price']:.2f} -> Exit {current_price:.2f} {profit_text} | Result: {result_text} | Win Rate: {win_rate:.1f}% ({stats['win']}W/{stats['loss']}L/{stats['breakeven']}BE) | Total P/L: {total_profit:.2f}")
            
            active_orders.remove(order)

# ==========================================
# Prediction Functions
# ==========================================
def predict_signal(current_data, current_entry_price, current_real_time):
    """Predict buy signal from features"""
    global active_orders, history_buffer
    
    # Calculate features
    features = calculate_features(current_data, history_buffer)
    
    if features is None:
        return  # Not enough data yet
    
    # Arrange features to match training order
    feature_cols = [
        'total_volume', 'net_flow', 'trade_count',
        'net_flow_ma5', 'net_flow_ma15', 'net_flow_ma30',
        'volume_ma5', 'volume_ma15',
        'net_flow_diff', 'net_flow_diff5',
        'price_change', 'price_change_ma5',
        'net_flow_ratio', 'cumulative_net_flow_30'
    ]
    
    X = pd.DataFrame([features])[feature_cols]
    
    # LightGBM Booster predict() returns probability directly
    prob_buy = model.predict(X)[0]
    
    timestamp = datetime.datetime.now().strftime('%H:%M:%S')
    
    # Check buy conditions
    if prob_buy >= CONFIDENCE_THRESHOLD :
        target_time = current_real_time + HOLDING_TIME
        
        print(f"{timestamp} | Buy signal! (Flow: {current_data['net_flow']:.4f}) | Confidence {prob_buy*100:.1f}% | Entry price: {current_entry_price:.2f}")
        
        active_orders.append({
            'entry_price': current_entry_price,
            'check_time': target_time
        })

# ==========================================
# WebSocket Handlers
# ==========================================
def on_message(ws, message):
    global current_second_data, history_buffer
    
    try:
        data = json.loads(message)
        
        price = float(data['p'])
        qty = float(data['q'])
        is_maker = data['m']
        ts = int(data['T'] / 1000)
        
        # Check active orders
        check_active_orders(price, ts)
        
        # Start first second
        if current_second_data['timestamp_sec'] is None:
            current_second_data['timestamp_sec'] = ts
        
        # If second changes -> save and predict
        if ts > current_second_data['timestamp_sec']:
            current_second_data['close_price'] = price
            
            # Add to history buffer
            history_buffer.append(current_second_data.copy())
            
            # Predict (need at least 5 seconds)
            if len(history_buffer) >= 5:
                predict_signal(current_second_data, price, ts)
            
            # Reset new second data
            current_second_data = {
                'net_flow': 0.0,
                'total_volume': 0.0,
                'trade_count': 0,
                'close_price': price,
                'timestamp_sec': ts
            }
        
        # Accumulate data in current second
        signed_vol = qty if not is_maker else -qty
        current_second_data['net_flow'] += signed_vol
        current_second_data['total_volume'] += qty
        current_second_data['trade_count'] += 1
        current_second_data['close_price'] = price
        
    except Exception as e:
        print(f"Error: {e}")

def on_error(ws, error):
    print(f"WebSocket Error: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### Disconnected ###")
    print(f"Summary: {stats['win']}W / {stats['loss']}L / {stats['breakeven']}BE | Total P/L: {total_profit:.2f}")

def on_open(ws):
    print(f"Connected to Binance ({SYMBOL.upper()}) successfully!")
    print(f"Waiting to accumulate 5 seconds of data before starting prediction...")

# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    print("="*50)
    print("BTC Scalping Bot (LightGBM)")
    print("="*50)
    
    ws = websocket.WebSocketApp(
        SOCKET,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close
    )
    ws.run_forever()