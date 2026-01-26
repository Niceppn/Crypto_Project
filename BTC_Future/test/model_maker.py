import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/test/btcusdt_future_training_data.csv' # 🔴 แก้ Path ไฟล์ข้อมูลดิบของคุณ
MODEL_OUTPUT = '/Users/Macbook/Collect_Crypto/BTC_Future/test/btcusdc_maker_model_lgbm.txt'     
CONFIDENCE_THRESHOLD = 0.65

print(f"🔄 Reading data from {RAW_FILE}...")
df = pd.read_csv(RAW_FILE)

# จัดการเรื่องเวลา
ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
df['datetime'] = pd.to_datetime(df[ts_col], unit='ms')
df = df.set_index('datetime')

# ==========================================
# 1. AGGREGATION (รวมข้อมูลเป็นรายวินาที)
# ==========================================
# สำคัญ: ต้องเอา min/max มาด้วย เพื่อจำลองการ Match ของ Maker
df['signed_volume'] = df.apply(lambda x: x['quantity'] if x['side'] == 'BUY' else -x['quantity'], axis=1)
df_1s = df.resample('1S').agg({
    'price': ['last', 'min', 'max'], 
    'quantity': 'sum',
    'signed_volume': 'sum',
    'side': 'count'
})
df_1s.columns = ['close_price', 'low_price', 'high_price', 'total_volume', 'net_flow', 'trade_count']

# ==========================================
# 2. FEATURE ENGINEERING (สร้างตัวแปร)
# ==========================================
print("⚙️ Generating Features...")

# Momentum & Flow Basic
df_1s['net_flow_ma5'] = df_1s['net_flow'].rolling(5).mean()
df_1s['net_flow_ma15'] = df_1s['net_flow'].rolling(15).mean()
df_1s['volume_ma5'] = df_1s['total_volume'].rolling(5).mean()
df_1s['net_flow_diff'] = df_1s['net_flow'].diff()
df_1s['price_change'] = df_1s['close_price'].pct_change() * 100

# 🆕 Maker Specific Features (จับความผันผวนและการย่อตัว)
df_1s['std_5'] = df_1s['close_price'].rolling(5).std() # ความผันผวนระยะสั้น
df_1s['dist_ma15'] = df_1s['close_price'] - df_1s['close_price'].rolling(15).mean() # ระยะห่างจากเส้นค่าเฉลี่ย

# RSI (Indicator ยอดฮิตจับ Oversold)
delta = df_1s['close_price'].diff()
gain = (delta.where(delta > 0, 0)).rolling(14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
rs = gain / (loss + 1e-10)
df_1s['rsi'] = 100 - (100 / (1 + rs))

# ==========================================
# 3. TARGET CREATION (สอน AI)
# ==========================================
print("🎯 Creating Targets (Dip & Rip)...")

# A. เงื่อนไขได้ของ (Filled): ภายใน 10 วิ ราคา Low ต้องลงมาต่ำกว่าหรือเท่ากับราคาปิดปัจจุบัน
indexer_fill = pd.api.indexers.FixedForwardWindowIndexer(window_size=10)
df_1s['future_min_low'] = df_1s['low_price'].rolling(window=indexer_fill).min().shift(-1)
is_filled = df_1s['future_min_low'] <= df_1s['close_price']

# B. เงื่อนไขกำไร (Profit): ภายใน 60 วิ ราคา High ต้องสูงกว่าราคาเข้า + 0.15%
indexer_profit = pd.api.indexers.FixedForwardWindowIndexer(window_size=60)
df_1s['future_max_high'] = df_1s['high_price'].rolling(window=indexer_profit).max().shift(-1)
target_price = df_1s['close_price'] * 1.0015 # เป้ากำไร 0.15%
is_profit = df_1s['future_max_high'] > target_price

# Target = 1 คือ "ต้องได้ของ" และ "ต้องกำไร"
df_1s['target'] = (is_filled & is_profit).astype(int)
df_1s.dropna(inplace=True)

# ==========================================
# 4. TRAINING
# ==========================================
feature_cols = [
    'total_volume', 'net_flow', 'trade_count',
    'net_flow_ma5', 'net_flow_ma15', 'volume_ma5',
    'net_flow_diff', 'price_change', 
    'std_5', 'dist_ma15', 'rsi'
]

X = df_1s[feature_cols]
y = df_1s['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
scale_pos_weight = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

print(f"🚀 Training LGBM Model... (Weight: {scale_pos_weight:.2f})")
model = lgb.LGBMClassifier(
    n_estimators=200, 
    learning_rate=0.03,
    num_leaves=20,
    max_depth=5,
    scale_pos_weight=scale_pos_weight,
    random_state=42, 
    verbose=-1
)
model.fit(X_train, y_train)

# Evaluation
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= CONFIDENCE_THRESHOLD).astype(int)
precision = precision_score(y_test, preds, zero_division=0)

print(f"\n📊 Result: Precision {precision*100:.2f}% (Threshold {CONFIDENCE_THRESHOLD})")
model.booster_.save_model(MODEL_OUTPUT)
print(f"💾 Model saved to: {MODEL_OUTPUT}")