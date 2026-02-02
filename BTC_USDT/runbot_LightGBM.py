import lightgbm as lgb
import pandas as pd
import numpy as np

MODEL_FILE = "/Users/Macbook/Collect_Crypto/BTC_USDT/btc_scalping_model_lgbm.txt"
CSV_FILE = "/Users/Macbook/Collect_Crypto/BTC_USDT/btc_training_data.csv"

# 1. เช็ด feature names จาก model จริง
model = lgb.Booster(model_file=MODEL_FILE)
print("=" * 60)
print("📌 MODEL FEATURE NAMES (จาก model จริง):")
print("=" * 60)
for i, name in enumerate(model.feature_name()):
    print(f"  [{i}] {name}")

# 2. เช็ด columns ใน CSV มี buy_volume ไหม
print("\n" + "=" * 60)
print("📌 CSV COLUMNS:")
print("=" * 60)
df = pd.read_csv(CSV_FILE, nrows=5)
print(df.columns.tolist())
print(df.head())

# 3. เทียบ — ใส่ feature ตามที่ model expect แล้วเช็ด prediction values
print("\n" + "=" * 60)
print("📌 TEST PREDICTIONS (ใส่ random values เพื่อเช็ด output range):")
print("=" * 60)

feature_names = model.feature_name()
# ใส่ dummy data ให้ prediction ออกมา
test_data = pd.DataFrame([{name: 1.0 for name in feature_names}])
pred = model.predict(test_data)
print(f"  Prediction (all 1.0): {pred[0]}")

test_data2 = pd.DataFrame([{name: 0.0 for name in feature_names}])
pred2 = model.predict(test_data2)
print(f"  Prediction (all 0.0): {pred2[0]}")

# 4. เทียบ feature names จาก model กับที่เราใช้ใน backtest
backtest_features = [
    'volume', 'net_flow', 'trade_count',
    'net_flow_ma5', 'net_flow_ma15',
    'volume_ma5', 'volume_ma15',
    'momentum_5', 'volatility_5',
    'dist_ma5', 'buy_volume_ratio', 'rsi'
]
print("\n" + "=" * 60)
print("📌 FEATURE MATCH CHECK:")
print("=" * 60)
model_features = model.feature_name()
for i, (mf, bf) in enumerate(zip(model_features, backtest_features)):
    match = "✅" if mf == bf else "❌"
    print(f"  [{i}] Model: {mf:25s} | Backtest: {bf:25s} {match}")

if len(model_features) != len(backtest_features):
    print(f"\n⚠️ LENGTH MISMATCH: Model={len(model_features)}, Backtest={len(backtest_features)}")