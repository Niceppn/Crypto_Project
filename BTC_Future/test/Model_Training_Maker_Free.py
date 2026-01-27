import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

# ==========================================
# ⚙️ 1. CONFIGURATION
# ==========================================
DATA_SOURCE_PATH = '/Users/Macbook/Collect_Crypto/BTC_Future/test'
MODEL_DEST_PATH = '/Users/Macbook/Collect_Crypto/BTC_Future/test'

# --- ปรับจูนกลยุทธ์ Zero Fee ---
PROFIT_TARGET_PCT = 1.0003  # กำไร 0.03% ก็เอา (เหมาะกับไม่มีค่าธรรมเนียม)
FILL_WINDOW = 20           # รอให้ราคาลงมา Match 20 วินาที
PROFIT_WINDOW = 90        # รอให้ราคาขึ้นไปขาย 2 นาที
CONFIDENCE_THRESHOLD = 0.55 # ลดระดับความมั่นใจลงเล็กน้อยเพื่อให้ AI ส่งสัญญาณบ่อยขึ้น

if not os.path.exists(MODEL_DEST_PATH):
    os.makedirs(MODEL_DEST_PATH)

csv_files = [f for f in os.listdir(DATA_SOURCE_PATH) if f.endswith('.csv')]

for csv_filename in csv_files:
    raw_file_path = os.path.join(DATA_SOURCE_PATH, csv_filename)
    model_filename = csv_filename.replace('.csv', '.txt')
    model_output_path = os.path.join(MODEL_DEST_PATH, model_filename)

    print(f"\n🚀 Processing: {csv_filename}")
    
    try:
        df = pd.read_csv(raw_file_path)
        ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
        df['datetime'] = pd.to_datetime(df[ts_col], unit='ms')
        df = df.set_index('datetime')

        # 1. AGGREGATION
        df['signed_volume'] = df.apply(lambda x: x['quantity'] if x['side'] == 'BUY' else -x['quantity'], axis=1)
        df_1s = df.resample('1s').agg({
            'price': ['last', 'min', 'max'], 
            'quantity': 'sum',
            'signed_volume': 'sum',
            'side': 'count'
        })
        df_1s.columns = ['close_price', 'low_price', 'high_price', 'total_volume', 'net_flow', 'trade_count']

        # 2. FEATURE ENGINEERING
        df_1s['net_flow_ma5'] = df_1s['net_flow'].rolling(5).mean()
        df_1s['net_flow_ma15'] = df_1s['net_flow'].rolling(15).mean()
        df_1s['volume_ma5'] = df_1s['total_volume'].rolling(5).mean()
        df_1s['net_flow_diff'] = df_1s['net_flow'].diff()
        df_1s['price_change'] = df_1s['close_price'].pct_change() * 100
        df_1s['std_5'] = df_1s['close_price'].rolling(5).std()
        df_1s['dist_ma15'] = df_1s['close_price'] - df_1s['close_price'].rolling(15).mean()

        delta = df_1s['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df_1s['rsi'] = 100 - (100 / (1 + rs))

        # 3. TARGET CREATION (Zero Fee Strategy)
        indexer_fill = pd.api.indexers.FixedForwardWindowIndexer(window_size=FILL_WINDOW)
        df_1s['future_min_low'] = df_1s['low_price'].rolling(window=indexer_fill).min().shift(-1)
        is_filled = df_1s['future_min_low'] <= df_1s['close_price']

        indexer_profit = pd.api.indexers.FixedForwardWindowIndexer(window_size=PROFIT_WINDOW)
        df_1s['future_max_high'] = df_1s['high_price'].rolling(window=indexer_profit).max().shift(-1)
        target_price = df_1s['close_price'] * PROFIT_TARGET_PCT 
        is_profit = df_1s['future_max_high'] > target_price

        df_1s['target'] = (is_filled & is_profit).astype(int)
        df_1s.dropna(inplace=True)

        y = df_1s['target']
        counts = y.value_counts()
        
        # เช็คว่ามี Class 1 (จุดทำกำไร) มากพอไหม
        pos_samples = counts.get(1, 0)
        print(f"📊 สถิติจุดทำกำไร: {pos_samples} จากทั้งหมด {len(y)}")

        if pos_samples < 10:
            print(f"⚠️ ข้าม {csv_filename}: จุดทำกำไรน้อยเกินไป ({pos_samples})")
            continue

        # 4. TRAINING
        feature_cols = ['total_volume', 'net_flow', 'trade_count', 'net_flow_ma5', 
                        'net_flow_ma15', 'volume_ma5', 'net_flow_diff', 'price_change', 
                        'std_5', 'dist_ma15', 'rsi']

        X = df_1s[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        scale_pos_weight = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

        model = lgb.LGBMClassifier(
            n_estimators=300, # เพิ่มจำนวนรอบการเรียนรู้
            learning_rate=0.02,
            num_leaves=31,
            max_depth=7,
            scale_pos_weight=scale_pos_weight,
            random_state=42, 
            verbose=-1
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= CONFIDENCE_THRESHOLD).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)

        model.booster_.save_model(model_output_path)
        print(f"✅ บันทึกสำเร็จ: {model_filename} | Precision: {precision*100:.2f}%")

    except Exception as e:
        print(f"❌ Error ในไฟล์ {csv_filename}: {e}")

print("\n--- DONE ---")

