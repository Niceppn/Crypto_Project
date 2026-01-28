import pandas as pd
import numpy as np
import lightgbm as lgb
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

# ==========================================
# ⚙️ 1. CONFIGURATION (ตั้งค่า)
# ==========================================
DATA_SOURCE_PATH = '/Users/Macbook/Collect_Crypto/BTC_Future/test'
MODEL_DEST_PATH = '/Users/Macbook/Collect_Crypto/BTC_Future/test'

# --- Strategy Parameters (ปรับจูนได้) ---
PROFIT_TARGET_PCT = 1.0003  # เป้ากำไร 0.03%
FILL_WINDOW = 20           # Window รอของเข้า (20 วินาที)
PROFIT_WINDOW = 300        # Window รอขายทำกำไร (2 นาที)
CONFIDENCE_THRESHOLD = 0.60 # ความมั่นใจขั้นต่ำ

if not os.path.exists(MODEL_DEST_PATH):
    os.makedirs(MODEL_DEST_PATH)

csv_files = [f for f in os.listdir(DATA_SOURCE_PATH) if f.endswith('.csv')]
print(f"📂 พบไฟล์ CSV ทั้งหมด: {len(csv_files)} ไฟล์")

for csv_filename in csv_files:
    raw_file_path = os.path.join(DATA_SOURCE_PATH, csv_filename)
    model_filename = csv_filename.replace('.csv', '.txt')
    model_output_path = os.path.join(MODEL_DEST_PATH, model_filename)

    print(f"\n{'='*40}")
    print(f"🚀 กำลังประมวลผล: {csv_filename}")
    print(f"{'='*40}")
    
    try:
        # 1. LOAD DATA
        df = pd.read_csv(raw_file_path)
        
        # ตรวจสอบชื่อคอลัมน์เวลา
        ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
        if ts_col not in df.columns:
            print(f"❌ ไม่พบคอลัมน์ timestamp (Columns: {df.columns})")
            continue

        df['datetime'] = pd.to_datetime(df[ts_col], unit='ms')
        df = df.set_index('datetime')
        
        # DEBUG: เช็คข้อมูลดิบ
        print(f"1️⃣  [Raw Data] จำนวนแถว: {len(df):,} | ช่วงเวลา: {df.index.max() - df.index.min()}")

        # 2. AGGREGATION (Resample 1s) & FILLING GAPS
        df['signed_volume'] = df.apply(lambda x: x['quantity'] if x['side'] == 'BUY' else -x['quantity'], axis=1)
        
        df_1s = df.resample('1s').agg({
            'price': ['last', 'min', 'max'], 
            'quantity': 'sum',
            'signed_volume': 'sum',
            'side': 'count'
        })
        df_1s.columns = ['close_price', 'low_price', 'high_price', 'total_volume', 'net_flow', 'trade_count']

        # 🔥 [CRITICAL FIX] ถมค่าว่างสำหรับวินาทีที่ไม่มีเทรด 🔥
        # 1. ราคา Close ให้ใช้ราคาล่าสุดก่อนหน้า (Forward Fill)
        df_1s['close_price'] = df_1s['close_price'].ffill()
        
        # 2. High/Low ถ้าไม่มีเทรด ให้เท่ากับ Close ของวินาทีนั้น
        df_1s['low_price'] = df_1s['low_price'].fillna(df_1s['close_price'])
        df_1s['high_price'] = df_1s['high_price'].fillna(df_1s['close_price'])
        
        # 3. Volume/Count ถ้าไม่มีเทรด ให้เป็น 0
        df_1s['total_volume'] = df_1s['total_volume'].fillna(0)
        df_1s['net_flow'] = df_1s['net_flow'].fillna(0)
        df_1s['trade_count'] = df_1s['trade_count'].fillna(0)
        
        # ลบเฉพาะช่วงต้นที่ยังไม่มีราคา (ถ้ามี)
        df_1s.dropna(subset=['close_price'], inplace=True)

        print(f"2️⃣  [Resampled & Filled] จำนวนแท่ง (1s): {len(df_1s):,} แถว")
        
        if len(df_1s) < PROFIT_WINDOW + 100:
            print(f"⚠️ ข้อมูลสั้นเกินไปหลัง Resample ({len(df_1s)} แถว) -> ข้าม")
            continue

        # 3. FEATURE ENGINEERING
        df_1s['net_flow_ma5'] = df_1s['net_flow'].rolling(5).mean()
        df_1s['net_flow_ma15'] = df_1s['net_flow'].rolling(15).mean()
        df_1s['volume_ma5'] = df_1s['total_volume'].rolling(5).mean()
        df_1s['net_flow_diff'] = df_1s['net_flow'].diff()
        
        # Fix Deprecation Warning
        df_1s['price_change'] = df_1s['close_price'].pct_change(fill_method=None) * 100
        
        df_1s['std_5'] = df_1s['close_price'].rolling(5).std()
        df_1s['dist_ma15'] = df_1s['close_price'] - df_1s['close_price'].rolling(15).mean()

        delta = df_1s['close_price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df_1s['rsi'] = 100 - (100 / (1 + rs))

        # 4. TARGET CREATION
        indexer_fill = pd.api.indexers.FixedForwardWindowIndexer(window_size=FILL_WINDOW)
        df_1s['future_min_low'] = df_1s['low_price'].rolling(window=indexer_fill).min().shift(-1)
        is_filled = df_1s['future_min_low'] <= df_1s['close_price']

        indexer_profit = pd.api.indexers.FixedForwardWindowIndexer(window_size=PROFIT_WINDOW)
        df_1s['future_max_high'] = df_1s['high_price'].rolling(window=indexer_profit).max().shift(-1)
        target_price = df_1s['close_price'] * PROFIT_TARGET_PCT 
        is_profit = df_1s['future_max_high'] > target_price

        df_1s['target'] = (is_filled & is_profit).astype(int)
        
        # ตัด NaN ที่เกิดจาก Rolling/Shift
        before_drop = len(df_1s)
        df_1s.dropna(inplace=True)
        
        print(f"3️⃣  [Final Training Set] พร้อมเทรน: {len(df_1s):,} แถว (ตัดไป {before_drop - len(df_1s)} แถว)")

        y = df_1s['target']
        counts = y.value_counts()
        pos_samples = counts.get(1, 0)
        
        print(f"📊 สถิติ Class: Win (1) = {pos_samples:,} | Loss (0) = {counts.get(0, 0):,}")

        if pos_samples < 50:
            print(f"⚠️ จุดทำกำไรน้อยเกินไป ({pos_samples}) -> ข้าม")
            continue

        # 5. TRAINING
        feature_cols = ['total_volume', 'net_flow', 'trade_count', 'net_flow_ma5', 
                        'net_flow_ma15', 'volume_ma5', 'net_flow_diff', 'price_change', 
                        'std_5', 'dist_ma15', 'rsi']

        X = df_1s[feature_cols]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Calculate Scale Pos Weight (แก้ปัญหา Imbalance)
        neg = y_train.value_counts().get(0, 1)
        pos = y_train.value_counts().get(1, 1)
        scale_weight = neg / pos if pos > 0 else 1.0

        print(f"⚖️  Scale Pos Weight: {scale_weight:.2f}")

        model = lgb.LGBMClassifier(
            n_estimators=500,        # เพิ่มรอบการเทรน
            learning_rate=0.01,      # ลด Learning rate ให้เรียนละเอียดขึ้น
            num_leaves=31,
            max_depth=7,
            scale_pos_weight=scale_weight,
            random_state=42, 
            verbose=-1
        )
        model.fit(X_train, y_train)

        # Evaluation
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= CONFIDENCE_THRESHOLD).astype(int)
        precision = precision_score(y_test, preds, zero_division=0)

        # Save Model
        model.booster_.save_model(model_output_path)
        print(f"✅ บันทึกโมเดลสำเร็จ: {model_filename}")
        print(f"🎯 Precision (Test Set): {precision*100:.2f}%")

    except Exception as e:
        print(f"❌ Error ในไฟล์ {csv_filename}: {e}")
        import traceback
        traceback.print_exc()

print("\n--- DONE ---")