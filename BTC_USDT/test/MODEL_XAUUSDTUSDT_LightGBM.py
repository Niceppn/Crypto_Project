import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score

# ==========================================
# 1. Configuration and Data Loading
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_USDT/test/BTCJPY.csv'
MODEL_FILE = '/Users/Macbook/Collect_Crypto/BTC_USDT/test/BTCJPY.txt'  # LightGBM uses .txt
CONFIDENCE_THRESHOLD = 0.7

print(f"Reading data from {RAW_FILE}...")
df = pd.read_csv(RAW_FILE)

# Convert timestamp
ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
df['datetime'] = pd.to_datetime(df[ts_col], unit='ms')
df = df.set_index('datetime')

# ==========================================
# 2. Feature Engineering
# ==========================================
print("Creating new features...")

# Signed Volume
df['signed_volume'] = df.apply(lambda x: x['quantity'] if x['side'] == 'BUY' else -x['quantity'], axis=1)

# Aggregate to 1 second intervals
df_1s = df.resample('1S').agg({
    'price': 'last',
    'quantity': 'sum',
    'signed_volume': 'sum',
    'side': 'count'
})

df_1s.rename(columns={
    'price': 'close_price',
    'quantity': 'total_volume',
    'signed_volume': 'net_flow',
    'side': 'trade_count'
}, inplace=True)

# ==========================================
# 🆕 Features ใหม่ (ช่วยให้แม่นขึ้น!)
# ==========================================
# Moving Averages ของ Net Flow
df_1s['net_flow_ma5'] = df_1s['net_flow'].rolling(5).mean()    # ค่าเฉลี่ย 5 วินาที
df_1s['net_flow_ma15'] = df_1s['net_flow'].rolling(15).mean()  # ค่าเฉลี่ย 15 วินาที
df_1s['net_flow_ma30'] = df_1s['net_flow'].rolling(30).mean()  # ค่าเฉลี่ย 30 วินาที

# Volume Moving Averages
df_1s['volume_ma5'] = df_1s['total_volume'].rolling(5).mean()
df_1s['volume_ma15'] = df_1s['total_volume'].rolling(15).mean()

# Net Flow Momentum (การเปลี่ยนแปลง)
df_1s['net_flow_diff'] = df_1s['net_flow'].diff()              # เปลี่ยนแปลงจากวินาทีก่อน
df_1s['net_flow_diff5'] = df_1s['net_flow'] - df_1s['net_flow'].shift(5)  # เปลี่ยนแปลงจาก 5 วินาทีก่อน

# Price Change
df_1s['price_change'] = df_1s['close_price'].pct_change() * 100  # % เปลี่ยนแปลงราคา
df_1s['price_change_ma5'] = df_1s['price_change'].rolling(5).mean()

# Net Flow Ratio (แรงซื้อ/ขายสัมพัทธ์)
df_1s['net_flow_ratio'] = df_1s['net_flow'] / (df_1s['total_volume'] + 1e-10)

# Cumulative Net Flow (สะสม 30 วินาที)
df_1s['cumulative_net_flow_30'] = df_1s['net_flow'].rolling(30).sum()

# ==========================================
# 🎯 3. สร้าง Target (15 วินาทีข้างหน้า)
# ==========================================
df_1s['future_price'] = df_1s['close_price'].shift(-10)
df_1s['target'] = (df_1s['future_price'] > df_1s['close_price']).astype(int)

# ลบ NaN
df_1s.dropna(inplace=True)

print(f"✅ เตรียมข้อมูลเสร็จสิ้น: {len(df_1s)} แถว")

# ==========================================
# 🧠 4. เตรียมสอนและฝึกฝน AI (Training)
# ==========================================
# Features ใหม่ทั้งหมด!
feature_cols = [
    'total_volume', 'net_flow', 'trade_count',
    'net_flow_ma5', 'net_flow_ma15', 'net_flow_ma30',
    'volume_ma5', 'volume_ma15',
    'net_flow_diff', 'net_flow_diff5',
    'price_change', 'price_change_ma5',
    'net_flow_ratio', 'cumulative_net_flow_30'
]

X = df_1s[feature_cols]
y = df_1s['target']

# แบ่งข้อมูล
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# คำนวณ scale weight
scale_pos_weight = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

print(f"🚀 กำลังฝึกฝน AI (ด้วย LightGBM)... [Scale Weight: {scale_pos_weight:.2f}]")

model = LGBMClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=4,
    num_leaves=15,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1  # ปิด warning
)
model.fit(X_train, y_train)

# ==========================================
# 🎯 5. วัดผล (Evaluation)
# ==========================================
probs = model.predict_proba(X_test)[:, 1]
predictions_tuned = (probs >= CONFIDENCE_THRESHOLD).astype(int)

precision = precision_score(y_test, predictions_tuned, zero_division=0)

print("\n" + "="*40)
print(f"📊 ผลลัพธ์ LightGBM (Threshold {CONFIDENCE_THRESHOLD*100}%)")
print("="*40)
print(f"✅ Precision (ความแม่นเข้าซื้อ): {precision * 100:.2f}%")
print(f"(ถ้า Bot บอกซื้อ 100 ครั้ง จะถูก {precision * 100:.0f} ครั้ง)")
print("-" * 40)
print(classification_report(y_test, predictions_tuned))

# ==========================================
# 💾 6. บันทึกโมเดล
# ==========================================
model.booster_.save_model(MODEL_FILE)
print(f"💾 บันทึกโมเดลไว้ที่: {MODEL_FILE}")

# ==========================================
# 📈 7. วาดกราฟ
# ==========================================
# Feature Importance
plt.figure(figsize=(12, 6))
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance.sort_values().plot(kind='barh', color='dodgerblue')
plt.title('🔥 Feature Importance (LightGBM)')
plt.xlabel('Importance')
plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, predictions_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Wait', 'Buy'],
            yticklabels=['Actual Wait', 'Actual Buy'])
plt.title(f'Confusion Matrix (Threshold > {CONFIDENCE_THRESHOLD})')
plt.ylabel('ความจริง')
plt.xlabel('AI ทายว่า')
plt.show()