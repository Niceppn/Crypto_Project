import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score

# ==========================================
# ⚙️ 1. ตั้งค่าและโหลดข้อมูล (Configuration)
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/Merge/merged_crypto_final.csv'
MODEL_FILE = 'merged_crypto_model_lgbm.txt'
CONFIDENCE_THRESHOLD = 0.60

print(f"🔄 กำลังอ่านข้อมูลจาก {RAW_FILE}...")
df = pd.read_csv(RAW_FILE)

# แปลง datetime เป็น index
df['datetime'] = pd.to_datetime(df['datetime'])
df = df.set_index('datetime')

print(f"✅ โหลดข้อมูลสำเร็จ: {len(df)} แถว")
print(f"ช่วงเวลา: {df.index.min()} ถึง {df.index.max()}")

# ==========================================
# 🛠️ 2. Feature Engineering ขั้นสูง
# ==========================================
print("⚙️ กำลังสร้าง Features ใหม่...")

# BTC Features
df['btc_price_change'] = df['btc_price'].pct_change() * 100
df['btc_price_ma5'] = df['btc_price'].rolling(5).mean()
df['btc_price_ma15'] = df['btc_price'].rolling(15).mean()
df['btc_volume_ma5'] = df['btc_volume'].rolling(5).mean()
df['btc_volume_ma15'] = df['btc_volume'].rolling(15).mean()

# XAU Features  
df['xau_price_change'] = df['xau_price'].pct_change() * 100
df['xau_price_ma5'] = df['xau_price'].rolling(5).mean()
df['xau_price_ma15'] = df['xau_price'].rolling(15).mean()
df['xau_volume_ma5'] = df['xau_volume'].rolling(5).mean()
df['xau_volume_ma15'] = df['xau_volume'].rolling(15).mean()

# Cross-Asset Features (ความสัมพันธ์ BTC-XAU)
df['price_ratio'] = df['btc_price'] / df['xau_price']  # อัตราส่วนราคา
df['price_ratio_change'] = df['price_ratio'].pct_change() * 100
df['price_ratio_ma5'] = df['price_ratio'].rolling(5).mean()

# Volume Features
df['total_volume'] = df['btc_volume'] + df['xau_volume']
df['btc_volume_ratio'] = df['btc_volume'] / (df['total_volume'] + 1e-10)
df['xau_volume_ratio'] = df['xau_volume'] / (df['total_volume'] + 1e-10)

# Momentum Features
df['btc_momentum'] = df['btc_price'] - df['btc_price'].shift(5)  # 5 วินาทีก่อน
df['xau_momentum'] = df['xau_price'] - df['xau_price'].shift(5)
df['momentum_diff'] = df['btc_momentum'] - df['xau_momentum']

# Volatility Features
df['btc_volatility'] = df['btc_price_change'].rolling(10).std()
df['xau_volatility'] = df['xau_price_change'].rolling(10).std()
df['volatility_ratio'] = df['btc_volatility'] / (df['xau_volatility'] + 1e-10)

# ==========================================
# 🎯 3. สร้าง Target Variables (หลายตัวเลือก)
# ==========================================
print("🎯 กำลังสร้าง Target...")

# Target 1: ทำนายราคา BTC 5 วินาทีข้างหน้า
df['btc_future_price'] = df['btc_price'].shift(-5)
df['target_btc_up'] = (df['btc_future_price'] > df['btc_price']).astype(int)

# Target 2: ทำนายราคา XAU 5 วินาทีข้างหน้า  
df['xau_future_price'] = df['xau_price'].shift(-5)
df['target_xau_up'] = (df['xau_future_price'] > df['xau_price']).astype(int)

# Target 3: ทำนายว่า BTC จะทำได้ดีกว่า XAU หรือไม่
df['btc_outperformance'] = (df['btc_price_change'] > df['xau_price_change']).astype(int)

# Target 4: ทำนายทิศทางของอัตราส่วนราคา
df['price_ratio_future'] = df['price_ratio'].shift(-5)
df['target_ratio_up'] = (df['price_ratio_future'] > df['price_ratio']).astype(int)

# เลือก Target หลัก (สามารถเปลี่ยนได้)
TARGET_CHOICE = 'btc_up'  # เปลี่ยนได้: 'btc_up', 'xau_up', 'btc_outperformance', 'ratio_up'

target_mapping = {
    'btc_up': 'target_btc_up',
    'xau_up': 'target_xau_up', 
    'btc_outperformance': 'btc_outperformance',
    'ratio_up': 'target_ratio_up'
}

df['target'] = df[target_mapping[TARGET_CHOICE]]

# ลบ NaN
df.dropna(inplace=True)

print(f"✅ เตรียมข้อมูลเสร็จ: {len(df)} แถว")
print(f"📊 Target: {TARGET_CHOICE}")
print(f"📈 Target Distribution: {df['target'].value_counts().to_dict()}")

# ==========================================
# 🧠 4. เลือก Features ทั้งหมด
# ==========================================
feature_cols = [
    # BTC Features
    'btc_price_change', 'btc_price_ma5', 'btc_price_ma15',
    'btc_volume_ma5', 'btc_volume_ma15', 'btc_momentum', 'btc_volatility',
    
    # XAU Features
    'xau_price_change', 'xau_price_ma5', 'xau_price_ma15', 
    'xau_volume_ma5', 'xau_volume_ma15', 'xau_momentum', 'xau_volatility',
    
    # Cross-Asset Features
    'price_ratio', 'price_ratio_change', 'price_ratio_ma5',
    'total_volume', 'btc_volume_ratio', 'xau_volume_ratio',
    'momentum_diff', 'volatility_ratio'
]

X = df[feature_cols]
y = df['target']

# ==========================================
# 🚀 5. แบ่งข้อมูลและฝึก AI
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# คำนวณ scale weight สำหรับ imbalance data
scale_pos_weight = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

print(f"🚀 กำลังฝึก AI LightGBM...")
print(f"📊 Features: {len(feature_cols)} ตัว")
print(f"⚖️ Scale Weight: {scale_pos_weight:.2f}")

model = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=20,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    verbose=-1
)

model.fit(X_train, y_train)

# ==========================================
# 🎯 6. วัดผลและประเมิน
# ==========================================
probs = model.predict_proba(X_test)[:, 1]
predictions_tuned = (probs >= CONFIDENCE_THRESHOLD).astype(int)

precision = precision_score(y_test, predictions_tuned, zero_division=0)

print("\n" + "="*50)
print(f"📊 ผลลัพธ์ LightGBM - Target: {TARGET_CHOICE}")
print(f"🎯 Confidence Threshold: {CONFIDENCE_THRESHOLD*100}%")
print("="*50)
print(f"✅ Precision: {precision * 100:.2f}%")
print(f"📈 Test Samples: {len(y_test)}")
print(f"🔢 Predictions > Threshold: {predictions_tuned.sum()}")
print("-" * 50)
print(classification_report(y_test, predictions_tuned))

# ==========================================
# 💾 7. บันทึกโมเดลและผลลัพธ์
# ==========================================
model.booster_.save_model(MODEL_FILE)
print(f"💾 บันทึกโมเดล: {MODEL_FILE}")

# บันทึก predictions
results_df = X_test.copy()
results_df['actual'] = y_test
results_df['predicted'] = predictions_tuned
results_df['probability'] = probs
results_df.to_csv('model_predictions.csv', index=True)
print("💾 บันทึกผลการทำนาย: model_predictions.csv")

# ==========================================
# 📈 8. วาดกราฟวิเคราะห์
# ==========================================
plt.style.use('default')

# Feature Importance
plt.figure(figsize=(12, 8))
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance_top15 = importance.sort_values(ascending=False).head(15)
importance_top15.sort_values().plot(kind='barh', color='dodgerblue')
plt.title(f'🔥 Top 15 Feature Importance\nTarget: {TARGET_CHOICE}', fontsize=14)
plt.xlabel('Importance Score')
plt.tight_layout()
plt.show()

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Wait', 'Action'],
            yticklabels=['Actual Wait', 'Actual Action'])
plt.title(f'Confusion Matrix (Threshold > {CONFIDENCE_THRESHOLD})\nTarget: {TARGET_CHOICE}')
plt.ylabel('ความจริง')
plt.xlabel('AI ทายว่า')
plt.tight_layout()
plt.show()

# Probability Distribution
plt.figure(figsize=(10, 6))
plt.hist(probs[y_test == 0], bins=30, alpha=0.7, label='Wait (0)', color='red')
plt.hist(probs[y_test == 1], bins=30, alpha=0.7, label='Action (1)', color='green')
plt.axvline(CONFIDENCE_THRESHOLD, color='black', linestyle='--', label=f'Threshold = {CONFIDENCE_THRESHOLD}')
plt.title(f'Probability Distribution\nTarget: {TARGET_CHOICE}')
plt.xlabel('Predicted Probability')
plt.ylabel('Frequency')
plt.legend()
plt.tight_layout()
plt.show()

# Time Series Prediction
plt.figure(figsize=(15, 6))
test_dates = y_test.index
plt.plot(test_dates, y_test, label='Actual', alpha=0.7, linewidth=1)
plt.plot(test_dates, predictions_tuned, label='Predicted', alpha=0.8, linewidth=1)
plt.title(f'Prediction Timeline\nTarget: {TARGET_CHOICE}')
plt.xlabel('Time')
plt.ylabel('Signal (0=Wait, 1=Action)')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("\n🎉 เสร็จสิ้นการสร้างโมเดล!")
print(f"📁 ไฟล์ที่สร้าง: {MODEL_FILE}, model_predictions.csv")
