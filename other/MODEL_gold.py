import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib  # เอาไว้ save โมเดล
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_score, RocCurveDisplay

# ==========================================
# ⚙️ 1. ตั้งค่าและโหลดข้อมูล (Configuration)
# ==========================================
RAW_FILE = 'xauusdt_training_data.csv'   
MODEL_FILE = 'xauusdt_scalping_model.json' 
CONFIDENCE_THRESHOLD = 0.65        

print(f"🔄 กำลังอ่านข้อมูลจาก {RAW_FILE}...")
df = pd.read_csv(RAW_FILE)

# แปลง timestamp (รองรับทั้ง timestamp และ timestamp_ms)
ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
df['datetime'] = pd.to_datetime(df[ts_col], unit='ms')
df = df.set_index('datetime')

# ==========================================
# 🛠️ 2. สร้าง Features (Feature Engineering)
# ==========================================
print("⚙️ กำลังแปลงข้อมูลเป็นกราฟ 1 วินาที และสร้าง Net Flow...")

# คำนวณ Signed Volume (ถ้า Buy เป็น +, Sell เป็น -)
# Logic: is_maker=True คือ Taker Sell (แรงขาย), is_maker=False คือ Taker Buy (แรงซื้อ)
df['signed_volume'] = df.apply(lambda x: x['quantity'] if x['side'] == 'BUY' else -x['quantity'], axis=1)

# ยุบรวมเป็นราย 1 วินาที
df_1s = df.resample('1S').agg({
    'price': 'last',
    'quantity': 'sum',
    'signed_volume': 'sum',
    'side': 'count'
})

# ตั้งชื่อคอลัมน์ใหม่ให้เข้าใจง่าย
df_1s.rename(columns={
    'price': 'close_price',
    'quantity': 'total_volume',
    'signed_volume': 'net_flow',  # <--- พระเอกของเรา
    'side': 'trade_count'
}, inplace=True)

# สร้างเฉลย (Target): ราคาอีก 15 วินาทีข้างหน้า > ราคาปัจจุบัน หรือไม่?
df_1s['future_price'] = df_1s['close_price'].shift(-5)  # 15 วินาที
df_1s['target'] = (df_1s['future_price'] > df_1s['close_price']).astype(int)

# ลบข้อมูลแถวสุดท้ายที่ไม่มีเฉลย (NaN)
df_1s.dropna(inplace=True)

print(f"✅ เตรียมข้อมูลเสร็จสิ้น: {len(df_1s)} แถว")

# ==========================================
# 🧠 3. เตรียมสอนและฝึกฝน AI (Training)
# ==========================================
# เลือกฟีเจอร์ที่จะสอน
feature_cols = ['total_volume', 'net_flow', 'trade_count']
X = df_1s[feature_cols]
y = df_1s['target']

# แบ่งข้อมูล (เรียน 80% / สอบ 20%) - ห้ามสลับลำดับ (shuffle=False) เพราะเป็น Time Series
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# คำนวณสัดส่วนเพื่อให้ AI สนใจขาขึ้น (Class 1) มากขึ้น
scale_pos_weight = float(y_train.value_counts()[0]) / y_train.value_counts()[1]

print(f"🤖 กำลังฝึกฝน AI (ด้วย XGBoost)... [Scale Weight: {scale_pos_weight:.2f}]")

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=3,
    scale_pos_weight=scale_pos_weight, # บังคับให้สนใจขาขึ้น
    random_state=42
)
model.fit(X_train, y_train)

# ==========================================
# 🎯 4. ปรับจูนและวัดผล (Tuning & Evaluation)
# ==========================================
# ให้ AI ทายออกมาเป็น % ความมั่นใจ (Probability)
probs = model.predict_proba(X_test)[:, 1]

# กรองด้วยเกณฑ์ความมั่นใจ (Threshold Tuning)
predictions_tuned = (probs >= CONFIDENCE_THRESHOLD).astype(int)

# วัดผล
accuracy = model.score(X_test, y_test) # Accuracy แบบปกติ
precision = precision_score(y_test, predictions_tuned, zero_division=0) # ความแม่นยำตอนเข้าซื้อ

print("\n" + "="*30)
print(f"📊 ผลลัพธ์การสอบ (Threshold {CONFIDENCE_THRESHOLD*100}%)")
print("="*30)
print(f"✅ ความแม่นยำในการเข้าซื้อ (Precision): {precision * 100:.2f}%")
print(f"(หมายความว่า: ถ้า Bot บอกให้ซื้อ 100 ครั้ง จะถูก {precision * 100:.0f} ครั้ง)")
print("-" * 30)
print(classification_report(y_test, predictions_tuned))

# ==========================================
# 💾 5. บันทึกโมเดล (Save Model)
# ==========================================
model.save_model(MODEL_FILE)
print(f"💾 บันทึกสมอง AI ไว้ที่ไฟล์: {MODEL_FILE} เรียบร้อยแล้ว!")

# ==========================================
# 📈 6. วาดกราฟสรุป (Visualization)
# ==========================================
# กราฟ 1: Feature Importance
plt.figure(figsize=(10, 4))
importance = pd.Series(model.feature_importances_, index=feature_cols)
importance.sort_values().plot(kind='barh', color='teal')
plt.title('Feature Importance (อะไรสำคัญที่สุด?)')
plt.tight_layout()
plt.show()

# กราฟ 2: Confusion Matrix (แบบจูนแล้ว)
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_test, predictions_tuned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=['Wait', 'Buy'],
            yticklabels=['Actual Wait', 'Actual Buy'])
plt.title(f'Confusion Matrix (Threshold > {CONFIDENCE_THRESHOLD})')
plt.ylabel('ความจริง')
plt.xlabel('AI ทายว่า')
plt.show()