import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# โหลดข้อมูล
print("📊 โหลดข้อมูล...")
df = pd.read_csv('BTC_USDT/btc_training_data.csv')
print(f"✅ โหลด {len(df):,} rows")

# เตรียม Features
feature_cols = ['total_volume', 'net_flow', 'trade_count']
X = df[feature_cols].values
y = df['label'].values

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# สร้าง Sequence (ดูย้อนหลัง 10 วินาที)
SEQUENCE_LENGTH = 10

def create_sequences(X, y, seq_length):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i+seq_length])
    return np.array(Xs), np.array(ys)

print(f"🔄 สร้าง Sequences (ย้อนหลัง {SEQUENCE_LENGTH} วินาที)...")
X_seq, y_seq = create_sequences(X_scaled, y, SEQUENCE_LENGTH)
print(f"✅ Shape: {X_seq.shape}")

# แบ่ง Train/Test
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

# สร้าง LSTM Model
print("🧠 สร้าง LSTM Model...")
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, len(feature_cols))),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# Train
print("🚀 เริ่ม Training...")
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=256,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ทดสอบ
loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n📊 ผลทดสอบ:")
print(f"   Accuracy: {accuracy*100:.2f}%")

# บันทึก
model.save('btc_lstm_model.h5')
print(f"✅ บันทึก Model: btc_lstm_model.h5")

# บันทึก Scaler
import joblib
joblib.dump(scaler, 'btc_scaler.pkl')
print(f"✅ บันทึก Scaler: btc_scaler.pkl")