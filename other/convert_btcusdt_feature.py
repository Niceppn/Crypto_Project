import pandas as pd

# 1. โหลดไฟล์ CSV ดิบ
df = pd.read_csv('btc_training_data.csv')

# แปลง timestamp เป็น datetime เพื่อให้ pandas เข้าใจเวลา
df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
df = df.set_index('datetime')

# 2. สร้าง Helper Column สำหรับคำนวณ Flow
# ถ้า side == BUY ให้ค่าเป็นบวก (+), ถ้า SELL ให้ค่าเป็นลบ (-)
df['signed_volume'] = df.apply(lambda x: x['quantity'] if x['side'] == 'BUY' else -x['quantity'], axis=1)

# 3. ยุบรวมข้อมูล (Resample) เป็นราย 1 วินาที ('1S')
# นี่คือการสร้าง Feature Engineering จริงๆ
ohlc_dict = {
    'price': 'last',             # ราคาปิดของวินาทีนีั้ (Close)
    'quantity': 'sum',           # Volume รวม (Total Volume)
    'signed_volume': 'sum',      # *** Net Flow (พระเอกของเรา) ***
    'side': 'count'              # Trade Count (นับจำนวน Transaction)
}

df_1s = df.resample('1S').agg(ohlc_dict)

# เปลี่ยนชื่อคอลัมน์ให้สื่อความหมาย
df_1s.rename(columns={
    'price': 'close_price',
    'quantity': 'total_volume',
    'signed_volume': 'net_flow',  # <--- ค่านี้สำคัญสุด ถ้าเป็นบวกเยอะๆ เตรียม Long
    'side': 'trade_count'         # <--- ค่านี้บอกความเดือด
}, inplace=True)

# 4. (Optional) สร้าง Label (เฉลย) สำหรับสอน AI
# ให้ AI ทายราคาในอีก 5 วินาทีข้างหน้า (Future Price)
# สร้าง Target: ถ้าราคาอีก 5 วิ > ราคาปัจจุบัน ให้เป็น 1 (ขึ้น), ถ้าไม่ใช่เป็น 0
df_1s['future_price_5s'] = df_1s['close_price'].shift(-5)
df_1s['target'] = (df_1s['future_price_5s'] > df_1s['close_price']).astype(int)

# ลบแถวที่มีค่าว่าง (ช่วงท้ายๆ ที่ไม่มีอนาคตให้เทียบ)
df_1s.dropna(inplace=True)

# 5. บันทึกผลลัพธ์
print(df_1s.head(10)) # โชว์ตัวอย่าง
df_1s.to_csv('processed_features.csv')
print("\n--- แปลงข้อมูลเสร็จสิ้น! บันทึกลง processed_features.csv แล้ว ---")