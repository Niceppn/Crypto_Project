import websocket
import json
import csv
import os
import time
from datetime import datetime

# --- ตั้งค่า ---
# คู่เหรียญ (ตัวเล็กหมด)
SYMBOL = "ethusdt"
# เลือก Stream เป็น aggTrade (Trade ที่เกิดขึ้นจริง)
SOCKET = f"wss://stream.binance.com:9443/ws/{SYMBOL}@aggTrade"
# ชื่อไฟล์ที่จะบันทึก
FILENAME = "eth_training_data.csv"
# --- ฟังก์ชันจัดการ CSV ---
def init_csv():
    """สร้างไฟล์และเขียน Header ถ้ายังไม่มีไฟล์"""
    if not os.path.exists(FILENAME):
        with open(FILENAME, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Header ของ CSV
            # timestamp: เวลา (ms)
            # price: ราคา
            # quantity: จำนวนเหรียญ
            # side: BUY หรือ SELL (แปลงจาก maker)
            # is_maker: ข้อมูลดิบ (True=Sell Taker, False=Buy Taker)
            writer.writerow(['timestamp', 'readable_time', 'price', 'quantity', 'side', 'is_maker'])
        print(f"สร้างไฟล์ {FILENAME} เรียบร้อยแล้ว")

def save_to_csv(data_row):
    """บันทึกข้อมูล 1 แถวลงไฟล์"""
    with open(FILENAME, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(data_row)

# --- WebSocket Callbacks ---
def on_message(ws, message):
    try:
        json_message = json.loads(message)
        
        # ดึงข้อมูลที่จำเป็น (Feature Extraction)
        timestamp = json_message['T'] # เวลาที่เกิด Trade
        price = float(json_message['p'])
        qty = float(json_message['q'])
        is_maker = json_message['m'] # True = คนขายเป็น Maker (แปลว่าคนซื้อเคาะขวา = แรงซื้อ) ? ผิด! ดูบรรทัดล่าง
        
        # แปลงความหมาย Maker/Taker ให้เข้าใจง่าย
        # ถ้า is_maker = True  --> Maker คือคนขาย (คนเคาะคือคนขาย) -> คือแรงขาย (SELL)
        # ถ้า is_maker = False --> Maker คือคนซื้อ (คนเคาะคือคนซื้อ) -> คือแรงซื้อ (BUY)
        side = "SELL" if is_maker else "BUY"
        
        # แปลงเวลาให้อ่านออก (เผื่อเปิดดูเอง)
        readable_time = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S.%f')

        # ปริ้นท์ดูในจอ (Option)
        color = "\033[92m" if side == "BUY" else "\033[91m" # เขียว/แดง
        reset = "\033[0m"
        print(f"{color}[{side}]{reset} Price: {price} | Vol: {qty}")

        # เตรียมข้อมูลลง CSV
        row = [timestamp, readable_time, price, qty, side, is_maker]
        save_to_csv(row)

    except Exception as e:
        print(f"Error: {e}")

def on_error(ws, error):
    print(f"Error เกิดขึ้น: {error}")

def on_close(ws, close_status_code, close_msg):
    print("### การเชื่อมต่อถูกตัด ###")

def on_open(ws):
    print("--- เชื่อมต่อ Binance Server สำเร็จ เริ่มเก็บข้อมูล... ---")

# --- Main Execution ---
if __name__ == "__main__":
    init_csv() # เช็คไฟล์ก่อน
    
    # เริ่มเชื่อมต่อ
    ws = websocket.WebSocketApp(SOCKET,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)
    
    # รันไปเรื่อยๆ
    ws.run_forever()