import websocket, json, datetime, sys, requests, threading, time
import pandas as pd
import lightgbm as lgb
from collections import deque
from binance.client import Client
from binance.enums import *

# ==========================================
# 1. CONFIGURATION
# ==========================================
SYMBOL_WS = "btcusdc"
SYMBOL_TRADE = "BTCUSDC"
MODEL_FILE = "btcusdc_training_data.txt"

# --- TELEGRAM ---
TG_TOKEN = "8552406124:AAGhfHsvF0B65FeefrvEPHxzlW3pwZcmMkY"
TG_CHAT_ID = "8440162744"

# --- API KEYS ---
API_KEY = "1EVQwptQguKnWL2ZG0aFNo4VQYfWj3pa2k6oxDT7JeLzjUeqPZqMsfPxsxBuPShy"
SECRET_KEY = "ePHw4rwFMTrkwwmdruClXQzOSX9WRvMVFulDDWeAjkZvrHAGkEAIkr3h1HeCsqyv"

# --- Strategy ---
CONFIDENCE_THRESHOLD = 0.40
CAPITAL_PER_TRADE = 200
HOLDING_TIME = 1000
PROFIT_TARGET_PCT = 0.00075
STOP_LOSS_PCT = 0.01
STATUS_REPORT_INTERVAL = 1800

# --- Maker Buy Settings ---
MAKER_BUY_OFFSET_PCT = 0.000   # 0.03% ต่ำกว่าราคาตลาด (Maker)
MAKER_ORDER_TIMEOUT = 60        # Timeout สำหรับ Limit Order

# --- Concurrent Positions ---
MAX_POSITIONS = 4
COOLDOWN_SECONDS = 180
SLOT2_COOLDOWN_SECONDS = 120
SLOT3_COOLDOWN_SECONDS = 60
SLOT4_COOLDOWN_SECONDS = 180

# ==========================================
# 2. CONNECT TO BINANCE
# ==========================================
try:
    client = Client(API_KEY, SECRET_KEY, testnet=True)
    client.FUTURES_URL = 'https://demo-fapi.binance.com'
    
    balance = client.futures_account_balance()
    usdt = next((item for item in balance if item["asset"] == "USDT"), None)
    print(f"\n✅ เชื่อมต่อ Binance Testnet สำเร็จ!")
    print(f"💰 เงินในพอร์ต: {usdt['balance']} USDT")

except Exception as e:
    print(f"❌ เชื่อมต่อไม่ได้: {e}")
    sys.exit()

# ==========================================
# GLOBAL VARS
# ==========================================
IS_RUNNING = True
HAS_EXISTING_POSITION = False  # Flag สำหรับตรวจสอบ Position เดิม
stats = {'win': 0, 'loss': 0, 'breakeven': 0, 'unfilled': 0}
total_pnl_cash = 0.0
active_orders = []
pending_orders = []
timeout_history = []
loss_history = []
last_trade_time_per_slot = [0] * MAX_POSITIONS
last_status_report_time = time.time()
last_update_id = 0
buffer = deque(maxlen=60)
current_sec = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'close': 0.0, 'low': 999999.0, 'ts': None}

# โหลดโมเดล
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    print(f"✅ Loaded AI Model OK")
except:
    print(f"❌ Model File Not Found"); sys.exit()

# ==========================================
# 3. TELEGRAM FUNCTIONS
# ==========================================
def send_tg_msg(msg):
    try: 
        requests.post(
            f"https://api.telegram.org/bot{TG_TOKEN}/sendMessage", 
            data={'chat_id': TG_CHAT_ID, 'text': msg, 'parse_mode': 'HTML'},
            timeout=5
        )
    except: pass

def send_status_report():
    global last_status_report_time
    
    current_time = time.time()
    if current_time - last_status_report_time >= STATUS_REPORT_INTERVAL:
        total_trades = stats['win'] + stats['loss'] + stats['breakeven']
        win_rate = (stats['win'] / total_trades * 100) if total_trades > 0 else 0
        
        send_tg_msg(
            f"📊 <b>AUTO REPORT</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"💰 Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
            f"✅ Win: {stats['win']} | ❌ Loss: {stats['loss']}\n"
            f"📈 Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"📋 Active: {len(active_orders)} | Pending: {len(pending_orders)}"
        )
        last_status_report_time = current_time

def get_telegram_updates():
    global last_update_id
    try:
        response = requests.get(
            f"https://api.telegram.org/bot{TG_TOKEN}/getUpdates",
            params={'offset': last_update_id + 1, 'timeout': 5},
            timeout=10
        )
        data = response.json()
        if data.get('ok') and data.get('result'):
            return data['result']
    except:
        pass
    return []

def handle_telegram_commands():
    global last_update_id, IS_RUNNING, active_orders, pending_orders, stats, total_pnl_cash
    global HOLDING_TIME, STOP_LOSS_PCT, PROFIT_TARGET_PCT, CONFIDENCE_THRESHOLD, CAPITAL_PER_TRADE
    global MAKER_BUY_OFFSET_PCT, MAKER_ORDER_TIMEOUT, HAS_EXISTING_POSITION
    
    updates = get_telegram_updates()
    for update in updates:
        if 'update_id' not in update:
            continue
        last_update_id = update['update_id']
        
        if 'message' not in update or not update['message']:
            continue
        
        if 'text' not in update['message'] or not update['message']['text']:
            continue
            
        message = update['message']['text'].strip()
        
        if message == '/status':
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            win_rate = (stats['win'] / total_trades * 100) if total_trades > 0 else 0
            
            try:
                balance = client.futures_account_balance()
                usdt = next((item for item in balance if item["asset"] == "USDT"), None)
                balance_text = f"💰 Balance: ${float(usdt['balance']):.2f}"
                
                # ดึง Position ปัจจุบัน
                positions = client.futures_position_information(symbol=SYMBOL_TRADE)
                pos_amt = 0
                pos_entry = 0
                pos_pnl = 0
                for pos in positions:
                    amt = float(pos['positionAmt'])
                    if amt > 0:
                        pos_amt = amt
                        pos_entry = float(pos['entryPrice'])
                        pos_pnl = float(pos['unRealizedProfit'])
                        break
                
                pos_text = f"📊 Position: {pos_amt} BTC @ ${pos_entry:.2f}\n💹 Unrealized: ${pos_pnl:.2f}" if pos_amt > 0 else "📊 Position: None"
            except:
                balance_text = "💰 Balance: N/A"
                pos_text = "📊 Position: N/A"
            
            send_tg_msg(
                f"📊 <b>BOT STATUS (MAKER ONLY)</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🤖 Status: {'🟢 RUNNING' if IS_RUNNING else '🔴 STOPPED'}\n"
                f"🔒 Has Existing Pos: {'⚠️ YES' if HAS_EXISTING_POSITION else '✅ NO'}\n"
                f"{balance_text}\n"
                f"{pos_text}\n"
                f"💵 Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"✅ Win: {stats['win']} | ❌ Loss: {stats['loss']}\n"
                f"😐 BE: {stats['breakeven']} | ⏳ Unfilled: {stats['unfilled']}\n"
                f"📈 Win Rate: <b>{win_rate:.1f}%</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📋 Active: {len(active_orders)}\n"
                f"⏱️ Pending: {len(pending_orders)}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⚙️ <b>MAKER SETTINGS:</b>\n"
                f"📉 Buy Offset: -{MAKER_BUY_OFFSET_PCT*100:.2f}%\n"
                f"⏱️ Timeout: {MAKER_ORDER_TIMEOUT}s\n"
                f"🎯 TP: {PROFIT_TARGET_PCT*100:.3f}%\n"
                f"🛑 SL: {STOP_LOSS_PCT*100:.2f}%"
            )
        
        elif message == '/position':
            try:
                positions = client.futures_position_information(symbol=SYMBOL_TRADE)
                open_orders = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
                
                pos_amt = 0
                pos_entry = 0
                pos_pnl = 0
                for pos in positions:
                    amt = float(pos['positionAmt'])
                    if amt > 0:
                        pos_amt = amt
                        pos_entry = float(pos['entryPrice'])
                        pos_pnl = float(pos['unRealizedProfit'])
                        break
                
                buy_orders = [o for o in open_orders if o['side'] == 'BUY']
                sell_orders = [o for o in open_orders if o['side'] == 'SELL']
                
                send_tg_msg(
                    f"📊 <b>POSITION DETAILS</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📈 Size: <b>{pos_amt} BTC</b>\n"
                    f"💰 Entry: ${pos_entry:.2f}\n"
                    f"💹 Unrealized PNL: ${pos_pnl:.2f}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📋 Open Buy Orders: {len(buy_orders)}\n"
                    f"📋 Open Sell Orders: {len(sell_orders)}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"🔢 Expected: {MAX_POSITIONS} × 0.003 = 0.012 BTC\n"
                    f"{'⚠️ POSITION MISMATCH!' if pos_amt > MAX_POSITIONS * 0.004 else '✅ Position OK'}"
                )
            except Exception as e:
                send_tg_msg(f"❌ Error: {e}")
        
        elif message.startswith('/set_offset'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    new_offset = float(parts[1])
                    MAKER_BUY_OFFSET_PCT = new_offset / 100
                    send_tg_msg(f"✅ Buy Offset: <b>-{new_offset}%</b> (Maker)")
                else:
                    send_tg_msg("❌ ใช้: /set_offset 0.03")
            except:
                send_tg_msg("❌ รูปแบบไม่ถูกต้อง")
        
        elif message.startswith('/set_timeout'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    MAKER_ORDER_TIMEOUT = int(parts[1])
                    send_tg_msg(f"✅ Timeout: <b>{MAKER_ORDER_TIMEOUT}s</b>")
                else:
                    send_tg_msg("❌ ใช้: /set_timeout 60")
            except:
                send_tg_msg("❌ รูปแบบไม่ถูกต้อง")
        
        elif message.startswith('/set_conf'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    CONFIDENCE_THRESHOLD = float(parts[1]) / 100
                    send_tg_msg(f"✅ AI Confidence: <b>{parts[1]}%</b>")
            except:
                send_tg_msg("❌ ใช้: /set_conf 40")
        
        elif message.startswith('/set_sl'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    STOP_LOSS_PCT = float(parts[1]) / 100
                    send_tg_msg(f"✅ Stop Loss: <b>{parts[1]}%</b>")
            except:
                send_tg_msg("❌ ใช้: /set_sl 0.7")
        
        elif message.startswith('/set_tp'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    PROFIT_TARGET_PCT = float(parts[1]) / 100
                    send_tg_msg(f"✅ Take Profit: <b>{parts[1]}%</b>")
            except:
                send_tg_msg("❌ ใช้: /set_tp 0.075")
        
        elif message == '/stop':
            IS_RUNNING = False
            send_tg_msg("🔴 <b>BOT STOPPED</b>\nจะไม่เปิด Position ใหม่")
        
        elif message == '/start':
            IS_RUNNING = True
            send_tg_msg("🟢 <b>BOT STARTED</b>")
        
        elif message == '/reset':
            # Reset flag และ arrays เพื่อเริ่มต้นใหม่
            HAS_EXISTING_POSITION = False
            active_orders.clear()
            pending_orders.clear()
            send_tg_msg(
                f"🔄 <b>BOT RESET</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"✅ Cleared internal state\n"
                f"⚠️ ตรวจสอบ Position บน Binance ด้วย!\n"
                f"ใช้ /position เพื่อดู"
            )
        
        elif message == '/closeall':
            closed_count = 0
            
            try:
                # ยกเลิก Open Orders ทั้งหมดบน Binance
                open_orders = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
                for order in open_orders:
                    cancel_order(SYMBOL_TRADE, order['orderId'])
                    closed_count += 1
                
                # ปิด Position ทั้งหมด
                positions = client.futures_position_information(symbol=SYMBOL_TRADE)
                for pos in positions:
                    amt = float(pos['positionAmt'])
                    if amt > 0:
                        client.futures_create_order(
                            symbol=SYMBOL_TRADE,
                            side='SELL',
                            type='MARKET',
                            quantity=amt
                        )
                        print(f"✅ Closed {amt} BTC Position")
                
                # Clear internal state
                active_orders.clear()
                pending_orders.clear()
                HAS_EXISTING_POSITION = False
                
                send_tg_msg(
                    f"✅ <b>CLOSE ALL COMPLETE</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📋 Cancelled {closed_count} Orders\n"
                    f"📊 Closed all Positions\n"
                    f"🔄 Internal state cleared"
                )
            except Exception as e:
                send_tg_msg(f"❌ Error: {e}")
        
        elif message == '/help':
            send_tg_msg(
                f"📚 <b>COMMANDS</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"/status - สถานะ Bot\n"
                f"/position - ดู Position จริง\n"
                f"/stop /start - หยุด/เริ่ม\n"
                f"/closeall - ปิดทั้งหมด\n"
                f"/reset - Reset state\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<b>SETTINGS:</b>\n"
                f"/set_offset [%] - Buy Offset\n"
                f"/set_timeout [s] - Timeout\n"
                f"/set_conf [%] - AI Conf\n"
                f"/set_sl [%] - Stop Loss\n"
                f"/set_tp [%] - Take Profit"
            )

def telegram_command_loop():
    while True:
        try:
            handle_telegram_commands()
        except Exception as e:
            print(f"❌ Telegram Error: {e}")
        time.sleep(5)

# ==========================================
# 4. SYNC EXISTING POSITIONS (CRITICAL!)
# ==========================================
def sync_existing_positions():
    """ตรวจสอบและจัดการ Position ที่มีอยู่บน Binance ก่อนเริ่ม Bot"""
    global active_orders, HAS_EXISTING_POSITION
    
    try:
        # ดึง Position ปัจจุบัน
        positions = client.futures_position_information(symbol=SYMBOL_TRADE)
        current_position = 0
        entry_price = 0
        
        for pos in positions:
            amt = float(pos['positionAmt'])
            if amt > 0:
                current_position = amt
                entry_price = float(pos['entryPrice'])
                break
        
        # ดึง Open Orders
        open_orders = client.futures_get_open_orders(symbol=SYMBOL_TRADE)
        buy_orders = [o for o in open_orders if o['side'] == 'BUY']
        sell_orders = [o for o in open_orders if o['side'] == 'SELL']
        
        print(f"\n📊 SYNC CHECK:")
        print(f"   Position: {current_position} BTC @ ${entry_price:.2f}")
        print(f"   Open Buy Orders: {len(buy_orders)}")
        print(f"   Open Sell Orders: {len(sell_orders)}")
        
        if current_position > 0:
            HAS_EXISTING_POSITION = True
            
            # คำนวณจำนวน slots ที่ใช้อยู่
            qty_per_slot = round(CAPITAL_PER_TRADE / entry_price, 3) if entry_price > 0 else 0.003
            estimated_slots = round(current_position / qty_per_slot) if qty_per_slot > 0 else 0
            
            print(f"\n⚠️ พบ Position เดิม!")
            print(f"   Estimated Slots: {estimated_slots}")
            print(f"   Expected Max: {MAX_POSITIONS} slots = {MAX_POSITIONS * qty_per_slot:.3f} BTC")
            
            if current_position > MAX_POSITIONS * qty_per_slot * 1.5:
                # Position ใหญ่เกินไป - มีปัญหา!
                send_tg_msg(
                    f"🚨 <b>WARNING: POSITION MISMATCH!</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📊 Position: <b>{current_position} BTC</b>\n"
                    f"💰 Entry: ${entry_price:.2f}\n"
                    f"📋 Open Sells: {len(sell_orders)}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"⚠️ Position ใหญ่กว่าที่คาด!\n"
                    f"🔢 Expected: {MAX_POSITIONS * qty_per_slot:.3f} BTC\n"
                    f"🔢 Actual: {current_position} BTC\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"⛔ Bot จะไม่เปิด Position ใหม่\n"
                    f"ใช้ /closeall เพื่อปิดทั้งหมด\n"
                    f"หรือ /reset หลังจัดการ Position"
                )
            else:
                send_tg_msg(
                    f"⚠️ <b>EXISTING POSITION FOUND</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📊 Position: {current_position} BTC\n"
                    f"💰 Entry: ${entry_price:.2f}\n"
                    f"📋 Open Sells: {len(sell_orders)}\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"⛔ Bot จะไม่เปิด Position ใหม่\n"
                    f"จนกว่า Position เดิมจะปิด\n"
                    f"หรือใช้ /closeall → /reset"
                )
            
            return True
        else:
            HAS_EXISTING_POSITION = False
            print(f"✅ ไม่มี Position เดิม - พร้อมเริ่มใหม่")
            
            # ยกเลิก Open Orders ที่ค้างอยู่ (ถ้ามี)
            if len(open_orders) > 0:
                print(f"🧹 ยกเลิก {len(open_orders)} Open Orders ที่ค้าง...")
                for order in open_orders:
                    cancel_order(SYMBOL_TRADE, order['orderId'])
                send_tg_msg(f"🧹 ยกเลิก {len(open_orders)} Orders ค้าง")
            
            return False
            
    except Exception as e:
        print(f"❌ Error syncing positions: {e}")
        send_tg_msg(f"❌ Sync Error: {e}")
        return False

def check_position_before_trade():
    """ตรวจสอบ Position ก่อนเปิด Order ใหม่ทุกครั้ง"""
    global HAS_EXISTING_POSITION
    
    try:
        positions = client.futures_position_information(symbol=SYMBOL_TRADE)
        current_position = 0
        
        for pos in positions:
            amt = float(pos['positionAmt'])
            if amt > 0:
                current_position = amt
                break
        
        # คำนวณ max allowed position
        max_allowed = MAX_POSITIONS * (CAPITAL_PER_TRADE / 78000) * 1.5  # ประมาณ 0.015 BTC
        
        if current_position > max_allowed:
            if not HAS_EXISTING_POSITION:
                HAS_EXISTING_POSITION = True
                send_tg_msg(
                    f"⚠️ <b>POSITION LIMIT REACHED</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📊 Current: {current_position} BTC\n"
                    f"📊 Max: {max_allowed:.4f} BTC\n"
                    f"⛔ หยุดเปิด Position ใหม่"
                )
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Check position error: {e}")
        return False

# ==========================================
# 5. TRADING FUNCTIONS (MAKER ONLY)
# ==========================================
def place_limit_buy(symbol, quantity, limit_price):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side='BUY',
            type='LIMIT',
            quantity=quantity,
            price=str(round(limit_price, 1)),
            timeInForce='GTC'
        )
        return order
    except Exception as e:
        print(f"❌ Error Limit Buy: {e}")
        return None

def place_limit_sell(symbol, quantity, limit_price):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side='SELL',
            type='LIMIT',
            quantity=quantity,
            price=str(round(limit_price, 1)),
            timeInForce='GTC'
        )
        return order
    except Exception as e:
        print(f"❌ Error Limit Sell: {e}")
        return None

def cancel_order(symbol, order_id):
    try:
        client.futures_cancel_order(symbol=symbol, orderId=order_id)
        return True
    except Exception as e:
        if "Unknown order" in str(e):
            return True
        print(f"❌ Error Cancel: {e}")
        return False

def close_position(symbol, quantity, reason):
    try:
        positions = client.futures_position_information(symbol=symbol)
        current_position = 0
        
        for pos in positions:
            if pos['positionSide'] == 'BOTH' or pos['positionSide'] == 'LONG':
                current_position = float(pos['positionAmt'])
                break
        
        if current_position <= 0 or current_position < quantity * 0.9:
            return True
        
        # ปิดแค่ quantity ที่ระบุ
        close_qty = min(quantity, current_position)
        
        client.futures_create_order(
            symbol=symbol,
            side='SELL',
            type='MARKET',
            quantity=close_qty
        )
        return True
    except Exception as e:
        print(f"❌ Error Close: {e}")
        return False

def check_pending_orders(current_price, current_ts):
    """ตรวจสอบ Pending Orders ว่า filled หรือ timeout"""
    global pending_orders, active_orders, stats, timeout_history
    
    for order in pending_orders[:]:
        # ตรวจสอบว่า filled หรือยัง
        if current_price <= order['limit_price']:
            # Filled! วาง TP Sell
            sell_order = place_limit_sell(SYMBOL_TRADE, order['quantity'], order['take_profit'])
            sell_order_id = sell_order.get('orderId') if sell_order else None
            
            active_orders.append({
                'entry': order['limit_price'],
                'quantity': order['quantity'],
                'take_profit': order['take_profit'],
                'stop_loss': order['stop_loss'],
                'exit_ts': current_ts + HOLDING_TIME,
                'entry_ts': current_ts,
                'sell_order_id': sell_order_id,
                'confidence': order.get('confidence', 0),
                'slot': order.get('slot', 0)
            })
            
            pending_orders.remove(order)
            
            send_tg_msg(
                f"🟢 <b>POSITION OPENED (MAKER)</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📥 Entry: ${order['limit_price']:.2f}\n"
                f"🎯 TP: ${order['take_profit']:.2f}\n"
                f"🛑 SL: ${order['stop_loss']:.2f}\n"
                f"🤖 AI: {order['confidence']*100:.2f}%\n"
                f"📊 Active: {len(active_orders)}/{MAX_POSITIONS}"
            )
            continue
        
        # ตรวจสอบ Timeout
        if current_ts >= order['timeout_ts']:
            stats['unfilled'] += 1
            print(f"\n⏳ Order Timeout @ ${order['limit_price']:.2f}")
            
            timeout_history.append({
                'time': datetime.datetime.now().strftime('%H:%M:%S'),
                'limit_price': order['limit_price'],
                'confidence': order.get('confidence', 0),
                'timeout': MAKER_ORDER_TIMEOUT
            })
            
            if order.get('order_id'):
                cancel_order(SYMBOL_TRADE, order['order_id'])
            
            pending_orders.remove(order)

def check_active_orders(current_price, current_ts):
    """ตรวจสอบ Active Orders สำหรับ TP/SL/Time Exit"""
    global stats, total_pnl_cash, loss_history, HAS_EXISTING_POSITION
    
    for order in active_orders[:]:
        is_exit = False
        reason = ""
        
        if current_price >= order['take_profit']:
            is_exit = True
            reason = "TP WIN (MAKER) 🎯"
        elif current_price <= order['stop_loss']:
            is_exit = True
            reason = "STOP LOSS (TAKER) 🛑"
        elif current_ts >= order['exit_ts']:
            is_exit = True
            reason = "TIME EXIT (TAKER) ⏳"

        if is_exit:
            if order.get('sell_order_id'):
                cancel_order(SYMBOL_TRADE, order['sell_order_id'])
            
            if "TP" not in reason:
                close_position(SYMBOL_TRADE, order['quantity'], reason)
            
            profit = (current_price - order['entry']) * order['quantity']
            total_pnl_cash += profit
            
            if profit > 0: 
                stats['win'] += 1
            elif profit < 0: 
                stats['loss'] += 1
                loss_history.append({
                    'time': datetime.datetime.now().strftime('%H:%M:%S'),
                    'slot': order['slot'],
                    'entry': order['entry'],
                    'exit': current_price,
                    'pnl': profit,
                    'reason': reason,
                    'confidence': order.get('confidence', 0)
                })
            else: 
                stats['breakeven'] += 1
            
            print(f"\n✅ CLOSED: ${current_price:.2f} | PNL: ${profit:.4f} | {reason}")
            
            send_tg_msg(
                f"{'✅' if profit >= 0 else '❌'} <b>CLOSED</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📥 Entry: ${order['entry']:.2f}\n"
                f"📤 Exit: ${current_price:.2f}\n"
                f"💰 PNL: <b>${profit:.4f}</b>\n"
                f"📝 {reason}\n"
                f"📊 Active: {len(active_orders)-1}/{MAX_POSITIONS}"
            )
            
            active_orders.remove(order)
            
            # ตรวจสอบว่าปิดหมดแล้วหรือยัง
            if len(active_orders) == 0 and len(pending_orders) == 0:
                HAS_EXISTING_POSITION = False
                print("✅ All positions closed - Ready for new trades")

# ==========================================
# 6. PREDICTION & SLOT MANAGEMENT
# ==========================================
def get_available_slot(current_ts):
    total_open = len(active_orders) + len(pending_orders)
    if total_open >= MAX_POSITIONS:
        return None
    
    if total_open == 0:
        return 0
    
    if total_open == 1 and len(active_orders) == 1:
        first_order = active_orders[0]
        entry_time = first_order.get('entry_ts', current_ts)
        if (current_ts - entry_time) >= SLOT2_COOLDOWN_SECONDS:
            return 1
    
    if total_open == 2 and len(active_orders) == 2:
        slot2 = next((o for o in active_orders if o['slot'] == 1), None)
        if slot2:
            entry_time = slot2.get('entry_ts', current_ts)
            if (current_ts - entry_time) >= SLOT3_COOLDOWN_SECONDS:
                return 2
    
    if total_open == 3 and len(active_orders) == 3:
        slot3 = next((o for o in active_orders if o['slot'] == 2), None)
        if slot3:
            entry_time = slot3.get('entry_ts', current_ts)
            if (current_ts - entry_time) >= SLOT4_COOLDOWN_SECONDS:
                return 3
    
    return None

def predict(data_list, last_price, current_ts):
    global last_trade_time_per_slot, pending_orders, HAS_EXISTING_POSITION
    
    if not IS_RUNNING: 
        return
    
    # ตรวจสอบ Position ก่อนเปิด Order ใหม่
    if HAS_EXISTING_POSITION:
        return
    
    # Double check กับ Binance ทุกๆ 30 วินาที
    if current_ts % 30 == 0:
        if not check_position_before_trade():
            return

    available_slot = get_available_slot(current_ts)
    if available_slot is None: 
        return

    df = pd.DataFrame(data_list)
    if len(df) < 15: 
        return
    
    feat = {
        'total_volume': df['total_volume'].iloc[-1], 
        'net_flow': df['net_flow'].iloc[-1],
        'trade_count': df['trade_count'].iloc[-1],
        'net_flow_ma5': df['net_flow'].rolling(5).mean().iloc[-1],
        'net_flow_ma15': df['net_flow'].rolling(15).mean().iloc[-1],
        'volume_ma5': df['total_volume'].rolling(5).mean().iloc[-1],
        'net_flow_diff': df['net_flow'].diff().iloc[-1],
        'price_change': df['close'].pct_change().iloc[-1] * 100,
        'std_5': df['close'].rolling(5).std().iloc[-1],
        'dist_ma15': df['close'].iloc[-1] - df['close'].rolling(15).mean().iloc[-1]
    }
    
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    feat['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10)))).iloc[-1]

    prob = model.predict(pd.DataFrame([feat]))[0]
    
    print(f"\rPrice: {last_price:.2f} | Prob: {prob*100:.2f}% | Active: {len(active_orders)} | Pending: {len(pending_orders)}", end="")

    if prob >= CONFIDENCE_THRESHOLD:
        try:
            # Maker Buy: ต่ำกว่าราคาตลาด
            limit_buy_price = last_price * (1 - MAKER_BUY_OFFSET_PCT)
            qty = round(CAPITAL_PER_TRADE / last_price, 3)
            
            tp = limit_buy_price * (1 + PROFIT_TARGET_PCT)
            sl = limit_buy_price * (1 - STOP_LOSS_PCT)
            
            print(f"\n⚡ [MAKER BUY] Slot {available_slot} @ ${limit_buy_price:.2f} (Market: ${last_price:.2f})")
            print(f"   🎯 TP: ${tp:.2f} | 🛑 SL: ${sl:.2f} | 🤖 AI: {prob*100:.2f}%")
            
            order_response = place_limit_buy(SYMBOL_TRADE, qty, limit_buy_price)
            
            if order_response:
                order_id = order_response.get('orderId')
                
                pending_orders.append({
                    'limit_price': limit_buy_price,
                    'quantity': qty,
                    'take_profit': tp,
                    'stop_loss': sl,
                    'timeout_ts': current_ts + MAKER_ORDER_TIMEOUT,
                    'order_id': order_id,
                    'confidence': prob,
                    'slot': available_slot
                })
                
                last_trade_time_per_slot[available_slot] = current_ts
                
                send_tg_msg(
                    f"⚡ <b>MAKER BUY PLACED</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"📉 Limit: ${limit_buy_price:.2f}\n"
                    f"📊 Market: ${last_price:.2f}\n"
                    f"🎯 TP: ${tp:.2f}\n"
                    f"🛑 SL: ${sl:.2f}\n"
                    f"🤖 AI: {prob*100:.2f}%\n"
                    f"⏱️ Timeout: {MAKER_ORDER_TIMEOUT}s\n"
                    f"📋 Slot: {available_slot + 1}/{MAX_POSITIONS}"
                )

        except Exception as e:
            print(f"\n❌ ERROR: {e}")

# ==========================================
# 7. WEBSOCKET RUNNER
# ==========================================
def on_message(ws, msg):
    global current_sec
    d = json.loads(msg)
    p, q, m, t = float(d['p']), float(d['q']), d['m'], int(d['T']/1000)
    
    if current_sec['ts'] is None: 
        current_sec['ts'] = t
    
    check_pending_orders(p, t)
    check_active_orders(p, t)
    send_status_report()
    
    if t > current_sec['ts']:
        buffer.append(current_sec.copy()) 
        predict(list(buffer), p, t)
        current_sec = {'net_flow': 0.0, 'total_volume': 0.0, 'trade_count': 0, 'close': p, 'low': p, 'ts': t}
    
    current_sec['net_flow'] += -q if m else q
    current_sec['total_volume'] += q
    current_sec['trade_count'] += 1
    current_sec['close'] = p
    if p < current_sec['low']: 
        current_sec['low'] = min(current_sec['low'], p)

def on_error(ws, error):
    print(f"\n❌ WebSocket Error: {error}")

def on_close(ws, close_status, close_msg):
    print(f"\n⚠️ WebSocket Closed: {close_status} - {close_msg}")

def on_open(ws):
    print("✅ WebSocket Connected")

# ==========================================
# 8. MAIN
# ==========================================
print(f"\n{'='*50}")
print(f"🚀 STARTING BOT (MAKER ONLY + SYNC)")
print(f"{'='*50}")
print(f"📉 Buy Offset: -{MAKER_BUY_OFFSET_PCT*100:.2f}% (Maker)")
print(f"⏱️ Timeout: {MAKER_ORDER_TIMEOUT}s")
print(f"📊 Max Positions: {MAX_POSITIONS}")
print(f"{'='*50}")

# CRITICAL: Sync positions ก่อนเริ่ม!
sync_existing_positions()

send_tg_msg(
    f"🚀 <b>BOT STARTED</b>\n"
    f"━━━━━━━━━━━━━━━━\n"
    f"📉 Buy Offset: -{MAKER_BUY_OFFSET_PCT*100:.2f}%\n"
    f"⏱️ Timeout: {MAKER_ORDER_TIMEOUT}s\n"
    f"🎯 TP: {PROFIT_TARGET_PCT*100:.3f}%\n"
    f"🛑 SL: {STOP_LOSS_PCT*100:.2f}%\n"
    f"📊 Max: {MAX_POSITIONS} positions\n"
    f"━━━━━━━━━━━━━━━━\n"
    f"🔄 Sync: {'⚠️ Has Position' if HAS_EXISTING_POSITION else '✅ Clean'}\n"
    f"━━━━━━━━━━━━━━━━\n"
    f"ใช้ /help เพื่อดูคำสั่ง"
)

# Start Telegram handler
telegram_thread = threading.Thread(target=telegram_command_loop, daemon=True)
telegram_thread.start()
print("✅ Telegram Handler Started")

# Start WebSocket with reconnect
while True:
    try:
        ws = websocket.WebSocketApp(
            f"wss://demo-fstream.binance.com/ws/{SYMBOL_WS}@aggTrade",
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        ws.run_forever(ping_interval=30, ping_timeout=10)
    except Exception as e:
        print(f"\n❌ Fatal Error: {e}")
    
    print("\n🔄 Reconnecting in 5 seconds...")
    time.sleep(5)