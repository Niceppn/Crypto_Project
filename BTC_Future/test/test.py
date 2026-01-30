import websocket, json, datetime, sys, requests, threading, time
import pandas as pd
import lightgbm as lgb
from collections import deque
from binance.client import Client
from binance.enums import *

# ==========================================
# 1. CONFIGURATION
# ==========================================
SYMBOL_WS = "btcusdc"       # ชื่อสำหรับ Websocket (ตัวเล็ก)
SYMBOL_TRADE = "BTCUSDC"    # ชื่อสำหรับส่งคำสั่ง (ตัวใหญ่)
MODEL_FILE = "/Users/Macbook/Collect_Crypto/BTC_Future/test/BTCUSDC.txt"

# --- TELEGRAM ---
TG_TOKEN = "8552406124:AAGhfHsvF0B65FeefrvEPHxzlW3pwZcmMkY"
TG_CHAT_ID = "8440162744"

# --- API KEYS ---
API_KEY = "1EVQwptQguKnWL2ZG0aFNo4VQYfWj3pa2k6oxDT7JeLzjUeqPZqMsfPxsxBuPShy"
SECRET_KEY = "ePHw4rwFMTrkwwmdruClXQzOSX9WRvMVFulDDWeAjkZvrHAGkEAIkr3h1HeCsqyv"

# --- Strategy (สามารถปรับผ่าน Telegram ได้) ---
CONFIDENCE_THRESHOLD = 0.40  # ความมั่นใจ AI (สามารถปรับได้)
CAPITAL_PER_TRADE = 200      # ทุนต่อไม้ (สามารถปรับได้)
HOLDING_TIME = 500           # วินาที (สามารถปรับได้)
PROFIT_TARGET_PCT = 0.0002   # % (สามารถปรับได้)
STOP_LOSS_PCT = 0.001        # % (สามารถปรับได้)
MAKER_BUY_OFFSET_PCT = 0.00001
MAKER_ORDER_TIMEOUT = 60     # Timeout ของ Limit Order (สามารถปรับได้)
STATUS_REPORT_INTERVAL = 1800  # 30 นาที

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
stats = {'win': 0, 'loss': 0, 'breakeven': 0, 'unfilled': 0}
total_pnl_cash = 0.0
active_orders = []
pending_orders = []
timeout_history = []  # เก็บประวัติ order ที่ timeout
last_trade_time = 0
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
    """ส่งรายงานสถานะทุก 30 นาที"""
    global last_status_report_time
    
    current_time = time.time()
    if current_time - last_status_report_time >= STATUS_REPORT_INTERVAL:
        total_trades = stats['win'] + stats['loss'] + stats['breakeven']
        win_rate = (stats['win'] / total_trades * 100) if total_trades > 0 else 0
        
        send_tg_msg(
            f"📊 <b>AUTO REPORT (30 min)</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"⏰ {datetime.datetime.now().strftime('%H:%M:%S')}\n"
            f"💰 Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"✅ Win: {stats['win']}\n"
            f"❌ Loss: {stats['loss']}\n"
            f"😐 BE: {stats['breakeven']}\n"
            f"⏳ Unfilled: {stats['unfilled']}\n"
            f"📈 Win Rate: <b>{win_rate:.1f}%</b>\n"
            f"━━━━━━━━━━━━━━━━\n"
            f"📋 Active: {len(active_orders)}\n"
            f"⏱️ Pending: {len(pending_orders)}"
        )
        last_status_report_time = current_time

# ==========================================
# TELEGRAM COMMAND HANDLER
# ==========================================
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
    global HOLDING_TIME, STOP_LOSS_PCT, PROFIT_TARGET_PCT, CONFIDENCE_THRESHOLD, CAPITAL_PER_TRADE, MAKER_ORDER_TIMEOUT
    
    updates = get_telegram_updates()
    for update in updates:
        last_update_id = update['update_id']
        
        if 'message' not in update or 'text' not in update['message']:
            continue
            
        message = update['message']['text'].strip()
        
        # /status
        if message == '/status':
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            win_rate = (stats['win'] / total_trades * 100) if total_trades > 0 else 0
            
            try:
                balance = client.futures_account_balance()
                usdt = next((item for item in balance if item["asset"] == "USDT"), None)
                balance_text = f"💰 Balance: ${float(usdt['balance']):.2f}"
            except:
                balance_text = "💰 Balance: N/A"
            
            send_tg_msg(
                f"📊 <b>BOT STATUS</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"🤖 Status: {'🟢 RUNNING' if IS_RUNNING else '🔴 STOPPED'}\n"
                f"{balance_text}\n"
                f"💵 Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"✅ Win: {stats['win']}\n"
                f"❌ Loss: {stats['loss']}\n"
                f"😐 BE: {stats['breakeven']}\n"
                f"⏳ Unfilled: {stats['unfilled']}\n"
                f"📈 Win Rate: <b>{win_rate:.1f}%</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📋 Active Orders: {len(active_orders)}\n"
                f"⏱️ Pending Orders: {len(pending_orders)}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"⚙️ <b>SETTINGS:</b>\n"
                f"🤖 AI Confidence: {CONFIDENCE_THRESHOLD*100:.0f}%\n"
                f"💰 Capital/Trade: ${CAPITAL_PER_TRADE}\n"
                f"⏰ Holding: {HOLDING_TIME}s\n"
                f"🎯 TP: {PROFIT_TARGET_PCT*100:.3f}%\n"
                f"🛑 SL: {STOP_LOSS_PCT*100:.3f}%\n"
                f"⏱️ Order Timeout: {MAKER_ORDER_TIMEOUT}s"
            )
        
        # /holding - ดูว่าถือไปกี่วินาทีแล้ว
        elif message == '/holding':
            if len(active_orders) == 0:
                send_tg_msg("ℹ️ ไม่มี Active Orders")
            else:
                current_ts = int(time.time())
                msg_list = ["⏱️ <b>HOLDING TIME</b>\n━━━━━━━━━━━━━━━━"]
                
                for i, order in enumerate(active_orders, 1):
                    entry_time = order['exit_ts'] - HOLDING_TIME
                    holding_sec = current_ts - entry_time
                    remaining_sec = order['exit_ts'] - current_ts
                    confidence = order.get('confidence', 0)
                    
                    msg_list.append(
                        f"\n<b>Order #{i}</b>\n"
                        f"📥 Entry: ${order['entry']:.2f}\n"
                        f"🤖 AI Conf: {confidence*100:.2f}%\n"
                        f"⏰ Holding: {holding_sec}s / {HOLDING_TIME}s\n"
                        f"⏳ Remaining: {remaining_sec}s\n"
                        f"🎯 TP: ${order['take_profit']:.2f}\n"
                        f"🛑 SL: ${order['stop_loss']:.2f}"
                    )
                
                send_tg_msg("\n".join(msg_list))
        
        # /timeout - ดูประวัติ order ที่ timeout
        elif message == '/timeout':
            if len(timeout_history) == 0:
                send_tg_msg("ℹ️ ยังไม่มี Order ที่ Timeout")
            else:
                msg_list = [f"⏳ <b>TIMEOUT HISTORY</b> (Last 10)\n━━━━━━━━━━━━━━━━"]
                
                # แสดง 10 รายการล่าสุด
                for i, order in enumerate(timeout_history[-10:], 1):
                    msg_list.append(
                        f"\n<b>#{i}</b> @ {order['time']}\n"
                        f"💵 Price: ${order['limit_price']:.2f}\n"
                        f"🤖 AI Conf: {order['confidence']*100:.2f}%\n"
                        f"⏱️ Timeout: {order['timeout']}s"
                    )
                
                send_tg_msg("\n".join(msg_list))
        
        # /set_conf 45 (ตั้งความมั่นใจเป็น 45%)
        elif message.startswith('/set_conf'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    new_conf = float(parts[1])
                    if 0 <= new_conf <= 100:
                        CONFIDENCE_THRESHOLD = new_conf / 100
                        send_tg_msg(
                            f"✅ <b>AI CONFIDENCE UPDATED</b>\n"
                            f"━━━━━━━━━━━━━━━━\n"
                            f"🤖 New Threshold: <b>{new_conf}%</b>\n"
                            f"ℹ️ จะใช้กับไม้ใหม่เท่านั้น"
                        )
                    else:
                        send_tg_msg("❌ ต้องอยู่ระหว่าง 0-100")
                else:
                    send_tg_msg("❌ ใช้: /set_conf 45")
            except:
                send_tg_msg("❌ รูปแบบไม่ถูกต้อง\nใช้: /set_conf 45")
        
        # /set_cap 150 (ตั้งทุนต่อไม้เป็น 150 USDT)
        elif message.startswith('/set_cap'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    new_cap = float(parts[1])
                    if new_cap >= 100:  # ขั้นต่ำของ Binance
                        CAPITAL_PER_TRADE = new_cap
                        send_tg_msg(
                            f"✅ <b>CAPITAL UPDATED</b>\n"
                            f"━━━━━━━━━━━━━━━━\n"
                            f"💰 New Capital: <b>${new_cap}</b>\n"
                            f"ℹ️ จะใช้กับไม้ใหม่เท่านั้น"
                        )
                    else:
                        send_tg_msg("❌ ต้องไม่ต่ำกว่า $100 (ขั้นต่ำ Binance)")
                else:
                    send_tg_msg("❌ ใช้: /set_cap 150")
            except:
                send_tg_msg("❌ รูปแบบไม่ถูกต้อง\nใช้: /set_cap 150")
        
        # /set_timeout 90 (ตั้ง timeout เป็น 90 วินาที)
        elif message.startswith('/set_timeout'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    new_timeout = int(parts[1])
                    if new_timeout >= 10:
                        MAKER_ORDER_TIMEOUT = new_timeout
                        send_tg_msg(
                            f"✅ <b>ORDER TIMEOUT UPDATED</b>\n"
                            f"━━━━━━━━━━━━━━━━\n"
                            f"⏱️ New Timeout: <b>{new_timeout}s</b>\n"
                            f"ℹ️ จะใช้กับไม้ใหม่เท่านั้น"
                        )
                    else:
                        send_tg_msg("❌ ต้องไม่ต่ำกว่า 10 วินาที")
                else:
                    send_tg_msg("❌ ใช้: /set_timeout 90")
            except:
                send_tg_msg("❌ รูปแบบไม่ถูกต้อง\nใช้: /set_timeout 90")
        
        # /set_sl 0.15 (ตั้ง Stop Loss เป็น 0.15%)
        elif message.startswith('/set_sl'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    new_sl = float(parts[1])
                    STOP_LOSS_PCT = new_sl / 100
                    send_tg_msg(
                        f"✅ <b>STOP LOSS UPDATED</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"🛑 New SL: <b>{new_sl}%</b>\n"
                        f"ℹ️ จะใช้กับไม้ใหม่เท่านั้น"
                    )
                else:
                    send_tg_msg("❌ ใช้: /set_sl 0.15")
            except:
                send_tg_msg("❌ รูปแบบไม่ถูกต้อง\nใช้: /set_sl 0.15")
        
        # /set_profit 0.03 (ตั้ง Take Profit เป็น 0.03%)
        elif message.startswith('/set_profit'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    new_tp = float(parts[1])
                    PROFIT_TARGET_PCT = new_tp / 100
                    send_tg_msg(
                        f"✅ <b>TAKE PROFIT UPDATED</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"🎯 New TP: <b>{new_tp}%</b>\n"
                        f"ℹ️ จะใช้กับไม้ใหม่เท่านั้น"
                    )
                else:
                    send_tg_msg("❌ ใช้: /set_profit 0.03")
            except:
                send_tg_msg("❌ รูปแบบไม่ถูกต้อง\nใช้: /set_profit 0.03")
        
        # /set_holding 600 (ตั้งเวลาถือเป็น 600 วินาที)
        elif message.startswith('/set_holding'):
            try:
                parts = message.split()
                if len(parts) == 2:
                    new_holding = int(parts[1])
                    HOLDING_TIME = new_holding
                    send_tg_msg(
                        f"✅ <b>HOLDING TIME UPDATED</b>\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"⏰ New Holding: <b>{new_holding}s</b>\n"
                        f"ℹ️ จะใช้กับไม้ใหม่เท่านั้น"
                    )
                else:
                    send_tg_msg("❌ ใช้: /set_holding 600")
            except:
                send_tg_msg("❌ รูปแบบไม่ถูกต้อง\nใช้: /set_holding 600")
        
        # /stop
        elif message == '/stop':
            IS_RUNNING = False
            send_tg_msg(
                f"🔴 <b>BOT STOPPED</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"Bot หยุดเทรดใหม่\n"
                f"Orders เก่ายังทำงานต่อ\n"
                f"ใช้ /start เพื่อเริ่มใหม่"
            )
        
        # /start
        elif message == '/start':
            IS_RUNNING = True
            send_tg_msg(
                f"🟢 <b>BOT STARTED</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"Bot กลับมาทำงานแล้ว!"
            )
        
        # /balance
        elif message == '/balance':
            try:
                balance = client.futures_account_balance()
                usdt = next((item for item in balance if item["asset"] == "USDT"), None)
                
                send_tg_msg(
                    f"💰 <b>ACCOUNT BALANCE</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"💵 Available: ${float(usdt['balance']):.2f}\n"
                    f"💸 Total PNL: ${total_pnl_cash:.4f}\n"
                    f"🌐 Server: DEMO"
                )
            except Exception as e:
                send_tg_msg(f"❌ Error: {e}")
        
        # /closeall
        elif message == '/closeall':
            if len(active_orders) == 0 and len(pending_orders) == 0:
                send_tg_msg("ℹ️ ไม่มี Orders ที่ต้องปิด")
            else:
                closed_count = 0
                
                for order in active_orders[:]:
                    if order.get('sell_order_id'):
                        cancel_order(SYMBOL_TRADE, order['sell_order_id'])
                    close_position(SYMBOL_TRADE, order['quantity'], "MANUAL CLOSE")
                    active_orders.remove(order)
                    closed_count += 1
                
                for order in pending_orders[:]:
                    if 'order_id' in order:
                        cancel_order(SYMBOL_TRADE, order['order_id'])
                    pending_orders.remove(order)
                    closed_count += 1
                
                send_tg_msg(
                    f"✅ <b>CLOSE ALL</b>\n"
                    f"━━━━━━━━━━━━━━━━\n"
                    f"ปิด/ยกเลิก {closed_count} Orders"
                )
        
        # /stats
        elif message == '/stats':
            total_trades = stats['win'] + stats['loss'] + stats['breakeven']
            win_rate = (stats['win'] / total_trades * 100) if total_trades > 0 else 0
            avg_pnl = total_pnl_cash / total_trades if total_trades > 0 else 0
            
            send_tg_msg(
                f"📈 <b>TRADING STATISTICS</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"📊 Total Trades: {total_trades}\n"
                f"✅ Win: {stats['win']} ({stats['win']/total_trades*100:.1f}%)\n"
                f"❌ Loss: {stats['loss']} ({stats['loss']/total_trades*100:.1f}%)\n"
                f"😐 Breakeven: {stats['breakeven']}\n"
                f"⏳ Unfilled: {stats['unfilled']}\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"💰 Total PNL: <b>${total_pnl_cash:.4f}</b>\n"
                f"📊 Avg PNL/Trade: ${avg_pnl:.4f}\n"
                f"📈 Win Rate: <b>{win_rate:.1f}%</b>"
            )
        
        # /help
        elif message == '/help':
            send_tg_msg(
                f"📚 <b>AVAILABLE COMMANDS</b>\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<b>📊 INFO</b>\n"
                f"/status - สถานะปัจจุบัน\n"
                f"/holding - ดูว่าถือไปกี่วิแล้ว\n"
                f"/timeout - ดู Orders ที่ Timeout\n"
                f"/balance - ยอดเงินในบัญชี\n"
                f"/stats - สถิติการเทรด\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<b>⚙️ SETTINGS</b>\n"
                f"/set_conf [%] - ความมั่นใจ AI\n"
                f"  ตัวอย่าง: /set_conf 45\n"
                f"/set_cap [USDT] - ทุนต่อไม้\n"
                f"  ตัวอย่าง: /set_cap 150\n"
                f"/set_timeout [วินาที] - Timeout\n"
                f"  ตัวอย่าง: /set_timeout 90\n"
                f"/set_sl [%] - Stop Loss\n"
                f"  ตัวอย่าง: /set_sl 0.15\n"
                f"/set_profit [%] - Take Profit\n"
                f"  ตัวอย่าง: /set_profit 0.03\n"
                f"/set_holding [วินาที] - เวลาถือ\n"
                f"  ตัวอย่าง: /set_holding 600\n"
                f"━━━━━━━━━━━━━━━━\n"
                f"<b>🎮 CONTROL</b>\n"
                f"/stop - หยุด Bot\n"
                f"/start - เริ่ม Bot ใหม่\n"
                f"/closeall - ปิด Orders ทั้งหมด"
            )

def telegram_command_loop():
    while True:
        try:
            handle_telegram_commands()
        except Exception as e:
            print(f"❌ Telegram Command Error: {e}")
        time.sleep(5)

# ==========================================
# 4. TRADING FUNCTIONS
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
        print(f"❌ Error Placing Limit Buy: {e}")
        return None

def place_limit_sell(symbol, quantity, limit_price):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side='SELL',
            type='LIMIT',
            quantity=quantity,
            price=str(round(limit_price, 1)),
            timeInForce='GTC',
            reduceOnly=True
        )
        return order
    except Exception as e:
        print(f"❌ Error Placing Limit Sell: {e}")
        return None

def cancel_order(symbol, order_id):
    try:
        client.futures_cancel_order(symbol=symbol, orderId=order_id)
        return True
    except Exception as e:
        print(f"❌ Error Cancelling: {e}")
        return False

def close_position(symbol, quantity, reason):
    try:
        order = client.futures_create_order(
            symbol=symbol,
            side='SELL',
            type='MARKET',
            quantity=quantity,
            reduceOnly=True
        )
        return True
    except Exception as e:
        print(f"❌ Error Closing: {e}")
        return False

def check_pending_orders(current_price, current_ts):
    global pending_orders, active_orders, stats, timeout_history
    
    for order in pending_orders[:]:
        if current_price <= order['limit_price']:
            print(f"\n✅ [FILLED] Maker Buy @ {order['limit_price']:.2f} | TP: {order['take_profit']:.2f} | SL: {order['stop_loss']:.2f}")
            
            sell_order = place_limit_sell(SYMBOL_TRADE, order['quantity'], order['take_profit'])
            
            if sell_order:
                sell_order_id = sell_order.get('orderId')
                print(f"📝 Limit Sell (TP) placed @ {order['take_profit']:.2f} | OrderID: {sell_order_id}")
                
                active_orders.append({
                    'entry': order['limit_price'],
                    'quantity': order['quantity'],
                    'take_profit': order['take_profit'],
                    'stop_loss': order['stop_loss'],
                    'exit_ts': current_ts + HOLDING_TIME,
                    'buy_order_id': order.get('order_id'),
                    'sell_order_id': sell_order_id,
                    'confidence': order.get('confidence', 0)  # เก็บความมั่นใจ
                })
            else:
                active_orders.append({
                    'entry': order['limit_price'],
                    'quantity': order['quantity'],
                    'take_profit': order['take_profit'],
                    'stop_loss': order['stop_loss'],
                    'exit_ts': current_ts + HOLDING_TIME,
                    'buy_order_id': order.get('order_id'),
                    'sell_order_id': None,
                    'confidence': order.get('confidence', 0)
                })
            
            pending_orders.remove(order)
        
        elif current_ts >= order['timeout_ts']:
            stats['unfilled'] += 1
            print(f"\n⏳ [UNFILLED] Limit Buy @ {order['limit_price']:.2f} (AI: {order.get('confidence', 0)*100:.2f}%) cancelled (timeout)")
            
            # เก็บประวัติ timeout
            timeout_history.append({
                'time': datetime.datetime.now().strftime('%H:%M:%S'),
                'limit_price': order['limit_price'],
                'confidence': order.get('confidence', 0),
                'timeout': MAKER_ORDER_TIMEOUT
            })
            
            # เก็บไว้แค่ 50 รายการล่าสุด
            if len(timeout_history) > 50:
                timeout_history.pop(0)
            
            if 'order_id' in order:
                cancel_order(SYMBOL_TRADE, order['order_id'])
            
            pending_orders.remove(order)

def check_orders(current_price, current_ts):
    global stats, total_pnl_cash
    
    for order in active_orders[:]:
        is_exit = False
        reason = ""
        is_tp_hit = False
        
        if current_price >= order['take_profit']:
            is_exit = True
            is_tp_hit = True
            reason = "TP WIN (MAKER) 🎯"
        elif current_price <= order['stop_loss']:
            is_exit = True
            reason = "STOP LOSS (TAKER) 🛑"
        elif current_ts >= order['exit_ts']:
            is_exit = True
            reason = "TIME EXIT (TAKER) ⏳"

        if is_exit:
            if not is_tp_hit and order.get('sell_order_id'):
                print(f"🔄 Cancelling Limit Sell Order...")
                cancel_order(SYMBOL_TRADE, order['sell_order_id'])
                success = close_position(SYMBOL_TRADE, order['quantity'], reason)
            else:
                success = True
            
            if success:
                profit = (current_price - order['entry']) * order['quantity']
                total_pnl_cash += profit
                
                if profit > 0: stats['win'] += 1
                elif profit < 0: stats['loss'] += 1
                else: stats['breakeven'] += 1
                
                confidence = order.get('confidence', 0)
                print(f"✅ SOLD: {current_price:.2f} | PNL: {profit:.4f} USDT | Total: {total_pnl_cash:.4f} | AI: {confidence*100:.2f}% | {reason}")
                
                active_orders.remove(order)

def predict(data_list, last_price, current_ts):
    global last_trade_time, active_orders, pending_orders
    
    if not IS_RUNNING: return
    if len(active_orders) > 0 or len(pending_orders) > 0 or (current_ts - last_trade_time < 30): return

    df = pd.DataFrame(data_list)
    if len(df) < 15: return
    
    feat = {
        'total_volume': df['total_volume'].iloc[-1], 'net_flow': df['net_flow'].iloc[-1],
        'trade_count': df['trade_count'].iloc[-1],
        'net_flow_ma5': df['net_flow'].rolling(5).mean().iloc[-1],
        'net_flow_ma15': df['net_flow'].rolling(15).mean().iloc[-1],
        'volume_ma5': df['total_volume'].rolling(5).mean().iloc[-1],
        'net_flow_diff': df['net_flow'].diff().iloc[-1],
        'price_change': df['close'].pct_change().iloc[-1] * 100,
        'std_5': df['close'].rolling(5).std().iloc[-1],
        'dist_ma15': df['close'].iloc[-1] - df['close'].rolling(15).mean().iloc[-1]
    }
    delta = df['close'].diff(); gain = (delta.where(delta > 0, 0)).rolling(14).mean(); loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    feat['rsi'] = 100 - (100 / (1 + (gain / (loss + 1e-10)))).iloc[-1]

    prob = model.predict(pd.DataFrame([feat]))[0]
    print(f"\rPrice: {last_price:.2f} | Prob: {prob*100:.2f}%", end="")

    if prob >= CONFIDENCE_THRESHOLD:
        try:
            limit_buy_price = last_price * (1 - MAKER_BUY_OFFSET_PCT)
            qty = round(CAPITAL_PER_TRADE / limit_buy_price, 3)
            
            tp = limit_buy_price * (1 + PROFIT_TARGET_PCT)
            sl = limit_buy_price * (1 - STOP_LOSS_PCT)
            
            print(f"\n⚡ [LIMIT BUY] @ {limit_buy_price:.2f} (Current: {last_price:.2f}) | AI: {prob*100:.2f}% | TP: {tp:.2f} | SL: {sl:.2f}")
            
            order_response = place_limit_buy(SYMBOL_TRADE, qty, limit_buy_price)
            
            if order_response:
                order_id = order_response.get('orderId')
                print(f"✅ Limit Order Placed | OrderID: {order_id}")
                
                pending_orders.append({
                    'limit_price': limit_buy_price,
                    'quantity': qty,
                    'take_profit': tp,
                    'stop_loss': sl,
                    'timeout_ts': current_ts + MAKER_ORDER_TIMEOUT,
                    'order_id': order_id,
                    'confidence': prob  # เก็บความมั่นใจของ AI
                })
                last_trade_time = current_ts

        except Exception as e:
            print(f"\n❌ BUY ERROR: {e}")

# ==========================================
# 5. WEBSOCKET RUNNER
# ==========================================
def on_message(ws, msg):
    global current_sec
    d = json.loads(msg)
    p, q, m, t = float(d['p']), float(d['q']), d['m'], int(d['T']/1000)
    
    if current_sec['ts'] is None: current_sec['ts'] = t
    
    check_pending_orders(p, t)
    check_orders(p, t)
    send_status_report()
    
    if t > current_sec['ts']:
        buffer.append(current_sec.copy()) 
        predict(list(buffer), p, t)
        current_sec = {'net_flow':0.0, 'total_volume':0.0, 'trade_count':0, 'close':p, 'low':p, 'ts':t}
    
    current_sec['net_flow'] += -q if m else q
    current_sec['total_volume'] += q
    current_sec['trade_count'] += 1
    current_sec['close'] = p
    if p < current_sec['low']: current_sec['low'] = min(current_sec['low'], p)

print(f"🚀 Bot Started (Silent Mode - แจ้งทุก 30 นาที)")
send_tg_msg(
    f"🚀 <b>AI BOT STARTED</b>\n"
    f"━━━━━━━━━━━━━━━━\n"
    f"🔕 Silent Mode: แจ้งทุก 30 นาที\n"
    f"ใช้ /status เพื่อดูสถานะ\n"
    f"ใช้ /help เพื่อดูคำสั่งทั้งหมด"
)

telegram_thread = threading.Thread(target=telegram_command_loop, daemon=True)
telegram_thread.start()
print("✅ Telegram Command Handler Started")

ws = websocket.WebSocketApp(f"wss://demo-fstream.binance.com/ws/{SYMBOL_WS}@aggTrade", on_message=on_message)
ws.run_forever()