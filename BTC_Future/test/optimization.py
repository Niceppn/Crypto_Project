import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from collections import deque, defaultdict
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
CSV_FILE = "/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv"
MODEL_FILE = "/Users/Macbook/Collect_Crypto/BTC_Future/test/btcusdc_training_data_regression.txt"

# ✅ แก้ 1: เปลี่ยน parameter ranges สำหรับ regression model
PARAMETER_RANGES = {
    'PREDICTION_THRESHOLD': (0.02, 0.08),      # แทน CONFIDENCE_THRESHOLD — ถ้า predicted change > นี้ ก็ buy
    'TP_RATIO': (0.5, 0.9),                    # แทน PROFIT_TARGET_PCT — TP = predicted_change * TP_RATIO
    'HOLDING_TIME': (60, 600),                  # ลดจาก 1800 เพราะ regression trade สั้นกว่า
    'STOP_LOSS_PCT': (0.0005, 0.002),           # ลดจาก 0.005 เพราะ avg loss จากตอนเทสต์คือ 0.0118%
    'MAKER_BUY_OFFSET_PCT': (0.00001, 0.0001),
    'MAKER_ORDER_TIMEOUT': (30, 120)            # ลดจาก 300
}

# Optimization Settings
N_TRIALS = 100
CAPITAL_PER_TRADE = 200

# ==========================================
# LOAD MODEL
# ==========================================
try:
    model = lgb.Booster(model_file=MODEL_FILE)
    print("✅ Model Loaded Successfully")
    
    # ===== SANITY CHECK: เทียบ feature order กับ model =====
    model_features = model.feature_name()
    expected_features = [
        'volume', 'volume_ma5', 'volume_ma15', 'net_flow', 'net_flow_ma5',
        'net_flow_ma15', 'buy_volume_ratio', 'dist_ma5', 'volatility_5',
        'rsi', 'momentum_5', 'trade_count'
    ]
    print(f"   Model features : {model_features}")
    print(f"   Expected order : {expected_features}")
    if model_features == expected_features:
        print("   ✅ Feature order match!")
    else:
        print("   ❌ FEATURE ORDER MISMATCH — หยุดเลยครับ")
        exit()
except:
    print("❌ Model File Not Found")
    exit()

# ==========================================
# BACKTESTING ENGINE
# ==========================================
class BacktestEngine:
    def __init__(self, df, model, params):
        self.df = df
        self.model = model
        self.params = params
        
        # Trading State
        self.active_orders = []
        self.pending_orders = []
        self.closed_trades = []
        
        # Statistics
        self.stats = {'win': 0, 'loss': 0, 'breakeven': 0, 'unfilled': 0}
        self.total_pnl = 0.0
        self.equity_curve = []
        
        # Time-based Statistics
        self.hourly_pnl = defaultdict(list)
        self.daily_pnl = defaultdict(list)
        
    def run(self):
        """รัน Backtest"""
        candles = self._aggregate_to_seconds()
        
        buffer = deque(maxlen=60)
        last_trade_time = 0
        
        for idx, row in candles.iterrows():
            current_ts = int(row['timestamp_ms'] / 1000)
            current_price = row['close']
            
            self._check_pending_orders(current_price, current_ts)
            self._check_active_orders(current_price, current_ts)
            
            # ✅ แก้ 2: เพิ่ม buy_volume เข้า buffer สำหรับ buy_volume_ratio
            buffer.append({
                'net_flow': row['net_flow'],
                'total_volume': row['total_volume'],
                'buy_volume': row['buy_volume'],       # เพิ่มใหม่
                'trade_count': row['trade_count'],
                'close': row['close'],
                'low': row['low']
            })
            
            # AI Prediction
            if len(buffer) >= 15:
                if len(self.active_orders) == 0 and len(self.pending_orders) == 0:
                    if current_ts - last_trade_time >= 30:
                        self._predict_and_trade(buffer, current_price, current_ts)
                        last_trade_time = current_ts
            
            self.equity_curve.append(self.total_pnl)
        
        return self._calculate_metrics()
    
    def _aggregate_to_seconds(self):
        """แปลง trade-by-trade เป็น 1-second candles"""
        df = self.df.copy()
        df['timestamp_s'] = df['timestamp_ms'] // 1000
        
        # คำนวณ net_flow
        df['signed_volume'] = df.apply(
            lambda x: x['quantity'] if x['side'] == 'BUY' else -x['quantity'], 
            axis=1
        )
        # ✅ แก้ 2: คำนวณ buy_volume แย่ง
        df['buy_vol'] = df.apply(
            lambda x: x['quantity'] if x['side'] == 'BUY' else 0, 
            axis=1
        )
        
        # Group by second
        candles = df.groupby('timestamp_s').agg({
            'price': ['first', 'max', 'min', 'last'],
            'quantity': 'sum',
            'signed_volume': 'sum',
            'buy_vol': 'sum',                          # เพิ่มใหม่
            'timestamp_ms': 'first'
        }).reset_index()
        
        candles.columns = ['timestamp_s', 'open', 'high', 'low', 'close', 
                          'total_volume', 'net_flow', 'buy_volume', 'timestamp_ms']  # เพิ่ม buy_volume
        candles['trade_count'] = df.groupby('timestamp_s').size().values
        
        return candles
    
    def _check_pending_orders(self, current_price, current_ts):
        """เช็ค pending orders"""
        for order in self.pending_orders[:]:
            if current_price <= order['limit_price']:
                self.active_orders.append({
                    'entry': order['limit_price'],
                    'quantity': order['quantity'],
                    'take_profit': order['take_profit'],
                    'stop_loss': order['stop_loss'],
                    'exit_ts': current_ts + self.params['HOLDING_TIME'],
                    'predicted_change': order['predicted_change'],  # แทน confidence
                    'entry_time': current_ts
                })
                self.pending_orders.remove(order)
            
            elif current_ts >= order['timeout_ts']:
                self.stats['unfilled'] += 1
                self.pending_orders.remove(order)
    
    def _check_active_orders(self, current_price, current_ts):
        """เช็ค active orders"""
        for order in self.active_orders[:]:
            exit_reason = None
            
            if current_price >= order['take_profit']:
                exit_reason = 'TP'
            elif current_price <= order['stop_loss']:
                exit_reason = 'SL'
            elif current_ts >= order['exit_ts']:
                exit_reason = 'TIME'
            
            if exit_reason:
                profit = (current_price - order['entry']) * order['quantity']
                self.total_pnl += profit
                
                if profit > 0: self.stats['win'] += 1
                elif profit < 0: self.stats['loss'] += 1
                else: self.stats['breakeven'] += 1
                
                dt = datetime.fromtimestamp(current_ts)
                self.hourly_pnl[dt.hour].append(profit)
                self.daily_pnl[dt.strftime('%A')].append(profit)
                
                self.closed_trades.append({
                    'entry': order['entry'],
                    'exit': current_price,
                    'pnl': profit,
                    'reason': exit_reason,
                    'predicted_change': order['predicted_change'],
                    'entry_time': order['entry_time'],
                    'exit_time': current_ts,
                    'hour': dt.hour,
                    'day': dt.strftime('%A')
                })
                
                self.active_orders.remove(order)
    
    def _predict_and_trade(self, buffer, current_price, current_ts):
        """ใช้ AI ทำนายและวาง order — ✅ แก้ 3: เปลี่ยน feature + logic หมด"""
        df = pd.DataFrame(list(buffer))
        
        # ===== Feature Engineering (ต้องตรงกับ training เนะ) =====
        # 1. volume (= total_volume)
        volume = df['total_volume'].iloc[-1]
        # 2. net_flow
        net_flow = df['net_flow'].iloc[-1]
        # 3. trade_count
        trade_count = df['trade_count'].iloc[-1]
        # 4. net_flow_ma5
        net_flow_ma5 = df['net_flow'].rolling(5).mean().iloc[-1]
        # 5. net_flow_ma15
        net_flow_ma15 = df['net_flow'].rolling(15).mean().iloc[-1]
        # 6. volume_ma5
        volume_ma5 = df['total_volume'].rolling(5).mean().iloc[-1]
        # 7. volume_ma15 ← เพิ่มใหม่
        volume_ma15 = df['total_volume'].rolling(15).mean().iloc[-1]
        # 8. momentum_5 ← เพิ่มใหม่ (price change over 5 periods)
        momentum_5 = (df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5] * 100 if len(df) >= 5 else 0.0
        # 9. volatility_5 ← เพิ่มใหม่ (std of close over 5 periods)
        volatility_5 = df['close'].rolling(5).std().iloc[-1]
        # 10. dist_ma5 ← เพิ่มใหม่ (price - ma5)
        dist_ma5 = df['close'].iloc[-1] - df['close'].rolling(5).mean().iloc[-1]
        # 11. buy_volume_ratio ← เพิ่มใหม่
        total_vol = df['total_volume'].iloc[-1]
        buy_vol = df['buy_volume'].iloc[-1]
        buy_volume_ratio = buy_vol / total_vol if total_vol > 0 else 0.5
        # 12. rsi
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rsi = (100 - (100 / (1 + (gain / (loss + 1e-10))))).iloc[-1]
        
        # ===== Feature dict (order ต้องตรงกับ model.feature_name() เนะ) =====
        # Model order: volume, volume_ma5, volume_ma15, net_flow, net_flow_ma5,
        #              net_flow_ma15, buy_volume_ratio, dist_ma5, volatility_5,
        #              rsi, momentum_5, trade_count
        feat = {
            'volume': volume,
            'volume_ma5': volume_ma5,
            'volume_ma15': volume_ma15,
            'net_flow': net_flow,
            'net_flow_ma5': net_flow_ma5,
            'net_flow_ma15': net_flow_ma15,
            'buy_volume_ratio': buy_volume_ratio,
            'dist_ma5': dist_ma5,
            'volatility_5': volatility_5,
            'rsi': rsi,
            'momentum_5': momentum_5,
            'trade_count': trade_count
        }
        
        # ===== AI Predict (regression → คืน predicted % change) =====
        predicted_change_pct = self.model.predict(pd.DataFrame([feat]))[0]
        
        # ===== Buy Logic (เทียบกับ PREDICTION_THRESHOLD แทน confidence) =====
        if predicted_change_pct > self.params['PREDICTION_THRESHOLD']:
            limit_price = current_price * (1 - self.params['MAKER_BUY_OFFSET_PCT'])
            qty = CAPITAL_PER_TRADE / limit_price
            
            # TP คำนวณจาก prediction เอง (dynamic TP)
            # predicted_change_pct เป็น % เยย ต้อง / 100 ก่อน
            tp_pct = (predicted_change_pct / 100) * self.params['TP_RATIO']
            tp = limit_price * (1 + tp_pct)
            
            # SL คงที่จาก parameter
            sl = limit_price * (1 - self.params['STOP_LOSS_PCT'])
            
            self.pending_orders.append({
                'limit_price': limit_price,
                'quantity': qty,
                'take_profit': tp,
                'stop_loss': sl,
                'timeout_ts': current_ts + self.params['MAKER_ORDER_TIMEOUT'],
                'predicted_change': predicted_change_pct  # บันทึก prediction เพื่อ tracking
            })
    
    def _calculate_metrics(self):
        """คำนวณ metrics"""
        total_trades = self.stats['win'] + self.stats['loss'] + self.stats['breakeven']
        
        if total_trades == 0:
            return {
                'total_pnl': 0,
                'win_rate': 0,
                'sharpe_ratio': -999,
                'max_drawdown': 0,
                'total_trades': 0,
                'hourly_best': None,
                'daily_best': None
            }
        
        win_rate = self.stats['win'] / total_trades * 100
        
        # Sharpe Ratio
        equity_curve = np.array(self.equity_curve)
        returns = np.diff(equity_curve)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252)
        else:
            sharpe_ratio = -999
        
        # Max Drawdown
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / (peak + 1)
        max_drawdown = np.min(drawdown) * 100
        
        # Time-based Analysis
        hourly_avg = {hour: np.mean(pnls) for hour, pnls in self.hourly_pnl.items()}
        daily_avg = {day: np.mean(pnls) for day, pnls in self.daily_pnl.items()}
        
        best_hour = max(hourly_avg, key=hourly_avg.get) if hourly_avg else None
        best_day = max(daily_avg, key=daily_avg.get) if daily_avg else None
        
        return {
            'total_pnl': self.total_pnl,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': total_trades,
            'stats': self.stats,
            'hourly_pnl': dict(self.hourly_pnl),
            'daily_pnl': dict(self.daily_pnl),
            'hourly_best': best_hour,
            'daily_best': best_day,
            'closed_trades': self.closed_trades
        }

# ==========================================
# OPTUNA OPTIMIZATION
# ==========================================
def objective(trial):
    # สุ่มค่า parameters (ใช้ key ใหม่ แทน CONFIDENCE_THRESHOLD + PROFIT_TARGET_PCT)
    params = {
        'PREDICTION_THRESHOLD': trial.suggest_float(
            'PREDICTION_THRESHOLD',
            PARAMETER_RANGES['PREDICTION_THRESHOLD'][0],
            PARAMETER_RANGES['PREDICTION_THRESHOLD'][1]
        ),
        'TP_RATIO': trial.suggest_float(
            'TP_RATIO',
            PARAMETER_RANGES['TP_RATIO'][0],
            PARAMETER_RANGES['TP_RATIO'][1]
        ),
        'HOLDING_TIME': trial.suggest_int(
            'HOLDING_TIME',
            PARAMETER_RANGES['HOLDING_TIME'][0],
            PARAMETER_RANGES['HOLDING_TIME'][1]
        ),
        'STOP_LOSS_PCT': trial.suggest_float(
            'STOP_LOSS_PCT',
            PARAMETER_RANGES['STOP_LOSS_PCT'][0],
            PARAMETER_RANGES['STOP_LOSS_PCT'][1]
        ),
        'MAKER_BUY_OFFSET_PCT': trial.suggest_float(
            'MAKER_BUY_OFFSET_PCT',
            PARAMETER_RANGES['MAKER_BUY_OFFSET_PCT'][0],
            PARAMETER_RANGES['MAKER_BUY_OFFSET_PCT'][1]
        ),
        'MAKER_ORDER_TIMEOUT': trial.suggest_int(
            'MAKER_ORDER_TIMEOUT',
            PARAMETER_RANGES['MAKER_ORDER_TIMEOUT'][0],
            PARAMETER_RANGES['MAKER_ORDER_TIMEOUT'][1]
        )
    }
    
    backtest = BacktestEngine(df_trades, model, params)
    results = backtest.run()
    
    trial.set_user_attr('total_pnl', results['total_pnl'])
    trial.set_user_attr('win_rate', results['win_rate'])
    trial.set_user_attr('max_drawdown', results['max_drawdown'])
    trial.set_user_attr('total_trades', results['total_trades'])
    trial.set_user_attr('hourly_best', results['hourly_best'])
    trial.set_user_attr('daily_best', results['daily_best'])
    
    if results['total_trades'] < 10:
        return -999
    
    score = (
        0.40 * results['sharpe_ratio'] +
        0.30 * (results['total_pnl'] / 1000) +
        0.20 * (results['win_rate'] / 100) +
        0.10 * (1 - abs(results['max_drawdown']) / 100)
    )
    
    return score

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 STARTING PARAMETER OPTIMIZATION (Regression Model)")
    print("=" * 60)
    
    print("\n📁 Loading CSV Data...")
    df_trades = pd.read_csv(CSV_FILE)
    print(f"✅ Loaded {len(df_trades):,} trades")
    print(f"📅 Date Range: {df_trades['readable_time'].iloc[0]} to {df_trades['readable_time'].iloc[-1]}")
    
    print(f"\n🔍 Starting Optimization ({N_TRIALS} trials)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    # แสดงผลลัพธ์
    print("\n" + "=" * 60)
    print("✅ OPTIMIZATION COMPLETE!")
    print("=" * 60)
    
    best_trial = study.best_trial
    print(f"\n🏆 BEST PARAMETERS:")
    print(f"{'─' * 60}")
    for key, value in best_trial.params.items():
        print(f"  {key:25s}: {value}")
    
    print(f"\n📊 PERFORMANCE METRICS:")
    print(f"{'─' * 60}")
    print(f"  Total PNL           : ${best_trial.user_attrs['total_pnl']:.2f}")
    print(f"  Win Rate            : {best_trial.user_attrs['win_rate']:.2f}%")
    print(f"  Score               : {best_trial.value:.4f}")
    print(f"  Max Drawdown        : {best_trial.user_attrs['max_drawdown']:.2f}%")
    print(f"  Total Trades        : {best_trial.user_attrs['total_trades']}")
    print(f"  Best Hour (UTC)     : {best_trial.user_attrs['hourly_best']}:00")
    print(f"  Best Day            : {best_trial.user_attrs['daily_best']}")
    
    # บันทึกผลลัพธ์
    print("\n💾 Saving Results...")
    
    with open('best_parameters.json', 'w') as f:
        json.dump(best_trial.params, f, indent=4)
    print("  ✅ best_parameters.json")
    
    report = {
        'best_params': best_trial.params,
        'metrics': {
            'total_pnl': best_trial.user_attrs['total_pnl'],
            'win_rate': best_trial.user_attrs['win_rate'],
            'score': best_trial.value,
            'max_drawdown': best_trial.user_attrs['max_drawdown'],
            'total_trades': best_trial.user_attrs['total_trades'],
            'best_hour': best_trial.user_attrs['hourly_best'],
            'best_day': best_trial.user_attrs['daily_best']
        },
        'all_trials': [
            {
                'number': t.number,
                'params': t.params,
                'score': t.value,
                'total_pnl': t.user_attrs.get('total_pnl', 0),
                'win_rate': t.user_attrs.get('win_rate', 0)
            }
            for t in study.trials
        ]
    }
    
    with open('optimization_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    print("  ✅ optimization_report.json")
    
    # Top 10 Trials
    print(f"\n📈 TOP 10 TRIALS:")
    print(f"{'─' * 110}")
    print(f"{'Trial':<8} {'Score':<10} {'PNL':<12} {'Win%':<8} {'Trades':<8} {'Pred>':<8} {'TP_R':<7} {'Hold(s)':<10}")
    print(f"{'─' * 110}")
    
    sorted_trials = sorted(study.trials, key=lambda t: t.value if t.value else -999, reverse=True)[:10]
    for t in sorted_trials:
        print(f"{t.number:<8} {t.value:<10.4f} ${t.user_attrs.get('total_pnl', 0):<11.2f} "
              f"{t.user_attrs.get('win_rate', 0):<7.1f} {t.user_attrs.get('total_trades', 0):<8} "
              f"{t.params.get('PREDICTION_THRESHOLD', 0):<7.3f} "
              f"{t.params.get('TP_RATIO', 0):<6.2f} "
              f"{t.params.get('HOLDING_TIME', 0):<10}")
    
    print("\n" + "=" * 60)
    print("🎉 DONE! ใช้ค่าที่ได้ในไฟล์ best_parameters.json ได้เลย")
    print("=" * 60)