"""
Backtest Comparison: Compare all 3 models
- Model 1: Multi-Class Classification
- Model 2: Dual Regression
- Model 3: Hybrid Approach
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import json
from datetime import datetime
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features,
    get_feature_columns
)

print("=" * 100)
print("🏁 BACKTEST COMPARISON: 3 MODELS")
print("=" * 100)

# ==========================================
# CONFIGURATION
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
INITIAL_CAPITAL = 1000.0  # USDT
POSITION_SIZE = 21.85  # USDT per trade
LEVERAGE = 20
COMMISSION_RATE = 0.0  # Assume maker fee = 0%

# Use subset for faster testing
USE_ALL_DATA = False
TEST_ROWS = 500000  # ~2-3 days of data

# ==========================================
# LOAD MODELS
# ==========================================
print("\n📂 Loading models...")

try:
    # Model 1: Multi-Class
    model1_classifier = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_multiclass.txt')

    # Model 2: Dual Regression
    model2_buy = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_buy_regressor.txt')
    model2_sell = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_sell_regressor.txt')

    # Model 3: Hybrid - Import class instead of unpickling
    import sys
    sys.path.insert(0, '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC')
    from model_3_hybrid import HybridMakerStrategy

    model3_strategy = HybridMakerStrategy(model1_classifier, model2_buy, model2_sell)

    print("✅ All models loaded successfully!")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("\n⚠️  Please train all models first:")
    print("   1. python model_1_multiclass.py")
    print("   2. python model_2_dual_regression.py")
    print("   3. python model_3_hybrid.py")
    exit(1)

# ==========================================
# LOAD AND PREPARE DATA
# ==========================================
print("\n📊 Loading test data...")

df = load_and_prepare_data(RAW_FILE, nrows=TEST_ROWS if not USE_ALL_DATA else None)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)
df_features = df_features.dropna()

# Use last 20% for backtesting
split_idx = int(len(df_features) * 0.8)
df_test = df_features.iloc[split_idx:].copy()

print(f"✅ Test period: {df_test.index[0]} → {df_test.index[-1]}")
print(f"   Total candles: {len(df_test):,}")

feature_cols = get_feature_columns()
X_test = df_test[feature_cols]

# ==========================================
# BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    """Backtest engine for maker strategy"""

    def __init__(self, df, initial_capital, position_size, leverage, commission_rate):
        self.df = df
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.leverage = leverage
        self.commission_rate = commission_rate

        self.capital = initial_capital
        self.trades = []
        self.equity_curve = []

    def place_buy_maker_order(self, idx, signal_time, entry_price, tp_pct=0.0015, sl_pct=0.0010,
                             timeout=10, hold_time=90):
        """Simulate BUY maker order"""
        # Check if order gets filled within timeout
        filled = False
        fill_time = None
        fill_price = entry_price

        for i in range(1, timeout + 1):
            if idx + i >= len(self.df):
                return None

            future_low = self.df.iloc[idx + i]['low']
            if future_low <= entry_price:
                filled = True
                fill_time = self.df.index[idx + i]
                fill_price = entry_price
                break

        if not filled:
            return {'status': 'unfilled', 'type': 'BUY'}

        # Order filled, now track position
        tp_price = fill_price * (1 + tp_pct)
        sl_price = fill_price * (1 - sl_pct)

        # Search for exit within hold_time
        exit_price = None
        exit_time = None
        exit_reason = 'timeout'

        for i in range(1, hold_time + 1):
            check_idx = idx + timeout + i
            if check_idx >= len(self.df):
                break

            candle = self.df.iloc[check_idx]

            # Check TP
            if candle['high'] >= tp_price:
                exit_price = tp_price
                exit_time = self.df.index[check_idx]
                exit_reason = 'take_profit'
                break

            # Check SL
            if candle['low'] <= sl_price:
                exit_price = sl_price
                exit_time = self.df.index[check_idx]
                exit_reason = 'stop_loss'
                break

        # If no exit, close at market
        if exit_price is None:
            exit_idx = min(idx + timeout + hold_time, len(self.df) - 1)
            exit_price = self.df.iloc[exit_idx]['close']
            exit_time = self.df.index[exit_idx]
            exit_reason = 'timeout'

        # Calculate PNL
        pnl_pct = (exit_price - fill_price) / fill_price
        pnl_usdt = self.position_size * self.leverage * pnl_pct
        pnl_usdt -= self.position_size * self.commission_rate * 2  # Entry + exit

        trade = {
            'status': 'completed',
            'type': 'BUY',
            'signal_time': signal_time,
            'fill_time': fill_time,
            'fill_price': fill_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct * 100,
            'pnl_usdt': pnl_usdt,
            'win': pnl_usdt > 0
        }

        self.capital += pnl_usdt
        return trade

    def place_sell_maker_order(self, idx, signal_time, entry_price, tp_pct=0.0015, sl_pct=0.0010,
                               timeout=10, hold_time=90):
        """Simulate SELL maker order"""
        # Check if order gets filled within timeout
        filled = False
        fill_time = None
        fill_price = entry_price

        for i in range(1, timeout + 1):
            if idx + i >= len(self.df):
                return None

            future_high = self.df.iloc[idx + i]['high']
            if future_high >= entry_price:
                filled = True
                fill_time = self.df.index[idx + i]
                fill_price = entry_price
                break

        if not filled:
            return {'status': 'unfilled', 'type': 'SELL'}

        # Order filled, now track position
        tp_price = fill_price * (1 - tp_pct)
        sl_price = fill_price * (1 + sl_pct)

        # Search for exit within hold_time
        exit_price = None
        exit_time = None
        exit_reason = 'timeout'

        for i in range(1, hold_time + 1):
            check_idx = idx + timeout + i
            if check_idx >= len(self.df):
                break

            candle = self.df.iloc[check_idx]

            # Check TP
            if candle['low'] <= tp_price:
                exit_price = tp_price
                exit_time = self.df.index[check_idx]
                exit_reason = 'take_profit'
                break

            # Check SL
            if candle['high'] >= sl_price:
                exit_price = sl_price
                exit_time = self.df.index[check_idx]
                exit_reason = 'stop_loss'
                break

        # If no exit, close at market
        if exit_price is None:
            exit_idx = min(idx + timeout + hold_time, len(self.df) - 1)
            exit_price = self.df.iloc[exit_idx]['close']
            exit_time = self.df.index[exit_idx]
            exit_reason = 'timeout'

        # Calculate PNL (short position)
        pnl_pct = (fill_price - exit_price) / fill_price
        pnl_usdt = self.position_size * self.leverage * pnl_pct
        pnl_usdt -= self.position_size * self.commission_rate * 2

        trade = {
            'status': 'completed',
            'type': 'SELL',
            'signal_time': signal_time,
            'fill_time': fill_time,
            'fill_price': fill_price,
            'exit_time': exit_time,
            'exit_price': exit_price,
            'exit_reason': exit_reason,
            'pnl_pct': pnl_pct * 100,
            'pnl_usdt': pnl_usdt,
            'win': pnl_usdt > 0
        }

        self.capital += pnl_usdt
        return trade

    def calculate_metrics(self):
        """Calculate performance metrics"""
        if not self.trades:
            return {}

        completed_trades = [t for t in self.trades if t['status'] == 'completed']

        if not completed_trades:
            return {'total_trades': 0}

        wins = [t for t in completed_trades if t['win']]
        losses = [t for t in completed_trades if not t['win']]

        total_pnl = sum(t['pnl_usdt'] for t in completed_trades)
        win_rate = len(wins) / len(completed_trades) if completed_trades else 0

        avg_win = np.mean([t['pnl_usdt'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['pnl_usdt'] for t in losses]) if losses else 0

        profit_factor = abs(sum(t['pnl_usdt'] for t in wins) / sum(t['pnl_usdt'] for t in losses)) if losses and sum(t['pnl_usdt'] for t in losses) != 0 else 0

        # Calculate max drawdown
        equity = self.initial_capital
        peak = equity
        max_dd = 0

        for trade in completed_trades:
            equity += trade['pnl_usdt']
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified)
        returns = [t['pnl_usdt'] / self.initial_capital for t in completed_trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        return {
            'total_trades': len(self.trades),
            'completed_trades': len(completed_trades),
            'unfilled_orders': len([t for t in self.trades if t['status'] == 'unfilled']),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'final_capital': self.capital,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100
        }

# ==========================================
# MODEL 1: MULTI-CLASS CLASSIFICATION
# ==========================================
print("\n" + "=" * 100)
print("🔷 MODEL 1: MULTI-CLASS CLASSIFICATION")
print("=" * 100)

bt1 = BacktestEngine(df_test, INITIAL_CAPITAL, POSITION_SIZE, LEVERAGE, COMMISSION_RATE)

print("Running backtest...")
for idx in range(len(df_test) - 100):
    features = X_test.iloc[idx].values.reshape(1, -1)
    probs = model1_classifier.predict(features)[0]
    signal = np.argmax(probs)
    confidence = probs[signal]

    if confidence < 0.70:
        continue

    current_price = df_test.iloc[idx]['close']
    signal_time = df_test.index[idx]

    if signal == 1:  # BUY
        entry_price = current_price * 0.9995
        trade = bt1.place_buy_maker_order(idx, signal_time, entry_price)
        if trade:
            bt1.trades.append(trade)

    elif signal == 2:  # SELL
        entry_price = current_price * 1.0005
        trade = bt1.place_sell_maker_order(idx, signal_time, entry_price)
        if trade:
            bt1.trades.append(trade)

metrics1 = bt1.calculate_metrics()

print("\n📊 Results:")
print(f"  Total trades:      {metrics1.get('total_trades', 0):5d}")
print(f"  Completed:         {metrics1.get('completed_trades', 0):5d}")
print(f"  Unfilled:          {metrics1.get('unfilled_orders', 0):5d}")
print(f"  Win rate:          {metrics1.get('win_rate', 0)*100:5.2f}%")
print(f"  Total PNL:         ${metrics1.get('total_pnl', 0):8.2f}")
print(f"  Return:            {metrics1.get('return_pct', 0):5.2f}%")
print(f"  Profit factor:     {metrics1.get('profit_factor', 0):5.2f}")
print(f"  Max drawdown:      {metrics1.get('max_drawdown', 0)*100:5.2f}%")
print(f"  Sharpe ratio:      {metrics1.get('sharpe_ratio', 0):5.2f}")

# ==========================================
# MODEL 2: DUAL REGRESSION
# ==========================================
print("\n" + "=" * 100)
print("🔶 MODEL 2: DUAL REGRESSION")
print("=" * 100)

bt2 = BacktestEngine(df_test, INITIAL_CAPITAL, POSITION_SIZE, LEVERAGE, COMMISSION_RATE)

print("Running backtest...")
for idx in range(len(df_test) - 100):
    features = X_test.iloc[idx].values.reshape(1, -1)
    upside = model2_buy.predict(features)[0]
    downside = model2_sell.predict(features)[0]

    current_price = df_test.iloc[idx]['close']
    signal_time = df_test.index[idx]

    # Decision logic
    if upside >= 0.20 and upside > downside * 1.2:
        entry_price = current_price * 0.9995
        trade = bt2.place_buy_maker_order(idx, signal_time, entry_price)
        if trade:
            bt2.trades.append(trade)

    elif downside >= 0.20 and downside > upside * 1.2:
        entry_price = current_price * 1.0005
        trade = bt2.place_sell_maker_order(idx, signal_time, entry_price)
        if trade:
            bt2.trades.append(trade)

metrics2 = bt2.calculate_metrics()

print("\n📊 Results:")
print(f"  Total trades:      {metrics2.get('total_trades', 0):5d}")
print(f"  Completed:         {metrics2.get('completed_trades', 0):5d}")
print(f"  Unfilled:          {metrics2.get('unfilled_orders', 0):5d}")
print(f"  Win rate:          {metrics2.get('win_rate', 0)*100:5.2f}%")
print(f"  Total PNL:         ${metrics2.get('total_pnl', 0):8.2f}")
print(f"  Return:            {metrics2.get('return_pct', 0):5.2f}%")
print(f"  Profit factor:     {metrics2.get('profit_factor', 0):5.2f}")
print(f"  Max drawdown:      {metrics2.get('max_drawdown', 0)*100:5.2f}%")
print(f"  Sharpe ratio:      {metrics2.get('sharpe_ratio', 0):5.2f}")

# ==========================================
# MODEL 3: HYBRID
# ==========================================
print("\n" + "=" * 100)
print("🔷 MODEL 3: HYBRID APPROACH")
print("=" * 100)

bt3 = BacktestEngine(df_test, INITIAL_CAPITAL, POSITION_SIZE, LEVERAGE, COMMISSION_RATE)

print("Running backtest...")
for idx in range(len(df_test) - 100):
    features = X_test.iloc[idx].values.reshape(1, -1)
    signal, confidence, metadata = model3_strategy.predict(features)

    if signal == 0 or confidence < 0.70:
        continue

    current_price = df_test.iloc[idx]['close']
    signal_time = df_test.index[idx]

    # Get dynamic parameters
    params = model3_strategy.get_dynamic_parameters(features, signal, confidence)

    if signal == 1:  # BUY
        entry_price = current_price * (1 - params['entry_offset_pct'])
        trade = bt3.place_buy_maker_order(
            idx, signal_time, entry_price,
            tp_pct=params['take_profit_pct'],
            sl_pct=params['stop_loss_pct']
        )
        if trade:
            bt3.trades.append(trade)

    elif signal == 2:  # SELL
        entry_price = current_price * (1 + params['entry_offset_pct'])
        trade = bt3.place_sell_maker_order(
            idx, signal_time, entry_price,
            tp_pct=params['take_profit_pct'],
            sl_pct=params['stop_loss_pct']
        )
        if trade:
            bt3.trades.append(trade)

metrics3 = bt3.calculate_metrics()

print("\n📊 Results:")
print(f"  Total trades:      {metrics3.get('total_trades', 0):5d}")
print(f"  Completed:         {metrics3.get('completed_trades', 0):5d}")
print(f"  Unfilled:          {metrics3.get('unfilled_orders', 0):5d}")
print(f"  Win rate:          {metrics3.get('win_rate', 0)*100:5.2f}%")
print(f"  Total PNL:         ${metrics3.get('total_pnl', 0):8.2f}")
print(f"  Return:            {metrics3.get('return_pct', 0):5.2f}%")
print(f"  Profit factor:     {metrics3.get('profit_factor', 0):5.2f}")
print(f"  Max drawdown:      {metrics3.get('max_drawdown', 0)*100:5.2f}%")
print(f"  Sharpe ratio:      {metrics3.get('sharpe_ratio', 0):5.2f}")

# ==========================================
# FINAL COMPARISON
# ==========================================
print("\n" + "=" * 100)
print("🏆 FINAL COMPARISON")
print("=" * 100)

comparison_df = pd.DataFrame({
    'Model': ['Multi-Class', 'Dual Regression', 'Hybrid'],
    'Total Trades': [
        metrics1.get('total_trades', 0),
        metrics2.get('total_trades', 0),
        metrics3.get('total_trades', 0)
    ],
    'Win Rate %': [
        metrics1.get('win_rate', 0) * 100,
        metrics2.get('win_rate', 0) * 100,
        metrics3.get('win_rate', 0) * 100
    ],
    'Total PNL $': [
        metrics1.get('total_pnl', 0),
        metrics2.get('total_pnl', 0),
        metrics3.get('total_pnl', 0)
    ],
    'Return %': [
        metrics1.get('return_pct', 0),
        metrics2.get('return_pct', 0),
        metrics3.get('return_pct', 0)
    ],
    'Profit Factor': [
        metrics1.get('profit_factor', 0),
        metrics2.get('profit_factor', 0),
        metrics3.get('profit_factor', 0)
    ],
    'Sharpe': [
        metrics1.get('sharpe_ratio', 0),
        metrics2.get('sharpe_ratio', 0),
        metrics3.get('sharpe_ratio', 0)
    ],
    'Max DD %': [
        metrics1.get('max_drawdown', 0) * 100,
        metrics2.get('max_drawdown', 0) * 100,
        metrics3.get('max_drawdown', 0) * 100
    ]
})

print("\n" + comparison_df.to_string(index=False))

# Find best model
best_by_return = comparison_df.loc[comparison_df['Total PNL $'].idxmax(), 'Model']
best_by_sharpe = comparison_df.loc[comparison_df['Sharpe'].idxmax(), 'Model']
best_by_winrate = comparison_df.loc[comparison_df['Win Rate %'].idxmax(), 'Model']

print(f"\n🏅 Best by Return:    {best_by_return}")
print(f"🏅 Best by Sharpe:    {best_by_sharpe}")
print(f"🏅 Best by Win Rate:  {best_by_winrate}")

# Save comparison
comparison_df.to_csv('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/backtest_comparison.csv', index=False)
print(f"\n✅ Comparison saved to: backtest_comparison.csv")

print("\n" + "=" * 100)
print("🎉 BACKTEST COMPARISON COMPLETE!")
print("=" * 100)
