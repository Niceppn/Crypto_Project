"""
Backtest with Optimized Parameters
Test all 3 models with optimal parameters from optimization
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features,
    get_feature_columns
)
from model_3_hybrid import HybridMakerStrategy

print("=" * 100)
print("🎯 BACKTEST WITH OPTIMIZED PARAMETERS")
print("=" * 100)

# ==========================================
# LOAD OPTIMAL PARAMETERS
# ==========================================
print("\n📂 Loading optimal parameters...")

with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/best_parameters.json', 'r') as f:
    best_params = json.load(f)

OPTIMAL_PARAMS = best_params['model1']

print("✅ Optimal parameters loaded:")
print(f"  Confidence threshold: {OPTIMAL_PARAMS['confidence_threshold']}")
print(f"  Maker offset:         {OPTIMAL_PARAMS['maker_offset_pct']} ({OPTIMAL_PARAMS['maker_offset_pct']*100:.2f}%)")
print(f"  Timeout:              {OPTIMAL_PARAMS['maker_timeout']}s")
print(f"  Take profit:          {OPTIMAL_PARAMS['take_profit_pct']} ({OPTIMAL_PARAMS['take_profit_pct']*100:.2f}%)")
print(f"  Stop loss:            {OPTIMAL_PARAMS['stop_loss_pct']} ({OPTIMAL_PARAMS['stop_loss_pct']*100:.2f}%)")
print(f"  Hold time:            {OPTIMAL_PARAMS['hold_time']}s")

# ==========================================
# CONFIGURATION
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
INITIAL_CAPITAL = 1000.0
POSITION_SIZE = 21.85
LEVERAGE = 20
COMMISSION_RATE = 0.0
USE_ALL_DATA = False
TEST_ROWS = 500000

# ==========================================
# LOAD MODELS
# ==========================================
print("\n📂 Loading models...")

model1_classifier = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_multiclass.txt')
model2_buy = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_buy_regressor.txt')
model2_sell = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_sell_regressor.txt')
model3_strategy = HybridMakerStrategy(model1_classifier, model2_buy, model2_sell)

print("✅ All models loaded")

# ==========================================
# LOAD DATA
# ==========================================
print("\n📊 Loading test data...")

df = load_and_prepare_data(RAW_FILE, nrows=TEST_ROWS if not USE_ALL_DATA else None)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)
df_features = df_features.dropna()

split_idx = int(len(df_features) * 0.8)
df_test = df_features.iloc[split_idx:].copy()

print(f"✅ Test period: {df_test.index[0]} → {df_test.index[-1]}")
print(f"   Total candles: {len(df_test):,}")

feature_cols = get_feature_columns()
X_test = df_test[feature_cols]

# ==========================================
# BACKTEST ENGINE (Same as before)
# ==========================================
class BacktestEngine:
    def __init__(self, df, initial_capital, position_size, leverage, commission_rate):
        self.df = df
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.leverage = leverage
        self.commission_rate = commission_rate
        self.capital = initial_capital
        self.trades = []

    def place_buy_maker_order(self, idx, signal_time, entry_price, tp_pct, sl_pct, timeout, hold_time):
        filled = False
        for i in range(1, timeout + 1):
            if idx + i >= len(self.df):
                return {'status': 'unfilled', 'type': 'BUY'}
            if self.df.iloc[idx + i]['low'] <= entry_price:
                filled = True
                fill_idx = idx + i
                break

        if not filled:
            return {'status': 'unfilled', 'type': 'BUY'}

        tp_price = entry_price * (1 + tp_pct)
        sl_price = entry_price * (1 - sl_pct)
        exit_price = None
        exit_reason = 'timeout'

        for i in range(1, hold_time + 1):
            check_idx = fill_idx + i
            if check_idx >= len(self.df):
                break
            candle = self.df.iloc[check_idx]
            if candle['high'] >= tp_price:
                exit_price = tp_price
                exit_reason = 'take_profit'
                break
            if candle['low'] <= sl_price:
                exit_price = sl_price
                exit_reason = 'stop_loss'
                break

        if exit_price is None:
            exit_idx = min(fill_idx + hold_time, len(self.df) - 1)
            exit_price = self.df.iloc[exit_idx]['close']

        pnl_pct = (exit_price - entry_price) / entry_price
        pnl_usdt = self.position_size * self.leverage * pnl_pct
        pnl_usdt -= self.position_size * self.commission_rate * 2

        self.capital += pnl_usdt
        return {
            'status': 'completed',
            'type': 'BUY',
            'pnl_usdt': pnl_usdt,
            'exit_reason': exit_reason,
            'win': pnl_usdt > 0
        }

    def place_sell_maker_order(self, idx, signal_time, entry_price, tp_pct, sl_pct, timeout, hold_time):
        filled = False
        for i in range(1, timeout + 1):
            if idx + i >= len(self.df):
                return {'status': 'unfilled', 'type': 'SELL'}
            if self.df.iloc[idx + i]['high'] >= entry_price:
                filled = True
                fill_idx = idx + i
                break

        if not filled:
            return {'status': 'unfilled', 'type': 'SELL'}

        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)
        exit_price = None
        exit_reason = 'timeout'

        for i in range(1, hold_time + 1):
            check_idx = fill_idx + i
            if check_idx >= len(self.df):
                break
            candle = self.df.iloc[check_idx]
            if candle['low'] <= tp_price:
                exit_price = tp_price
                exit_reason = 'take_profit'
                break
            if candle['high'] >= sl_price:
                exit_price = sl_price
                exit_reason = 'stop_loss'
                break

        if exit_price is None:
            exit_idx = min(fill_idx + hold_time, len(self.df) - 1)
            exit_price = self.df.iloc[exit_idx]['close']

        pnl_pct = (entry_price - exit_price) / entry_price
        pnl_usdt = self.position_size * self.leverage * pnl_pct
        pnl_usdt -= self.position_size * self.commission_rate * 2

        self.capital += pnl_usdt
        return {
            'status': 'completed',
            'type': 'SELL',
            'pnl_usdt': pnl_usdt,
            'exit_reason': exit_reason,
            'win': pnl_usdt > 0
        }

    def calculate_metrics(self):
        if not self.trades:
            return {'total_trades': 0}

        completed_trades = [t for t in self.trades if t['status'] == 'completed']
        if not completed_trades:
            return {'total_trades': len(self.trades), 'completed_trades': 0}

        wins = [t for t in completed_trades if t['win']]
        losses = [t for t in completed_trades if not t['win']]

        total_pnl = sum(t['pnl_usdt'] for t in completed_trades)
        win_rate = len(wins) / len(completed_trades) if completed_trades else 0

        profit_factor = abs(sum(t['pnl_usdt'] for t in wins) / sum(t['pnl_usdt'] for t in losses)) if losses and sum(t['pnl_usdt'] for t in losses) != 0 else 0

        returns = [t['pnl_usdt'] / self.initial_capital for t in completed_trades]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        equity = self.initial_capital
        peak = equity
        max_dd = 0
        for trade in completed_trades:
            equity += trade['pnl_usdt']
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return {
            'total_trades': len(self.trades),
            'completed_trades': len(completed_trades),
            'unfilled_orders': len([t for t in self.trades if t['status'] == 'unfilled']),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'profit_factor': profit_factor,
            'max_drawdown': max_dd,
            'sharpe_ratio': sharpe,
            'final_capital': self.capital,
            'return_pct': (self.capital - self.initial_capital) / self.initial_capital * 100
        }

# ==========================================
# MODEL 1: MULTI-CLASS (Optimized)
# ==========================================
print("\n" + "=" * 100)
print("🔷 MODEL 1: MULTI-CLASS CLASSIFICATION (Optimized)")
print("=" * 100)

bt1 = BacktestEngine(df_test, INITIAL_CAPITAL, POSITION_SIZE, LEVERAGE, COMMISSION_RATE)

for idx in range(len(df_test) - 100):
    features = X_test.iloc[idx].values.reshape(1, -1)
    probs = model1_classifier.predict(features)[0]
    signal = np.argmax(probs)
    confidence = probs[signal]

    if confidence < OPTIMAL_PARAMS['confidence_threshold']:
        continue

    current_price = df_test.iloc[idx]['close']

    if signal == 1:  # BUY
        entry_price = current_price * (1 - OPTIMAL_PARAMS['maker_offset_pct'])
        trade = bt1.place_buy_maker_order(
            idx, df_test.index[idx], entry_price,
            OPTIMAL_PARAMS['take_profit_pct'],
            OPTIMAL_PARAMS['stop_loss_pct'],
            OPTIMAL_PARAMS['maker_timeout'],
            OPTIMAL_PARAMS['hold_time']
        )
        if trade:
            bt1.trades.append(trade)

    elif signal == 2:  # SELL
        entry_price = current_price * (1 + OPTIMAL_PARAMS['maker_offset_pct'])
        trade = bt1.place_sell_maker_order(
            idx, df_test.index[idx], entry_price,
            OPTIMAL_PARAMS['take_profit_pct'],
            OPTIMAL_PARAMS['stop_loss_pct'],
            OPTIMAL_PARAMS['maker_timeout'],
            OPTIMAL_PARAMS['hold_time']
        )
        if trade:
            bt1.trades.append(trade)

metrics1 = bt1.calculate_metrics()

print("\n📊 Results (Optimized):")
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
# MODEL 2: DUAL REGRESSION (Optimized)
# ==========================================
print("\n" + "=" * 100)
print("🔶 MODEL 2: DUAL REGRESSION (Optimized)")
print("=" * 100)

bt2 = BacktestEngine(df_test, INITIAL_CAPITAL, POSITION_SIZE, LEVERAGE, COMMISSION_RATE)

for idx in range(len(df_test) - 100):
    features = X_test.iloc[idx].values.reshape(1, -1)
    upside = model2_buy.predict(features)[0]
    downside = model2_sell.predict(features)[0]

    current_price = df_test.iloc[idx]['close']

    # Use similar threshold logic but adjusted for regression
    if upside >= 0.20 and upside > downside * 1.2:
        entry_price = current_price * (1 - OPTIMAL_PARAMS['maker_offset_pct'])
        trade = bt2.place_buy_maker_order(
            idx, df_test.index[idx], entry_price,
            OPTIMAL_PARAMS['take_profit_pct'],
            OPTIMAL_PARAMS['stop_loss_pct'],
            OPTIMAL_PARAMS['maker_timeout'],
            OPTIMAL_PARAMS['hold_time']
        )
        if trade:
            bt2.trades.append(trade)

    elif downside >= 0.20 and downside > upside * 1.2:
        entry_price = current_price * (1 + OPTIMAL_PARAMS['maker_offset_pct'])
        trade = bt2.place_sell_maker_order(
            idx, df_test.index[idx], entry_price,
            OPTIMAL_PARAMS['take_profit_pct'],
            OPTIMAL_PARAMS['stop_loss_pct'],
            OPTIMAL_PARAMS['maker_timeout'],
            OPTIMAL_PARAMS['hold_time']
        )
        if trade:
            bt2.trades.append(trade)

metrics2 = bt2.calculate_metrics()

print("\n📊 Results (Optimized):")
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
# MODEL 3: HYBRID (Optimized)
# ==========================================
print("\n" + "=" * 100)
print("💎 MODEL 3: HYBRID APPROACH (Optimized)")
print("=" * 100)

# Update hybrid strategy parameters
model3_strategy.classification_threshold = OPTIMAL_PARAMS['confidence_threshold']

bt3 = BacktestEngine(df_test, INITIAL_CAPITAL, POSITION_SIZE, LEVERAGE, COMMISSION_RATE)

for idx in range(len(df_test) - 100):
    features = X_test.iloc[idx].values.reshape(1, -1)
    signal, confidence, metadata = model3_strategy.predict(features)

    if signal == 0 or confidence < OPTIMAL_PARAMS['confidence_threshold']:
        continue

    current_price = df_test.iloc[idx]['close']

    if signal == 1:  # BUY
        entry_price = current_price * (1 - OPTIMAL_PARAMS['maker_offset_pct'])
        trade = bt3.place_buy_maker_order(
            idx, df_test.index[idx], entry_price,
            OPTIMAL_PARAMS['take_profit_pct'],
            OPTIMAL_PARAMS['stop_loss_pct'],
            OPTIMAL_PARAMS['maker_timeout'],
            OPTIMAL_PARAMS['hold_time']
        )
        if trade:
            bt3.trades.append(trade)

    elif signal == 2:  # SELL
        entry_price = current_price * (1 + OPTIMAL_PARAMS['maker_offset_pct'])
        trade = bt3.place_sell_maker_order(
            idx, df_test.index[idx], entry_price,
            OPTIMAL_PARAMS['take_profit_pct'],
            OPTIMAL_PARAMS['stop_loss_pct'],
            OPTIMAL_PARAMS['maker_timeout'],
            OPTIMAL_PARAMS['hold_time']
        )
        if trade:
            bt3.trades.append(trade)

metrics3 = bt3.calculate_metrics()

print("\n📊 Results (Optimized):")
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
print("🏆 FINAL COMPARISON (Before vs After Optimization)")
print("=" * 100)

comparison_df = pd.DataFrame({
    'Model': ['Multi-Class', 'Dual Regression', 'Hybrid'],
    'Total Trades': [
        metrics1.get('total_trades', 0),
        metrics2.get('total_trades', 0),
        metrics3.get('total_trades', 0)
    ],
    'Completed': [
        metrics1.get('completed_trades', 0),
        metrics2.get('completed_trades', 0),
        metrics3.get('completed_trades', 0)
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
    ]
})

print("\n" + comparison_df.to_string(index=False))

best_by_return = comparison_df.loc[comparison_df['Total PNL $'].idxmax(), 'Model']
best_by_sharpe = comparison_df.loc[comparison_df['Sharpe'].idxmax(), 'Model']
best_by_winrate = comparison_df.loc[comparison_df['Win Rate %'].idxmax(), 'Model']

print(f"\n🏅 Best by Return:    {best_by_return}")
print(f"🏅 Best by Sharpe:    {best_by_sharpe}")
print(f"🏅 Best by Win Rate:  {best_by_winrate}")

# Save comparison
comparison_df.to_csv('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/backtest_optimized_comparison.csv', index=False)
print(f"\n✅ Comparison saved to: backtest_optimized_comparison.csv")

print("\n" + "=" * 100)
print("🎉 OPTIMIZED BACKTEST COMPLETE!")
print("=" * 100)
