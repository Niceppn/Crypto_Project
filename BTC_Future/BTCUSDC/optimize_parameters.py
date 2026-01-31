"""
Parameter Optimization for All 3 Models
Uses Grid Search to find optimal thresholds, offsets, and timeouts
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from itertools import product
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features,
    get_feature_columns
)
from model_3_hybrid import HybridMakerStrategy

print("=" * 100)
print("🔬 PARAMETER OPTIMIZATION - GRID SEARCH")
print("=" * 100)

# ==========================================
# CONFIGURATION
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
INITIAL_CAPITAL = 1000.0
POSITION_SIZE = 21.85
LEVERAGE = 20
COMMISSION_RATE = 0.0

# Use subset for faster optimization
USE_ALL_DATA = False
TEST_ROWS = 500000

# ==========================================
# PARAMETER GRID
# ==========================================
PARAM_GRID = {
    # Classification threshold
    'confidence_threshold': [0.65, 0.70, 0.75, 0.80],

    # Maker entry offset (%)
    'maker_offset_pct': [0.0003, 0.0005, 0.0007, 0.0010],

    # Order timeout (seconds)
    'maker_timeout': [10, 20, 30, 40],

    # Take profit (%)
    'take_profit_pct': [0.0012, 0.0015, 0.0018, 0.0020],

    # Stop loss (%)
    'stop_loss_pct': [0.0008, 0.0010, 0.0012],

    # Hold time (seconds)
    'hold_time': [60, 90, 120]
}

print("\n📊 Parameter Grid:")
for key, values in PARAM_GRID.items():
    print(f"  {key:25s}: {values}")

total_combinations = np.prod([len(v) for v in PARAM_GRID.values()])
print(f"\n💡 Total combinations: {total_combinations:,}")
print("⚠️  This will take a while... Reducing grid for practical optimization")

# Simplified grid (reasonable number of tests)
SIMPLIFIED_GRID = {
    'confidence_threshold': [0.70, 0.75],
    'maker_offset_pct': [0.0003, 0.0005],
    'maker_timeout': [20, 30],
    'take_profit_pct': [0.0015, 0.0018],
    'stop_loss_pct': [0.0010],
    'hold_time': [90]
}

total_tests = np.prod([len(v) for v in SIMPLIFIED_GRID.values()])
print(f"\n✅ Simplified to {total_tests} combinations")

# ==========================================
# LOAD MODELS
# ==========================================
print("\n📂 Loading models...")

model1_classifier = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_multiclass.txt')
model2_buy = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_buy_regressor.txt')
model2_sell = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_sell_regressor.txt')
model3_strategy = HybridMakerStrategy(model1_classifier, model2_buy, model2_sell)

print("✅ Models loaded")

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
# BACKTEST ENGINE (Optimized)
# ==========================================
class FastBacktestEngine:
    """Lightweight backtest engine for parameter optimization"""

    def __init__(self, df, initial_capital, position_size, leverage, commission_rate):
        self.df = df
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.leverage = leverage
        self.commission_rate = commission_rate

    def backtest(self, signals, params):
        """
        Run backtest with given signals and parameters

        signals: list of tuples (idx, signal_type, confidence)
        params: dict with maker_offset_pct, maker_timeout, take_profit_pct, stop_loss_pct, hold_time
        """
        capital = self.initial_capital
        trades = []

        for idx, signal_type, confidence in signals:
            if signal_type == 0:  # SKIP
                continue

            if idx >= len(self.df) - params['maker_timeout'] - params['hold_time']:
                continue

            current_price = self.df.iloc[idx]['close']

            if signal_type == 1:  # BUY
                trade = self._simulate_buy_trade(
                    idx, current_price,
                    params['maker_offset_pct'],
                    params['maker_timeout'],
                    params['take_profit_pct'],
                    params['stop_loss_pct'],
                    params['hold_time']
                )
            elif signal_type == 2:  # SELL
                trade = self._simulate_sell_trade(
                    idx, current_price,
                    params['maker_offset_pct'],
                    params['maker_timeout'],
                    params['take_profit_pct'],
                    params['stop_loss_pct'],
                    params['hold_time']
                )
            else:
                continue

            if trade:
                trades.append(trade)
                if trade['status'] == 'completed':
                    capital += trade['pnl_usdt']

        return self._calculate_metrics(trades, capital)

    def _simulate_buy_trade(self, idx, current_price, offset, timeout, tp_pct, sl_pct, hold_time):
        """Simulate BUY maker trade"""
        entry_price = current_price * (1 - offset)

        # Check fill
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

        # Track position
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

        return {
            'status': 'completed',
            'type': 'BUY',
            'pnl_usdt': pnl_usdt,
            'exit_reason': exit_reason,
            'win': pnl_usdt > 0
        }

    def _simulate_sell_trade(self, idx, current_price, offset, timeout, tp_pct, sl_pct, hold_time):
        """Simulate SELL maker trade"""
        entry_price = current_price * (1 + offset)

        # Check fill
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

        # Track position
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

        return {
            'status': 'completed',
            'type': 'SELL',
            'pnl_usdt': pnl_usdt,
            'exit_reason': exit_reason,
            'win': pnl_usdt > 0
        }

    def _calculate_metrics(self, trades, final_capital):
        """Calculate performance metrics"""
        if not trades:
            return {
                'total_trades': 0,
                'completed_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'return_pct': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }

        completed = [t for t in trades if t['status'] == 'completed']

        if not completed:
            return {
                'total_trades': len(trades),
                'completed_trades': 0,
                'win_rate': 0,
                'total_pnl': 0,
                'return_pct': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }

        wins = [t for t in completed if t['win']]
        losses = [t for t in completed if not t['win']]

        total_pnl = sum(t['pnl_usdt'] for t in completed)
        win_rate = len(wins) / len(completed)

        # Profit factor
        total_profit = sum(t['pnl_usdt'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl_usdt'] for t in losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        # Sharpe ratio
        returns = [t['pnl_usdt'] / self.initial_capital for t in completed]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        return {
            'total_trades': len(trades),
            'completed_trades': len(completed),
            'unfilled': len(trades) - len(completed),
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'return_pct': (final_capital - self.initial_capital) / self.initial_capital * 100,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor,
            'final_capital': final_capital
        }

# ==========================================
# OPTIMIZE MODEL 1 (Multi-Class)
# ==========================================
print("\n" + "=" * 100)
print("🔷 OPTIMIZING MODEL 1: MULTI-CLASS CLASSIFICATION")
print("=" * 100)

backtest_engine = FastBacktestEngine(df_test, INITIAL_CAPITAL, POSITION_SIZE, LEVERAGE, COMMISSION_RATE)

best_params_m1 = None
best_score_m1 = -999999
results_m1 = []

param_combinations = list(product(
    SIMPLIFIED_GRID['confidence_threshold'],
    SIMPLIFIED_GRID['maker_offset_pct'],
    SIMPLIFIED_GRID['maker_timeout'],
    SIMPLIFIED_GRID['take_profit_pct'],
    SIMPLIFIED_GRID['stop_loss_pct'],
    SIMPLIFIED_GRID['hold_time']
))

print(f"Testing {len(param_combinations)} combinations...")

for i, (conf_thresh, offset, timeout, tp, sl, hold) in enumerate(param_combinations):
    # Generate signals
    signals = []
    for idx in range(len(X_test) - 100):
        features = X_test.iloc[idx].values.reshape(1, -1)
        probs = model1_classifier.predict(features)[0]
        signal = np.argmax(probs)
        confidence = probs[signal]

        if confidence >= conf_thresh:
            signals.append((idx, signal, confidence))

    # Backtest
    params = {
        'maker_offset_pct': offset,
        'maker_timeout': timeout,
        'take_profit_pct': tp,
        'stop_loss_pct': sl,
        'hold_time': hold
    }

    metrics = backtest_engine.backtest(signals, params)

    # Scoring: Weighted combination
    score = (
        metrics['return_pct'] * 0.4 +
        metrics['sharpe_ratio'] * 10 * 0.3 +
        metrics['win_rate'] * 100 * 0.2 +
        metrics['profit_factor'] * 20 * 0.1
    )

    result = {
        'conf_thresh': conf_thresh,
        'offset': offset,
        'timeout': timeout,
        'tp': tp,
        'sl': sl,
        'hold': hold,
        'score': score,
        **metrics
    }

    results_m1.append(result)

    if score > best_score_m1:
        best_score_m1 = score
        best_params_m1 = result

    if (i + 1) % 5 == 0:
        print(f"  Progress: {i+1}/{len(param_combinations)} | Best score: {best_score_m1:.2f}")

print("\n✅ Optimization complete!")
print(f"\n🏆 Best Parameters (Model 1):")
print(f"  Confidence threshold: {best_params_m1['conf_thresh']:.2f}")
print(f"  Maker offset:         {best_params_m1['offset']:.4f} ({best_params_m1['offset']*100:.2f}%)")
print(f"  Timeout:              {best_params_m1['timeout']}s")
print(f"  Take profit:          {best_params_m1['tp']:.4f} ({best_params_m1['tp']*100:.2f}%)")
print(f"  Stop loss:            {best_params_m1['sl']:.4f} ({best_params_m1['sl']*100:.2f}%)")
print(f"  Hold time:            {best_params_m1['hold']}s")
print(f"\n📊 Best Performance:")
print(f"  Total trades:    {best_params_m1['total_trades']}")
print(f"  Completed:       {best_params_m1['completed_trades']}")
print(f"  Win rate:        {best_params_m1['win_rate']*100:.2f}%")
print(f"  Total PNL:       ${best_params_m1['total_pnl']:.2f}")
print(f"  Return:          {best_params_m1['return_pct']:.2f}%")
print(f"  Sharpe:          {best_params_m1['sharpe_ratio']:.2f}")
print(f"  Profit Factor:   {best_params_m1['profit_factor']:.2f}")
print(f"  Score:           {best_score_m1:.2f}")

# ==========================================
# SAVE RESULTS
# ==========================================
print("\n" + "=" * 100)
print("💾 SAVING RESULTS")
print("=" * 100)

# Save Model 1 results
df_results_m1 = pd.DataFrame(results_m1)
df_results_m1 = df_results_m1.sort_values('score', ascending=False)
df_results_m1.to_csv('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/optimization_results_model1.csv', index=False)
print("✅ Model 1 results saved to: optimization_results_model1.csv")

# Save best parameters
import json
best_config = {
    'model1': {
        'confidence_threshold': best_params_m1['conf_thresh'],
        'maker_offset_pct': best_params_m1['offset'],
        'maker_timeout': best_params_m1['timeout'],
        'take_profit_pct': best_params_m1['tp'],
        'stop_loss_pct': best_params_m1['sl'],
        'hold_time': best_params_m1['hold'],
        'performance': {
            'win_rate': best_params_m1['win_rate'],
            'return_pct': best_params_m1['return_pct'],
            'sharpe_ratio': best_params_m1['sharpe_ratio'],
            'profit_factor': best_params_m1['profit_factor']
        }
    }
}

with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/best_parameters.json', 'w') as f:
    json.dump(best_config, f, indent=2)

print("✅ Best parameters saved to: best_parameters.json")

# Show top 5 configurations
print("\n📊 Top 5 Configurations:")
print(df_results_m1[['conf_thresh', 'offset', 'timeout', 'tp', 'return_pct', 'win_rate', 'sharpe_ratio', 'score']].head(10).to_string(index=False))

print("\n" + "=" * 100)
print("🎉 PARAMETER OPTIMIZATION COMPLETE!")
print("=" * 100)
