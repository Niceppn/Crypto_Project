"""
Quick Test: Different Confidence Thresholds
Find optimal confidence level
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features,
    get_feature_columns
)

print("=" * 100)
print("🧪 TESTING DIFFERENT CONFIDENCE THRESHOLDS")
print("=" * 100)

# Load data (use last 20% as test set - includes both Window 4 & 5)
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
df = load_and_prepare_data(RAW_FILE, nrows=None)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)

# Create targets
FILL_WINDOW = 15
PROFIT_WINDOW = 90
FILL_OFFSET = 0.0005
PROFIT_TARGET = 0.0020

# BUY
future_min_low = df_features['low'].rolling(window=FILL_WINDOW, min_periods=1).min().shift(-1)
entry_price_buy = df_features['close'] * (1 - FILL_OFFSET)
is_filled_buy = future_min_low <= entry_price_buy
future_max_high = df_features['high'].rolling(window=PROFIT_WINDOW, min_periods=1).max().shift(-(FILL_WINDOW + 1))
target_price_buy = df_features['close'] * (1 + PROFIT_TARGET)
is_profit_buy = future_max_high >= target_price_buy
df_features['buy_target'] = (is_filled_buy & is_profit_buy).astype(int)

# SELL
future_max_high = df_features['high'].rolling(window=FILL_WINDOW, min_periods=1).max().shift(-1)
entry_price_sell = df_features['close'] * (1 + FILL_OFFSET)
is_filled_sell = future_max_high >= entry_price_sell
future_min_low = df_features['low'].rolling(window=PROFIT_WINDOW, min_periods=1).min().shift(-(FILL_WINDOW + 1))
target_price_sell = df_features['close'] * (1 - PROFIT_TARGET)
is_profit_sell = future_min_low <= target_price_sell
df_features['sell_target'] = (is_filled_sell & is_profit_sell).astype(int)

df_features['target'] = 0
df_features.loc[df_features['buy_target'] == 1, 'target'] = 1
df_features.loc[df_features['sell_target'] == 1, 'target'] = 2

both_targets = (df_features['buy_target'] == 1) & (df_features['sell_target'] == 1)
if both_targets.sum() > 0:
    df_features.loc[both_targets & (df_features['net_flow'] > 0), 'target'] = 1
    df_features.loc[both_targets & (df_features['net_flow'] <= 0), 'target'] = 2

df_features = df_features.dropna()

# Use test set
split_idx = int(len(df_features) * 0.8)
test_data = df_features.iloc[split_idx:]

print(f"Test set: {len(test_data):,} samples")

# Load model
model = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_improved.txt')
with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_improved_metadata.json', 'r') as f:
    metadata = json.load(f)
selected_features = metadata['features']

# Predict
X_test = test_data[selected_features]
y_test_proba = model.predict(X_test)
y_test_pred = y_test_proba.argmax(axis=1)
y_test_conf = y_test_proba.max(axis=1)

# Load parameters
with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/best_parameters.json', 'r') as f:
    params = json.load(f)['model1']

MAKER_OFFSET_PCT = params['maker_offset_pct']
MAKER_TIMEOUT = params['maker_timeout']
TAKE_PROFIT_PCT = params['take_profit_pct']
STOP_LOSS_PCT = params['stop_loss_pct']
HOLD_TIME = params['hold_time']

# Volume & ATR thresholds
VOLUME_THRESHOLD = df_features['volume_ma30'].quantile(0.50)
ATR_THRESHOLD = df_features['atr_5'].quantile(0.25)

# Test different confidence levels
confidence_levels = [0.85, 0.90, 0.92, 0.95, 0.97, 0.99]

print(f"\n{'='*100}")
print(f"{'Confidence':>12s} | {'Signals':>8s} | {'Trades':>7s} | {'Win Rate':>9s} | {'Return':>8s} | {'Sharpe':>7s}")
print(f"{'='*100}")

results = []

for conf_threshold in confidence_levels:
    # Apply filter
    filter_mask = (
        (y_test_conf >= conf_threshold) &
        (test_data['volume_ma30'].values >= VOLUME_THRESHOLD) &
        (test_data['atr_5'].values >= ATR_THRESHOLD)
    )

    test_copy = test_data.copy()
    test_copy['prediction'] = 0
    test_copy.loc[filter_mask, 'prediction'] = y_test_pred[filter_mask]

    signals_count = filter_mask.sum()

    # Backtest
    trades = []

    for i in range(len(test_copy)):
        if test_copy.iloc[i]['prediction'] == 0:
            continue

        signal = 'BUY' if test_copy.iloc[i]['prediction'] == 1 else 'SELL'
        signal_price = test_copy.iloc[i]['close']

        if signal == 'BUY':
            entry_price = signal_price * (1 - MAKER_OFFSET_PCT)
        else:
            entry_price = signal_price * (1 + MAKER_OFFSET_PCT)

        # Check fill
        filled = False
        fill_idx = None

        for j in range(i, min(i + MAKER_TIMEOUT, len(test_copy))):
            if signal == 'BUY':
                if test_copy.iloc[j]['low'] <= entry_price:
                    filled = True
                    fill_idx = j
                    break
            else:
                if test_copy.iloc[j]['high'] >= entry_price:
                    filled = True
                    fill_idx = j
                    break

        if not filled:
            continue

        # Check exit
        exit_price = None

        for k in range(fill_idx, min(fill_idx + HOLD_TIME, len(test_copy))):
            current_price = test_copy.iloc[k]['close']

            if signal == 'BUY':
                if current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                    exit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                    break
                elif current_price <= entry_price * (1 - STOP_LOSS_PCT):
                    exit_price = entry_price * (1 - STOP_LOSS_PCT)
                    break
            else:
                if current_price <= entry_price * (1 - TAKE_PROFIT_PCT):
                    exit_price = entry_price * (1 - TAKE_PROFIT_PCT)
                    break
                elif current_price >= entry_price * (1 + STOP_LOSS_PCT):
                    exit_price = entry_price * (1 + STOP_LOSS_PCT)
                    break

        if exit_price is None:
            exit_price = test_copy.iloc[min(fill_idx + HOLD_TIME - 1, len(test_copy) - 1)]['close']

        if signal == 'BUY':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({'pnl_pct': pnl_pct})

    if len(trades) > 0:
        df_trades = pd.DataFrame(trades)
        wins = (df_trades['pnl_pct'] > 0).sum()
        win_rate = wins / len(df_trades)
        total_return = df_trades['pnl_pct'].sum()

        returns = df_trades['pnl_pct'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0

        print(f"{conf_threshold:>12.2f} | {signals_count:>8,} | {len(trades):>7,} | {win_rate*100:>8.2f}% | {total_return*100:>7.2f}% | {sharpe:>7.2f}")

        results.append({
            'confidence': conf_threshold,
            'signals': signals_count,
            'trades': len(trades),
            'win_rate': win_rate,
            'return_pct': total_return * 100,
            'sharpe': sharpe
        })
    else:
        print(f"{conf_threshold:>12.2f} | {signals_count:>8,} | {0:>7,} | {0:>8.2f}% | {0:>7.2f}% | {0:>7.2f}")

print(f"{'='*100}")

# Find best
df_results = pd.DataFrame(results)
best_sharpe_idx = df_results['sharpe'].idxmax()
best_return_idx = df_results['return_pct'].idxmax()
best_winrate_idx = df_results['win_rate'].idxmax()

print(f"\n💡 RECOMMENDATIONS:")
print(f"\n  Best Sharpe ratio:")
print(f"    Confidence: {df_results.loc[best_sharpe_idx, 'confidence']:.2f}")
print(f"    Trades: {int(df_results.loc[best_sharpe_idx, 'trades'])}")
print(f"    Win rate: {df_results.loc[best_sharpe_idx, 'win_rate']*100:.2f}%")
print(f"    Return: {df_results.loc[best_sharpe_idx, 'return_pct']:+.2f}%")
print(f"    Sharpe: {df_results.loc[best_sharpe_idx, 'sharpe']:.2f}")

print(f"\n  Best return:")
print(f"    Confidence: {df_results.loc[best_return_idx, 'confidence']:.2f}")
print(f"    Trades: {int(df_results.loc[best_return_idx, 'trades'])}")
print(f"    Win rate: {df_results.loc[best_return_idx, 'win_rate']*100:.2f}%")
print(f"    Return: {df_results.loc[best_return_idx, 'return_pct']:+.2f}%")

print(f"\n  Best win rate:")
print(f"    Confidence: {df_results.loc[best_winrate_idx, 'confidence']:.2f}")
print(f"    Trades: {int(df_results.loc[best_winrate_idx, 'trades'])}")
print(f"    Win rate: {df_results.loc[best_winrate_idx, 'win_rate']*100:.2f}%")
print(f"    Return: {df_results.loc[best_winrate_idx, 'return_pct']:+.2f}%")

print(f"\n✅ Testing complete")
