"""
Backtest Improved Regularized Model
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

print("=" * 100)
print("📊 BACKTESTING IMPROVED MODEL")
print("=" * 100)

# Load metadata
with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_improved_metadata.json', 'r') as f:
    metadata = json.load(f)

selected_features = metadata['features']
print(f"\n✅ Loaded {len(selected_features)} selected features")

# Load model
model = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_improved.txt')
print("✅ Model loaded")

# Load optimal parameters
with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/best_parameters.json', 'r') as f:
    params = json.load(f)['model1']

CONFIDENCE_THRESHOLD = params['confidence_threshold']
MAKER_OFFSET_PCT = params['maker_offset_pct']
MAKER_TIMEOUT = params['maker_timeout']
TAKE_PROFIT_PCT = params['take_profit_pct']
STOP_LOSS_PCT = params['stop_loss_pct']
HOLD_TIME = params['hold_time']

print(f"\n📋 Parameters:")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"  Maker offset: {MAKER_OFFSET_PCT:.4f}")
print(f"  Maker timeout: {MAKER_TIMEOUT}s")
print(f"  Take profit: {TAKE_PROFIT_PCT:.4f}")
print(f"  Stop loss: {STOP_LOSS_PCT:.4f}")
print(f"  Hold time: {HOLD_TIME}s")

# Load data
print("\n📂 Loading data...")
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
df = load_and_prepare_data(RAW_FILE, nrows=None)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)
df_features = df_features.dropna()

# Split
split_idx = int(len(df_features) * 0.8)
df_test = df_features.iloc[split_idx:].copy()

print(f"Test set: {len(df_test):,} rows")

# Predict
X_test = df_test[selected_features]
y_pred_proba = model.predict(X_test)

# Get predictions
max_proba = y_pred_proba.max(axis=1)
y_pred = y_pred_proba.argmax(axis=1)

# Filter by confidence
high_confidence_mask = max_proba >= CONFIDENCE_THRESHOLD
df_test['prediction'] = 0
df_test['confidence'] = max_proba
df_test.loc[high_confidence_mask, 'prediction'] = y_pred[high_confidence_mask]

print(f"\n🎯 Predictions:")
print(f"  Total signals: {high_confidence_mask.sum():,} ({high_confidence_mask.sum()/len(df_test)*100:.2f}%)")
print(f"  BUY signals:   {(df_test['prediction'] == 1).sum():,}")
print(f"  SELL signals:  {(df_test['prediction'] == 2).sum():,}")

# Backtest
print("\n" + "=" * 100)
print("💰 RUNNING BACKTEST")
print("=" * 100)

trades = []

for i in range(len(df_test)):
    if df_test.iloc[i]['prediction'] == 0:
        continue

    signal = 'BUY' if df_test.iloc[i]['prediction'] == 1 else 'SELL'
    confidence = df_test.iloc[i]['confidence']
    signal_price = df_test.iloc[i]['close']
    signal_time = df_test.index[i]

    # Check if maker order fills
    if signal == 'BUY':
        entry_price = signal_price * (1 - MAKER_OFFSET_PCT)
    else:
        entry_price = signal_price * (1 + MAKER_OFFSET_PCT)

    # Check fill within timeout
    filled = False
    fill_time = None

    for j in range(i, min(i + MAKER_TIMEOUT, len(df_test))):
        if signal == 'BUY':
            if df_test.iloc[j]['low'] <= entry_price:
                filled = True
                fill_time = df_test.index[j]
                break
        else:
            if df_test.iloc[j]['high'] >= entry_price:
                filled = True
                fill_time = df_test.index[j]
                break

    if not filled:
        trades.append({
            'signal': signal,
            'signal_price': signal_price,
            'entry_price': entry_price,
            'confidence': confidence,
            'filled': False,
            'pnl_pct': 0
        })
        continue

    # Find fill index
    fill_idx = df_test.index.get_loc(fill_time)

    # Check TP/SL/Hold time
    exit_price = None
    exit_reason = None

    for k in range(fill_idx, min(fill_idx + HOLD_TIME, len(df_test))):
        current_price = df_test.iloc[k]['close']

        if signal == 'BUY':
            pnl_pct = (current_price - entry_price) / entry_price

            if current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                exit_price = entry_price * (1 + TAKE_PROFIT_PCT)
                exit_reason = 'TP'
                break
            elif current_price <= entry_price * (1 - STOP_LOSS_PCT):
                exit_price = entry_price * (1 - STOP_LOSS_PCT)
                exit_reason = 'SL'
                break
        else:
            pnl_pct = (entry_price - current_price) / entry_price

            if current_price <= entry_price * (1 - TAKE_PROFIT_PCT):
                exit_price = entry_price * (1 - TAKE_PROFIT_PCT)
                exit_reason = 'TP'
                break
            elif current_price >= entry_price * (1 + STOP_LOSS_PCT):
                exit_price = entry_price * (1 + STOP_LOSS_PCT)
                exit_reason = 'SL'
                break

    if exit_price is None:
        exit_price = df_test.iloc[min(fill_idx + HOLD_TIME - 1, len(df_test) - 1)]['close']
        exit_reason = 'TIMEOUT'

    # Calculate PnL
    if signal == 'BUY':
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    trades.append({
        'signal': signal,
        'signal_price': signal_price,
        'entry_price': entry_price,
        'exit_price': exit_price,
        'exit_reason': exit_reason,
        'confidence': confidence,
        'filled': True,
        'pnl_pct': pnl_pct
    })

# Results
df_trades = pd.DataFrame(trades)

print(f"\n📊 Trade Summary:")
print(f"  Total signals:     {len(df_trades):,}")
print(f"  Filled orders:     {df_trades['filled'].sum():,} ({df_trades['filled'].sum()/len(df_trades)*100:.2f}%)")
print(f"  Completed trades:  {df_trades['filled'].sum():,}")

if df_trades['filled'].sum() > 0:
    completed = df_trades[df_trades['filled'] == True]

    wins = (completed['pnl_pct'] > 0).sum()
    losses = (completed['pnl_pct'] < 0).sum()
    win_rate = wins / len(completed)

    total_return = completed['pnl_pct'].sum()
    avg_win = completed[completed['pnl_pct'] > 0]['pnl_pct'].mean() if wins > 0 else 0
    avg_loss = completed[completed['pnl_pct'] < 0]['pnl_pct'].mean() if losses > 0 else 0

    print(f"\n💰 Performance:")
    print(f"  Win rate:          {win_rate*100:.2f}%")
    print(f"  Total return:      {total_return*100:.2f}%")
    print(f"  Average win:       {avg_win*100:.4f}%")
    print(f"  Average loss:      {avg_loss*100:.4f}%")

    # Calculate Sharpe
    returns = completed['pnl_pct'].values
    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0
    print(f"  Sharpe ratio:      {sharpe:.4f}")

    # Profit factor
    total_wins = completed[completed['pnl_pct'] > 0]['pnl_pct'].sum()
    total_losses = abs(completed[completed['pnl_pct'] < 0]['pnl_pct'].sum())
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    print(f"  Profit factor:     {profit_factor:.4f}")

    # By exit reason
    print(f"\n📋 Exit Reasons:")
    for reason in ['TP', 'SL', 'TIMEOUT']:
        count = (completed['exit_reason'] == reason).sum()
        pct = count / len(completed) * 100
        avg_pnl = completed[completed['exit_reason'] == reason]['pnl_pct'].mean() if count > 0 else 0
        print(f"  {reason:10s}: {count:4d} ({pct:5.2f}%) | Avg PnL: {avg_pnl*100:+.4f}%")

    # By signal type
    print(f"\n📊 By Signal Type:")
    for signal_type in ['BUY', 'SELL']:
        signal_trades = completed[completed['signal'] == signal_type]
        if len(signal_trades) > 0:
            signal_wins = (signal_trades['pnl_pct'] > 0).sum()
            signal_win_rate = signal_wins / len(signal_trades)
            signal_return = signal_trades['pnl_pct'].sum()
            print(f"  {signal_type:4s}: {len(signal_trades):4d} trades | Win rate: {signal_win_rate*100:.2f}% | Return: {signal_return*100:+.2f}%")

print("\n" + "=" * 100)
print("✅ BACKTEST COMPLETE")
print("=" * 100)
