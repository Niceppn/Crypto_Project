"""
Walk-Forward with Simple Filter
Approach: High confidence + Volume requirement + Volatility requirement
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
print("🔄 WALK-FORWARD WITH SIMPLE FILTER")
print("=" * 100)

# Load data
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
df = load_and_prepare_data(RAW_FILE, nrows=None)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)

# Target parameters (stricter)
FILL_WINDOW = 15
PROFIT_WINDOW = 90
FILL_OFFSET = 0.0005
PROFIT_TARGET = 0.0020

print("\n🎯 Creating STRICTER targets...")

# BUY target
future_min_low = df_features['low'].rolling(window=FILL_WINDOW, min_periods=1).min().shift(-1)
entry_price_buy = df_features['close'] * (1 - FILL_OFFSET)
is_filled_buy = future_min_low <= entry_price_buy

future_max_high = df_features['high'].rolling(window=PROFIT_WINDOW, min_periods=1).max().shift(-(FILL_WINDOW + 1))
target_price_buy = df_features['close'] * (1 + PROFIT_TARGET)
is_profit_buy = future_max_high >= target_price_buy

df_features['buy_target'] = (is_filled_buy & is_profit_buy).astype(int)

# SELL target
future_max_high = df_features['high'].rolling(window=FILL_WINDOW, min_periods=1).max().shift(-1)
entry_price_sell = df_features['close'] * (1 + FILL_OFFSET)
is_filled_sell = future_max_high >= entry_price_sell

future_min_low = df_features['low'].rolling(window=PROFIT_WINDOW, min_periods=1).min().shift(-(FILL_WINDOW + 1))
target_price_sell = df_features['close'] * (1 - PROFIT_TARGET)
is_profit_sell = future_min_low <= target_price_sell

df_features['sell_target'] = (is_filled_sell & is_profit_sell).astype(int)

# Multi-class target
df_features['target'] = 0
df_features.loc[df_features['buy_target'] == 1, 'target'] = 1
df_features.loc[df_features['sell_target'] == 1, 'target'] = 2

# Handle conflicts
both_targets = (df_features['buy_target'] == 1) & (df_features['sell_target'] == 1)
if both_targets.sum() > 0:
    df_features.loc[both_targets & (df_features['net_flow'] > 0), 'target'] = 1
    df_features.loc[both_targets & (df_features['net_flow'] <= 0), 'target'] = 2

df_features = df_features.dropna()

print(f"✅ Dataset: {len(df_features):,} samples")

# Calculate filter thresholds from overall data
CONFIDENCE_THRESHOLD = 0.92  # Increased from 0.75
VOLUME_THRESHOLD = df_features['volume_ma30'].quantile(0.50)  # Median
ATR_THRESHOLD = df_features['atr_5'].quantile(0.25)  # 25th percentile

print(f"\n📋 Filter Parameters:")
print(f"  Confidence threshold: {CONFIDENCE_THRESHOLD}")
print(f"  Volume threshold:     {VOLUME_THRESHOLD:.4f}")
print(f"  ATR threshold:        {ATR_THRESHOLD:.4f}")

# Load trading parameters
with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/best_parameters.json', 'r') as f:
    params = json.load(f)['model1']

MAKER_OFFSET_PCT = params['maker_offset_pct']
MAKER_TIMEOUT = params['maker_timeout']
TAKE_PROFIT_PCT = params['take_profit_pct']
STOP_LOSS_PCT = params['stop_loss_pct']
HOLD_TIME = params['hold_time']

print(f"\n📋 Trading Parameters:")
print(f"  Maker offset: {MAKER_OFFSET_PCT:.4f}")
print(f"  TP/SL: {TAKE_PROFIT_PCT:.4f}/{STOP_LOSS_PCT:.4f}")
print(f"  Hold time: {HOLD_TIME}s")

# Walk-forward setup
NUM_WINDOWS = 5
TRAIN_PCT = 0.8

window_size = len(df_features) // NUM_WINDOWS
results = []

print(f"\n" + "=" * 100)
print(f"🔄 TESTING {NUM_WINDOWS} WINDOWS WITH FILTER")
print("=" * 100)

for window_idx in range(NUM_WINDOWS):
    print(f"\n{'='*100}")
    print(f"Window {window_idx + 1}/{NUM_WINDOWS}")
    print(f"{'='*100}")

    # Define window
    start_idx = window_idx * window_size
    end_idx = start_idx + window_size if window_idx < NUM_WINDOWS - 1 else len(df_features)

    window_data = df_features.iloc[start_idx:end_idx]

    # Split train/val
    split_idx = int(len(window_data) * TRAIN_PCT)
    train_data = window_data.iloc[:split_idx]
    val_data = window_data.iloc[split_idx:]

    print(f"Period: {window_data.index[0]} to {window_data.index[-1]}")
    print(f"Train: {len(train_data):,} | Val: {len(val_data):,}")

    # Feature selection on training data
    feature_cols = get_feature_columns()
    X_train = train_data[feature_cols]
    y_train = train_data['target']

    selector = SelectKBest(mutual_info_classif, k=40)
    X_train_selected = selector.fit_transform(X_train, y_train)

    selected_mask = selector.get_support()
    selected_features = [f for f, s in zip(feature_cols, selected_mask) if s]

    print(f"Selected {len(selected_features)} features")

    # Train model with regularization
    from sklearn.utils.class_weight import compute_class_weight

    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        max_depth=4,
        min_child_samples=200,
        reg_alpha=0.1,
        reg_lambda=0.1,
        subsample=0.7,
        colsample_bytree=0.7,
        subsample_freq=5,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(
        X_train_selected, y_train,
        eval_set=[(X_train_selected, y_train)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0)
        ]
    )

    print(f"✅ Model trained (best iteration: {model.best_iteration_})")

    # Predict on validation
    X_val = val_data[selected_features]
    y_pred_proba = model.predict_proba(X_val)

    max_proba = y_pred_proba.max(axis=1)
    y_pred = y_pred_proba.argmax(axis=1)

    # Apply SIMPLE FILTER
    filter_mask = (
        (max_proba >= CONFIDENCE_THRESHOLD) &
        (val_data['volume_ma30'].values >= VOLUME_THRESHOLD) &
        (val_data['atr_5'].values >= ATR_THRESHOLD)
    )

    val_data_copy = val_data.copy()
    val_data_copy['prediction'] = 0
    val_data_copy.loc[filter_mask, 'prediction'] = y_pred[filter_mask]

    signals_count = filter_mask.sum()
    signals_pct = signals_count / len(val_data) * 100

    print(f"\n🎯 Filter Results:")
    print(f"  Before filter:  {(max_proba >= 0.75).sum():,} signals ({(max_proba >= 0.75).sum()/len(val_data)*100:.1f}%)")
    print(f"  After filter:   {signals_count:,} signals ({signals_pct:.1f}%)")
    print(f"  Reduction:      {((max_proba >= 0.75).sum() - signals_count):,} signals filtered out")

    # Backtest
    trades = []

    for i in range(len(val_data_copy)):
        if val_data_copy.iloc[i]['prediction'] == 0:
            continue

        signal = 'BUY' if val_data_copy.iloc[i]['prediction'] == 1 else 'SELL'
        signal_price = val_data_copy.iloc[i]['close']

        if signal == 'BUY':
            entry_price = signal_price * (1 - MAKER_OFFSET_PCT)
        else:
            entry_price = signal_price * (1 + MAKER_OFFSET_PCT)

        # Check fill
        filled = False
        fill_idx = None

        for j in range(i, min(i + MAKER_TIMEOUT, len(val_data_copy))):
            if signal == 'BUY':
                if val_data_copy.iloc[j]['low'] <= entry_price:
                    filled = True
                    fill_idx = j
                    break
            else:
                if val_data_copy.iloc[j]['high'] >= entry_price:
                    filled = True
                    fill_idx = j
                    break

        if not filled:
            continue

        # Check exit
        exit_price = None

        for k in range(fill_idx, min(fill_idx + HOLD_TIME, len(val_data_copy))):
            current_price = val_data_copy.iloc[k]['close']

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
            exit_price = val_data_copy.iloc[min(fill_idx + HOLD_TIME - 1, len(val_data_copy) - 1)]['close']

        if signal == 'BUY':
            pnl_pct = (exit_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - exit_price) / entry_price

        trades.append({'pnl_pct': pnl_pct})

    # Calculate metrics
    if len(trades) > 0:
        df_trades = pd.DataFrame(trades)
        wins = (df_trades['pnl_pct'] > 0).sum()
        win_rate = wins / len(df_trades)
        total_return = df_trades['pnl_pct'].sum()

        returns = df_trades['pnl_pct'].values
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(len(returns)) if np.std(returns) > 0 else 0

        total_wins = df_trades[df_trades['pnl_pct'] > 0]['pnl_pct'].sum()
        total_losses = abs(df_trades[df_trades['pnl_pct'] < 0]['pnl_pct'].sum())
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        print(f"\n💰 Results:")
        print(f"  Trades: {len(df_trades)}")
        print(f"  Win rate: {win_rate*100:.2f}%")
        print(f"  Return: {total_return*100:+.2f}%")
        print(f"  Sharpe: {sharpe:.2f}")
        print(f"  Profit factor: {profit_factor:.2f}")

        results.append({
            'window': window_idx + 1,
            'period_start': str(window_data.index[0]),
            'period_end': str(window_data.index[-1]),
            'signals_before': int((max_proba >= 0.75).sum()),
            'signals_after': int(signals_count),
            'trades': len(df_trades),
            'win_rate': win_rate,
            'return_pct': total_return * 100,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor
        })
    else:
        print(f"\n⚠️  No trades executed")
        results.append({
            'window': window_idx + 1,
            'period_start': str(window_data.index[0]),
            'period_end': str(window_data.index[-1]),
            'signals_before': int((max_proba >= 0.75).sum()),
            'signals_after': int(signals_count),
            'trades': 0,
            'win_rate': 0,
            'return_pct': 0,
            'sharpe_ratio': 0,
            'profit_factor': 0
        })

# Summary
print(f"\n" + "=" * 100)
print("📊 WALK-FORWARD SUMMARY (WITH FILTER)")
print("=" * 100)

df_results = pd.DataFrame(results)
df_results.to_csv('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/walk_forward_filtered_results.csv', index=False)

print(f"\n📋 Results by Window:")
print(df_results[['window', 'signals_after', 'trades', 'win_rate', 'return_pct', 'sharpe_ratio']].to_string(index=False))

# Statistics
valid_results = df_results[df_results['trades'] > 0]

if len(valid_results) > 0:
    mean_win_rate = valid_results['win_rate'].mean()
    std_win_rate = valid_results['win_rate'].std()
    mean_return = valid_results['return_pct'].mean()
    std_return = valid_results['return_pct'].std()
    mean_sharpe = valid_results['sharpe_ratio'].mean()
    std_sharpe = valid_results['sharpe_ratio'].std()
    profitable_windows = (valid_results['return_pct'] > 0).sum()

    print(f"\n📊 Statistics:")
    print(f"  Mean win rate:      {mean_win_rate*100:.2f}% ± {std_win_rate*100:.2f}%")
    print(f"  Mean return:        {mean_return:+.2f}% ± {std_return:.2f}%")
    print(f"  Mean Sharpe:        {mean_sharpe:.2f} ± {std_sharpe:.2f}")
    print(f"  Profitable windows: {profitable_windows}/{NUM_WINDOWS} ({profitable_windows/NUM_WINDOWS*100:.0f}%)")

    # Compare to original
    print(f"\n📊 Comparison to Original (No Filter):")
    print(f"  Original consistency: 40% (2/5 windows)")
    print(f"  Filtered consistency: {profitable_windows/NUM_WINDOWS*100:.0f}% ({profitable_windows}/{NUM_WINDOWS} windows)")

    # Rating
    consistency = profitable_windows / NUM_WINDOWS

    if consistency >= 0.8 and mean_return > 0.5:
        rating = "🟢 EXCELLENT"
        verdict = "Ready for production testing!"
    elif consistency >= 0.6 and mean_return > 0:
        rating = "🟡 GOOD"
        verdict = "Improved! Continue development"
    elif consistency >= 0.4:
        rating = "🟠 FAIR"
        verdict = "Marginal improvement"
    else:
        rating = "🔴 POOR"
        verdict = "Filter didn't help significantly"

    print(f"\n🏆 Consistency Score: {consistency*100:.0f}%")
    print(f"Rating: {rating}")
    print(f"Verdict: {verdict}")

    # Save summary
    summary = {
        'approach': 'simple_filter',
        'filter_params': {
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'volume_threshold': float(VOLUME_THRESHOLD),
            'atr_threshold': float(ATR_THRESHOLD)
        },
        'statistics': {
            'mean_win_rate': mean_win_rate,
            'std_win_rate': std_win_rate,
            'mean_return_pct': mean_return,
            'std_return_pct': std_return,
            'mean_sharpe': mean_sharpe,
            'std_sharpe': std_sharpe,
            'profitable_windows': int(profitable_windows),
            'consistency_score': consistency
        },
        'comparison': {
            'original_consistency': 0.4,
            'filtered_consistency': consistency,
            'improvement': consistency - 0.4
        },
        'rating': rating,
        'verdict': verdict
    }

    with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/walk_forward_filtered_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Results saved")

print(f"\n" + "=" * 100)
print("✅ FILTERED WALK-FORWARD COMPLETE")
print("=" * 100)
