"""
Walk-Forward Optimization
Tests model consistency across different time periods

Strategy:
1. Split data into N windows (e.g., 10 windows)
2. For each window:
   - Train on 80% of window
   - Validate on remaining 20%
3. Analyze consistency of performance across windows
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from datetime import datetime
import json
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features,
    create_buy_maker_target,
    create_sell_maker_target,
    get_feature_columns
)

print("=" * 100)
print("🔄 WALK-FORWARD OPTIMIZATION")
print("=" * 100)

# ==========================================
# CONFIGURATION
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
USE_ALL_DATA = True
NUM_WINDOWS = 5  # Split data into 5 windows

# Best parameters from previous optimization
OPTIMAL_PARAMS = {
    'confidence_threshold': 0.75,
    'maker_offset_pct': 0.0003,
    'maker_timeout': 20,
    'take_profit_pct': 0.0018,
    'stop_loss_pct': 0.0010,
    'hold_time': 90
}

INITIAL_CAPITAL = 1000.0
POSITION_SIZE = 21.85
LEVERAGE = 20
COMMISSION_RATE = 0.0

# ==========================================
# LOAD AND PREPARE DATA
# ==========================================
print("\n📂 Loading full dataset...")

df = load_and_prepare_data(RAW_FILE, nrows=None if USE_ALL_DATA else 500000)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)

print("\n🎯 Creating targets...")
df_features['buy_target'] = create_buy_maker_target(df_features)
df_features['sell_target'] = create_sell_maker_target(df_features)

# Create multi-class target
df_features['target'] = 0
df_features.loc[df_features['buy_target'] == 1, 'target'] = 1
df_features.loc[df_features['sell_target'] == 1, 'target'] = 2

# Handle conflicts
both_targets = (df_features['buy_target'] == 1) & (df_features['sell_target'] == 1)
if both_targets.sum() > 0:
    df_features.loc[both_targets & (df_features['net_flow'] > 0), 'target'] = 1
    df_features.loc[both_targets & (df_features['net_flow'] <= 0), 'target'] = 2

df_features = df_features.dropna()

print(f"✅ Dataset ready: {len(df_features):,} candles")
print(f"   Period: {df_features.index[0]} → {df_features.index[-1]}")

feature_cols = get_feature_columns()
print(f"   Features: {len(feature_cols)}")

# ==========================================
# SPLIT INTO WINDOWS
# ==========================================
print(f"\n📊 Splitting data into {NUM_WINDOWS} windows...")

window_size = len(df_features) // NUM_WINDOWS
windows = []

for i in range(NUM_WINDOWS):
    start_idx = i * window_size
    end_idx = (i + 1) * window_size if i < NUM_WINDOWS - 1 else len(df_features)

    window_data = df_features.iloc[start_idx:end_idx]

    # Split each window into train (80%) and validate (20%)
    split_idx = int(len(window_data) * 0.8)
    train_data = window_data.iloc[:split_idx]
    val_data = window_data.iloc[split_idx:]

    windows.append({
        'id': i + 1,
        'start': window_data.index[0],
        'end': window_data.index[-1],
        'train': train_data,
        'val': val_data,
        'train_size': len(train_data),
        'val_size': len(val_data)
    })

    print(f"  Window {i+1}: {window_data.index[0]} → {window_data.index[-1]}")
    print(f"    Train: {len(train_data):,} | Val: {len(val_data):,}")

# ==========================================
# LIGHTWEIGHT BACKTEST ENGINE
# ==========================================
class FastBacktest:
    """Fast backtest for walk-forward"""

    def __init__(self, df, params):
        self.df = df
        self.params = params
        self.initial_capital = INITIAL_CAPITAL
        self.position_size = POSITION_SIZE
        self.leverage = LEVERAGE

    def run(self, predictions):
        """
        predictions: list of (idx, signal, confidence)
        """
        capital = self.initial_capital
        trades = []

        for idx, signal, confidence in predictions:
            if signal == 0 or confidence < self.params['confidence_threshold']:
                continue

            if idx >= len(self.df) - 120:  # Need buffer
                continue

            current_price = self.df.iloc[idx]['close']

            if signal == 1:  # BUY
                trade = self._simulate_buy(idx, current_price)
            elif signal == 2:  # SELL
                trade = self._simulate_sell(idx, current_price)
            else:
                continue

            if trade:
                trades.append(trade)
                if trade['status'] == 'completed':
                    capital += trade['pnl']

        return self._calculate_metrics(trades, capital)

    def _simulate_buy(self, idx, current_price):
        entry = current_price * (1 - self.params['maker_offset_pct'])

        # Check fill
        filled = False
        for i in range(1, self.params['maker_timeout'] + 1):
            if idx + i >= len(self.df):
                return {'status': 'unfilled'}
            if self.df.iloc[idx + i]['low'] <= entry:
                filled = True
                fill_idx = idx + i
                break

        if not filled:
            return {'status': 'unfilled'}

        # Track exit
        tp = entry * (1 + self.params['take_profit_pct'])
        sl = entry * (1 - self.params['stop_loss_pct'])

        exit_price = None
        for i in range(1, self.params['hold_time'] + 1):
            check_idx = fill_idx + i
            if check_idx >= len(self.df):
                break
            candle = self.df.iloc[check_idx]
            if candle['high'] >= tp:
                exit_price = tp
                break
            if candle['low'] <= sl:
                exit_price = sl
                break

        if exit_price is None:
            exit_idx = min(fill_idx + self.params['hold_time'], len(self.df) - 1)
            exit_price = self.df.iloc[exit_idx]['close']

        pnl_pct = (exit_price - entry) / entry
        pnl = self.position_size * self.leverage * pnl_pct

        return {'status': 'completed', 'pnl': pnl, 'win': pnl > 0}

    def _simulate_sell(self, idx, current_price):
        entry = current_price * (1 + self.params['maker_offset_pct'])

        filled = False
        for i in range(1, self.params['maker_timeout'] + 1):
            if idx + i >= len(self.df):
                return {'status': 'unfilled'}
            if self.df.iloc[idx + i]['high'] >= entry:
                filled = True
                fill_idx = idx + i
                break

        if not filled:
            return {'status': 'unfilled'}

        tp = entry * (1 - self.params['take_profit_pct'])
        sl = entry * (1 + self.params['stop_loss_pct'])

        exit_price = None
        for i in range(1, self.params['hold_time'] + 1):
            check_idx = fill_idx + i
            if check_idx >= len(self.df):
                break
            candle = self.df.iloc[check_idx]
            if candle['low'] <= tp:
                exit_price = tp
                break
            if candle['high'] >= sl:
                exit_price = sl
                break

        if exit_price is None:
            exit_idx = min(fill_idx + self.params['hold_time'], len(self.df) - 1)
            exit_price = self.df.iloc[exit_idx]['close']

        pnl_pct = (entry - exit_price) / entry
        pnl = self.position_size * self.leverage * pnl_pct

        return {'status': 'completed', 'pnl': pnl, 'win': pnl > 0}

    def _calculate_metrics(self, trades, final_capital):
        if not trades:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'return_pct': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }

        completed = [t for t in trades if t['status'] == 'completed']
        if not completed:
            return {
                'total_trades': len(trades),
                'win_rate': 0,
                'return_pct': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }

        wins = [t for t in completed if t['win']]
        losses = [t for t in completed if not t['win']]

        win_rate = len(wins) / len(completed)
        return_pct = (final_capital - self.initial_capital) / self.initial_capital * 100

        total_profit = sum(t['pnl'] for t in wins) if wins else 0
        total_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        returns = [t['pnl'] / self.initial_capital for t in completed]
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        return {
            'total_trades': len(trades),
            'completed_trades': len(completed),
            'win_rate': win_rate,
            'return_pct': return_pct,
            'sharpe_ratio': sharpe,
            'profit_factor': profit_factor
        }

# ==========================================
# WALK-FORWARD LOOP
# ==========================================
print("\n" + "=" * 100)
print("🔄 RUNNING WALK-FORWARD OPTIMIZATION")
print("=" * 100)

results = []

for window in windows:
    print(f"\n{'='*100}")
    print(f"📊 WINDOW {window['id']}/{NUM_WINDOWS}")
    print(f"{'='*100}")
    print(f"Period: {window['start']} → {window['end']}")

    # Prepare data
    X_train = window['train'][feature_cols]
    y_train = window['train']['target']
    X_val = window['val'][feature_cols]
    y_val = window['val']['target']

    print(f"\nTraining set: {len(X_train):,}")
    print(f"Validation set: {len(X_val):,}")

    # Calculate class weights
    from sklearn.utils.class_weight import compute_class_weight
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    # Train model
    print(f"\n🚀 Training model...")
    model = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        n_estimators=200,  # Reduced for speed
        learning_rate=0.02,
        num_leaves=31,
        max_depth=6,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    model.fit(X_train, y_train)
    print("✅ Training complete")

    # Generate predictions
    predictions = []
    for idx in range(len(X_val)):
        features = X_val.iloc[idx].values.reshape(1, -1)
        probs = model.predict_proba(features)[0]
        signal = np.argmax(probs)
        confidence = probs[signal]
        predictions.append((idx, signal, confidence))

    # Backtest
    print(f"\n📈 Running backtest...")
    backtest = FastBacktest(window['val'], OPTIMAL_PARAMS)
    metrics = backtest.run(predictions)

    # Store results
    result = {
        'window': window['id'],
        'period_start': str(window['start']),
        'period_end': str(window['end']),
        'train_size': window['train_size'],
        'val_size': window['val_size'],
        **metrics
    }
    results.append(result)

    # Print window results
    print(f"\n📊 Window {window['id']} Results:")
    print(f"  Total trades:     {metrics['total_trades']:5d}")
    print(f"  Completed:        {metrics.get('completed_trades', 0):5d}")
    print(f"  Win rate:         {metrics['win_rate']*100:5.2f}%")
    print(f"  Return:           {metrics['return_pct']:5.2f}%")
    print(f"  Sharpe:           {metrics['sharpe_ratio']:5.2f}")
    print(f"  Profit Factor:    {metrics['profit_factor']:5.2f}")

# ==========================================
# AGGREGATE RESULTS
# ==========================================
print("\n" + "=" * 100)
print("📊 WALK-FORWARD SUMMARY")
print("=" * 100)

results_df = pd.DataFrame(results)

print("\n📋 All Windows:")
print(results_df[['window', 'win_rate', 'return_pct', 'sharpe_ratio', 'profit_factor']].to_string(index=False))

# Statistics
print("\n📈 Performance Statistics:")
print(f"  Mean Win Rate:       {results_df['win_rate'].mean()*100:.2f}% (±{results_df['win_rate'].std()*100:.2f}%)")
print(f"  Mean Return:         {results_df['return_pct'].mean():.2f}% (±{results_df['return_pct'].std():.2f}%)")
print(f"  Mean Sharpe:         {results_df['sharpe_ratio'].mean():.2f} (±{results_df['sharpe_ratio'].std():.2f})")
print(f"  Mean Profit Factor:  {results_df['profit_factor'].mean():.2f} (±{results_df['profit_factor'].std():.2f})")

# Consistency metrics
print("\n🎯 Consistency Metrics:")
positive_returns = (results_df['return_pct'] > 0).sum()
print(f"  Profitable windows:  {positive_returns}/{NUM_WINDOWS} ({positive_returns/NUM_WINDOWS*100:.0f}%)")

good_sharpe = (results_df['sharpe_ratio'] > 1.0).sum()
print(f"  Sharpe > 1.0:        {good_sharpe}/{NUM_WINDOWS} ({good_sharpe/NUM_WINDOWS*100:.0f}%)")

good_winrate = (results_df['win_rate'] > 0.40).sum()
print(f"  Win rate > 40%:      {good_winrate}/{NUM_WINDOWS} ({good_winrate/NUM_WINDOWS*100:.0f}%)")

# Stability score (lower is better)
stability_score = (
    results_df['return_pct'].std() / abs(results_df['return_pct'].mean()) if results_df['return_pct'].mean() != 0 else 999
)
print(f"\n📉 Stability Score:     {stability_score:.2f} (CV of returns, lower=better)")

# Best/Worst windows
best_window = results_df.loc[results_df['return_pct'].idxmax()]
worst_window = results_df.loc[results_df['return_pct'].idxmin()]

print(f"\n🏆 Best Window:  #{best_window['window']} ({best_window['return_pct']:.2f}% return)")
print(f"⚠️  Worst Window: #{worst_window['window']} ({worst_window['return_pct']:.2f}% return)")

# ==========================================
# SAVE RESULTS
# ==========================================
print("\n" + "=" * 100)
print("💾 SAVING RESULTS")
print("=" * 100)

results_df.to_csv('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/walk_forward_results.csv', index=False)
print("✅ Results saved to: walk_forward_results.csv")

# Save summary
summary = {
    'num_windows': NUM_WINDOWS,
    'features_used': len(feature_cols),
    'parameters': OPTIMAL_PARAMS,
    'statistics': {
        'mean_win_rate': float(results_df['win_rate'].mean()),
        'std_win_rate': float(results_df['win_rate'].std()),
        'mean_return_pct': float(results_df['return_pct'].mean()),
        'std_return_pct': float(results_df['return_pct'].std()),
        'mean_sharpe': float(results_df['sharpe_ratio'].mean()),
        'std_sharpe': float(results_df['sharpe_ratio'].std()),
        'profitable_windows': int(positive_returns),
        'stability_score': float(stability_score)
    },
    'best_window': int(best_window['window']),
    'worst_window': int(worst_window['window'])
}

with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/walk_forward_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("✅ Summary saved to: walk_forward_summary.json")

# ==========================================
# INTERPRETATION
# ==========================================
print("\n" + "=" * 100)
print("💡 INTERPRETATION")
print("=" * 100)

avg_return = results_df['return_pct'].mean()
avg_sharpe = results_df['sharpe_ratio'].mean()
consistency = positive_returns / NUM_WINDOWS

if consistency >= 0.8 and avg_sharpe > 1.5:
    rating = "🟢 EXCELLENT"
    message = "Model is highly consistent and profitable across time periods. Ready for production!"
elif consistency >= 0.6 and avg_sharpe > 1.0:
    rating = "🟡 GOOD"
    message = "Model shows good consistency. Consider paper trading before live deployment."
elif consistency >= 0.4 and avg_sharpe > 0.5:
    rating = "🟠 MODERATE"
    message = "Model has moderate consistency. More improvements needed before deployment."
else:
    rating = "🔴 POOR"
    message = "Model lacks consistency. Significant improvements required."

print(f"\n{rating}")
print(f"{message}")
print(f"\n📊 Key Metrics:")
print(f"  Consistency:   {consistency*100:.0f}% profitable")
print(f"  Avg Return:    {avg_return:.2f}%")
print(f"  Avg Sharpe:    {avg_sharpe:.2f}")

print("\n" + "=" * 100)
print("🎉 WALK-FORWARD OPTIMIZATION COMPLETE!")
print("=" * 100)
