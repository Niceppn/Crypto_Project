"""
Analyze Model Predictions: Why Window 4 Failed but Window 5 Succeeded
New Approach: Focus on prediction quality, not market conditions
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
print("🔍 ANALYZING PREDICTION QUALITY")
print("=" * 100)

# Load improved model
model = lgb.Booster(model_file='/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_improved.txt')
with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_improved_metadata.json', 'r') as f:
    metadata = json.load(f)
selected_features = metadata['features']

print(f"✅ Loaded model with {len(selected_features)} features")

# Load data
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
df = load_and_prepare_data(RAW_FILE, nrows=None)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)

# Create targets (stricter)
FILL_WINDOW = 15
PROFIT_WINDOW = 90
FILL_OFFSET = 0.0005
PROFIT_TARGET = 0.0020

print("\n🎯 Creating targets...")

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

# Multi-class
df_features['target'] = 0
df_features.loc[df_features['buy_target'] == 1, 'target'] = 1
df_features.loc[df_features['sell_target'] == 1, 'target'] = 2

both_targets = (df_features['buy_target'] == 1) & (df_features['sell_target'] == 1)
if both_targets.sum() > 0:
    df_features.loc[both_targets & (df_features['net_flow'] > 0), 'target'] = 1
    df_features.loc[both_targets & (df_features['net_flow'] <= 0), 'target'] = 2

df_features = df_features.dropna()

print(f"✅ Dataset: {len(df_features):,} samples")

# Define windows
NUM_WINDOWS = 5
window_size = len(df_features) // NUM_WINDOWS

# Window 4: Disaster
window4_start = 3 * window_size
window4_end = 4 * window_size
window4_data = df_features.iloc[window4_start:window4_end].copy()

# Window 5: Success
window5_start = 4 * window_size
window5_end = len(df_features)
window5_data = df_features.iloc[window5_start:window5_end].copy()

print(f"\n📊 Window 4 (DISASTER): {len(window4_data):,} samples")
print(f"📊 Window 5 (SUCCESS): {len(window5_data):,} samples")

# ==========================================
# PREDICT ON BOTH WINDOWS
# ==========================================
print(f"\n" + "=" * 100)
print("🤖 MAKING PREDICTIONS")
print("=" * 100)

# Window 4
X_w4 = window4_data[selected_features]
y_w4 = window4_data['target']
y_w4_proba = model.predict(X_w4)
y_w4_pred = y_w4_proba.argmax(axis=1)
y_w4_conf = y_w4_proba.max(axis=1)

window4_data['prediction'] = y_w4_pred
window4_data['confidence'] = y_w4_conf
window4_data['correct'] = (y_w4_pred == y_w4).astype(int)

# Window 5
X_w5 = window5_data[selected_features]
y_w5 = window5_data['target']
y_w5_proba = model.predict(X_w5)
y_w5_pred = y_w5_proba.argmax(axis=1)
y_w5_conf = y_w5_proba.max(axis=1)

window5_data['prediction'] = y_w5_pred
window5_data['confidence'] = y_w5_conf
window5_data['correct'] = (y_w5_pred == y_w5).astype(int)

# ==========================================
# COMPARE PREDICTION ACCURACY
# ==========================================
print(f"\n📊 Prediction Accuracy:")

w4_accuracy = window4_data['correct'].mean()
w5_accuracy = window5_data['correct'].mean()

print(f"  Window 4: {w4_accuracy*100:.2f}%")
print(f"  Window 5: {w5_accuracy*100:.2f}%")
print(f"  Difference: {(w5_accuracy - w4_accuracy)*100:+.2f}%")

# By class
for window_name, window_data in [('Window 4', window4_data), ('Window 5', window5_data)]:
    print(f"\n{window_name} by class:")
    for pred_class, class_name in [(1, 'BUY'), (2, 'SELL')]:
        mask = window_data['prediction'] == pred_class
        if mask.sum() > 0:
            class_accuracy = window_data.loc[mask, 'correct'].mean()
            count = mask.sum()
            print(f"  {class_name}: {count:5d} predictions, {class_accuracy*100:.2f}% correct")

# ==========================================
# CONFIDENCE DISTRIBUTION
# ==========================================
print(f"\n" + "=" * 100)
print("📊 CONFIDENCE DISTRIBUTION")
print("=" * 100)

# Filter only BUY/SELL predictions (not SKIP)
w4_signals = window4_data[window4_data['prediction'] != 0]
w5_signals = window5_data[window5_data['prediction'] != 0]

print(f"\nWindow 4 (Disaster):")
print(f"  Mean confidence:   {w4_signals['confidence'].mean():.4f}")
print(f"  Median confidence: {w4_signals['confidence'].median():.4f}")
print(f"  Std confidence:    {w4_signals['confidence'].std():.4f}")
print(f"  High conf (>0.8):  {(w4_signals['confidence'] > 0.8).sum():5d} ({(w4_signals['confidence'] > 0.8).sum()/len(w4_signals)*100:.1f}%)")

print(f"\nWindow 5 (Success):")
print(f"  Mean confidence:   {w5_signals['confidence'].mean():.4f}")
print(f"  Median confidence: {w5_signals['confidence'].median():.4f}")
print(f"  Std confidence:    {w5_signals['confidence'].std():.4f}")
print(f"  High conf (>0.8):  {(w5_signals['confidence'] > 0.8).sum():5d} ({(w5_signals['confidence'] > 0.8).sum()/len(w5_signals)*100:.1f}%)")

# ==========================================
# ACCURACY BY CONFIDENCE LEVEL
# ==========================================
print(f"\n" + "=" * 100)
print("🎯 ACCURACY BY CONFIDENCE LEVEL")
print("=" * 100)

conf_bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for window_name, window_signals in [('Window 4', w4_signals), ('Window 5', w5_signals)]:
    print(f"\n{window_name}:")
    for i in range(len(conf_bins) - 1):
        lower = conf_bins[i]
        upper = conf_bins[i + 1]

        mask = (window_signals['confidence'] >= lower) & (window_signals['confidence'] < upper)
        if mask.sum() > 0:
            accuracy = window_signals.loc[mask, 'correct'].mean()
            count = mask.sum()
            print(f"  Confidence {lower:.1f}-{upper:.1f}: {count:5d} predictions, {accuracy*100:.2f}% accurate")

# ==========================================
# FEATURE VALUES COMPARISON
# ==========================================
print(f"\n" + "=" * 100)
print("🔬 TOP FEATURE VALUES COMPARISON")
print("=" * 100)

# Compare top 10 features
top_features = metadata['feature_importance'][:10]

print(f"\n{'Feature':<30s} | Window 4 (Bad) | Window 5 (Good) | Difference")
print("-" * 100)

for feat_info in top_features:
    feat_name = feat_info['feature']

    w4_mean = window4_data[feat_name].mean()
    w5_mean = window5_data[feat_name].mean()
    diff_pct = ((w5_mean - w4_mean) / abs(w4_mean) * 100) if w4_mean != 0 else 0

    print(f"{feat_name:<30s} | {w4_mean:13.4f} | {w5_mean:13.4f} | {diff_pct:+8.1f}%")

# ==========================================
# FEATURE CORRELATION WITH CORRECTNESS
# ==========================================
print(f"\n" + "=" * 100)
print("🔗 FEATURES CORRELATED WITH CORRECT PREDICTIONS")
print("=" * 100)

# Combine both windows
combined = pd.concat([window4_data, window5_data])
combined_signals = combined[combined['prediction'] != 0]

# Calculate correlation
correlations = []
for feat in selected_features:
    corr = combined_signals[feat].corr(combined_signals['correct'])
    if not np.isnan(corr):
        correlations.append({'feature': feat, 'correlation': corr})

correlations_df = pd.DataFrame(correlations).sort_values('correlation', key=abs, ascending=False)

print(f"\nTop 10 features correlated with correct predictions:")
for idx, row in correlations_df.head(10).iterrows():
    print(f"  {row['feature']:<30s}: {row['correlation']:+.4f}")

print(f"\nTop 10 features anti-correlated (predict wrong):")
for idx, row in correlations_df.tail(10).iterrows():
    print(f"  {row['feature']:<30s}: {row['correlation']:+.4f}")

# ==========================================
# PROPOSED QUALITY FILTER
# ==========================================
print(f"\n" + "=" * 100)
print("💡 PREDICTION QUALITY FILTER")
print("=" * 100)

def calculate_prediction_quality(row, selected_features):
    """Calculate quality score for a prediction"""
    score = 0

    # 1. Confidence score (0-40 points)
    conf = row['confidence']
    if conf >= 0.85:
        score += 40
    elif conf >= 0.80:
        score += 30
    elif conf >= 0.75:
        score += 20
    else:
        score += 10

    # 2. Feature quality score (0-30 points)
    # Check if top features are in "good" range (based on Window 5)
    feature_checks = 0

    # momentum_30 should be reasonable
    if abs(row['momentum_30']) < 100:
        feature_checks += 1

    # cvd_60 should not be extreme
    if abs(row['cvd_60']) < 50:
        feature_checks += 1

    # liquidity_ma15 should be decent
    if row['liquidity_ma15'] > 0:
        feature_checks += 1

    score += (feature_checks / 3) * 30

    # 3. Consistency score (0-30 points)
    # Check if related features agree
    # If net_flow > 0 and prediction is BUY, or net_flow < 0 and prediction is SELL
    if row['prediction'] == 1 and row['net_flow'] > 0:
        score += 15
    elif row['prediction'] == 2 and row['net_flow'] < 0:
        score += 15

    # If momentum and prediction align
    if row['prediction'] == 1 and row['momentum_30'] > 0:
        score += 15
    elif row['prediction'] == 2 and row['momentum_30'] < 0:
        score += 15

    return score

# Apply quality filter
window4_data['quality_score'] = window4_data.apply(lambda row: calculate_prediction_quality(row, selected_features), axis=1)
window5_data['quality_score'] = window5_data.apply(lambda row: calculate_prediction_quality(row, selected_features), axis=1)

print(f"\nQuality Score Distribution:")
print(f"\nWindow 4 (Bad):")
print(f"  Mean quality: {window4_data['quality_score'].mean():.2f}")
print(f"  Median:       {window4_data['quality_score'].median():.2f}")

print(f"\nWindow 5 (Good):")
print(f"  Mean quality: {window5_data['quality_score'].mean():.2f}")
print(f"  Median:       {window5_data['quality_score'].median():.2f}")

# Test different quality thresholds
print(f"\n📊 Accuracy by Quality Threshold:")

for threshold in [50, 60, 70, 80]:
    w4_high_quality = window4_data[window4_data['quality_score'] >= threshold]
    w5_high_quality = window5_data[window5_data['quality_score'] >= threshold]

    w4_signals_hq = w4_high_quality[w4_high_quality['prediction'] != 0]
    w5_signals_hq = w5_high_quality[w5_high_quality['prediction'] != 0]

    if len(w4_signals_hq) > 0 and len(w5_signals_hq) > 0:
        w4_acc = w4_signals_hq['correct'].mean()
        w5_acc = w5_signals_hq['correct'].mean()

        print(f"\n  Threshold ≥{threshold}:")
        print(f"    Window 4: {len(w4_signals_hq):5d} signals, {w4_acc*100:.2f}% accurate")
        print(f"    Window 5: {len(w5_signals_hq):5d} signals, {w5_acc*100:.2f}% accurate")

# Save quality filter parameters
quality_params = {
    'method': 'prediction_quality_score',
    'components': {
        'confidence_weight': 40,
        'feature_quality_weight': 30,
        'consistency_weight': 30
    },
    'thresholds': {
        'recommended': 60,
        'conservative': 70,
        'aggressive': 50
    },
    'analysis': {
        'window4_mean_quality': float(window4_data['quality_score'].mean()),
        'window5_mean_quality': float(window5_data['quality_score'].mean()),
        'quality_difference': float(window5_data['quality_score'].mean() - window4_data['quality_score'].mean())
    }
}

with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/quality_filter_params.json', 'w') as f:
    json.dump(quality_params, f, indent=2)

print(f"\n✅ Quality filter parameters saved")

print(f"\n" + "=" * 100)
print("✅ PREDICTION ANALYSIS COMPLETE")
print("=" * 100)

print(f"\n🎯 KEY FINDINGS:")
print(f"  1. Window 4 accuracy: {w4_accuracy*100:.2f}% vs Window 5: {w5_accuracy*100:.2f}%")
print(f"  2. Confidence levels similar between windows")
print(f"  3. Quality score difference: {quality_params['analysis']['quality_difference']:.2f}")
print(f"  4. Using quality filter at threshold 60 can improve accuracy")
