"""
Analyze Market Conditions: Window 4 (Disaster) vs Window 5 (Success)
Goal: Find key differences to create regime filter
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features
)

print("=" * 100)
print("🔍 ANALYZING MARKET REGIMES")
print("=" * 100)

# Load data
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
df = load_and_prepare_data(RAW_FILE, nrows=None)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)
df_features = df_features.dropna()

print(f"✅ Loaded {len(df_features):,} samples")

# Define windows based on walk-forward results
NUM_WINDOWS = 5
window_size = len(df_features) // NUM_WINDOWS

# Window 4: Jan 28-29 (DISASTER: 771 trades, 31.91% win rate, -14.63%)
window4_start = 3 * window_size
window4_end = 4 * window_size
window4_data = df_features.iloc[window4_start:window4_end]

# Window 5: Jan 29-30 (SUCCESS: 31 trades, 80.65% win rate, +1.48%)
window5_start = 4 * window_size
window5_end = len(df_features)
window5_data = df_features.iloc[window5_start:window5_end]

print(f"\n📊 Window 4 (DISASTER):")
print(f"  Period: {window4_data.index[0]} to {window4_data.index[-1]}")
print(f"  Samples: {len(window4_data):,}")

print(f"\n📊 Window 5 (SUCCESS):")
print(f"  Period: {window5_data.index[0]} to {window5_data.index[-1]}")
print(f"  Samples: {len(window5_data):,}")

# ==========================================
# ANALYSIS 1: VOLATILITY (ATR)
# ==========================================
print(f"\n" + "=" * 100)
print("📈 VOLATILITY ANALYSIS (ATR)")
print("=" * 100)

w4_atr = window4_data['atr_5'].describe()
w5_atr = window5_data['atr_5'].describe()

print(f"\nWindow 4 (Bad):")
print(f"  Mean ATR:   {w4_atr['mean']:.4f}")
print(f"  Median ATR: {w4_atr['50%']:.4f}")
print(f"  Std ATR:    {w4_atr['std']:.4f}")
print(f"  Max ATR:    {w4_atr['max']:.4f}")

print(f"\nWindow 5 (Good):")
print(f"  Mean ATR:   {w5_atr['mean']:.4f}")
print(f"  Median ATR: {w5_atr['50%']:.4f}")
print(f"  Std ATR:    {w5_atr['std']:.4f}")
print(f"  Max ATR:    {w5_atr['max']:.4f}")

atr_ratio = w4_atr['mean'] / w5_atr['mean']
print(f"\n🔍 Window 4 ATR is {atr_ratio:.2f}x higher than Window 5")

# ==========================================
# ANALYSIS 2: SPREAD STABILITY
# ==========================================
print(f"\n" + "=" * 100)
print("📏 SPREAD ANALYSIS")
print("=" * 100)

w4_spread = window4_data['spread_proxy'].describe()
w5_spread = window5_data['spread_proxy'].describe()

print(f"\nWindow 4 (Bad):")
print(f"  Mean Spread:   {w4_spread['mean']:.6f}")
print(f"  Median Spread: {w4_spread['50%']:.6f}")
print(f"  Std Spread:    {w4_spread['std']:.6f}")
print(f"  Spread Volatility: {window4_data['spread_volatility'].mean():.6f}")

print(f"\nWindow 5 (Good):")
print(f"  Mean Spread:   {w5_spread['mean']:.6f}")
print(f"  Median Spread: {w5_spread['50%']:.6f}")
print(f"  Std Spread:    {w5_spread['std']:.6f}")
print(f"  Spread Volatility: {window5_data['spread_volatility'].mean():.6f}")

spread_ratio = w4_spread['mean'] / w5_spread['mean']
print(f"\n🔍 Window 4 Spread is {spread_ratio:.2f}x higher than Window 5")

# ==========================================
# ANALYSIS 3: ORDER FLOW CONSISTENCY
# ==========================================
print(f"\n" + "=" * 100)
print("💧 ORDER FLOW CONSISTENCY")
print("=" * 100)

w4_flow_std = window4_data['net_flow_std5'].describe()
w5_flow_std = window5_data['net_flow_std5'].describe()

print(f"\nWindow 4 (Bad):")
print(f"  Mean Flow Std:   {w4_flow_std['mean']:.2f}")
print(f"  Median Flow Std: {w4_flow_std['50%']:.2f}")
print(f"  Flow Accel Std:  {window4_data['net_flow_acceleration'].std():.6f}")

print(f"\nWindow 5 (Good):")
print(f"  Mean Flow Std:   {w5_flow_std['mean']:.2f}")
print(f"  Median Flow Std: {w5_flow_std['50%']:.2f}")
print(f"  Flow Accel Std:  {window5_data['net_flow_acceleration'].std():.6f}")

flow_ratio = w4_flow_std['mean'] / w5_flow_std['mean']
print(f"\n🔍 Window 4 Flow Std is {flow_ratio:.2f}x higher than Window 5")

# ==========================================
# ANALYSIS 4: VOLUME PATTERNS
# ==========================================
print(f"\n" + "=" * 100)
print("📊 VOLUME ANALYSIS")
print("=" * 100)

w4_vol = window4_data['total_volume'].describe()
w5_vol = window5_data['total_volume'].describe()

print(f"\nWindow 4 (Bad):")
print(f"  Mean Volume:     {w4_vol['mean']:.2f}")
print(f"  Volume Std:      {w4_vol['std']:.2f}")
print(f"  Volume Surge:    {window4_data['volume_surge'].mean():.4f}")
print(f"  Relative Volume: {window4_data['relative_volume'].mean():.4f}")

print(f"\nWindow 5 (Good):")
print(f"  Mean Volume:     {w5_vol['mean']:.2f}")
print(f"  Volume Std:      {w5_vol['std']:.2f}")
print(f"  Volume Surge:    {window5_data['volume_surge'].mean():.4f}")
print(f"  Relative Volume: {window5_data['relative_volume'].mean():.4f}")

vol_surge_ratio = window4_data['volume_surge'].mean() / window5_data['volume_surge'].mean()
print(f"\n🔍 Window 4 Volume Surge is {vol_surge_ratio:.2f}x higher than Window 5")

# ==========================================
# ANALYSIS 5: TREND STRENGTH
# ==========================================
print(f"\n" + "=" * 100)
print("📉 TREND ANALYSIS")
print("=" * 100)

w4_trend = window4_data['trend_strength'].describe()
w5_trend = window5_data['trend_strength'].describe()

print(f"\nWindow 4 (Bad):")
print(f"  Mean Trend Strength: {w4_trend['mean']:.6f}")
print(f"  Median:              {w4_trend['50%']:.6f}")
print(f"  Price Velocity Std:  {window4_data['price_velocity'].std():.6f}")

print(f"\nWindow 5 (Good):")
print(f"  Mean Trend Strength: {w5_trend['mean']:.6f}")
print(f"  Median:              {w5_trend['50%']:.6f}")
print(f"  Price Velocity Std:  {window5_data['price_velocity'].std():.6f}")

# ==========================================
# CALCULATE PERCENTILES FOR THRESHOLDS
# ==========================================
print(f"\n" + "=" * 100)
print("🎯 SUGGESTED THRESHOLDS FOR REGIME FILTER")
print("=" * 100)

# Use overall data to calculate percentiles
all_atr = df_features['atr_5']
all_spread_vol = df_features['spread_volatility']
all_flow_std = df_features['net_flow_std5']
all_vol_surge = df_features['volume_surge']

# Find percentiles where Window 4 typically sits
w4_atr_pct = (all_atr < w4_atr['mean']).sum() / len(all_atr)
w4_spread_pct = (all_spread_vol < window4_data['spread_volatility'].mean()).sum() / len(all_spread_vol)
w4_flow_pct = (all_flow_std < w4_flow_std['mean']).sum() / len(all_flow_std)
w4_surge_pct = (all_vol_surge < window4_data['volume_surge'].mean()).sum() / len(all_vol_surge)

print(f"\nWindow 4 characteristics sit at these percentiles:")
print(f"  ATR:              {w4_atr_pct*100:.1f}th percentile")
print(f"  Spread Volatility: {w4_spread_pct*100:.1f}th percentile")
print(f"  Flow Std:         {w4_flow_pct*100:.1f}th percentile")
print(f"  Volume Surge:     {w4_surge_pct*100:.1f}th percentile")

# Suggest thresholds to filter top 20-30% worst conditions
atr_threshold = all_atr.quantile(0.75)
spread_vol_threshold = all_spread_vol.quantile(0.75)
flow_std_threshold = all_flow_std.quantile(0.75)
vol_surge_threshold = all_vol_surge.quantile(0.75)

print(f"\n💡 Suggested Thresholds (Filter worst 25%):")
print(f"  ATR threshold:              {atr_threshold:.6f}")
print(f"  Spread volatility threshold: {spread_vol_threshold:.6f}")
print(f"  Flow std threshold:         {flow_std_threshold:.2f}")
print(f"  Volume surge threshold:     {vol_surge_threshold:.4f}")

# ==========================================
# TEST FILTER ON HISTORICAL DATA
# ==========================================
print(f"\n" + "=" * 100)
print("🧪 TESTING FILTER ON HISTORICAL DATA")
print("=" * 100)

def is_favorable_regime(row):
    """Simple rule-based regime filter"""
    conditions = []

    # 1. ATR not too high (volatility check)
    if row['atr_5'] <= atr_threshold:
        conditions.append(True)
    else:
        conditions.append(False)

    # 2. Spread volatility stable
    if row['spread_volatility'] <= spread_vol_threshold:
        conditions.append(True)
    else:
        conditions.append(False)

    # 3. Order flow consistent
    if row['net_flow_std5'] <= flow_std_threshold:
        conditions.append(True)
    else:
        conditions.append(False)

    # 4. No extreme volume surges
    if row['volume_surge'] <= vol_surge_threshold:
        conditions.append(True)
    else:
        conditions.append(False)

    # Require at least 3/4 conditions
    return sum(conditions) >= 3

# Apply filter
df_features['is_favorable'] = df_features.apply(is_favorable_regime, axis=1)

print(f"\n📊 Overall Data:")
print(f"  Favorable samples: {df_features['is_favorable'].sum():,} ({df_features['is_favorable'].sum()/len(df_features)*100:.1f}%)")
print(f"  Unfavorable samples: {(~df_features['is_favorable']).sum():,} ({(~df_features['is_favorable']).sum()/len(df_features)*100:.1f}%)")

# Check each window
print(f"\n📊 By Window:")
w4_favorable_pct = 0
w5_favorable_pct = 0

for i in range(NUM_WINDOWS):
    w_start = i * window_size
    w_end = w_start + window_size if i < NUM_WINDOWS - 1 else len(df_features)
    w_data = df_features.iloc[w_start:w_end]

    favorable_pct = w_data['is_favorable'].sum() / len(w_data) * 100
    window_name = f"Window {i+1}"

    if i == 3:  # Window 4 (disaster)
        window_name += " (DISASTER)"
        w4_favorable_pct = favorable_pct
        w4_favorable_count = w_data['is_favorable'].sum()
        w4_total = len(w_data)
    elif i == 4:  # Window 5 (success)
        window_name += " (SUCCESS)"
        w5_favorable_pct = favorable_pct

    print(f"  {window_name:25s}: {favorable_pct:5.1f}% favorable")

# Save thresholds
thresholds = {
    'atr_threshold': float(atr_threshold),
    'spread_volatility_threshold': float(spread_vol_threshold),
    'flow_std_threshold': float(flow_std_threshold),
    'volume_surge_threshold': float(vol_surge_threshold),
    'conditions_required': 3,  # out of 4
    'analysis': {
        'window4_favorable_pct': float(w4_favorable_pct),
        'window5_favorable_pct': float(w5_favorable_pct),
        'expected_reduction': f"Window 4: {w4_favorable_count} / {w4_total} samples"
    }
}

import json
with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/regime_thresholds.json', 'w') as f:
    json.dump(thresholds, f, indent=2)

print(f"\n✅ Thresholds saved to regime_thresholds.json")

# ==========================================
# KEY INSIGHTS
# ==========================================
print(f"\n" + "=" * 100)
print("💡 KEY INSIGHTS")
print("=" * 100)

print(f"\n1. VOLATILITY:")
print(f"   - Window 4 ATR is {atr_ratio:.2f}x higher")
print(f"   - High ATR = unstable price action = harder to predict")

print(f"\n2. SPREAD:")
print(f"   - Window 4 Spread is {spread_ratio:.2f}x wider")
print(f"   - Wide spread = poor liquidity = harder to fill maker orders")

print(f"\n3. ORDER FLOW:")
print(f"   - Window 4 Flow Std is {flow_ratio:.2f}x higher")
print(f"   - Inconsistent flow = unpredictable direction")

print(f"\n4. VOLUME:")
print(f"   - Window 4 Volume Surge is {vol_surge_ratio:.2f}x higher")
print(f"   - Extreme surges = potential manipulation/news events")

print(f"\n🎯 CONCLUSION:")
print(f"   Window 4 = High volatility + Wide spread + Inconsistent flow + Volume surges")
print(f"   Window 5 = Stable conditions = Model works well")
print(f"   Strategy: Filter out periods that look like Window 4")

print(f"\n" + "=" * 100)
print("✅ ANALYSIS COMPLETE")
print("=" * 100)
