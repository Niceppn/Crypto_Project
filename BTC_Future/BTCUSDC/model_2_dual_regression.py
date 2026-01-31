"""
Model 2: Dual Regression Models
- Buy Model: Predicts maximum upside % in next 5 minutes
- Sell Model: Predicts maximum downside % in next 5 minutes
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
import pickle
import matplotlib.pyplot as plt
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features,
    create_regression_targets,
    get_feature_columns
)

# ==========================================
# CONFIGURATION
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
BUY_MODEL_OUTPUT = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_buy_regressor.txt'
SELL_MODEL_OUTPUT = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_sell_regressor.txt'
METADATA_OUTPUT = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_dual_regression_metadata.json'

# Use all data or limit for testing
USE_ALL_DATA = True
TEST_ROWS = 500000

# Regression window (5 minutes = 300 seconds)
PREDICTION_WINDOW = 300

print("=" * 80)
print("📈 MODEL 2: DUAL REGRESSION MODELS")
print("=" * 80)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
df = load_and_prepare_data(RAW_FILE, nrows=None if USE_ALL_DATA else TEST_ROWS)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)

# ==========================================
# 2. CREATE REGRESSION TARGETS
# ==========================================
print("\n🎯 Creating regression targets...")

upside_pct, downside_pct = create_regression_targets(df_features, window=PREDICTION_WINDOW)
df_features['upside_pct'] = upside_pct
df_features['downside_pct'] = downside_pct

# Clean data
df_features = df_features.dropna()

print(f"\n📊 Target Statistics:")
print(f"  Upside % (max gain in {PREDICTION_WINDOW}s):")
print(f"    Mean:   {df_features['upside_pct'].mean():7.4f}%")
print(f"    Median: {df_features['upside_pct'].median():7.4f}%")
print(f"    Std:    {df_features['upside_pct'].std():7.4f}%")
print(f"    Max:    {df_features['upside_pct'].max():7.4f}%")
print(f"\n  Downside % (max loss in {PREDICTION_WINDOW}s):")
print(f"    Mean:   {df_features['downside_pct'].mean():7.4f}%")
print(f"    Median: {df_features['downside_pct'].median():7.4f}%")
print(f"    Std:    {df_features['downside_pct'].std():7.4f}%")
print(f"    Max:    {df_features['downside_pct'].max():7.4f}%")

# ==========================================
# 3. PREPARE FEATURES
# ==========================================
feature_cols = get_feature_columns()
X = df_features[feature_cols]

# Train-test split (temporal order preserved)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]

print(f"\n📈 Training set: {len(X_train):,} samples")
print(f"📉 Test set: {len(X_test):,} samples")

# ==========================================
# 4. TRAIN BUY MODEL (Upside Predictor)
# ==========================================
print("\n" + "=" * 80)
print("🟢 TRAINING BUY MODEL (Upside Predictor)")
print("=" * 80)

y_upside_train = df_features['upside_pct'].iloc[:split_idx]
y_upside_test = df_features['upside_pct'].iloc[split_idx:]

buy_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

buy_model.fit(
    X_train, y_upside_train,
    eval_set=[(X_test, y_upside_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

# Evaluate
y_upside_pred = buy_model.predict(X_test)

buy_mae = mean_absolute_error(y_upside_test, y_upside_pred)
buy_rmse = np.sqrt(mean_squared_error(y_upside_test, y_upside_pred))
buy_r2 = r2_score(y_upside_test, y_upside_pred)

print(f"\n✅ Buy Model Performance:")
print(f"  MAE:  {buy_mae:.4f}%")
print(f"  RMSE: {buy_rmse:.4f}%")
print(f"  R²:   {buy_r2:.4f}")

# ==========================================
# 5. TRAIN SELL MODEL (Downside Predictor)
# ==========================================
print("\n" + "=" * 80)
print("🔴 TRAINING SELL MODEL (Downside Predictor)")
print("=" * 80)

y_downside_train = df_features['downside_pct'].iloc[:split_idx]
y_downside_test = df_features['downside_pct'].iloc[split_idx:]

sell_model = lgb.LGBMRegressor(
    objective='regression',
    n_estimators=300,
    learning_rate=0.02,
    num_leaves=31,
    max_depth=6,
    min_child_samples=100,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

sell_model.fit(
    X_train, y_downside_train,
    eval_set=[(X_test, y_downside_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

# Evaluate
y_downside_pred = sell_model.predict(X_test)

sell_mae = mean_absolute_error(y_downside_test, y_downside_pred)
sell_rmse = np.sqrt(mean_squared_error(y_downside_test, y_downside_pred))
sell_r2 = r2_score(y_downside_test, y_downside_pred)

print(f"\n✅ Sell Model Performance:")
print(f"  MAE:  {sell_mae:.4f}%")
print(f"  RMSE: {sell_rmse:.4f}%")
print(f"  R²:   {sell_r2:.4f}")

# ==========================================
# 6. FEATURE IMPORTANCE
# ==========================================
print("\n" + "=" * 80)
print("🔝 FEATURE IMPORTANCE")
print("=" * 80)

# Buy model features
buy_feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': buy_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🟢 Top 15 Features for BUY (Upside Prediction):")
for idx, row in buy_feature_importance.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:8.2f}")

# Sell model features
sell_feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': sell_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔴 Top 15 Features for SELL (Downside Prediction):")
for idx, row in sell_feature_importance.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:8.2f}")

# ==========================================
# 7. THRESHOLD ANALYSIS
# ==========================================
print("\n" + "=" * 80)
print("🎚️  THRESHOLD ANALYSIS")
print("=" * 80)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30]

for threshold in thresholds:
    # Buy signals
    buy_signals = y_upside_pred >= threshold
    buy_actual = y_upside_test[buy_signals]
    buy_profit_rate = (buy_actual >= 0.15).sum() / len(buy_actual) if len(buy_actual) > 0 else 0

    # Sell signals
    sell_signals = y_downside_pred >= threshold
    sell_actual = y_downside_test[sell_signals]
    sell_profit_rate = (sell_actual >= 0.15).sum() / len(sell_actual) if len(sell_actual) > 0 else 0

    print(f"\n📊 Threshold: {threshold:.2f}%")
    print(f"  BUY signals:  {buy_signals.sum():6,} | Win rate: {buy_profit_rate*100:5.2f}% | Avg actual: {buy_actual.mean():.4f}%")
    print(f"  SELL signals: {sell_signals.sum():6,} | Win rate: {sell_profit_rate*100:5.2f}% | Avg actual: {sell_actual.mean():.4f}%")

# ==========================================
# 8. COMBINED SIGNAL ANALYSIS
# ==========================================
print("\n" + "=" * 80)
print("🎯 COMBINED SIGNAL ANALYSIS (Buy vs Sell)")
print("=" * 80)

# Find strong directional signals
strong_buy = y_upside_pred >= 0.20
strong_sell = y_downside_pred >= 0.20

print(f"\n📊 Strong Directional Signals:")
print(f"  Strong BUY (upside ≥ 0.20%):   {strong_buy.sum():6,}")
print(f"  Strong SELL (downside ≥ 0.20%): {strong_sell.sum():6,}")
print(f"  Both strong (conflict):         {(strong_buy & strong_sell).sum():6,}")

# Analyze when both models agree
both_low = (y_upside_pred < 0.10) & (y_downside_pred < 0.10)
print(f"  Both weak (skip):               {both_low.sum():6,}")

# Win rate when models disagree strongly
dominant_buy = (y_upside_pred >= 0.20) & (y_upside_pred > y_downside_pred * 1.5)
dominant_sell = (y_downside_pred >= 0.20) & (y_downside_pred > y_upside_pred * 1.5)

if dominant_buy.sum() > 0:
    dominant_buy_winrate = (y_upside_test[dominant_buy] >= 0.15).sum() / dominant_buy.sum()
    print(f"\n  Dominant BUY (upside > 1.5x downside):")
    print(f"    Signals: {dominant_buy.sum():6,} | Win rate: {dominant_buy_winrate*100:5.2f}%")

if dominant_sell.sum() > 0:
    dominant_sell_winrate = (y_downside_test[dominant_sell] >= 0.15).sum() / dominant_sell.sum()
    print(f"  Dominant SELL (downside > 1.5x upside):")
    print(f"    Signals: {dominant_sell.sum():6,} | Win rate: {dominant_sell_winrate*100:5.2f}%")

# ==========================================
# 9. SAVE MODELS
# ==========================================
print("\n" + "=" * 80)
print("💾 SAVING MODELS")
print("=" * 80)

# Save models
buy_model.booster_.save_model(BUY_MODEL_OUTPUT)
sell_model.booster_.save_model(SELL_MODEL_OUTPUT)

print(f"✅ Buy model saved to: {BUY_MODEL_OUTPUT}")
print(f"✅ Sell model saved to: {SELL_MODEL_OUTPUT}")

# Save metadata
metadata = {
    'model_type': 'dual_regression',
    'prediction_window': PREDICTION_WINDOW,
    'features': feature_cols,
    'num_features': len(feature_cols),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'buy_model': {
        'mae': float(buy_mae),
        'rmse': float(buy_rmse),
        'r2': float(buy_r2),
        'feature_importance': buy_feature_importance.head(15).to_dict('records')
    },
    'sell_model': {
        'mae': float(sell_mae),
        'rmse': float(sell_rmse),
        'r2': float(sell_r2),
        'feature_importance': sell_feature_importance.head(15).to_dict('records')
    },
    'target_statistics': {
        'upside': {
            'mean': float(df_features['upside_pct'].mean()),
            'median': float(df_features['upside_pct'].median()),
            'std': float(df_features['upside_pct'].std())
        },
        'downside': {
            'mean': float(df_features['downside_pct'].mean()),
            'median': float(df_features['downside_pct'].median()),
            'std': float(df_features['downside_pct'].std())
        }
    },
    'hyperparameters': {
        'n_estimators': 300,
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 6,
        'min_child_samples': 100
    }
}

with open(METADATA_OUTPUT, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Metadata saved to: {METADATA_OUTPUT}")

print("\n" + "=" * 80)
print("🎉 MODEL 2 TRAINING COMPLETE!")
print("=" * 80)
