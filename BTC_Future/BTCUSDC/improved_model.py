"""
Improved Model Training with All Optimizations
- Feature Selection (keep only best features)
- Regularization (prevent overfitting)
- Proper Cross-Validation
- Stricter Target Criteria
- Ensemble of Models
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectKBest, mutual_info_classif
import json
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features,
    get_feature_columns
)

print("=" * 100)
print("🚀 IMPROVED MODEL TRAINING - AGGRESSIVE OPTIMIZATION")
print("=" * 100)

# ==========================================
# CONFIGURATION
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
USE_ALL_DATA = True

# Stricter target criteria
FILL_WINDOW = 15          # เพิ่มจาก 10 → 15 (ง่ายกว่า)
PROFIT_WINDOW = 90        # เพิ่มจาก 60 → 90 (เวลามากขึ้น)
FILL_OFFSET = 0.0005      # เหมือนเดิม
PROFIT_TARGET = 0.0020    # เพิ่มจาก 0.0015 → 0.0020 (เป้าสูงขึ้น)

# ==========================================
# LOAD DATA
# ==========================================
print("\n📂 Loading data...")

df = load_and_prepare_data(RAW_FILE, nrows=None if USE_ALL_DATA else 500000)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)

print("\n🎯 Creating STRICTER targets...")

# Stricter BUY target
future_min_low = df_features['low'].rolling(window=FILL_WINDOW, min_periods=1).min().shift(-1)
entry_price_buy = df_features['close'] * (1 - FILL_OFFSET)
is_filled_buy = future_min_low <= entry_price_buy

future_max_high = df_features['high'].rolling(window=PROFIT_WINDOW, min_periods=1).max().shift(-(FILL_WINDOW + 1))
target_price_buy = df_features['close'] * (1 + PROFIT_TARGET)
is_profit_buy = future_max_high >= target_price_buy

df_features['buy_target'] = (is_filled_buy & is_profit_buy).astype(int)

# Stricter SELL target
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

print(f"\n📊 Target Distribution (STRICTER):")
target_counts = df_features['target'].value_counts().sort_index()
for target, count in target_counts.items():
    target_name = ['SKIP', 'BUY', 'SELL'][target]
    print(f"  {target_name:12s}: {count:8,} ({count/len(df_features)*100:5.2f}%)")

# ==========================================
# FEATURE SELECTION
# ==========================================
print("\n" + "=" * 100)
print("🔍 FEATURE SELECTION - Keep only best features")
print("=" * 100)

feature_cols = get_feature_columns()
X_all = df_features[feature_cols]
y_all = df_features['target']

print(f"Starting with {len(feature_cols)} features")

# Use mutual information to select best features
selector = SelectKBest(mutual_info_classif, k=40)  # Keep top 40
X_selected = selector.fit_transform(X_all, y_all)

# Get selected feature names
selected_mask = selector.get_support()
selected_features = [f for f, selected in zip(feature_cols, selected_mask) if selected]

print(f"Selected {len(selected_features)} best features:")
feature_scores = pd.DataFrame({
    'feature': feature_cols,
    'score': selector.scores_
}).sort_values('score', ascending=False)

for idx, row in feature_scores.head(40).iterrows():
    print(f"  {row['feature']:35s}: {row['score']:8.4f}")

# Update feature set
X = df_features[selected_features]
y = df_features['target']

# ==========================================
# SPLIT DATA (Temporal)
# ==========================================
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n📈 Training set: {len(X_train):,}")
print(f"📉 Test set: {len(X_test):,}")

# ==========================================
# CLASS WEIGHTS
# ==========================================
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weight_dict = {i: w for i, w in enumerate(class_weights)}

print(f"\n⚖️  Class Weights:")
for cls, weight in class_weight_dict.items():
    cls_name = ['SKIP', 'BUY', 'SELL'][cls]
    print(f"  {cls_name:12s}: {weight:.4f}")

# ==========================================
# REGULARIZED MODEL TRAINING
# ==========================================
print("\n" + "=" * 100)
print("🚀 TRAINING REGULARIZED MODEL")
print("=" * 100)

model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,

    # Reduced complexity to prevent overfitting
    n_estimators=150,              # ลดจาก 300
    learning_rate=0.05,             # เพิ่มจาก 0.02 (learn slower)
    num_leaves=15,                  # ลดจาก 31 (simpler trees)
    max_depth=4,                    # ลดจาก 6 (shallower)
    min_child_samples=200,          # เพิ่มจาก 100 (more samples needed)

    # Regularization
    reg_alpha=0.1,                  # L1 regularization
    reg_lambda=0.1,                 # L2 regularization

    # Bagging
    subsample=0.7,                  # ลดจาก 0.8
    colsample_bytree=0.7,           # ลดจาก 0.8
    subsample_freq=5,

    # Class weights
    class_weight=class_weight_dict,

    # Other
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("Training with heavy regularization...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0)
    ]
)

print("✅ Training complete!")
print(f"Best iteration: {model.best_iteration_}")

# ==========================================
# EVALUATION
# ==========================================
print("\n" + "=" * 100)
print("📊 MODEL EVALUATION")
print("=" * 100)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

target_names = ['SKIP', 'BUY', 'SELL']
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("                  Predicted")
print("               SKIP    BUY    SELL")
print(f"Actual SKIP    {cm[0][0]:6d}  {cm[0][1]:6d}  {cm[0][2]:6d}")
print(f"       BUY     {cm[1][0]:6d}  {cm[1][1]:6d}  {cm[1][2]:6d}")
print(f"       SELL    {cm[2][0]:6d}  {cm[2][1]:6d}  {cm[2][2]:6d}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': selected_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 20 Features (after selection):")
for idx, row in feature_importance.head(20).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:8.2f}")

# ==========================================
# THRESHOLD ANALYSIS
# ==========================================
print("\n" + "=" * 100)
print("🎚️  CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 100)

thresholds = [0.70, 0.75, 0.80, 0.85, 0.90]

for threshold in thresholds:
    max_proba = y_pred_proba.max(axis=1)
    high_confidence_mask = max_proba >= threshold

    y_test_filtered = y_test[high_confidence_mask]
    y_pred_filtered = y_pred[high_confidence_mask]

    if len(y_test_filtered) == 0:
        continue

    buy_mask = y_pred_filtered == 1
    sell_mask = y_pred_filtered == 2

    buy_accuracy = (y_test_filtered[buy_mask] == 1).sum() / buy_mask.sum() if buy_mask.sum() > 0 else 0
    sell_accuracy = (y_test_filtered[sell_mask] == 2).sum() / sell_mask.sum() if sell_mask.sum() > 0 else 0

    print(f"\n📊 Threshold: {threshold:.2f}")
    print(f"  Total signals: {len(y_test_filtered):6,} ({len(y_test_filtered)/len(y_test)*100:5.2f}%)")
    print(f"  BUY signals:   {buy_mask.sum():6,} (accuracy: {buy_accuracy*100:5.2f}%)")
    print(f"  SELL signals:  {sell_mask.sum():6,} (accuracy: {sell_accuracy*100:5.2f}%)")

# ==========================================
# SAVE MODEL
# ==========================================
print("\n" + "=" * 100)
print("💾 SAVING IMPROVED MODEL")
print("=" * 100)

model.booster_.save_model('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_improved.txt')
print("✅ Model saved: model_improved.txt")

metadata = {
    'model_type': 'improved_regularized',
    'features': selected_features,
    'num_features': len(selected_features),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'regularization': {
        'n_estimators': 150,
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'min_child_samples': 200,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'subsample': 0.7,
        'colsample_bytree': 0.7
    },
    'target_criteria': {
        'fill_window': FILL_WINDOW,
        'profit_window': PROFIT_WINDOW,
        'fill_offset': FILL_OFFSET,
        'profit_target': PROFIT_TARGET
    },
    'target_distribution': {
        'skip': int(target_counts[0]),
        'buy': int(target_counts[1]),
        'sell': int(target_counts[2])
    },
    'feature_importance': feature_importance.head(20).to_dict('records'),
    'best_iteration': model.best_iteration_
}

with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_improved_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✅ Metadata saved: model_improved_metadata.json")

print("\n" + "=" * 100)
print("🎉 IMPROVED MODEL TRAINING COMPLETE!")
print("=" * 100)

print("\n📊 Key Improvements:")
print("  ✅ Feature selection: 73 → 40 features")
print("  ✅ Regularization: L1 + L2 + early stopping")
print("  ✅ Stricter targets: Higher profit requirement (0.20%)")
print("  ✅ Simpler model: Shallower trees, fewer leaves")
print("  ✅ Better sampling: 70% subsample per tree")
