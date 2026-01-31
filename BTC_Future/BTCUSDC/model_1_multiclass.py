"""
Model 1: Multi-Class Classification
3 Classes: SKIP (0), BUY_MAKER (1), SELL_MAKER (2)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import pickle
from feature_engineering import (
    load_and_prepare_data,
    aggregate_to_1s,
    create_features,
    create_buy_maker_target,
    create_sell_maker_target,
    get_feature_columns
)

# ==========================================
# CONFIGURATION
# ==========================================
RAW_FILE = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv'
MODEL_OUTPUT = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_multiclass.txt'
METADATA_OUTPUT = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_multiclass_metadata.json'

# Use all data or limit for testing
USE_ALL_DATA = True  # Set to False for quick testing
TEST_ROWS = 500000

print("=" * 80)
print("🎯 MODEL 1: MULTI-CLASS CLASSIFICATION")
print("=" * 80)

# ==========================================
# 1. LOAD AND PREPARE DATA
# ==========================================
df = load_and_prepare_data(RAW_FILE, nrows=None if USE_ALL_DATA else TEST_ROWS)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)

# ==========================================
# 2. CREATE TARGETS
# ==========================================
print("\n🎯 Creating multi-class targets...")

df_features['buy_target'] = create_buy_maker_target(df_features)
df_features['sell_target'] = create_sell_maker_target(df_features)

# Create 3-class target
# 0 = SKIP, 1 = BUY_MAKER, 2 = SELL_MAKER
df_features['target'] = 0
df_features.loc[df_features['buy_target'] == 1, 'target'] = 1
df_features.loc[df_features['sell_target'] == 1, 'target'] = 2

# Handle conflict: if both buy and sell are 1, choose based on order flow
both_targets = (df_features['buy_target'] == 1) & (df_features['sell_target'] == 1)
if both_targets.sum() > 0:
    print(f"⚠️  Found {both_targets.sum()} conflicting signals, resolving with order flow...")
    df_features.loc[both_targets & (df_features['net_flow'] > 0), 'target'] = 1
    df_features.loc[both_targets & (df_features['net_flow'] <= 0), 'target'] = 2

# Clean data
df_features = df_features.dropna()

print(f"\n📊 Target Distribution:")
target_counts = df_features['target'].value_counts().sort_index()
for target, count in target_counts.items():
    target_name = ['SKIP', 'BUY_MAKER', 'SELL_MAKER'][target]
    print(f"  {target_name:12s}: {count:8,} ({count/len(df_features)*100:5.2f}%)")

# ==========================================
# 3. PREPARE FEATURES
# ==========================================
feature_cols = get_feature_columns()
X = df_features[feature_cols]
y = df_features['target']

# Train-test split (temporal order preserved)
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"\n📈 Training set: {len(X_train):,} samples")
print(f"📉 Test set: {len(X_test):,} samples")

# ==========================================
# 4. CALCULATE CLASS WEIGHTS
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
    cls_name = ['SKIP', 'BUY_MAKER', 'SELL_MAKER'][cls]
    print(f"  {cls_name:12s}: {weight:.4f}")

# ==========================================
# 5. TRAIN MODEL
# ==========================================
print(f"\n🚀 Training LightGBM Multi-Class Classifier...")

model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=3,
    n_estimators=300,
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

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)]
)

print("✅ Training complete!")

# ==========================================
# 6. EVALUATE MODEL
# ==========================================
print("\n" + "=" * 80)
print("📊 MODEL EVALUATION")
print("=" * 80)

# Predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Classification report
target_names = ['SKIP', 'BUY_MAKER', 'SELL_MAKER']
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred, target_names=target_names, digits=4))

# Confusion matrix
print("\n🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print("                  Predicted")
print("               SKIP    BUY    SELL")
print(f"Actual SKIP    {cm[0][0]:6d}  {cm[0][1]:6d}  {cm[0][2]:6d}")
print(f"       BUY     {cm[1][0]:6d}  {cm[1][1]:6d}  {cm[1][2]:6d}")
print(f"       SELL    {cm[2][0]:6d}  {cm[2][1]:6d}  {cm[2][2]:6d}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔝 Top 20 Features:")
for idx, row in feature_importance.head(20).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:8.2f}")

# ==========================================
# 7. CONFIDENCE THRESHOLD ANALYSIS
# ==========================================
print("\n" + "=" * 80)
print("🎚️  CONFIDENCE THRESHOLD ANALYSIS")
print("=" * 80)

thresholds = [0.60, 0.65, 0.70, 0.75, 0.80]

for threshold in thresholds:
    # Get max probability for each prediction
    max_proba = y_pred_proba.max(axis=1)
    high_confidence_mask = max_proba >= threshold

    # Filter to high confidence predictions
    y_test_filtered = y_test[high_confidence_mask]
    y_pred_filtered = y_pred[high_confidence_mask]

    if len(y_test_filtered) == 0:
        continue

    # Calculate metrics for BUY and SELL only
    buy_mask = y_pred_filtered == 1
    sell_mask = y_pred_filtered == 2

    buy_accuracy = (y_test_filtered[buy_mask] == 1).sum() / buy_mask.sum() if buy_mask.sum() > 0 else 0
    sell_accuracy = (y_test_filtered[sell_mask] == 2).sum() / sell_mask.sum() if sell_mask.sum() > 0 else 0

    print(f"\n📊 Threshold: {threshold:.2f}")
    print(f"  Total signals: {len(y_test_filtered):6,} ({len(y_test_filtered)/len(y_test)*100:5.2f}% of test set)")
    print(f"  BUY signals:   {buy_mask.sum():6,} (accuracy: {buy_accuracy*100:5.2f}%)")
    print(f"  SELL signals:  {sell_mask.sum():6,} (accuracy: {sell_accuracy*100:5.2f}%)")

# ==========================================
# 8. SAVE MODEL
# ==========================================
print("\n" + "=" * 80)
print("💾 SAVING MODEL")
print("=" * 80)

# Save LightGBM model
model.booster_.save_model(MODEL_OUTPUT)
print(f"✅ Model saved to: {MODEL_OUTPUT}")

# Save metadata
metadata = {
    'model_type': 'multiclass_classification',
    'num_classes': 3,
    'class_names': target_names,
    'features': feature_cols,
    'num_features': len(feature_cols),
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'target_distribution': {
        'skip': int(target_counts[0]),
        'buy': int(target_counts[1]),
        'sell': int(target_counts[2])
    },
    'class_weights': class_weight_dict,
    'feature_importance': feature_importance.head(20).to_dict('records'),
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
print("🎉 MODEL 1 TRAINING COMPLETE!")
print("=" * 80)
