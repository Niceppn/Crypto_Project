"""
Model 3: Hybrid Approach
Combines Classification (direction) + Regression (magnitude)

Step 1: Multi-class classifier decides direction (SKIP/BUY/SELL)
Step 2: Regression models predict magnitude (how much upside/downside)
Step 3: Use both to make optimal trading decisions
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import json
from feature_engineering import get_feature_columns

print("=" * 80)
print("🎯 MODEL 3: HYBRID APPROACH")
print("=" * 80)

print("\nℹ️  This model combines:")
print("  1️⃣  Multi-Class Classifier (Model 1) - for direction")
print("  2️⃣  Dual Regressors (Model 2) - for magnitude")
print("\n✅ Both models should be trained first!")

# ==========================================
# LOAD PRE-TRAINED MODELS
# ==========================================
print("\n📂 Loading pre-trained models...")

MULTICLASS_MODEL = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_multiclass.txt'
BUY_REGRESSOR = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_buy_regressor.txt'
SELL_REGRESSOR = '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_sell_regressor.txt'

try:
    classifier = lgb.Booster(model_file=MULTICLASS_MODEL)
    buy_regressor = lgb.Booster(model_file=BUY_REGRESSOR)
    sell_regressor = lgb.Booster(model_file=SELL_REGRESSOR)
    print("✅ All models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("\n⚠️  Please train Model 1 and Model 2 first:")
    print("   1. python model_1_multiclass.py")
    print("   2. python model_2_dual_regression.py")
    exit(1)

# ==========================================
# HYBRID DECISION LOGIC
# ==========================================
class HybridMakerStrategy:
    """
    Hybrid trading strategy combining classification and regression
    """

    def __init__(self, classifier, buy_regressor, sell_regressor):
        self.classifier = classifier
        self.buy_regressor = buy_regressor
        self.sell_regressor = sell_regressor

        # Default parameters (can be optimized)
        self.classification_threshold = 0.70
        self.min_upside_pct = 0.18
        self.min_downside_pct = 0.18
        self.magnitude_weight = 0.4  # How much to weight magnitude vs confidence

    def predict(self, features):
        """
        Make trading decision using hybrid approach

        Returns:
        - signal: 0 (SKIP), 1 (BUY), 2 (SELL)
        - confidence: Combined confidence score (0-1)
        - metadata: Additional info for analysis
        """
        # Step 1: Get classification probabilities
        class_probs = self.classifier.predict(features)[0]  # [prob_skip, prob_buy, prob_sell]
        class_prediction = np.argmax(class_probs)
        class_confidence = class_probs[class_prediction]

        # Step 2: Get regression predictions
        predicted_upside = self.buy_regressor.predict(features)[0]
        predicted_downside = self.sell_regressor.predict(features)[0]

        # Step 3: Hybrid decision logic
        signal = 0  # Default: SKIP
        confidence = 0.0
        reason = "No signal"

        # Case 1: Classifier says BUY
        if class_prediction == 1 and class_confidence >= self.classification_threshold:
            # Check if magnitude confirms the signal
            if predicted_upside >= self.min_upside_pct:
                # Strong buy signal
                signal = 1
                # Combine classification confidence with magnitude
                magnitude_factor = min(predicted_upside / 0.30, 1.0)  # Cap at 0.30%
                confidence = (class_confidence * (1 - self.magnitude_weight) +
                            magnitude_factor * self.magnitude_weight)
                reason = f"BUY: Class={class_confidence:.3f}, Upside={predicted_upside:.3f}%"
            else:
                reason = f"BUY rejected: Upside too low ({predicted_upside:.3f}%)"

        # Case 2: Classifier says SELL
        elif class_prediction == 2 and class_confidence >= self.classification_threshold:
            # Check if magnitude confirms the signal
            if predicted_downside >= self.min_downside_pct:
                # Strong sell signal
                signal = 2
                magnitude_factor = min(predicted_downside / 0.30, 1.0)
                confidence = (class_confidence * (1 - self.magnitude_weight) +
                            magnitude_factor * self.magnitude_weight)
                reason = f"SELL: Class={class_confidence:.3f}, Downside={predicted_downside:.3f}%"
            else:
                reason = f"SELL rejected: Downside too low ({predicted_downside:.3f}%)"

        # Case 3: Classifier says SKIP but regression shows strong move
        elif class_prediction == 0:
            # Check if regression models strongly disagree
            if predicted_upside >= 0.25 and predicted_upside > predicted_downside * 2.0:
                # Regression overrides classifier
                signal = 1
                confidence = min(predicted_upside / 0.30, 0.85)  # Max 0.85 without classifier
                reason = f"BUY (regression override): Upside={predicted_upside:.3f}%"

            elif predicted_downside >= 0.25 and predicted_downside > predicted_upside * 2.0:
                signal = 2
                confidence = min(predicted_downside / 0.30, 0.85)
                reason = f"SELL (regression override): Downside={predicted_downside:.3f}%"
            else:
                reason = f"SKIP: Class={class_confidence:.3f}"

        metadata = {
            'class_probs': class_probs.tolist(),
            'class_prediction': int(class_prediction),
            'class_confidence': float(class_confidence),
            'predicted_upside': float(predicted_upside),
            'predicted_downside': float(predicted_downside),
            'reason': reason
        }

        return signal, confidence, metadata

    def get_dynamic_parameters(self, features, signal, confidence):
        """
        Get dynamic trading parameters based on prediction magnitude

        Returns:
        - entry_offset_pct: How far from current price to place limit order
        - take_profit_pct: Target profit percentage
        - stop_loss_pct: Stop loss percentage
        - position_size_multiplier: Position size adjustment
        """
        # Get magnitude prediction
        if signal == 1:  # BUY
            magnitude = self.buy_regressor.predict(features)[0]
        elif signal == 2:  # SELL
            magnitude = self.sell_regressor.predict(features)[0]
        else:
            return None

        # Base parameters
        base_entry_offset = 0.0005  # 0.05%
        base_tp = 0.0015  # 0.15%
        base_sl = 0.0010  # 0.10%
        base_size = 1.0

        # Adjust based on magnitude and confidence
        if magnitude >= 0.30:
            # Very strong signal - aggressive parameters
            tp_pct = 0.0025  # 0.25%
            entry_offset = 0.0003  # 0.03% (closer to market)
            sl_pct = 0.0012  # 0.12%
            size_mult = min(confidence * 1.5, 1.8)

        elif magnitude >= 0.22:
            # Strong signal - normal parameters
            tp_pct = 0.0020  # 0.20%
            entry_offset = 0.0005
            sl_pct = 0.0010
            size_mult = min(confidence * 1.2, 1.4)

        elif magnitude >= 0.18:
            # Moderate signal - conservative parameters
            tp_pct = 0.0015
            entry_offset = 0.0005
            sl_pct = 0.0010
            size_mult = confidence

        else:
            # Weak signal - very conservative
            tp_pct = 0.0012  # 0.12%
            entry_offset = 0.0007  # 0.07%
            sl_pct = 0.0010
            size_mult = max(confidence * 0.8, 0.5)

        return {
            'entry_offset_pct': entry_offset,
            'take_profit_pct': tp_pct,
            'stop_loss_pct': sl_pct,
            'position_size_multiplier': size_mult,
            'magnitude': magnitude
        }

# ==========================================
# CREATE STRATEGY INSTANCE
# ==========================================
strategy = HybridMakerStrategy(classifier, buy_regressor, sell_regressor)

print("\n" + "=" * 80)
print("⚙️  HYBRID STRATEGY CONFIGURATION")
print("=" * 80)
print(f"  Classification threshold: {strategy.classification_threshold:.2f}")
print(f"  Min upside % (BUY):       {strategy.min_upside_pct:.2f}%")
print(f"  Min downside % (SELL):    {strategy.min_downside_pct:.2f}%")
print(f"  Magnitude weight:         {strategy.magnitude_weight:.2f}")

# ==========================================
# TEST ON SAMPLE DATA
# ==========================================
print("\n" + "=" * 80)
print("🧪 TESTING HYBRID STRATEGY")
print("=" * 80)

print("\nLoading test data...")
from feature_engineering import load_and_prepare_data, aggregate_to_1s, create_features

df = load_and_prepare_data(
    '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv',
    nrows=100000
)
df_1s = aggregate_to_1s(df)
df_features = create_features(df_1s)
df_features = df_features.dropna()

# Test on last 1000 samples
feature_cols = get_feature_columns()
X_test = df_features[feature_cols].iloc[-1000:]

print(f"Testing on {len(X_test)} samples...")

signals = []
confidences = []
reasons = []

for idx, features in X_test.iterrows():
    signal, confidence, metadata = strategy.predict(features.values.reshape(1, -1))
    signals.append(signal)
    confidences.append(confidence)
    reasons.append(metadata['reason'])

# Analyze results
signals_series = pd.Series(signals)
print(f"\n📊 Signal Distribution:")
print(f"  SKIP:  {(signals_series == 0).sum():5d} ({(signals_series == 0).sum()/len(signals_series)*100:5.2f}%)")
print(f"  BUY:   {(signals_series == 1).sum():5d} ({(signals_series == 1).sum()/len(signals_series)*100:5.2f}%)")
print(f"  SELL:  {(signals_series == 2).sum():5d} ({(signals_series == 2).sum()/len(signals_series)*100:5.2f}%)")

print(f"\n🎯 Confidence Statistics:")
buy_conf = [c for s, c in zip(signals, confidences) if s == 1]
sell_conf = [c for s, c in zip(signals, confidences) if s == 2]

if buy_conf:
    print(f"  BUY signals:  Mean={np.mean(buy_conf):.3f}, Min={np.min(buy_conf):.3f}, Max={np.max(buy_conf):.3f}")
if sell_conf:
    print(f"  SELL signals: Mean={np.mean(sell_conf):.3f}, Min={np.min(sell_conf):.3f}, Max={np.max(sell_conf):.3f}")

print(f"\n📝 Sample Decisions (last 20 signals):")
for i, (signal, conf, reason) in enumerate(list(zip(signals, confidences, reasons))[-20:]):
    signal_name = ['SKIP', 'BUY', 'SELL'][signal]
    print(f"  {signal_name:5s} (conf={conf:.3f}): {reason}")

# ==========================================
# SAVE HYBRID STRATEGY CONFIG
# ==========================================
print("\n" + "=" * 80)
print("💾 SAVING HYBRID STRATEGY")
print("=" * 80)

hybrid_config = {
    'model_type': 'hybrid',
    'version': '1.0',
    'models': {
        'classifier': MULTICLASS_MODEL,
        'buy_regressor': BUY_REGRESSOR,
        'sell_regressor': SELL_REGRESSOR
    },
    'parameters': {
        'classification_threshold': strategy.classification_threshold,
        'min_upside_pct': strategy.min_upside_pct,
        'min_downside_pct': strategy.min_downside_pct,
        'magnitude_weight': strategy.magnitude_weight
    },
    'features': feature_cols,
    'decision_logic': {
        'step1': 'Classification model predicts direction and confidence',
        'step2': 'Regression models predict magnitude (upside/downside)',
        'step3': 'Combine both: signal only if direction AND magnitude agree',
        'override': 'Strong regression signal can override classifier SKIP'
    },
    'dynamic_parameters': {
        'entry_offset': '0.03-0.07% based on magnitude',
        'take_profit': '0.12-0.25% based on magnitude',
        'stop_loss': '0.10-0.12% based on magnitude',
        'position_size': '0.5x-1.8x based on confidence and magnitude'
    }
}

with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/model_hybrid_config.json', 'w') as f:
    json.dump(hybrid_config, f, indent=2)

print("✅ Hybrid strategy config saved!")

# Save strategy class for later use
import pickle
with open('/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/hybrid_strategy.pkl', 'wb') as f:
    pickle.dump(strategy, f)

print("✅ Hybrid strategy object saved!")

print("\n" + "=" * 80)
print("🎉 MODEL 3 (HYBRID) READY!")
print("=" * 80)
print("\n💡 Usage in trading bot:")
print("  1. Load strategy: strategy = pickle.load(open('hybrid_strategy.pkl', 'rb'))")
print("  2. Get signal: signal, confidence, metadata = strategy.predict(features)")
print("  3. Get params: params = strategy.get_dynamic_parameters(features, signal, confidence)")
print("  4. Place order with dynamic TP/SL/Size")
