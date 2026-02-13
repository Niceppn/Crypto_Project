#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Model Training V3 SPOT - For V2 Multi-Stream Data (No Funding)
Uses V2 CSV export (OHLC + Order Book) — Spot ไม่มี Funding Rate
Maker Strategy: Limit Buy at best_bid → Take Profit

Features: 33 features from 4 groups
  1. Price/OHLC (candle patterns, MA, RSI, momentum, volatility)
  2. Volume Flow (buy/sell ratio, net_flow MAs, volume spike)
  3. Order Book (spread, imbalance, bid/ask ratio)
  4. Cross Features (imbalance×flow, spread×volume)

Usage:
  python "train_model_v3 spot.py" --data-dir /path/to/csv/folder
  python "train_model_v3 spot.py" --data-dir /path/to/csv/folder --profit-target 0.0003
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import sys
import argparse
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, classification_report, confusion_matrix

# ==========================================
# ⚙️ Configuration
# ==========================================
MODEL_DEST_PATH = os.path.join(os.path.dirname(__file__), "..", "models")

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}", flush=True)

# ==========================================
# 📂 1. Data Loading
# ==========================================
def load_v2_data(data_path):
    """Load V2 CSV from file or directory"""
    
    if os.path.isfile(data_path):
        # Single CSV file
        log(f"📄 Loading file: {os.path.basename(data_path)}")
        df = pd.read_csv(data_path)
        log(f"   {len(df):,} rows")
    elif os.path.isdir(data_path):
        # Directory of CSVs
        csv_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
        if not csv_files:
            raise ValueError(f"No CSV files found in {data_path}")
        log(f"📂 Found {len(csv_files)} CSV file(s)")
        all_dfs = []
        for f in csv_files:
            filepath = os.path.join(data_path, f)
            d = pd.read_csv(filepath)
            log(f"   📄 {f}: {len(d):,} rows")
            all_dfs.append(d)
        df = pd.concat(all_dfs, ignore_index=True)
    else:
        raise ValueError(f"Path not found: {data_path}")
    
    df.columns = df.columns.str.strip()
    
    # Detect column format
    # V2 export has: symbol,timestamp_ms,readable_time,open,high,low,close,...
    required_cols = ['open', 'high', 'low', 'close', 'best_bid', 'best_ask', 'book_imbalance']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing V2 columns: {missing}. Is this a V2 export?")
    
    # Sort by timestamp
    ts_col = 'timestamp_ms' if 'timestamp_ms' in df.columns else 'timestamp'
    df = df.sort_values(ts_col).reset_index(drop=True)
    
    log(f"📊 Total: {len(df):,} rows loaded")
    log(f"   Time range: {df['readable_time'].iloc[0]} → {df['readable_time'].iloc[-1]}")
    log(f"   Price range: ${df['close'].min():.1f} - ${df['close'].max():.1f}")
    
    return df

# ==========================================
# 🧠 2. Feature Engineering (~35 features)
# ==========================================
def create_features(df):
    """Create 33 features from V2 data (4 groups, no funding)"""
    log("🧠 Creating 33 features (SPOT, no funding)...")
    
    feat = df.copy()
    
    # ─────────────────────────────────────
    # Group 1: Price / OHLC Features (14)
    # ─────────────────────────────────────
    
    # Candle patterns
    feat['candle_body'] = feat['close'] - feat['open']                          # + = bullish, - = bearish
    feat['candle_range'] = feat['high'] - feat['low']                           # volatility in 1s
    feat['upper_shadow'] = feat['high'] - feat[['open', 'close']].max(axis=1)   # selling pressure
    feat['lower_shadow'] = feat[['open', 'close']].min(axis=1) - feat['low']    # buying support
    
    # Price change
    feat['price_change_1'] = feat['close'].pct_change(fill_method=None) * 100   # % change 1s
    feat['price_change_5'] = feat['close'].pct_change(5, fill_method=None) * 100
    
    # Moving Averages
    feat['ma5'] = feat['close'].rolling(5).mean()
    feat['ma15'] = feat['close'].rolling(15).mean()
    feat['ma30'] = feat['close'].rolling(30).mean()
    feat['dist_ma15'] = feat['close'] - feat['ma15']                            # distance from MA15
    feat['dist_ma30'] = feat['close'] - feat['ma30']
    
    # Volatility
    feat['std_5'] = feat['close'].rolling(5).std()
    feat['std_15'] = feat['close'].rolling(15).std()
    
    # RSI 14
    delta = feat['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    feat['rsi_14'] = 100 - (100 / (1 + rs))
    
    # Momentum
    feat['momentum_5'] = feat['close'] - feat['close'].shift(5)
    feat['momentum_15'] = feat['close'] - feat['close'].shift(15)
    
    # ─────────────────────────────────────
    # Group 2: Volume Flow Features (8)
    # ─────────────────────────────────────
    
    # Buy/Sell ratio
    feat['volume_ratio'] = feat['buy_volume'] / (feat['sell_volume'] + 1e-10)   # >1 = more buying
    
    # Net flow MAs
    feat['net_flow_ma5'] = feat['net_flow'].rolling(5).mean()
    feat['net_flow_ma15'] = feat['net_flow'].rolling(15).mean()
    feat['net_flow_diff'] = feat['net_flow'].diff()                             # acceleration
    
    # Volume MAs
    feat['volume_ma5'] = feat['total_volume'].rolling(5).mean()
    feat['volume_ma15'] = feat['total_volume'].rolling(15).mean()
    
    # Volume spike (unusual volume)
    feat['volume_spike'] = feat['total_volume'] / (feat['volume_ma15'] + 1e-10)
    
    # Cumulative flow (10s)
    feat['cum_flow_10'] = feat['net_flow'].rolling(10).sum()
    
    # ─────────────────────────────────────
    # Group 3: Order Book Features (8) ⭐
    # ─────────────────────────────────────
    
    # Spread
    feat['spread_ma5'] = feat['spread'].rolling(5).mean()
    feat['spread_change'] = feat['spread'].diff()                               # widening = uncertainty
    
    # Book imbalance
    feat['imbalance_ma5'] = feat['book_imbalance'].rolling(5).mean()
    feat['imbalance_ma15'] = feat['book_imbalance'].rolling(15).mean()
    feat['imbalance_change'] = feat['book_imbalance'].diff()                    # direction change
    
    # Bid/Ask qty ratio
    feat['bid_ask_ratio'] = feat['bid_qty'] / (feat['ask_qty'] + 1e-10)

    # ─────────────────────────────────────
    # Group 4: Cross Features (3)
    # ─────────────────────────────────────
    
    # Imbalance × Net Flow (both positive = strong buy signal)
    feat['imbalance_x_flow'] = feat['book_imbalance'] * feat['net_flow']
    
    # Spread × Volume (high volume + wide spread = potential breakout)
    feat['spread_x_volume'] = feat['spread'] * feat['total_volume']
    
    # Price-Volume correlation (10s rolling)
    feat['price_vol_corr'] = feat['close'].rolling(10).corr(feat['total_volume'])
    
    log(f"   ✅ Created features, shape: {feat.shape}")
    
    return feat

# ==========================================
# 🎯 3. Target Creation (Maker Strategy)
# ==========================================
def create_target(df, profit_target_pct, fill_window, profit_window):
    """
    Maker Strategy Target:
    - Entry: Place limit BUY at best_bid
    - Fill condition: future low <= best_bid within FILL_WINDOW seconds
    - Profit condition: future high > best_bid × (1 + profit_target) within PROFIT_WINDOW seconds
    - Target = 1 if FILLED AND PROFITABLE
    """
    log(f"🎯 Creating target (Maker Strategy)...")
    log(f"   Entry: best_bid | Target: +{profit_target_pct*100:.3f}%")
    log(f"   Fill window: {fill_window}s | Profit window: {profit_window}s")
    
    feat = df.copy()
    
    # Fill check: does price come down to our bid within FILL_WINDOW seconds?
    indexer_fill = pd.api.indexers.FixedForwardWindowIndexer(window_size=fill_window)
    feat['future_min_low'] = feat['low'].rolling(window=indexer_fill).min().shift(-1)
    is_filled = feat['future_min_low'] <= feat['best_bid']
    
    # Profit check: does price go up from our entry (best_bid) within PROFIT_WINDOW seconds?
    indexer_profit = pd.api.indexers.FixedForwardWindowIndexer(window_size=profit_window)
    feat['future_max_high'] = feat['high'].rolling(window=indexer_profit).max().shift(-1)
    target_price = feat['best_bid'] * (1 + profit_target_pct)  # ✅ Fixed bug from old code
    is_profit = feat['future_max_high'] > target_price
    
    # Combined: filled AND profitable
    feat['target'] = (is_filled & is_profit).astype(int)
    
    return feat

# ==========================================
# 📊 4. Feature List
# ==========================================
FEATURE_COLS = [
    # Group 1: Price/OHLC (14)
    'candle_body', 'candle_range', 'upper_shadow', 'lower_shadow',
    'price_change_1', 'price_change_5',
    'dist_ma15', 'dist_ma30',
    'std_5', 'std_15',
    'rsi_14',
    'momentum_5', 'momentum_15',
    'trade_count',

    # Group 2: Volume Flow (8)
    'volume_ratio', 'net_flow_ma5', 'net_flow_ma15', 'net_flow_diff',
    'volume_ma5', 'volume_spike', 'cum_flow_10',
    'total_volume',

    # Group 3: Order Book (8)
    'spread', 'spread_ma5', 'spread_change',
    'book_imbalance', 'imbalance_ma5', 'imbalance_ma15', 'imbalance_change',
    'bid_ask_ratio',

    # Group 4: Cross (3)
    'imbalance_x_flow', 'spread_x_volume', 'price_vol_corr',
]

# ==========================================
# 🚀 5. Training
# ==========================================
def train(df, args):
    """Train LightGBM model"""
    
    # Step 1: Create features
    df = create_features(df)
    
    # Step 2: Create target
    df = create_target(df, args.profit_target, args.fill_window, args.profit_window)
    
    feature_cols = FEATURE_COLS.copy()

    # Step 3: Drop NaN
    before = len(df)
    df.dropna(subset=feature_cols + ['target'], inplace=True)
    log(f"📋 Training set: {len(df):,} rows (dropped {before - len(df)} from rolling/shift)")
    log(f"📊 Features: {len(feature_cols)}")
    
    # Step 4: Check class balance
    y = df['target']
    counts = y.value_counts()
    pos = counts.get(1, 0)
    neg = counts.get(0, 0)
    
    log(f"📊 Class distribution:")
    log(f"   Win  (1) = {pos:,} ({pos/(pos+neg)*100:.1f}%)")
    log(f"   Loss (0) = {neg:,} ({neg/(pos+neg)*100:.1f}%)")
    
    if pos < 50:
        raise ValueError(f"Not enough positive samples ({pos}). Need at least 50. Try collecting more data.")
    
    # Step 5: Prepare X, y
    X = df[feature_cols]
    
    # Time-based split (no shuffle!)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    log(f"🔀 Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Scale pos weight for imbalance
    neg_train = y_train.value_counts().get(0, 1)
    pos_train = y_train.value_counts().get(1, 1)
    scale_weight = neg_train / pos_train if pos_train > 0 else 1.0
    
    log(f"⚖️  Scale pos weight: {scale_weight:.2f}")
    
    # Step 6: Train LightGBM
    log("🚀 Training LightGBM...")
    
    model = lgb.LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=31,
        max_depth=args.max_depth,
        scale_pos_weight=scale_weight,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    model.fit(X_train, y_train)
    
    # Step 7: Evaluate
    log("📊 Evaluating model...")
    
    probs = model.predict_proba(X_test)[:, 1]
    
    # Test multiple thresholds
    log(f"\n{'='*50}")
    log(f"📈 Precision at different confidence thresholds:")
    log(f"{'='*50}")
    
    best_threshold = args.confidence
    best_precision = 0
    
    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        preds = (probs >= threshold).astype(int)
        p = precision_score(y_test, preds, zero_division=0)
        signal_count = preds.sum()
        log(f"   Threshold {threshold:.2f}: Precision = {p*100:.1f}% | Signals = {signal_count}")
        
        if p > best_precision and signal_count >= 5:
            best_precision = p
            best_threshold = threshold
    
    # Final evaluation with best threshold
    final_preds = (probs >= best_threshold).astype(int)
    final_precision = precision_score(y_test, final_preds, zero_division=0)
    
    log(f"\n{'='*50}")
    log(f"🏆 Best Threshold: {best_threshold:.2f}")
    log(f"🎯 Final Precision: {final_precision*100:.1f}%")
    log(f"📍 Signals in test set: {final_preds.sum()}")
    log(f"{'='*50}")
    
    # Feature Importance
    log("\n📊 Top 15 Feature Importance:")
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for i, row in importance.head(15).iterrows():
        bar = '█' * int(row['importance'] / importance['importance'].max() * 30)
        log(f"   {row['feature']:25s} {row['importance']:6.0f} {bar}")
    
    # Step 8: Save model
    if not os.path.exists(MODEL_DEST_PATH):
        os.makedirs(MODEL_DEST_PATH)
    
    model_filename = f"v3_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    model_path = os.path.join(MODEL_DEST_PATH, model_filename)
    model.booster_.save_model(model_path)
    
    # Save metadata
    meta = {
        'version': 'v3',
        'features': feature_cols,
        'n_features': len(feature_cols),
        'best_threshold': best_threshold,
        'precision': final_precision,
        'training_rows': len(X_train),
        'test_rows': len(X_test),
        'profit_target_pct': args.profit_target,
        'fill_window': args.fill_window,
        'profit_window': args.profit_window,
        'class_distribution': {'win': int(pos), 'loss': int(neg)},
        'top_features': importance.head(10)['feature'].tolist(),
        'trained_at': datetime.now().isoformat()
    }
    
    meta_path = model_path.replace('.txt', '_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    log(f"\n✅ Model saved: {model_filename}")
    log(f"📋 Metadata saved: {os.path.basename(meta_path)}")
    
    return model, final_precision

# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description='Train AI Model V3 (V2 Multi-Stream Data)')
    
    # Data source (file or directory)
    default_csv = os.path.join(os.path.dirname(__file__), 'core', 'csv', 'BTCFDUSD_SPOT_v2_trades_2026-02-13.csv')
    parser.add_argument('--data', type=str, default=default_csv,
                        help='Path to CSV file or directory of CSV files')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='(Alias for --data) Directory containing V2 CSV files')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Filter by symbol (e.g. BTCUSDC). If not set, uses all data.')
    
    # Strategy parameters
    parser.add_argument('--profit-target', type=float, default=0.0003,
                        help='Profit target %% (default: 0.0003 = 0.03%%)')
    parser.add_argument('--fill-window', type=int, default=20,
                        help='Fill window in seconds (default: 20)')
    parser.add_argument('--profit-window', type=int, default=300,
                        help='Profit window in seconds (default: 300 = 5min)')
    parser.add_argument('--confidence', type=float, default=0.60,
                        help='Confidence threshold (default: 0.60)')
    # Model hyperparameters
    parser.add_argument('--n-estimators', type=int, default=500,
                        help='Number of trees (default: 500)')
    parser.add_argument('--learning-rate', type=float, default=0.01,
                        help='Learning rate (default: 0.01)')
    parser.add_argument('--max-depth', type=int, default=7,
                        help='Max tree depth (default: 7)')
    
    args = parser.parse_args()
    
    # Resolve data path (--data or --data-dir)
    data_path = args.data or args.data_dir
    if not data_path:
        parser.error("Must specify --data <path> or --data-dir <path>")
    
    log("=" * 60)
    log("🧠 AI Model Training V3 SPOT - 33 Features (No Funding)")
    log("=" * 60)
    log(f"📂 Data: {data_path}")
    if args.symbol:
        log(f"🪙 Symbol filter: {args.symbol}")
    log(f"🎯 Profit target: {args.profit_target*100:.3f}%")
    log(f"⏱️  Fill window: {args.fill_window}s | Profit window: {args.profit_window}s")
    log(f"📊 LightGBM: {args.n_estimators} trees, lr={args.learning_rate}, depth={args.max_depth}")
    log("🏪 Mode: SPOT (no funding features)")
    log("=" * 60)
    
    try:
        # Load data
        df = load_v2_data(data_path)
        
        # Filter by symbol
        if args.symbol:
            if 'symbol' in df.columns:
                df = df[df['symbol'] == args.symbol].reset_index(drop=True)
                log(f"🪙 Filtered to {args.symbol}: {len(df):,} rows")
                if len(df) == 0:
                    symbols = df['symbol'].unique() if 'symbol' in df.columns else []
                    raise ValueError(f"No data for symbol '{args.symbol}'. Available: {symbols}")
            else:
                log(f"⚠️  No 'symbol' column found, using all data")
        
        # Train
        model, precision = train(df, args)
        
        log(f"\n🎉 Training completed! Precision: {precision*100:.1f}%")
        
    except Exception as e:
        log(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
