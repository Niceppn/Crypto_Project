"""
Feature Engineering Module for Maker Strategy
Shared feature engineering for all models
"""

import pandas as pd
import numpy as np
from typing import Tuple

def load_and_prepare_data(file_path: str, nrows: int = None) -> pd.DataFrame:
    """Load and prepare raw trade data"""
    print(f"📂 Loading data from {file_path}...")
    df = pd.read_csv(file_path, nrows=nrows)

    # Convert timestamp
    df['datetime'] = pd.to_datetime(df['timestamp_ms'], unit='ms')
    df = df.set_index('datetime')

    # Signed volume
    df['signed_volume'] = df.apply(
        lambda x: x['quantity'] if x['side'] == 'BUY' else -x['quantity'],
        axis=1
    )

    print(f"✅ Loaded {len(df):,} trades")
    return df

def aggregate_to_1s(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tick data to 1-second OHLCV"""
    print("⏱️  Aggregating to 1-second candles...")

    df_1s = df.resample('1s').agg({
        'price': ['first', 'last', 'min', 'max'],
        'quantity': 'sum',
        'signed_volume': 'sum',
        'side': 'count'
    })

    df_1s.columns = ['open', 'close', 'low', 'high', 'total_volume', 'net_flow', 'trade_count']

    # Remove empty periods
    df_1s = df_1s[df_1s['trade_count'] > 0].copy()

    # Calculate buy/sell volumes
    buy_volume = df[df['side'] == 'BUY'].resample('1s')['quantity'].sum()
    sell_volume = df[df['side'] == 'SELL'].resample('1s')['quantity'].sum()

    df_1s['buy_volume'] = buy_volume.reindex(df_1s.index, fill_value=0)
    df_1s['sell_volume'] = sell_volume.reindex(df_1s.index, fill_value=0)

    print(f"✅ Created {len(df_1s):,} 1-second candles")
    return df_1s

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive feature set (30+ features)

    Categories:
    1. Order Flow Features
    2. Volume Profile Features
    3. Price Action Features
    4. Microstructure Features
    5. Technical Indicators
    6. Market Regime Features
    """
    print("🔧 Creating features...")
    df = df.copy()

    # ========================================
    # 1. ORDER FLOW FEATURES
    # ========================================
    # Basic ratios
    df['buy_sell_ratio'] = df['buy_volume'] / (df['sell_volume'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_volume'] - df['sell_volume']) / (df['total_volume'] + 1e-10)
    df['aggressive_buy_ratio'] = df['buy_volume'] / (df['total_volume'] + 1e-10)

    # Time-series momentum
    df['net_flow_ma5'] = df['net_flow'].rolling(5).mean()
    df['net_flow_ma10'] = df['net_flow'].rolling(10).mean()
    df['net_flow_ma30'] = df['net_flow'].rolling(30).mean()
    df['net_flow_std5'] = df['net_flow'].rolling(5).std()
    df['net_flow_acceleration'] = df['net_flow'].diff().diff()

    # Cumulative volume delta (CVD)
    df['cvd_60'] = df['net_flow'].rolling(60).sum()
    df['cvd_300'] = df['net_flow'].rolling(300).sum()

    # ========================================
    # 2. VOLUME PROFILE FEATURES
    # ========================================
    df['volume_ma5'] = df['total_volume'].rolling(5).mean()
    df['volume_ma15'] = df['total_volume'].rolling(15).mean()
    df['volume_ma30'] = df['total_volume'].rolling(30).mean()
    df['volume_std5'] = df['total_volume'].rolling(5).std()
    df['volume_surge'] = df['total_volume'] / (df['volume_ma15'] + 1e-10)

    # Relative volume (compared to 1-hour average)
    df['relative_volume'] = df['total_volume'] / (df['total_volume'].rolling(3600).mean() + 1e-10)

    # Volume by side percentages
    df['buy_volume_pct'] = df['buy_volume'] / (df['total_volume'] + 1e-10) * 100
    df['sell_volume_pct'] = df['sell_volume'] / (df['total_volume'] + 1e-10) * 100

    # ========================================
    # 3. PRICE ACTION FEATURES
    # ========================================
    # OHLC patterns
    df['hl_range'] = df['high'] - df['low']
    df['oc_diff'] = df['close'] - df['open']
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['body_to_range_ratio'] = df['body_size'] / (df['hl_range'] + 1e-10)

    # Momentum
    df['price_velocity'] = df['close'].diff()
    df['price_acceleration'] = df['price_velocity'].diff()
    df['price_change_pct_1'] = df['close'].pct_change() * 100
    df['price_change_pct_5'] = df['close'].pct_change(5) * 100
    df['price_change_pct_15'] = df['close'].pct_change(15) * 100
    df['momentum_5'] = df['close'].diff(5)
    df['momentum_15'] = df['close'].diff(15)

    # Moving averages
    df['ma5'] = df['close'].rolling(5).mean()
    df['ma15'] = df['close'].rolling(15).mean()
    df['ma30'] = df['close'].rolling(30).mean()
    df['dist_from_ma5'] = (df['close'] - df['ma5']) / (df['ma5'] + 1e-10) * 100
    df['dist_from_ma15'] = (df['close'] - df['ma15']) / (df['ma15'] + 1e-10) * 100
    df['dist_from_ma30'] = (df['close'] - df['ma30']) / (df['ma30'] + 1e-10) * 100
    df['ma5_slope'] = df['ma5'].diff()

    # ========================================
    # 4. MICROSTRUCTURE FEATURES
    # ========================================
    df['avg_trade_size'] = df['total_volume'] / (df['trade_count'] + 1e-10)
    df['trade_frequency'] = df['trade_count']

    # Price tick direction
    df['price_tick'] = np.sign(df['close'].diff())
    df['tick_momentum'] = df['price_tick'].rolling(5).sum()

    # ========================================
    # 5. TECHNICAL INDICATORS
    # ========================================
    # RSI (5 and 14 period)
    def calculate_rsi(series, period):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    df['rsi_5'] = calculate_rsi(df['close'], 5)
    df['rsi_14'] = calculate_rsi(df['close'], 14)

    # Bollinger Bands position
    bb_period = 20
    df['bb_middle'] = df['close'].rolling(bb_period).mean()
    df['bb_std'] = df['close'].rolling(bb_period).std()
    df['bb_upper'] = df['bb_middle'] + 2 * df['bb_std']
    df['bb_lower'] = df['bb_middle'] - 2 * df['bb_std']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

    # ATR (Average True Range)
    tr1 = df['hl_range']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    df['tr'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['atr_5'] = df['tr'].rolling(5).mean()

    # ========================================
    # 6. MARKET REGIME FEATURES
    # ========================================
    # Volatility regime
    df['volatility_60'] = df['close'].pct_change().rolling(60).std() * 100
    df['volatility_regime'] = (df['volatility_60'] - df['volatility_60'].rolling(300).mean()) / (df['volatility_60'].rolling(300).std() + 1e-10)

    # Trend strength (ADX-like)
    df['price_range_60'] = df['high'].rolling(60).max() - df['low'].rolling(60).min()
    df['trend_strength'] = abs(df['close'].rolling(60).mean().diff(5)) / (df['price_range_60'] + 1e-10)

    # ========================================
    # 7. ADVANCED MICROSTRUCTURE FEATURES
    # ========================================
    print("🔬 Creating advanced microstructure features...")

    # Order Book Imbalance Proxy (using aggressive vs passive volume)
    # In real orderbook: imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
    # Proxy: BUY taker = aggressive buy = hitting asks, SELL maker = passive sell = on bids
    df['aggressive_buy_volume'] = df['buy_volume']  # Takers hitting asks
    df['passive_sell_volume'] = df['sell_volume']   # Makers on bids
    df['order_book_imbalance_proxy'] = (df['aggressive_buy_volume'] - df['passive_sell_volume']) / (df['total_volume'] + 1e-10)
    df['ob_imbalance_ma5'] = df['order_book_imbalance_proxy'].rolling(5).mean()
    df['ob_imbalance_ma15'] = df['order_book_imbalance_proxy'].rolling(15).mean()

    # Spread Proxy (HL range as % of price)
    df['spread_proxy'] = (df['high'] - df['low']) / df['close'] * 100
    df['spread_ma5'] = df['spread_proxy'].rolling(5).mean()
    df['spread_volatility'] = df['spread_proxy'].rolling(15).std()

    # Liquidity Depth Proxy (volume concentration)
    # High volume + low spread = good liquidity
    df['liquidity_score'] = df['total_volume'] / (df['spread_proxy'] + 1e-10)
    df['liquidity_ma15'] = df['liquidity_score'].rolling(15).mean()

    # Trade Size Distribution
    df['large_trade_ratio'] = (df['total_volume'] / (df['trade_count'] + 1)) / (df['avg_trade_size'].rolling(60).mean() + 1e-10)
    df['large_trade_indicator'] = (df['large_trade_ratio'] > 2.0).astype(int)

    # Volume Concentration (Gini-like)
    # High concentration = few large trades, Low = many small trades
    df['volume_concentration'] = df['total_volume'] / (df['trade_count'] * df['avg_trade_size'] + 1e-10)

    # ========================================
    # 8. TIME-BASED FEATURES
    # ========================================
    print("⏰ Creating time-based features...")

    # Hour of day (cyclical encoding)
    df['hour'] = df.index.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Minute within hour
    df['minute'] = df.index.minute
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)

    # Day of week (cyclical)
    df['dayofweek'] = df.index.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # ========================================
    # 9. MOMENTUM & ACCELERATION FEATURES
    # ========================================
    print("🚀 Creating momentum features...")

    # Price momentum at different scales
    df['momentum_3'] = df['close'].diff(3)
    df['momentum_10'] = df['close'].diff(10)
    df['momentum_30'] = df['close'].diff(30)

    # Acceleration (2nd derivative)
    df['acceleration_5'] = df['momentum_5'].diff()
    df['acceleration_15'] = df['momentum_15'].diff()

    # Momentum strength
    df['momentum_strength'] = abs(df['momentum_5']) / (df['atr_5'] + 1e-10)

    # ========================================
    # 10. ORDER FLOW PRESSURE FEATURES
    # ========================================
    print("💪 Creating order flow pressure features...")

    # Buying/Selling pressure intensity
    df['buy_pressure'] = df['buy_volume'] / (df['total_volume'].rolling(30).mean() + 1e-10)
    df['sell_pressure'] = df['sell_volume'] / (df['total_volume'].rolling(30).mean() + 1e-10)
    df['pressure_diff'] = df['buy_pressure'] - df['sell_pressure']

    # Sustained flow (CVD momentum)
    df['cvd_momentum_5'] = df['cvd_60'].diff(5)
    df['cvd_momentum_15'] = df['cvd_60'].diff(15)

    # Flow reversal detection
    df['flow_reversal'] = ((df['net_flow'] * df['net_flow'].shift(1)) < 0).astype(int)

    print(f"✅ Created {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'total_volume', 'net_flow', 'trade_count']])} features")

    return df

def create_buy_maker_target(df: pd.DataFrame,
                            fill_window: int = 10,
                            profit_window: int = 60,
                            fill_offset_pct: float = 0.0005,
                            profit_pct: float = 0.0015) -> pd.Series:
    """
    Create BUY MAKER target

    Conditions:
    1. Within fill_window seconds: price must dip to fill order (close * (1 - fill_offset_pct))
    2. After filled, within profit_window seconds: price must reach TP (close * (1 + profit_pct))
    """
    # Condition 1: Order gets filled
    future_min_low = df['low'].rolling(window=fill_window, min_periods=1).min().shift(-1)
    entry_price = df['close'] * (1 - fill_offset_pct)
    is_filled = future_min_low <= entry_price

    # Condition 2: Reaches profit target after fill
    future_max_high = df['high'].rolling(window=profit_window, min_periods=1).max().shift(-(fill_window + 1))
    target_price = df['close'] * (1 + profit_pct)
    is_profit = future_max_high >= target_price

    return (is_filled & is_profit).astype(int)

def create_sell_maker_target(df: pd.DataFrame,
                             fill_window: int = 10,
                             profit_window: int = 60,
                             fill_offset_pct: float = 0.0005,
                             profit_pct: float = 0.0015) -> pd.Series:
    """
    Create SELL MAKER target

    Conditions:
    1. Within fill_window seconds: price must rise to fill order (close * (1 + fill_offset_pct))
    2. After filled, within profit_window seconds: price must reach TP (close * (1 - profit_pct))
    """
    # Condition 1: Order gets filled
    future_max_high = df['high'].rolling(window=fill_window, min_periods=1).max().shift(-1)
    entry_price = df['close'] * (1 + fill_offset_pct)
    is_filled = future_max_high >= entry_price

    # Condition 2: Reaches profit target after fill
    future_min_low = df['low'].rolling(window=profit_window, min_periods=1).min().shift(-(fill_window + 1))
    target_price = df['close'] * (1 - profit_pct)
    is_profit = future_min_low <= target_price

    return (is_filled & is_profit).astype(int)

def create_regression_targets(df: pd.DataFrame,
                              window: int = 300) -> Tuple[pd.Series, pd.Series]:
    """
    Create regression targets for upside and downside potential

    Returns:
    - upside_pct: Maximum % gain possible in next window seconds
    - downside_pct: Maximum % loss possible in next window seconds (positive value)
    """
    # Maximum upside
    future_max_high = df['high'].rolling(window=window, min_periods=1).max().shift(-1)
    upside_pct = (future_max_high - df['close']) / df['close'] * 100

    # Maximum downside (as positive value)
    future_min_low = df['low'].rolling(window=window, min_periods=1).min().shift(-1)
    downside_pct = (df['close'] - future_min_low) / df['close'] * 100

    return upside_pct, downside_pct

def get_feature_columns() -> list:
    """Return list of feature columns to use for training"""
    return [
        # Order Flow (11 features)
        'net_flow', 'buy_sell_ratio', 'order_flow_imbalance', 'aggressive_buy_ratio',
        'net_flow_ma5', 'net_flow_ma10', 'net_flow_ma30', 'net_flow_std5', 'net_flow_acceleration',
        'cvd_60', 'cvd_300',

        # Volume (9 features)
        'total_volume', 'volume_ma5', 'volume_ma15', 'volume_ma30', 'volume_std5', 'volume_surge',
        'relative_volume', 'buy_volume_pct', 'sell_volume_pct',

        # Price Action (14 features)
        'hl_range', 'oc_diff', 'body_to_range_ratio', 'upper_wick', 'lower_wick',
        'price_velocity', 'price_acceleration',
        'price_change_pct_1', 'price_change_pct_5', 'price_change_pct_15',
        'momentum_5', 'momentum_15',
        'dist_from_ma5', 'dist_from_ma15', 'dist_from_ma30', 'ma5_slope',

        # Microstructure (4 features)
        'trade_count', 'avg_trade_size', 'trade_frequency', 'tick_momentum',

        # Technical Indicators (4 features)
        'rsi_5', 'rsi_14', 'bb_position', 'atr_5',

        # Market Regime (2 features)
        'volatility_regime', 'trend_strength',

        # Advanced Microstructure (9 features) - NEW
        'order_book_imbalance_proxy', 'ob_imbalance_ma5', 'ob_imbalance_ma15',
        'spread_proxy', 'spread_ma5', 'spread_volatility',
        'liquidity_score', 'liquidity_ma15',
        'volume_concentration',

        # Time-based (6 features) - NEW
        'hour_sin', 'hour_cos', 'minute_sin', 'minute_cos', 'dow_sin', 'dow_cos',

        # Momentum & Acceleration (6 features) - NEW
        'momentum_3', 'momentum_10', 'momentum_30',
        'acceleration_5', 'acceleration_15', 'momentum_strength',

        # Order Flow Pressure (6 features) - NEW
        'buy_pressure', 'sell_pressure', 'pressure_diff',
        'cvd_momentum_5', 'cvd_momentum_15', 'flow_reversal'
    ]

if __name__ == "__main__":
    # Test the feature engineering pipeline
    print("=" * 60)
    print("🧪 Testing Feature Engineering Pipeline")
    print("=" * 60)

    # Load sample data
    df = load_and_prepare_data(
        '/Users/Macbook/Collect_Crypto/BTC_Future/BTCUSDC/btcusdc_training_data.csv',
        nrows=100000
    )

    # Aggregate
    df_1s = aggregate_to_1s(df)

    # Create features
    df_features = create_features(df_1s)

    # Create targets
    df_features['buy_target'] = create_buy_maker_target(df_features)
    df_features['sell_target'] = create_sell_maker_target(df_features)
    df_features['upside_pct'], df_features['downside_pct'] = create_regression_targets(df_features)

    # Clean data
    df_features = df_features.dropna()

    print("\n" + "=" * 60)
    print("📊 FEATURE STATISTICS")
    print("=" * 60)
    print(f"Total candles: {len(df_features):,}")
    print(f"Feature count: {len(get_feature_columns())}")
    print(f"\nTarget distribution:")
    print(f"  BUY opportunities: {df_features['buy_target'].sum():,} ({df_features['buy_target'].mean()*100:.2f}%)")
    print(f"  SELL opportunities: {df_features['sell_target'].sum():,} ({df_features['sell_target'].mean()*100:.2f}%)")
    print(f"\nRegression targets:")
    print(f"  Avg upside: {df_features['upside_pct'].mean():.4f}%")
    print(f"  Avg downside: {df_features['downside_pct'].mean():.4f}%")

    print("\n✅ Feature engineering pipeline working correctly!")
