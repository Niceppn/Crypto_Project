"""
Real-time Feature Engineering for Trading Bot
คำนวณ 40 features จาก buffer สำหรับ prediction
"""

import pandas as pd
import numpy as np

def calculate_features(data_list):
    """
    คำนวณ 40 features จาก list ของ second-by-second data

    Parameters:
        data_list: list of dict, แต่ละ dict มี keys:
                   'close', 'low', 'high', 'net_flow', 'total_volume', 'trade_count', 'ts'

    Returns:
        dict ของ features 40 ตัว, หรือ None ถ้า data ไม่พอ
    """

    if len(data_list) < 30:  # ต้องมีอย่างน้อย 30 seconds
        return None

    df = pd.DataFrame(data_list)

    # ตรวจสอบว่ามี columns ครบไหม
    required_cols = ['close', 'low', 'total_volume', 'net_flow', 'trade_count']
    for col in required_cols:
        if col not in df.columns:
            return None

    try:
        # ใช้ row สุดท้าย (ปัจจุบัน)
        last_idx = len(df) - 1

        features = {}

        # ==========================================
        # ORDER FLOW FEATURES
        # ==========================================
        features['net_flow'] = df['net_flow'].iloc[last_idx]

        # Rolling MA
        features['net_flow_ma5'] = df['net_flow'].rolling(5, min_periods=1).mean().iloc[last_idx]
        features['net_flow_ma10'] = df['net_flow'].rolling(10, min_periods=1).mean().iloc[last_idx]
        features['net_flow_ma30'] = df['net_flow'].rolling(30, min_periods=1).mean().iloc[last_idx]

        # Std & Acceleration
        features['net_flow_std5'] = df['net_flow'].rolling(5, min_periods=1).std().iloc[last_idx]
        features['net_flow_acceleration'] = df['net_flow'].diff().iloc[last_idx] if len(df) > 1 else 0

        # CVD (Cumulative Volume Delta)
        features['cvd_60'] = df['net_flow'].tail(60).sum() if len(df) >= 60 else df['net_flow'].sum()
        features['cvd_300'] = df['net_flow'].tail(300).sum() if len(df) >= 300 else df['net_flow'].sum()

        # ==========================================
        # VOLUME FEATURES
        # ==========================================
        features['total_volume'] = df['total_volume'].iloc[last_idx]
        features['volume_ma5'] = df['total_volume'].rolling(5, min_periods=1).mean().iloc[last_idx]
        features['volume_ma15'] = df['total_volume'].rolling(15, min_periods=1).mean().iloc[last_idx]
        features['volume_ma30'] = df['total_volume'].rolling(30, min_periods=1).mean().iloc[last_idx]
        features['volume_std5'] = df['total_volume'].rolling(5, min_periods=1).std().iloc[last_idx]

        # Volume surge
        vol_ma = df['total_volume'].rolling(30, min_periods=1).mean().iloc[last_idx]
        features['volume_surge'] = df['total_volume'].iloc[last_idx] / vol_ma if vol_ma > 0 else 1.0

        # Relative volume
        features['relative_volume'] = features['volume_surge']

        # ==========================================
        # PRICE FEATURES
        # ==========================================
        # High-Low range (สร้าง column ก่อน)
        if 'high' in df.columns:
            df['hl_range'] = df['high'] - df['low']
        else:
            df['hl_range'] = 0

        features['hl_range'] = df['hl_range'].iloc[last_idx]

        # Price changes
        features['price_change_pct_1'] = df['close'].pct_change(1).iloc[last_idx] * 100
        features['price_change_pct_15'] = df['close'].pct_change(15).iloc[last_idx] * 100 if len(df) >= 15 else 0

        # Distance from MA
        ma5 = df['close'].rolling(5, min_periods=1).mean().iloc[last_idx]
        ma15 = df['close'].rolling(15, min_periods=1).mean().iloc[last_idx]
        ma30 = df['close'].rolling(30, min_periods=1).mean().iloc[last_idx]

        features['dist_from_ma5'] = df['close'].iloc[last_idx] - ma5
        features['dist_from_ma15'] = df['close'].iloc[last_idx] - ma15
        features['dist_from_ma30'] = df['close'].iloc[last_idx] - ma30

        # MA slope
        if len(df) >= 2:
            features['ma5_slope'] = df['close'].rolling(5, min_periods=1).mean().diff().iloc[last_idx]
        else:
            features['ma5_slope'] = 0

        # ==========================================
        # MOMENTUM FEATURES
        # ==========================================
        features['momentum_3'] = df['close'].diff(3).iloc[last_idx] if len(df) >= 3 else 0
        features['momentum_10'] = df['close'].diff(10).iloc[last_idx] if len(df) >= 10 else 0
        features['momentum_30'] = df['close'].diff(30).iloc[last_idx] if len(df) >= 30 else 0

        # Acceleration
        if len(df) >= 6:
            mom5 = df['close'].diff(5)
            features['acceleration_15'] = mom5.diff().iloc[last_idx]
        else:
            features['acceleration_15'] = 0

        # ==========================================
        # TECHNICAL INDICATORS
        # ==========================================
        # ATR (simplified - using high-low range)
        # hl_range column ถูกสร้างไว้แล้วด้านบน
        if len(df) >= 5:
            features['atr_5'] = df['hl_range'].rolling(5, min_periods=1).mean().iloc[last_idx]
        else:
            features['atr_5'] = features['hl_range']

        # Spread proxy (simplified - using price velocity)
        features['spread_proxy'] = abs(df['close'].diff().iloc[last_idx]) if len(df) > 1 else 0
        features['spread_ma5'] = abs(df['close'].diff()).rolling(5, min_periods=1).mean().iloc[last_idx]
        features['spread_volatility'] = abs(df['close'].diff()).rolling(5, min_periods=1).std().iloc[last_idx]

        # Liquidity score (proxy using volume)
        features['liquidity_ma15'] = df['total_volume'].rolling(15, min_periods=1).sum().iloc[last_idx]

        # CVD momentum
        cvd = df['net_flow'].cumsum()
        if len(df) >= 5:
            features['cvd_momentum_5'] = cvd.diff(5).iloc[last_idx]
            features['cvd_momentum_15'] = cvd.diff(15).iloc[last_idx] if len(df) >= 15 else 0
        else:
            features['cvd_momentum_5'] = 0
            features['cvd_momentum_15'] = 0

        # Trade count
        features['trade_count'] = df['trade_count'].iloc[last_idx]

        # ==========================================
        # FILL MISSING (ถ้ามี NaN ให้เป็น 0)
        # ==========================================
        for key in features:
            if pd.isna(features[key]):
                features[key] = 0.0

        return features

    except Exception as e:
        print(f"❌ Feature calculation error: {e}")
        return None


def get_selected_features():
    """
    Return list ของ 40 features ที่ model ใช้ (ตามลำดับที่ถูกต้อง)
    """
    return [
        'volume_ma30',
        'momentum_30',
        'liquidity_ma15',
        'cvd_60',
        'spread_volatility',
        'dist_from_ma15',
        'cvd_300',
        'volume_ma15',
        'cvd_momentum_15',
        'price_change_pct_15',
        'net_flow_ma30',
        'atr_5',
        'spread_ma5',
        'momentum_10',
        'net_flow_ma10',
        'momentum_3',
        'dist_from_ma5',
        'volume_ma5',
        'acceleration_15',
        'net_flow',
        'ma5_slope',
        'cvd_momentum_5',
        'net_flow_ma5',
        'dist_from_ma30',
        'volume_std5',
        'total_volume',
        'net_flow_std5',
        'spread_proxy',
        'relative_volume',
        'hl_range',
        'volume_surge',
        'trade_count',
        'net_flow_acceleration',
        'price_change_pct_1',
        # Note: เหลืออีก 6 features แต่ไม่มีใน simplified version
        # จะ fill ด้วย 0 หรือใช้ proxy
        'hour_sin',      # dummy - ใช้ 0
        'hour_cos',      # dummy - ใช้ 0
        'minute_sin',    # dummy - ใช้ 0
        'minute_cos',    # dummy - ใช้ 0
        'dow_sin',       # dummy - ใช้ 0
        'dow_cos'        # dummy - ใช้ 0
    ]


def prepare_features_for_model(features_dict):
    """
    จัดเรียง features ตามลำดับที่ model ต้องการ (40 features)

    Returns:
        list of values ตามลำดับที่ถูกต้อง
    """
    selected_features = get_selected_features()

    # สร้าง list ตามลำดับ
    feature_values = []
    for feat_name in selected_features:
        if feat_name in features_dict:
            feature_values.append(features_dict[feat_name])
        else:
            # Features ที่ไม่มี (เช่น time features) ใช้ 0
            feature_values.append(0.0)

    return feature_values
