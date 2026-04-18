from __future__ import annotations
import numpy as np
import pandas as pd

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss.replace(0, np.nan))
    return 100 - (100 / (1 + rs))

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    g = out.groupby('asset', group_keys=False)

    for lag in [1, 3, 6, 12]:
        out[f'ret_{lag}'] = g['close'].pct_change(lag)

    for win in [5, 10, 20]:
        out[f'sma_{win}'] = g['close'].transform(lambda s: s.rolling(win).mean())
        out[f'ema_{win}'] = g['close'].transform(lambda s: s.ewm(span=win, adjust=False).mean())

    out['rsi_14'] = g['close'].transform(lambda s: compute_rsi(s, 14))
    out['vol_12'] = g['close'].pct_change().groupby(out['asset']).transform(lambda s: s.rolling(12).std())
    out['zscore_20'] = g['close'].transform(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
    out['range_pct'] = (out['high'] - out['low']) / out['close'].replace(0, np.nan)
    out['body_pct'] = (out['close'] - out['open']) / out['open'].replace(0, np.nan)
    out['volume_zscore_20'] = g['volume'].transform(lambda s: (s - s.rolling(20).mean()) / s.rolling(20).std())
    out['hour'] = out['timestamp'].dt.hour
    out['dayofweek'] = out['timestamp'].dt.dayofweek
    out['month'] = out['timestamp'].dt.month
    return out
