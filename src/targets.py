from __future__ import annotations
import numpy as np
import pandas as pd

LABEL_TO_ID = {'sell': 0, 'hold': 1, 'buy': 2}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}

def build_target(df: pd.DataFrame, horizon: int = 12, buy_threshold: float = 0.004, sell_threshold: float = -0.004) -> pd.DataFrame:
    out = df.copy()
    future_ret = out.groupby('asset')['close'].shift(-horizon) / out['close'] - 1.0
    conditions = [future_ret <= sell_threshold, future_ret >= buy_threshold]
    choices = ['sell', 'buy']
    out['future_return'] = future_ret
    out['target_label'] = np.select(conditions, choices, default='hold')
    out['target'] = out['target_label'].map(LABEL_TO_ID)
    return out
